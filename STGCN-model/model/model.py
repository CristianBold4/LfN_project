import torch
import torch.nn as nn
from model.layers import STConvBlock, OutputBlock
import tqdm
import numpy as np


class STGCN_model(nn.Module):
    """
    Composed by ST blocks + 1 Output block with Chebyshev graph convolution
    """

    def __init__(self, args, blocks, n_sensors):
        super(STGCN_model, self).__init__()
        modules = []
        for b in range(len(blocks) - 3):
            modules.append(STConvBlock(args.Kt, args.Ks, n_sensors, blocks[b][-1], blocks[b + 1], args.gc_type,
                                       args.gso, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.M - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_sensors, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0])
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0])
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x


def train(loss, args, opt, scheduler, model, train_iter, val_iter, es):
    """
    Function to train the model and save to args.savepath the best model found
    :param loss: the loss function
    :param args: args parameters
    :param opt: optimizer
    :param scheduler: scheduler
    :param model: model to train
    :param train_iter: preprocessed torch train dataset
    :param val_iter: preprocessed torch validation dataset
    :param es: early stopping
    :return: the minimum validation loss
    """
    min_val_loss = 1
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            opt.zero_grad()
            l.backward()
            opt.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val(model, val_iter, loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), args.savepath)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000
        train_losses.append(l_sum / n)
        val_losses.append(val_loss)
        print('Epoch: {:03d} | lr: {:.20f} | Train loss: {:.6f} | Val loss {:.6f} | GPU occupy: {:.6f} Mib'.\
              format(epoch + 1, opt.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if es.step(val_loss):
            print('Early stopping.')
            break

    # # Train loss plot
    # epochs = np.arange(epoch + 1).tolist()
    #
    # # plt.plot(epochs, train_losses, color="blue", label="Train Loss")
    # # plt.xlabel("Epochs")
    # # plt.ylabel("Training Loss")
    # # plt.title("Training loss plot")
    # #
    # # plt.show()

    return min_val_loss


@torch.no_grad()
def val(model, val_iter, loss):
    """
    Function to perform validation on the model
    :param model: the model
    :param val_iter: torch preprocessed validation dataset
    :param loss: loss function
    :return: validation loss
    """
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


def test(scaler, loss, model, test_iter, args):
    """
    Function to perform test on the model and compute metrics
    :param scaler: scaler to rescaler the data
    :param loss: loss function
    :param model: the model
    :param test_iter: preprocessed torch test dataset
    :param args: args parameters
    :return: performance metrics MAE, MAPE, WMAPE, RMSE
    """
    model.eval()
    # compute test loss as MSE
    l_sum, n = 0.0, 0
    mae, sum_y, mape, mse = [], [], [], []
    with torch.no_grad():
        for x, y in test_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(y_pred.cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()

    MAE = np.array(mae).mean()
    RMSE = np.sqrt(np.array(mse).mean())
    # -- weighted MAPE
    WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y)) * 100
    MAPE = np.array(mape).mean() * 100
    test_loss = l_sum / n

    print(f'Dataset {args.dataset} | Test Loss {test_loss:.6f} | MAE {MAE:.6f} | MAPE {MAPE:.6f} | WMAPE {WMAPE:.6f}'
          f'| RMSE {RMSE:.6f}')

    return test_loss
