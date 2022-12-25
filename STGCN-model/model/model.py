import torch
import torch.nn as nn
import torch.optim as optim
from model.layers import STConvBlock, OutputBlock
import tqdm
import numpy as np


class STGCN_model(nn.Module):
    """
    Composed by 2 ST blocks + 1 Output block with Chebyshev graph convolution
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


def train(loss, args, opt, scheduler, model, train_iter, val_iter):
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
        val_loss = compute_loss(loss, model, val_iter)
        print('Epoch: {:03d} | lr: {:.20f} | Train loss: {:.6f} | Val loss {:.6f}'.\
              format(epoch+1, opt.param_groups[0]['lr'], l_sum / n, val_loss))


def compute_loss(loss, model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return l_sum / n


def test(scaler, loss, model, test_iter, args):
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
    MAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y)) * 100
    test_loss = l_sum / n

    print(f'Dataset {args.dataset} | Test Loss {test_loss:.6f} | MAE {MAE:.6f} | MAPE {MAPE:.6f} | RMSE {RMSE:.6f}')




