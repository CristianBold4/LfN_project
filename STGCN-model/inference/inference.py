from data_loader.data_loader import *
from model.model import *
import matplotlib.pyplot as plt


def run_inference(args, blocks, n_sensors, device, n_day=1, n_route=0, day_slot=288):
    """
    Function to run inference and plot the predicted vs true speed graphic
    :param args: args parameters
    :param blocks: channel blocks of the model
    :param n_sensors: number of routes = sensors (= size of adjacency matrix)
    :param device: CUDA/CPU device
    :param n_day: number of day to plot
    :param n_route: number of route = sensor to plot
    :param day_slot: number of slots in a day (usually 5 min slot = 288 slots in a day)
    :return:
    """
    best_model = STGCN_model(args, blocks, n_sensors).to(device)
    best_model.load_state_dict(torch.load(args.savepath))

    # reload data with batch_size = 1, more suitable for report's plots
    args.batch_size = 1
    scaler, _, _, test_iter = load_data(args, device)

    pred_speeds = []
    true_speeds = []

    best_model.eval()
    with torch.no_grad():
        for x, y in test_iter:
            y_pred = best_model(x).view(len(x), -1)
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(y_pred.cpu().numpy()).reshape(-1)
            # print(f'True: {y[n_route]}, predicted: {y_pred[n_route]}')
            pred_speeds.append(y_pred[n_route])
            true_speeds.append(y[n_route])

    day_true = []
    day_pred = []

    for i in range(n_day*day_slot, (n_day+1)*day_slot, 1):
        day_true.append(true_speeds[i])
        day_pred.append(pred_speeds[i])

    day = np.arange(day_slot).tolist()
    l1, = plt.plot(day, day_pred, color='red')
    l2, = plt.plot(day, day_true, color='blue')

    plt.legend((l1, l2), ["Predicted speed", "True speed"])

    plt.xlabel("Day time")
    plt.ylabel("Speed [mph]")
    plt.xticks(np.arange(min(day) + 4*12, max(day)+2, 6*12), labels=('04:00', '10:00', '16:00', '22:00'))

    plt.show()





