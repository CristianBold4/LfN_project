import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data
import math

from sklearn import preprocessing

data_folder = './datasets'


def load_weight_matrix(dataset_name):
    """
    Function that loads the weight matrix
    :param dataset_name: path to dataset
    :return: weight matrix, number of sensor (= adj.shape[0])
    """
    dataset_path = os.path.join(data_folder, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    n = adj.shape[0]
    # convert adj to Compressed Sparse Column format
    adj = adj.tocsc()
    return adj, n


def data_transform(data, M, H, device):
    """
    Function that preprocess data in a suitable way for training, given M, H
    :param data: respective dataset
    :param M: M = number of historical slot
    :param H: H = number of steps to predict
    :param device: CUDA/CPU device
    :return: processed X, Y for the respective dataset
    """
    n_sensors = data.shape[1]
    total_len = len(data)
    window_len = total_len - M - H

    X = np.zeros([window_len, 1, M, n_sensors])
    Y = np.zeros([window_len, n_sensors])

    for i in range(window_len):
        start = i
        end = i + M
        X[i, :, :, :] = data[start: end].reshape(1, M, n_sensors)
        Y[i] = data[end + H - 1]

    return torch.Tensor(X).to(device), torch.Tensor(Y).to(device)


def load_data(args, device):
    """
    Function that loads and preprocesses the data
    :param args: args parameters
    :param device: CUDA/GPU device
    :return: scaler: the scaler used to standardize the data, train iterations
    validation iterations and test iterations datasets.
    """
    print(f'Loading dataset {args.dataset}...')
    M = args.M
    H = args.H

    # load velocities data from csv
    dataset_path = os.path.join(data_folder, args.dataset)
    df = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    # train-val-test split using (70, 15, 15)
    len_train = int(math.floor(df.shape[0] * 0.7))
    len_val = int(math.floor(df.shape[0] * 0.15))
    len_test = df.shape[0] - len_train - len_val

    # print(f'{len_train}, {len_val}, {len_test}')

    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]

    # scale the data
    scaler = preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    # prepare the data and labels
    X_train, Y_train = data_transform(train, M, H, device)
    X_val, Y_val = data_transform(val, M, H, device)
    X_test, Y_test = data_transform(test, M, H, device)

    train_data = torch.utils.data.TensorDataset(X_train, Y_train)
    train_iter = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = torch.utils.data.TensorDataset(X_val, Y_val)
    val_iter = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)
    test_iter = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return scaler, train_iter, val_iter, test_iter
