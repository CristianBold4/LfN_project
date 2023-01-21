import logging

import numpy as np
from utils.parser import parse_parameters
from data_loader.data_loader import *
from model.model import *
from utils import early_stopping

import utils.utils as U
import torch.nn as nn
import torch.optim as optim
import random


def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for a multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.INFO)

    set_env(42)

    args, blocks, device = parse_parameters()

    print(f'Training configs: {args}')

    W, n_sensors = load_weight_matrix(args.dataset)
    scaler, train_iter, val_iter, test_iter = load_data(args, device)

    # compute gso
    gso = U.compute_gso(W, args.gso_type)
    if args.gc_type == 'cheb_graph_conv':
        gso = U.compute_cheby_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    # batch_sizes = [16, 32, 64]
    # st_blocks = [2, 3]
    # lrs = [0.01, 0.005, 0.001]
    # K = [2, 3]
    # best_loss = 1
    # best_params = args
    #
    # print('Starting Grid Search')
    #
    # for b in batch_sizes:
    #     for st in st_blocks:
    #         for lr in lrs:
    #             for Kt in K:
    #                 for Ks in K:
    #
    #                     args.batch_size = b
    #                     args.st_blocks = st
    #                     args.lr = lr
    #                     args.Kt = Kt
    #                     args.Ks = Ks
    #                     print(f'Training configs: {args}')
    #
    #                     # prepare model for training
    #                     model = STGCN_model(args, blocks, n_sensors).to(device)
    #                     loss = nn.MSELoss()
    #                     opt = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4, amsgrad=False)
    #                     scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
    #                     es = early_stopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)
    #
    #                     train(loss, args, opt, scheduler, model, train_iter, val_iter, es)
    #                     test_loss = test(scaler, loss, model, test_iter, args)
    #                     if test_loss < best_loss:
    #                         best_loss = test_loss
    #                         best_params = args

    # prepare model for training
    model = STGCN_model(args, blocks, n_sensors).to(device)
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
    es = early_stopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    train(loss, args, opt, scheduler, model, train_iter, val_iter, es)
    test_loss = test(scaler, loss, model, test_iter, args)

