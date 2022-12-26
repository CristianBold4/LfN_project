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


if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.INFO)

    args, blocks, device = parse_parameters()

    W, n_sensors = load_weight_matrix(args.dataset)
    scaler, train_iter, val_iter, test_iter = load_data(args, device)

    # compute gso
    gso = U.compute_gso(W, args.gso_type)
    if args.gc_type == 'cheb_graph_conv':
        gso = U.compute_cheby_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    # prepare model for training
    model = STGCN_model(args, blocks, n_sensors).to(device)
    # print(blocks)
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
    es = early_stopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    train(loss, args, opt, scheduler, model, train_iter, val_iter, es)
    test(scaler, loss, model, test_iter, args)
