import logging

import numpy as np
from utils.parser import parse_parameters
from data_loader.data_loader import *
from model.model import *
import utils.utils as U
import torch.nn as nn
import torch.optim as optim

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
    # print(model)
    loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.95)

    train(loss, args, optimizer, scheduler, model=model, train_iter=train_iter, val_iter=val_iter)
    test(scaler, loss, model, test_iter, args)










