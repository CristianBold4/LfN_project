import argparse
import torch


def parse_parameters():
    """
    Function to parse parameters and to compute channel blocks
    :return: args parsed, channel blocks computed, device used
    """
    parser = argparse.ArgumentParser(description='STGCN_model')
    parser.add_argument('--M', type=int, default=12, help='Number of slot of historical data to use, default = 12')
    parser.add_argument('--H', type=int, default=3, help='Number of steps away in the future in which make the '
                                                         'prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default = 32')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs, default = 100')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate, default = 0.001')
    parser.add_argument('--opt', type=str, default='adam', help='Optimization method, default = adam')
    # parser.add_argument('--adj', type=str, help='Path to the adjacency matrix', default='./dataset/PeMSD7_W_228.csv')
    parser.add_argument('--dataset', type=str, help='name of the dataset', choices=['pemsd7-m', 'metr-la', 'pems-bay'],
                        default='pemsd7-m')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap',
                                                                                 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA", default=False)
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience")
    parser.add_argument("--Kt", type=int, default=3, help='Temporal kernel size')
    parser.add_argument('--Ks', type=int, default=3, help='Spatial kernel size')
    parser.add_argument('--st_blocks', type=int, default=2, help='Number of spatio temporal blocks')
    parser.add_argument('-gc_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--savepath', type=str, default='STGCN_model.pt', help="Path to save the model")
    parser.add_argument('--inference', type=bool, default=0, help="Flag to run inference with saved model")

    args = parser.parse_args()

    # print(f'Training configs: {args}')
    if torch.cuda.is_available() and not args.disable_cuda:
        print(f'GPU is available: {torch.cuda.get_device_name(0)}')
    else:
        print('GPU NOT available')

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    Ko = args.M - (args.Kt - 1) * 2 * args.st_blocks

    # build the channel sizes blocks using the bottleneck design
    blocks = [[1]]
    for b in range(args.st_blocks):
        blocks.append([64, 16, 64])

    blocks.append([128, 128]) if Ko > 0 else blocks.append([128])
    blocks.append([1])

    # Default blocks design -> [[1], [64, 16, 64], [64, 16, 64], [128, 128], [1], where
    # [1] is the channel size (i.e., the quantity to predict, e.g., only the speed = 1)
    # [64, 16, 64] are the channels for a single ST blocks, composed by two Temporal Blocks and a Spatial Conv block
    # [128, 128] ???

    return args, blocks, device
