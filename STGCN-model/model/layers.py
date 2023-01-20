import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.utils import _pair


class Align(nn.Module):
    """
    Used to align the input channel numbers and the output channel numbers
    """

    def __init__(self, in_channels, out_channels):
        super(Align, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        if self.in_channels > self.out_channels:
            x = self.align_conv(x)
        elif self.in_channels < self.out_channels:
            batch_size, _, timestep, n_sensors = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.out_channels - self.in_channels, timestep,
                                           n_sensors]).to(x)], dim=1)
        else:
            x = x

        return x


class CausalConv1d(nn.Conv1d):
    """
    Causal convolutions are a type of convolution used for temporal data which ensures the model cannot violate the
    ordering in which we model the data
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=False, dilation=1):
        if padding:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.__padding, dilation=dilation)

    def forward(self, x):
        output = super(CausalConv1d, self).forward(x)
        if self.__padding != 0:
            return output[:, :, : -self.__padding]

        return output


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=False, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if padding:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = _pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation)

    def forward(self, x):
        if self.__padding != 0:
            x = F.pad(x, (self.left_padding[1], 0, self.left_padding[0], 0))
        output = super(CausalConv2d, self).forward(x)

        return output


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, in_channels, out_channels, n):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.align = Align(in_channels, out_channels)
        self.conv = CausalConv2d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=(Kt, 1),
                                 padding=False, dilation=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_aligned = self.align(x)[:, :, self.Kt - 1:, :]
        x_conv = self.conv(x)

        P = x_conv[:, :self.out_channels, :, :]
        Q = x_conv[:, -self.out_channels:, :, :]

        # GLU: P * sigmoid(Q)
        output = torch.mul((P + x_aligned), self.sigmoid(Q))

        return output


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, Ks, gso):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        # init parameters
        init.kaiming_normal_(self.weight)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1/math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        return torch.add(second_mul, self.bias)


class ChebGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, Ks, gso):
        super(ChebGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        # init parameters
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1/math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: Ks must be a positive integer, received {self.Ks}')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k-1]) - x_list[k-2])

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        return torch.add(cheb_graph_conv, self.bias)


class GraphConvLayer(nn.Module):
    def __init__(self, gc_type, in_channels, out_channels, Ks, gso):
        super(GraphConvLayer, self).__init__()
        self.gc_type = gc_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align = Align(in_channels, out_channels)
        self.Ks = Ks
        self.gso = gso
        if gc_type == 'cheb_graph_conv':
            self.gc = ChebGraphConv(out_channels, out_channels, Ks, gso)
        elif gc_type == 'graph_conv':
            self.gc = GraphConv(out_channels, out_channels, Ks, gso)
        else:
            raise ValueError(f'Unidentified graph convolutional type')

    def forward(self, x):
        x_gc_in = self.align(x)
        x_gc = self.gc(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        output = torch.add(x_gc, x_gc_in)
        return output


class STConvBlock(nn.Module):
    """
    STConv Block contains 'TGTND' structure
    - T: Gated Temporal Convolution Layer
    - G: Graph Convolutional Layer using Chebyshev Approximation
    - N: Normalization Layer
    - D: Dropout
    """

    def __init__(self, Kt, Ks, n_sensors, last_block_channel, channels, gc_type, gso, dropout_rate):
        super(STConvBlock, self).__init__()
        self.temp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_sensors)
        self.graph_conv = GraphConvLayer(gc_type, channels[0], channels[1], Ks, gso)
        self.temp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_sensors)
        self.norm_layer = nn.LayerNorm([n_sensors, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.temp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.temp_conv2(x)
        x = self.norm_layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):
    """
    Output block contains 'TNFF' structure
    - T: Gated Temporal Convolutional Layer
    - N: Normalization Layer
    - F: Fully-Connected Layer
    """

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_sensors, dropout_rate):
        super(OutputBlock, self).__init__()
        self.temp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_sensors)
        self.norm_layer = nn.LayerNorm([n_sensors, channels[0]])
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1])
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.temp_conv1(x)
        x = self.norm_layer(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x