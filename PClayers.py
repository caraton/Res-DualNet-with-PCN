import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

# copy.deepcopy()로 깊은 복사 수행

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from PCfunctions import *


class AvgPoolLayer(nn.Module):
    def __init__(self, in_channels, stride, device="cpu"):
        super().__init__()
        self.in_channels = in_channels
        # self.kernel_size = kernel_size
        self.stride = stride
        self.device = device

        self.avg = nn.AvgPool2d(kernel_size=3, stride=self.stride, padding=1)

    def forward(self, x):
        x = self.avg(x)
        # x의 size는 (batch_size, input_channel_number, input_size, input_size)

        return x

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        return dfdt

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, x, y):
        return 0

    def _initialize_weights(self):
        pass


class ShortcutPath(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        return dfdt

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, x, y):
        return 0

    def _initialize_weights(self):
        pass


class AddLayer(nn.Module):
    def __init__(self, concat):
        super().__init__()
        self.concat = concat

    def forward(self, x):
        if self.concat:
            self.pre_activation = x
        else:
            w = torch.split(x, x.shape[1] // 2, dim=1)
            self.pre_activation = w[0] + w[1]

        return relu(self.pre_activation)

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        dX = torch.split(dfdt, dfdt.shape[1] // 2, dim=1)
        dX1 = dX[0]
        dX2 = dX[1]

        return dX1, dX2

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, x, y):
        return 0

    def _initialize_weights(self):
        pass


class ShuffleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.CS = nn.ChannelShuffle(groups=2)
        # pytorch CUDA에 nn.ChannelShuffle이 아직 구현안됨

    def forward(self, x):
        x = torch.split(x, x.shape[1] // 4, dim=1)
        x = torch.cat((x[0], x[2], x[1], x[3]), dim=1)

        return x

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        dX = torch.split(dfdt, dfdt.shape[1] // 2, dim=1)
        dX1 = dX[0]
        dX2 = dX[1]

        return dX1, dX2

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, x, y):
        return 0

    def _initialize_weights(self):
        pass


class AdaptiveAvgPoolLayer(nn.Module):
    def __init__(self, in_channels, device="cpu"):
        super().__init__()
        self.in_channels = in_channels
        # self.kernel_size = kernel_size
        self.device = device

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.avg(x)
        # x의 size는 (batch_size, input_channel_number, 1, 1)
        x = self.flat(x)
        # x의 size는 (batch_size, input_channel_number)

        return x

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        return dfdt

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, x, y):
        return 0

    def _initialize_weights(self):
        pass
