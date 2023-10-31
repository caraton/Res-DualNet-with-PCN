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


class DPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        device="cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=2 * self.out_channels,
            groups=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            device=device,
        )

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        output_size = math.floor((x.size(2) + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        x = self.conv(x).reshape(-1, self.out_channels, 2, output_size, output_size)
        # self.conv(x)의 size는 (batch_size, 2 * self.out_channels, output_size, output_size)
        # input channel 하나 마다 1x3x3 필터를 2개 적용해서 output channel 2개씩 생성

        # reshape로 (batch_size, self.out_channels, 2, output_size, output_size) 형태로 변경

        x = torch.sum(x, dim=2)
        # 각 input channel 별 2개씩 있던 output을 합쳐서
        # 각 input channel 마다 output channel 한개로 변경
        # (batch_size, self.out_channels, output_size, output_size) 형태

        return x


class ConvAG(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        learning_rate,
        stride=1,
        padding=1,
        groups=1,
        # bias=False,
        # BN 할때는 conv bias 무조건 False
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.eps = 0.00001
        self.momentum = momentum

        self.device = device

        # self.conv = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.in_channels,
        #         out_channels=self.out_channels,
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         padding=self.padding,
        #         bias=False,
        #         device=self.device,
        #     ),
        #     nn.BatchNorm2d(
        #         num_features=self.out_channels, eps=self.eps, momentum=self.momentum, device=self.device
        #     ),
        #     nn.ReLU(),
        # )

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=False,
            device=self.device,
        )

        self.BN = nn.BatchNorm2d(
            num_features=self.out_channels, eps=self.eps, momentum=self.momentum, device=self.device
        )

        self.relu = nn.ReLU()

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)
        self.MSE = nn.MSELoss(reduction="sum").to(self.device)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print(f"==>> forward x.requires_grad: {x.requires_grad}")
        x = self.conv(x)
        # print(f"==>> forward conv(x).grad_fn: {x.grad_fn}")

        x = self.BN(x)
        # print(f"==>> forward BN(x).grad_fn: {x.grad_fn}")
        x = self.relu(x)
        # print(f"==>> forward relu(x).grad_fn: {x.grad_fn}")

        return x

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        return dfdt

    def update_weights(self, x, y):
        # print(f"==>> uw x.requires_grad: {x.requires_grad}")
        x = x.detach().clone().requires_grad_(requires_grad=True)
        # print(f"==>> uw x.requires_grad: {x.requires_grad}")

        p = self.forward(x)
        # print(f"==>> uw x.requires_grad: {x.requires_grad}")
        # print(f"==>> uw p.grad_fn: {p.grad_fn}")
        loss = 0.5 * self.MSE(p, y)
        # print(f"==>> uw loss.grad_fn: {loss.grad_fn}")
        # loss.requires_grad = True
        loss_mean = loss / p.size(0)
        # print(f"==>> uw loss.grad_fn: {loss.grad_fn}")
        # p.size(0) == batch_size

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # return loss.item()
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def _initialize_weights(self):
        # nn.init.kaiming_normal_(self.conv[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        # if self.conv[0].bias is not None:
        if self.conv.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            # nn.init.constant_(self.conv[0].bias, 0)
            nn.init.constant_(self.conv.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    def set_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group["lr"] = new_lr


class DualPathConvAG(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        learning_rate,
        stride=1,
        padding=1,
        # bias=False,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=2 * self.out_channels,
            groups=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            device=self.device,
        )

        self.eps = 0.00001
        self.momentum = momentum

        self.BN = nn.BatchNorm2d(
            num_features=self.out_channels, eps=self.eps, momentum=self.momentum, device=self.device
        )
        self.relu = nn.ReLU()

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)
        self.MSE = nn.MSELoss(reduction="sum").to(self.device)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        output_size = math.floor((x.size(2) + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        x = self.conv(x).reshape(-1, self.out_channels, 2, output_size, output_size)
        # self.conv(x)의 size는 (batch_size, 2 * self.out_channels, output_size, output_size)
        # input channel 하나 마다 1x3x3 필터를 2개 적용해서 output channel 2개씩 생성

        # reshape로 (batch_size, self.out_channels, 2, output_size, output_size) 형태로 변경

        x = torch.sum(x, dim=2)
        # 각 input channel 별 2개씩 있던 output을 합쳐서
        # 각 input channel 마다 output channel 한개로 변경
        # (batch_size, self.out_channels, output_size, output_size) 형태

        x = self.BN(x)
        x = self.relu(x)

        return x

    def backward(self, x, e):
        _, dfdt = torch.autograd.functional.vjp(self.forward, x, e)

        return dfdt

    def update_weights(self, x, y):
        x = x.detach().clone().requires_grad_(requires_grad=True)

        p = self.forward(x)
        loss = 0.5 * self.MSE(p, y)
        # loss.requires_grad = True
        loss_mean = loss / p.size(0)
        # p.size(0) == batch_size

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), loss_mean.item()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.conv.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            nn.init.constant_(self.conv.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    def set_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group["lr"] = new_lr
