import torch
import torchvision.datasets as dsets

# torchvision.datasets 을 이용해 ImageNet 데이터셋을 관리
import torchvision.transforms as transforms

# MNIST등의 데이터셋에 들어있는 데이터들을 원하는 모양으로 변환할때 사용하는 모듈

from torch import FloatTensor
from torch import optim

# from torch import FloatTensor, optim 같이 한줄로 합쳐도 됨 (as로 새이름 정하지 않을경우)

from torch.optim.lr_scheduler import ReduceLROnPlateau

# 학습 진행이 느려지면 자동으로 lr값을 조정해주는 module

from torch.utils.data import DataLoader

# 미니배치 데이터 로딩을 도울 모듈

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import json

# n01443537 같이 되어있는 클래스 이름들을 goldfish 와 같이 쉽게 바꿔줄 때 사용할 파일이 JSON파일
import os

# os.path.join(save_path, filename) 으로 파일 경로 합칠 때 사용
import shutil

# shutil.copyfile(path_a, path_b) a 경로의 파일을 b 경로에 복사

import scipy


import torchsummary

# 모델 구조 표로 정리해서 보여주는 모듈
# torchsummary.summary(model, input_size=(3, 224, 224), batch_size=64) 와 같이 사용

from torchvision import models

# pretrained 된 모델들을 담고 있는 모듈

import torchvision.transforms.functional as visionF

# 이미지 표시에 쓰이는 visionF.to_pil_image(img) 함수등 여러 함수 포함

from torchvision.utils import make_grid

# 이미지들을 표시할 grid 생성


import time
import datetime

# 시간 측정에 사용

import math

import copy

# copy.deepcopy()로 깊은 복사 수행

from PClayers import *
from PCConvLayers import *
from PCFCLayers import *
from PCfunctions import *
from PCutils import *


class PCResBlock0(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        for i in reversed(range(1, self.num_xs - 1)):
            p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 2:
                err = self.Xs[i + 1] - p
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                    err = y_r - p
                else:
                    err = y - self.X_shortcuts["shortcut"] - p

            _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Xs[i], err)

            e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - p

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Xs[0], err)

        return dfdt

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            if i < self.num_xs - 2:
                p = self.residual[i](self.Xs[i].detach())
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                p = self.residual[i](self.Xs[i].detach())

                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.X_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# ‘fixed prediction assumption’ 미적용
# ===> 훈련 안됨
class PCResBlock(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        for i in reversed(range(1, self.num_xs - 1)):
            p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 2:
                err = self.Xs[i + 1] - p
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                    err = y_r - p
                else:
                    err = y - self.X_shortcuts["shortcut"] - p

            _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Xs[i], err)

            e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - p

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Xs[0], err)

        return dfdt

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            if i < self.num_xs - 2:
                p = self.residual[i](self.Xs[i].detach())
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                p = self.residual[i](self.Xs[i].detach())

                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.X_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산 안함
class PCResBlock2(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}
        self.U_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()
        self.Us[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.U_shortcuts["shortcut"] = self.X_shortcuts["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())
        self.U_shortcuts["add"] = self.X_shortcuts["add"].clone()

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        for i in reversed(range(1, self.num_xs - 1)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 2:
                # err = self.Xs[i + 1] - p
                err = self.Xs[i + 1] - self.Us[i + 1]
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                    err = y_r - self.Us[-1]
                else:
                    err = y - self.U_shortcuts["shortcut"] - self.Us[-1]

            _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        # p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - self.Us[1]

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        return dfdt

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.U_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산함
class PCResBlock3(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}
        self.U_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()
        self.Us[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.U_shortcuts["shortcut"] = self.X_shortcuts["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())
        self.U_shortcuts["add"] = self.X_shortcuts["add"].clone()

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.Xs[0] = x.clone()
        # self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        err = self.X_shortcuts["add"] - self.U_shortcuts["add"]
        u_concat = torch.cat((self.Us[-1], self.U_shortcuts["shortcut"]), dim=1)
        dx_r, dx_s = self.add.backward(u_concat, err)
        if self.concat:
            e_cur_s = self.X_shortcuts["shortcut"] - self.U_shortcuts["shortcut"]
            self.X_shortcuts["shortcut"] -= decay * infer_rate * (e_cur_s - dx_s)

            err_s = self.X_shortcuts["shortcut"] - self.U_shortcuts["shortcut"]
            dfdt_s = self.shortcut.backward(self.Us[0], err_s)
        else:
            dfdt_s = dx_s

        for i in reversed(range(1, self.num_xs)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 1:
                # err = self.Xs[i + 1] - p
                err = self.Xs[i + 1] - self.Us[i + 1]

                _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)
            else:
                dfdt = dx_r

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        # p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - self.Us[1]

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        return dfdt + dfdt_s

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.U_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산 안함
class PCDPBlock(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                DPConv(  # DPConv는 block에서 weight 초기화가 안되므로 class 내부에서 초기화하기
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                device=device,
            ),
            nn.Sequential(
                DPConv(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}
        self.U_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()
        self.Us[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.U_shortcuts["shortcut"] = self.X_shortcuts["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())
        self.U_shortcuts["add"] = self.X_shortcuts["add"].clone()

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        for i in reversed(range(1, self.num_xs - 1)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 2:
                # err = self.Xs[i + 1] - p
                err = self.Xs[i + 1] - self.Us[i + 1]
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                    err = y_r - self.Us[-1]
                else:
                    err = y - self.U_shortcuts["shortcut"] - self.Us[-1]

            _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        # p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - self.Us[1]

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        return dfdt

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.U_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산함
class PCDPBlock2(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                DPConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                device=device,
            ),
            nn.Sequential(
                DPConv(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}
        self.U_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()
        self.Us[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.U_shortcuts["shortcut"] = self.X_shortcuts["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())
        self.U_shortcuts["add"] = self.X_shortcuts["add"].clone()

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.Xs[0] = x.clone()
        # self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        err = self.X_shortcuts["add"] - self.U_shortcuts["add"]
        u_concat = torch.cat((self.Us[-1], self.U_shortcuts["shortcut"]), dim=1)
        dx_r, dx_s = self.add.backward(u_concat, err)
        if self.concat:
            e_cur_s = self.X_shortcuts["shortcut"] - self.U_shortcuts["shortcut"]
            self.X_shortcuts["shortcut"] -= decay * infer_rate * (e_cur_s - dx_s)

            err_s = self.X_shortcuts["shortcut"] - self.U_shortcuts["shortcut"]
            dfdt_s = self.shortcut.backward(self.Us[0], err_s)
        else:
            dfdt_s = dx_s

        for i in reversed(range(1, self.num_xs)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 1:
                # err = self.Xs[i + 1] - p
                err = self.Xs[i + 1] - self.Us[i + 1]

                _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)
            else:
                dfdt = dx_r

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        # p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - self.Us[1]

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        return dfdt + dfdt_s

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.U_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산함
# SeqIL 미적용
class PCDPBlock3(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                DPConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                device=device,
            ),
            nn.Sequential(
                DPConv(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}
        self.U_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()
        self.Us[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.U_shortcuts["shortcut"] = self.X_shortcuts["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())
        self.U_shortcuts["add"] = self.X_shortcuts["add"].clone()

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, y_old, decay, infer_rate):
        self.Xs[0] = x.clone()
        # self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        err = y_old - self.U_shortcuts["add"]
        u_concat = torch.cat((self.Us[-1], self.U_shortcuts["shortcut"]), dim=1)
        dx_r, dx_s = self.add.backward(u_concat, err)
        if self.concat:
            e_cur_s = self.X_shortcuts["shortcut"] - self.U_shortcuts["shortcut"]
            self.X_shortcuts["shortcut"] -= decay * infer_rate * (e_cur_s - dx_s)

            err_s = e_cur_s
            dfdt_s = self.shortcut.backward(self.Us[0], err_s)
        else:
            dfdt_s = dx_s

        for i in reversed(range(1, self.num_xs)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 1:
                # err = self.Xs[i + 1] - p
                err = e_cur

                _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)
            else:
                dfdt = dx_r

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        # p = self.residual[0](self.Xs[0])

        err = e_cur

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        return dfdt + dfdt_s

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.U_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산함
# SeqIL 미적용
# decay 제거
# dx에 torch.clamp 적용
class PCDPBlock4(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        # self.groups = 4
        # self.out_channels = out_channels
        self.out_channels = in_channels
        # output 채널의 개수를 conv로 늘리지 않고
        # x = torch.cat((x, x_shortcut), dim=1) 로 늘리기 때문에
        # in_channels로 변경

        # BatchNorm 과정에서 bias를 추가하기 때문에 conv layer에서는 bias를 빼준다
        # https://stats.stackexchange.com/questions/482305/batch-normalization-and-the-need-for-bias-in-neural-networks

        self.residual = nn.Sequential(
            nn.Sequential(
                DPConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                device=device,
            ),
            nn.Sequential(
                DPConv(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.out_channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(in_channels=in_channels, stride=stride, device=device)
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_shortcuts = {"shortcut": [], "add": []}
        self.U_shortcuts = {"shortcut": [], "add": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()
        self.Us[0] = x.clone()

        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.U_shortcuts["shortcut"] = self.X_shortcuts["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_shortcuts["shortcut"]), dim=1)
        self.X_shortcuts["add"] = self.add(x_concat.detach())
        self.U_shortcuts["add"] = self.X_shortcuts["add"].clone()

        return self.X_shortcuts["add"]

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        # print(f"==>> x_shortcut.shape: {x_shortcut.shape}")

        x = self.residual(x)
        # print(f"==>> x.shape: {x.shape}")

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.add(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, y_old, decay, infer_rate):
        self.Xs[0] = x.clone()
        # self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        err = y_old - self.U_shortcuts["add"]
        u_concat = torch.cat((self.Us[-1], self.U_shortcuts["shortcut"]), dim=1)
        dx_r, dx_s = self.add.backward(u_concat, err)
        if self.concat:
            e_cur_s = self.X_shortcuts["shortcut"] - self.U_shortcuts["shortcut"]
            self.X_shortcuts["shortcut"] -= infer_rate * (e_cur_s - torch.clamp(dx_s, -50, 50))

            err_s = e_cur_s
            dfdt_s = torch.clamp(self.shortcut.backward(self.Us[0], err_s), -50, 50)
        else:
            dfdt_s = torch.clamp(dx_s, -50, 50)

        for i in reversed(range(1, self.num_xs)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 1:
                # err = self.Xs[i + 1] - p
                err = e_cur

                _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)
                dfdt = torch.clamp(dfdt, -50, 50)
            else:
                dfdt = torch.clamp(dx_r, -50, 50)

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX

        # p = self.residual[0](self.Xs[0])

        err = e_cur

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        return torch.clamp(dfdt, -50, 50) + dfdt_s

    def update_weights(self, x, y):
        self.Xs[0] = x.clone()
        self.X_shortcuts["shortcut"] = self.shortcut(self.Xs[0].detach())
        self.X_shortcuts["add"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())
            else:
                if self.concat:
                    y_r, y_s = torch.split(
                        self.X_shortcuts["add"], self.X_shortcuts["add"].shape[1] // 2, dim=1
                    )
                else:
                    y_r = self.X_shortcuts["add"] - self.U_shortcuts["shortcut"]

                loss = 0.5 * self.MSE(p, y_r.detach())

            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()
            # losses.append(loss.item())

        # return losses
        # backward 후에 계산한 loss는 실제 loss값과는 다르다

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


# shortcut pass dfdt 계산함
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 학습안됨?
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class PCShuffleV2Block(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        stride=1,
        momentum=0.01,
        device="cpu",
        init_weights=True,
    ):
        # 다른 block이랑 똑같게 groups는 인수에서 일단 뺌
        super().__init__()

        self.shortcut = nn.Sequential()
        self.split = True
        # identity mapping

        self.channels = in_channels // 2

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCResBlock.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참

            self.split = False
            self.channels = in_channels

            self.shortcut = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.channels,
                        out_channels=self.channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=self.channels,
                        device=device,
                    ),
                    nn.BatchNorm2d(num_features=self.channels, momentum=momentum, device=device),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                        device=device,
                    ),
                    nn.BatchNorm2d(num_features=self.channels, momentum=momentum, device=device),
                    nn.ReLU(),
                ),
            )

        self.residual = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.channels,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.channels, momentum=momentum, device=device),
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    device=device,
                ),
                nn.BatchNorm2d(num_features=self.channels, momentum=momentum, device=device),
                nn.ReLU(),
            ),
        )

        self.shuffle = ShuffleLayer()

        self.learning_rate = learning_rate
        # self.optim = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.num_xs = len(self.residual) + 1
        self.Xs = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.Us = [torch.randn(1, 1) for i in range(self.num_xs)]
        self.X_dict = {"input": [], "shortcut": [], "residual": [], "shuffle": []}
        self.U_dict = {"input": [], "shortcut": [], "residual": [], "shuffle": []}

        self.optims = []
        for i in range(self.num_xs - 1):
            self.optims.append(optim.Adam(self.residual[i].parameters(), lr=self.learning_rate))

        self.num_shortcuts = len(self.shortcut) + 1
        self.optims_short = []
        self.X_shorts = [torch.randn(1, 1) for i in range(self.num_shortcuts)]
        self.U_shorts = [torch.randn(1, 1) for i in range(self.num_shortcuts)]
        if not self.split:
            for i in range(self.num_shortcuts - 2):
                self.optims_short.append(optim.Adam(self.shortcut[i].parameters(), lr=self.learning_rate))

            self.optims[-1] = optim.Adam(
                [{"params": self.residual[-1].parameters()}, {"params": self.shortcut[-1].parameters()}],
                lr=self.learning_rate,
            )

        self.MSE = nn.MSELoss(reduction="sum").to(device)

        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.X_dict["input"] = x.clone()
        self.U_dict["input"] = x.clone()

        if self.split:
            x = torch.split(x, x.shape[1] // 2, dim=1)
            self.Xs[0] = x[0].clone()
            self.Us[0] = x[0].clone()
            self.X_shorts[0] = x[1].clone()
            self.U_shorts[0] = x[1].clone()

            self.X_dict["shortcut"] = self.shortcut(self.Xs[0].detach())
            self.U_dict["shortcut"] = self.X_dict["shortcut"].clone()
        else:
            self.Xs[0] = x.clone()
            self.Us[0] = x.clone()
            self.X_shorts[0] = x.clone()
            self.U_shorts[0] = x.clone()

            for i in range(1, self.num_shortcuts):
                self.X_shorts[i] = self.shortcut[i - 1](self.X_shorts[i - 1].detach())
                self.U_shorts[i] = self.X_shorts[i].clone()
            self.X_dict["shortcut"] = self.X_shorts[-1].clone()
            self.U_dict["shortcut"] = self.X_dict["shortcut"].clone()

        for i in range(1, self.num_xs):
            self.Xs[i] = self.residual[i - 1](self.Xs[i - 1].detach())
            self.Us[i] = self.Xs[i].clone()
        # self.X_dict["residual"] = self.Xs[-1].clone()
        # self.U_dict["residual"] = self.X_dict["residual"].clone()

        x_concat = torch.cat((self.Xs[-1], self.X_dict["shortcut"]), dim=1)
        self.X_dict["shuffle"] = self.shuffle(x_concat.detach())
        self.U_dict["shuffle"] = self.X_dict["shuffle"].clone()

        return self.X_dict["shuffle"]

    def forward(self, x):
        if self.split:
            x = torch.split(x, x.shape[1] // 2, dim=1)
            x_shortcut = self.shortcut(x[1])

            x = self.residual(x[0])
        else:
            x_shortcut = self.shortcut(x)

            x = self.residual(x)

        x_concat = torch.cat((x, x_shortcut), dim=1)
        x = self.shuffle(x_concat)

        return x

    @torch.no_grad()
    def backward(self, x, y, decay, infer_rate):
        self.X_dict["shuffle"] = y.clone()

        err = self.X_dict["shuffle"] - self.U_dict["shuffle"]
        u_concat = torch.cat((self.Us[-1], self.U_dict["shortcut"]), dim=1)
        dx_r, dx_s = self.shuffle.backward(u_concat, err)

        # self.X_dict["input"] = x.clone()

        for i in reversed(range(1, self.num_xs)):
            # p = self.residual[i](self.Xs[i])

            if i < self.num_xs - 1:
                # err = self.Xs[i + 1] - p
                err = self.Xs[i + 1] - self.Us[i + 1]

                _, dfdt = torch.autograd.functional.vjp(self.residual[i], self.Us[i], err)
            else:
                dfdt = dx_r

            # e_cur = self.Xs[i] - self.residual[i - 1](self.Xs[i - 1])
            e_cur = self.Xs[i] - self.Us[i]
            dX = decay * infer_rate * (e_cur - dfdt)
            self.Xs[i] -= dX
        # p = self.residual[0](self.Xs[0])

        err = self.Xs[1] - self.Us[1]

        _, dfdt = torch.autograd.functional.vjp(self.residual[0], self.Us[0], err)

        if self.split:
            return torch.cat((dfdt, dx_s), dim=1)
        else:
            for i in reversed(range(1, self.num_shortcuts)):
                if i < self.num_shortcuts - 1:
                    err_s = self.X_shorts[i + 1] - self.U_shorts[i + 1]

                    _, dfdt_s = torch.autograd.functional.vjp(self.shortcut[i], self.U_shorts[i], err_s)
                else:
                    dfdt_s = dx_s

                e_cur_s = self.X_shorts[i] - self.U_shorts[i]
                dX = decay * infer_rate * (e_cur_s - dfdt_s)
                self.X_shorts[i] -= dX

            self.X_dict["shortcut"] = self.X_shorts[-1].clone()

            err_s = self.X_shorts[1] - self.U_shorts[1]
            _, dfdt_s = torch.autograd.functional.vjp(self.shortcut[0], self.U_shorts[0], err_s)
            return dfdt + dfdt_s

    def update_weights(self, x, y):
        if self.split:
            x = torch.split(x, x.shape[1] // 2, dim=1)
            self.Xs[0] = x[0].clone()
            self.X_shorts[0] = x[1].clone()
        else:
            self.Xs[0] = x.clone()
            self.X_shorts[0] = x.clone()

        self.X_dict["shuffle"] = y.clone()

        # losses = []

        for i in range(self.num_xs - 1):
            p = self.residual[i](self.Us[i].detach())
            if i < self.num_xs - 2:
                loss = 0.5 * self.MSE(p, self.Xs[i + 1].detach())

                self.optims[i].zero_grad()
                loss.backward()
                self.optims[i].step()

        if self.split:
            p = torch.cat((p, self.U_dict["shortcut"]), dim=1)
            p = self.shuffle(p)
            loss = 0.5 * self.MSE(p, self.X_dict["shuffle"].detach())

            self.optims[-1].zero_grad()
            loss.backward()
            self.optims[-1].step()
        else:
            for i in range(self.num_shortcuts - 1):
                p_s = self.shortcut[i](self.U_shorts[i].detach())
                if i < self.num_shortcuts - 2:
                    loss_s = 0.5 * self.MSE(p_s, self.X_shorts[i + 1].detach())

                    self.optims_short[i].zero_grad()
                    loss_s.backward()
                    self.optims_short[i].step()

            p_c = torch.cat((p, p_s), dim=1)
            p_c = self.shuffle(p_c)

            loss = 0.5 * self.MSE(p_c, self.X_dict["shuffle"].detach())

            self.optims[-1].zero_grad()
            loss.backward()
            self.optims[-1].step()

    def set_lr(self, new_lr):
        for optim in self.optims:
            for param_group in optim.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Conv2d):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # mode 의 'fan_in' 과 'fan_out' 에 대한 설명
                # fan_in은 여러번 forward 방향 계산을 반복할 때 weights 에 곱해서 얻어진 output에 활성함수를 적용할 때
                # 정해진 활성함수가 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것이고
                # fan_out은 반대로 backprop 방향 계산을 반복할 때 활성함수의 미분이 잘 작동하는 범위에 계속 있도록 weights 초기화를 하는 것
                # so all in all it doesn't matter much but it's more about what you are after.
                # I assume that if you suspect your backward pass might be more "chaotic" (greater variance)
                # it is worth changing the mode to fan_out. This might happen when the loss oscillates a lot
                # (e.g. very easy examples followed by very hard ones).

                # nn.init.normal_(m.weight, mean=0.0, std=10.0)

                # https://nittaku.tistory.com/269
                # https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화
