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
from PClayersBN import *
from PCfunctions import *


class PCBlockTest(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self, in_channels, out_channels, input_size, learning_rate, stride=1, momentum=None, device="cpu"
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

        self.DP1 = DualPathConvLayer(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            input_size=input_size,
            learning_rate=learning_rate,
            stride=stride,
            f=relu,
            df=d_relu,
            device=device,
        )
        # self.BN1 = nn.BatchNorm2d(self.out_channels)
        # self.relu = RELU()

        self.PW = ConvLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            input_size=input_size // stride,
            learning_rate=learning_rate,
            stride=1,
            padding=0,
            device=device,
        )

        self.DP2 = DualPathConvLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            input_size=input_size // stride,
            learning_rate=learning_rate,
            stride=1,
            f=relu,
            df=d_relu,
            device=device,
        )

        # self.BN2 = nn.BatchNorm2d(self.out_channels)

        # self.relu = RELU()

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCBlockTest.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(
                in_channels=in_channels, input_size=input_size, stride=stride, device=device
            )
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.Xs = {"input": [], "DP1": [], "DP2": [], "PW": [], "shortcut": [], "add": []}
        self.Us = {"input": [], "DP1": [], "DP2": [], "PW": [], "shortcut": [], "add": []}
        # self.Es = {'input':[], 'DP1':[], 'DP2':[], 'PW':[], 'shortcut':[], 'add':[]}
        self.pred_errors = {"input": [], "DP1": [], "DP2": [], "PW": [], "shortcut": [], "add": []}

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs["input"] = x.clone()

        x_shortcut = self.shortcut(self.Xs["input"].detach())
        self.Xs["shortcut"] = x_shortcut

        x = self.DP1(self.Xs["input"].detach())
        self.Xs["DP1"] = x
        x = self.PW(x.detach())
        self.Xs["PW"] = x
        x = self.DP2(x.detach())
        self.Xs["DP2"] = x

        x = self.add(x.detach(), x_shortcut.detach())
        self.Xs["add"] = x

        return x

    def forward(self, x):
        x_shortcut = self.shortcut(x)

        x = self.DP1(x)
        # x = self.BN1(x)
        # x = self.relu(x)

        x = self.PW(x)

        # y = torch.zeros_like(x).to(device)
        # for i, x_s in enumerate(torch.split(x, 2 * self.out_channels // self.groups, dim=1)):
        #     y += self.DWs[i](x_s)

        x = self.DP2(x)

        # x = self.BN2(y)
        # x = self.relu(x)

        x = self.add(x, x_shortcut)

        return x

    # PCNet infer에서 매번 수행
    @torch.no_grad()
    def prediction(self, e):
        # self.Us['input'] = u.clone()
        self.pred_errors["input"] = e

        self.Us["shortcut"] = self.shortcut(self.Xs["input"])
        self.pred_errors["shortcut"] = self.Xs["shortcut"] - self.Us["shortcut"]

        self.Us["DP1"] = self.DP1(self.Xs["input"])
        self.pred_errors["DP1"] = self.Xs["DP1"] - self.Us["DP1"]

        self.Us["PW"] = self.PW(self.Xs["DP1"])
        self.pred_errors["PW"] = self.Xs["PW"] - self.Us["PW"]

        self.Us["DP2"] = self.DP2(self.Xs["PW"])
        self.pred_errors["DP2"] = self.Xs["DP2"] - self.Us["DP2"]

        self.Us["add"] = self.add(self.Xs["DP2"], self.Xs["shortcut"])

        return self.Us["add"]

    @torch.no_grad()
    def infer(self, x, decay, infer_rate):
        self.Xs["add"] = x.clone()
        self.pred_errors["add"] = self.Xs["add"] - self.Us["add"]

        dX1, dX2 = self.add.backward(self.pred_errors["add"])

        self.Xs["DP2"] -= decay * infer_rate * (self.pred_errors["DP2"] - dX1)
        self.Xs["shortcut"] -= decay * infer_rate * (self.pred_errors["shortcut"] - dX2)
        # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
        self.pred_errors["DP2"] = self.Xs["DP2"] - self.Us["DP2"]
        self.pred_errors["shortcut"] = self.Xs["shortcut"] - self.Us["shortcut"]

        dX = self.DP2.backward(self.pred_errors["DP2"])

        self.Xs["PW"] -= decay * infer_rate * (self.pred_errors["PW"] - dX)
        # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
        self.pred_errors["PW"] = self.Xs["PW"] - self.Us["PW"]

        dX = self.PW.backward(self.pred_errors["PW"])

        self.Xs["DP1"] -= decay * infer_rate * (self.pred_errors["DP1"] - dX)
        # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
        self.pred_errors["DP1"] = self.Xs["DP1"] - self.Us["DP1"]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        dX = self.DP1.backward(self.pred_errors["DP1"])

        dX2 = self.shortcut.backward(self.pred_errors["shortcut"])

        self.Xs["input"] -= decay * infer_rate * (self.pred_errors["input"] - dX - dX2)
        # self.Xs['input']이 두개로 갈라진 DP1과 shortcut에 같이 사용되므로
        # 역방향일때는 DP1, shortcut 각각에서 구해진 ^ε(DP1),^ε(shortcut)를 더해야 함
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return self.Xs["input"]

    @torch.no_grad()
    def update_weights(self):
        return [
            self.DP1.update_weights(self.pred_errors["DP1"]),
            self.PW.update_weights(self.pred_errors["PW"]),
            self.DP2.update_weights(self.pred_errors["DP2"]),
        ]

    def set_lr(self, new_lr):
        self.DP1.set_lr(new_lr)
        self.PW.set_lr(new_lr)
        self.DP2.set_lr(new_lr)

    # def _initialize_weights(self):
    #     return


class PCBlockBN(nn.Module):
    # Bottleneck 에서만 꼭 필요한 부분들도 미리 추가해두면
    # Bottleneck 코드 작성 시 편하다.
    expansion = 1
    # bottleneck 구조에서 쓰임

    def __init__(
        self, in_channels, out_channels, input_size, learning_rate, stride=1, momentum=0.01, device="cpu"
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

        self.DP1 = DualPathConvBN(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            input_size=input_size,
            learning_rate=learning_rate,
            stride=stride,
            momentum=momentum,
            f=relu,
            df=d_relu,
            device=device,
        )
        # self.BN1 = nn.BatchNorm2d(self.out_channels)
        # self.relu = RELU()

        self.PW = ConvLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            input_size=input_size // stride,
            learning_rate=learning_rate,
            stride=1,
            padding=0,
            device=device,
        )

        self.DP2 = DualPathConvBN(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            input_size=input_size // stride,
            learning_rate=learning_rate,
            stride=1,
            momentum=momentum,
            f=relu,
            df=d_relu,
            device=device,
        )

        # self.BN2 = nn.BatchNorm2d(self.out_channels)

        # self.relu = RELU()

        self.shortcut = ShortcutPath(device=device)
        self.concat = False
        # identity mapping

        # filter 갯수가 두배가 되는 블록인 경우에는 identity mapping 대신 1x1 conv로 projection 수행
        if stride != 1 or in_channels != PCBlockTest.expansion * out_channels:
            # @@@  이 조건문의 out_channels는 self.out_channels로 바꾸면 안된다.  @@@
            # in_channels != ResBlock.expansion * out_channels 은 각 층에서 제일 앞에 있는 블록에서만 참
            self.shortcut = AvgPoolLayer(
                in_channels=in_channels, input_size=input_size, stride=stride, device=device
            )
            self.concat = True

        self.add = AddLayer(concat=self.concat)

        self.Xs = {"input": [], "DP1": [], "DP2": [], "PW": [], "shortcut": [], "add": []}
        self.Us = {"input": [], "DP1": [], "DP2": [], "PW": [], "shortcut": [], "add": []}
        # self.Es = {'input':[], 'DP1':[], 'DP2':[], 'PW':[], 'shortcut':[], 'add':[]}
        self.pred_errors = {"input": [], "DP1": [], "DP2": [], "PW": [], "shortcut": [], "add": []}

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs["input"] = x.clone()

        x_shortcut = self.shortcut(self.Xs["input"].detach())
        self.Xs["shortcut"] = x_shortcut

        x = self.DP1(self.Xs["input"].detach(), init=True)
        self.Xs["DP1"] = x
        x = self.PW(x.detach())
        self.Xs["PW"] = x
        x = self.DP2(x.detach(), init=True)
        self.Xs["DP2"] = x

        x = self.add(x.detach(), x_shortcut.detach())
        self.Xs["add"] = x

        return x

    def forward(self, x):
        x_shortcut = self.shortcut(x)

        x = self.DP1(x)
        # x = self.BN1(x)
        # x = self.relu(x)

        x = self.PW(x)

        # y = torch.zeros_like(x).to(device)
        # for i, x_s in enumerate(torch.split(x, 2 * self.out_channels // self.groups, dim=1)):
        #     y += self.DWs[i](x_s)

        x = self.DP2(x)

        # x = self.BN2(y)
        # x = self.relu(x)

        x = self.add(x, x_shortcut)

        return x

    # PCNet infer에서 매번 수행
    @torch.no_grad()
    def prediction(self, e):
        # self.Us['input'] = u.clone()
        self.pred_errors["input"] = e

        self.Us["shortcut"] = self.shortcut(self.Xs["input"])
        self.pred_errors["shortcut"] = self.Xs["shortcut"] - self.Us["shortcut"]

        self.Us["DP1"] = self.DP1(self.Xs["input"])
        self.pred_errors["DP1"] = self.Xs["DP1"] - self.Us["DP1"]

        self.Us["PW"] = self.PW(self.Xs["DP1"])
        self.pred_errors["PW"] = self.Xs["PW"] - self.Us["PW"]

        self.Us["DP2"] = self.DP2(self.Xs["PW"])
        self.pred_errors["DP2"] = self.Xs["DP2"] - self.Us["DP2"]

        self.Us["add"] = self.add(self.Xs["DP2"], self.Xs["shortcut"])

        return self.Us["add"]

    @torch.no_grad()
    def infer(self, x, decay, infer_rate):
        self.Xs["add"] = x.clone()
        self.pred_errors["add"] = self.Xs["add"] - self.Us["add"]

        dX1, dX2 = self.add.backward(self.pred_errors["add"])

        self.Xs["DP2"] -= decay * infer_rate * (self.pred_errors["DP2"] - dX1)
        self.Xs["shortcut"] -= decay * infer_rate * (self.pred_errors["shortcut"] - dX2)
        # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
        self.pred_errors["DP2"] = self.Xs["DP2"] - self.Us["DP2"]
        self.pred_errors["shortcut"] = self.Xs["shortcut"] - self.Us["shortcut"]

        dX = self.DP2.backward(self.pred_errors["DP2"])

        self.Xs["PW"] -= decay * infer_rate * (self.pred_errors["PW"] - dX)
        # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
        self.pred_errors["PW"] = self.Xs["PW"] - self.Us["PW"]

        dX = self.PW.backward(self.pred_errors["PW"])

        self.Xs["DP1"] -= decay * infer_rate * (self.pred_errors["DP1"] - dX)
        # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
        self.pred_errors["DP1"] = self.Xs["DP1"] - self.Us["DP1"]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        dX = self.DP1.backward(self.pred_errors["DP1"])

        dX2 = self.shortcut.backward(self.pred_errors["shortcut"])

        self.Xs["input"] -= decay * infer_rate * (self.pred_errors["input"] - dX - dX2)
        # self.Xs['input']이 두개로 갈라진 DP1과 shortcut에 같이 사용되므로
        # 역방향일때는 DP1, shortcut 각각에서 구해진 ^ε(DP1),^ε(shortcut)를 더해야 함
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return self.Xs["input"]

    @torch.no_grad()
    def update_weights(self):
        return [
            self.DP1.update_weights(self.pred_errors["DP1"]),
            self.PW.update_weights(self.pred_errors["PW"]),
            self.DP2.update_weights(self.pred_errors["DP2"]),
        ]

    def set_lr(self, new_lr):
        self.DP1.set_lr(new_lr)
        self.PW.set_lr(new_lr)
        self.DP2.set_lr(new_lr)

    # def _initialize_weights(self):
    #     return


class PCSequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
        self.length = len(self.blocks)

        self.Xs = [[] for i in range(self.length + 1)]
        self.Us = [[] for i in range(self.length + 1)]
        self.pred_errors = [[] for i in range(self.length + 1)]

    @torch.no_grad()
    def _initialize_Xs(self, x):
        self.Xs[0] = x.clone()

        for i in range(self.length):
            self.Xs[i + 1] = self.blocks[i]._initialize_Xs(self.Xs[i].detach())

        return self.Xs[-1]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

    @torch.no_grad()
    def prediction(self, e):
        # self.Us[0] = u.clone()
        self.pred_errors[0] = e

        for i in range(self.length):
            self.Us[i + 1] = self.blocks[i].prediction(self.pred_errors[i])
            self.pred_errors[i + 1] = self.Xs[i + 1] - self.Us[i + 1]

        return self.Us[-1]

    @torch.no_grad()
    def infer(self, x, decay, infer_rate):
        self.Xs[-1] = x

        for i in reversed(range(self.length)):
            self.Xs[i] = self.blocks[i].infer(self.Xs[i + 1], decay, infer_rate)

        return self.Xs[0]

    @torch.no_grad()
    def update_weights(self):
        dWs = []
        for block in self.blocks:
            dWs.extend(block.update_weights())

        return dWs

    def set_lr(self, new_lr):
        for block in self.blocks:
            block.set_lr(new_lr)


class PCNet(nn.Module):
    def __init__(
        self,
        block_name,
        num_block_inlayers,
        learning_rate=0.1,
        n_iter_dx=25,
        infer_rate=0.05,
        beta=100,
        momentum=0.01,
        num_classes=10,
        init_weights=True,
        device="cpu",
    ):
        # block은 ResBlock 또는 Bottleneck 클래스 이름을 받아서 저장
        # num_block_inlayers 는 각 층마다 residual block이 몇개씩 들어가는지 정보를 담은 list
        super().__init__()

        self.in_channels = 32
        # 3x3 conv와 3x3 max pool을 지나면 32 채널이 됨
        self.input_size = 32

        self.learning_rate = learning_rate
        self.n_iter_dx = n_iter_dx
        self.infer_rate = infer_rate
        self.beta = beta
        self.gamma = 1 / (1 + self.beta)

        self.momentum = momentum

        self.num_classes = num_classes

        self.device = device

        self.conv1 = ConvBN(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            input_size=self.input_size,
            stride=1,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            f=relu,
            df=d_relu,
            device=self.device,
        )

        self.conv2_x = self._make_layer(
            block_name=block_name, out_channels=32, num_blocks=num_block_inlayers[0], stride=1
        )
        self.conv3_x = self._make_layer(
            block_name=block_name, out_channels=64, num_blocks=num_block_inlayers[1], stride=1
        )
        self.conv4_x = self._make_layer(
            block_name=block_name, out_channels=128, num_blocks=num_block_inlayers[2], stride=2
        )
        self.conv5_x = self._make_layer(
            block_name=block_name, out_channels=256, num_blocks=num_block_inlayers[3], stride=2
        )
        # residual block들을 담은 4개의 층 생성

        self.avg_pool = AdaptiveAvgPoolLayer(in_channels=256, input_size=8, device=self.device)
        # filter 갯수는 유지하고 Width와 Height는 지정한 값으로 average pooling을 수행해 준다
        # ex: 256, 8, 8 을 256, 1, 1 로 변경

        self.fc = FCtoSoftMax(
            in_features=256 * block_name.expansion,
            out_features=num_classes,
            learning_rate=self.learning_rate,
            bias=True,
            device=self.device,
        )
        self.softmax = nn.Softmax(dim=1)

        self.Xs = {
            "input": [],
            "conv1": [],
            "conv2_x": [],
            "conv3_x": [],
            "conv4_x": [],
            "conv5_x": [],
            "avg_pool": [],
            "fc": [],
        }
        self.Us = {
            "input": [],
            "conv1": [],
            "conv2_x": [],
            "conv3_x": [],
            "conv4_x": [],
            "conv5_x": [],
            "avg_pool": [],
            "fc": [],
        }
        self.pred_errors = {
            "input": [],
            "conv1": [],
            "conv2_x": [],
            "conv3_x": [],
            "conv4_x": [],
            "conv5_x": [],
            "avg_pool": [],
            "fc": [],
        }

    # residual block들을 지정된 갯수만큼 생성해 nn.Sequential() 안에 넣어서 반환해주는 함수
    def _make_layer(self, block_name, out_channels, num_blocks, stride):
        # block은 Resnet 클래스가 받아놓은 ResBlock 또는 Bottleneck 클래스 이름을 내려받음
        # 여기서의 out_channels은 residual block 안의 3x3 conv의 out_channels 갯수
        # num_blocks 는 이 층에 들어갈 residual block 갯수
        # stride는 이 층 제일 첫번째 residual block에서 down sampling을 할경우 1이 아닌 값을 넣어준다.

        strides = [stride] + [1] * (num_blocks - 1)
        # [stride, 1, 1, ...., 1] 꼴

        blocks = []
        # residual block들을 담을 list

        for stride in strides:
            blocks.append(
                block_name(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    input_size=self.input_size,
                    learning_rate=self.learning_rate,
                    stride=stride,
                    momentum=self.momentum,
                    device=self.device,
                )
            )
            # list 에 residual block 만들어서 저장

            self.in_channels = out_channels * block_name.expansion
            # residual block 생성 뒤에 코드가 있어서
            # 한 층에서 다음 층으로 넘어가 만나는 첫 residual block에서 down sampling 하거나
            # bottleneck 처럼 expansion이 4 인경우에도 문제없음

            if stride == 2:
                self.input_size = self.input_size // 2

        return PCSequential(blocks)
        # list 인 blocks 를 * 로 unpack해서 nn.Sequential()에 입력

    @torch.no_grad()
    def _initialize_Xs(self, x):
        # self.Xs = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        self.Xs["input"] = x.clone()
        self.Us["input"] = x.clone()

        self.Xs["conv1"] = self.conv1(self.Xs["input"].detach(), init=True)

        self.Xs["conv2_x"] = self.conv2_x._initialize_Xs(self.Xs["conv1"].detach())
        self.Xs["conv3_x"] = self.conv3_x._initialize_Xs(self.Xs["conv2_x"].detach())
        self.Xs["conv4_x"] = self.conv4_x._initialize_Xs(self.Xs["conv3_x"].detach())
        self.Xs["conv5_x"] = self.conv5_x._initialize_Xs(self.Xs["conv4_x"].detach())

        self.Xs["avg_pool"] = self.avg_pool(self.Xs["conv5_x"].detach())

        self.Xs["fc"] = self.fc(self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1).detach())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        # avg_pool까지 하면 [batch_size, 256 * expansion, 1, 1] 형태
        x = x.view(x.size(0), -1)
        # [batch_size, 256 * expansion] 꼴로 변경
        # x = torch.flatten(x, start_dim=1) 도 동일한 결과
        # # flatten에 start_dim=1을 넣어주면 dim=0 인 batch_size 부분은 그대로 두고 두번째부터 flatten 수행

        x = self.fc(x)

        return x

    def no_grad_forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2_x(x)
            x = self.conv3_x(x)
            x = self.conv4_x(x)
            x = self.conv5_x(x)
            x = self.avg_pool(x)
            # avg_pool까지 하면 [batch_size, 256 * expansion, 1, 1] 형태
            x = x.view(x.size(0), -1)
            # [batch_size, 256 * expansion] 꼴로 변경
            # x = torch.flatten(x, start_dim=1) 도 동일한 결과
            # # flatten에 start_dim=1을 넣어주면 dim=0 인 batch_size 부분은 그대로 두고 두번째부터 flatten 수행

            x = self.fc(x)

            return x

    @torch.no_grad()
    def prediction(self):
        # self.Us = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        # self.Us['input'] = u.clone()
        # self.pred_errors['input'] = self.Xs['input'] - self.Us['input']
        # self.pred_errors['input'] = []
        # ∵ self.Xs['input'] == self.Us['input']

        self.Us["conv1"] = self.conv1(self.Xs["input"])
        self.pred_errors["conv1"] = self.Xs["conv1"] - self.Us["conv1"]

        self.Us["conv2_x"] = self.conv2_x.prediction(self.pred_errors["conv1"])
        self.pred_errors["conv2_x"] = self.Xs["conv2_x"] - self.Us["conv2_x"]

        self.Us["conv3_x"] = self.conv3_x.prediction(self.pred_errors["conv2_x"])
        self.pred_errors["conv3_x"] = self.Xs["conv3_x"] - self.Us["conv3_x"]

        self.Us["conv4_x"] = self.conv4_x.prediction(self.pred_errors["conv3_x"])
        self.pred_errors["conv4_x"] = self.Xs["conv4_x"] - self.Us["conv4_x"]

        self.Us["conv5_x"] = self.conv5_x.prediction(self.pred_errors["conv4_x"])
        self.pred_errors["conv5_x"] = self.Xs["conv5_x"] - self.Us["conv5_x"]

        self.Us["avg_pool"] = self.avg_pool(self.Xs["conv5_x"])
        self.pred_errors["avg_pool"] = self.Xs["avg_pool"] - self.Us["avg_pool"]

        self.Us["fc"] = self.fc(self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1))

        return self.Us["fc"]

    def infer(self, input, label):
        # 여기서 label은 onehot encoding 된 상태
        self._initialize_Xs(input)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@  Free := ½∑l∑N∑out∑out{(x-μ)^2}  @@
        # 먼저 constraint가 x(0) == input 밖에 없을 때는
        # x(lmax)는 constraint없이 자유롭게 값이 변경 가능하다.
        # 따라서 Free 는 경사하강법을 통해 최소값이 0까지 근사가능하고
        # 그때  x = μ 가 된다.
        # μ는 사실 역전파알고리즘 기반 신경망의 output계산과 동일한 형태에 인풋만 x이기 때문에
        # 이때는 x(l+1) = μ(l+1) = f(x(l)) 이므로 역전파기반 신경망의 forward pass와 동일하게
        # x를 l=0부터 l=max까지 다 계산가능하다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # self.Xs['fc'] = label.clone() #setup final label
        # x(lmax) == label 이라는 constraint가 추가되어 x(0)에 더해 x(lmax)도 이제 고정되고
        # 나머지 x(l1)~x(lmax -1) 만 inference 과정에서 수렴
        with torch.no_grad():
            self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.Xs["fc"]
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 128번째 줄

            self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # inference loop 들어가기 전 self.Us 와 self.pred_errors 초기화
            # @@ 하지 않으면 loop i = 0 일 때 self.pred_errors["avg_pool"] = [] 상태라 에러 발생 @@

        for i in range(self.n_iter_dx):
            decay = 1 / (1 + i)
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 132번째 줄
            with torch.no_grad():
                # self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
                # @@계산량 줄이기위해 변경@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                self.Us["fc"] = self.fc(self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1))
                self.pred_errors["fc"] = self.Xs["fc"] - self.Us["fc"]
                # 이제는 self.Us["fc"]를 제외하고 self.Us는 self.n_iter_dx번 도는 동안 맨처음 한번만 갱신
                # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 는 매번 갱신하고
                # https://github.com/BerenMillidge/PredictiveCodingBackprop/blob/master/cnn.py는 여기서와 같이
                # self.n_iter_dx번 마나 한번만 갱신
                # ===> 1 epoch 당 시간이 절반으로 줄어듬
                # @@계산량 줄이기위해 변경@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # self.prediction()은 self.Us['fc']이고
                # softmax 함수가 이미 적용됨

            dX = self.fc.backward(
                x=self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1), e=self.pred_errors["fc"]
            ).reshape(-1, 256, 1, 1)
            # self.fc.backward(x, e) 는 (batch_size, 256 * expansion) 꼴을 반환
            # self.Xs['avg_pool'] 는 (batch_size, 256 * expansion, 1, 1) 형태 이므로
            # dX를 (batch_size, 256 * expansion, 1, 1) 형태로 바꿔준다.
            with torch.no_grad():
                self.Xs["avg_pool"] -= decay * self.infer_rate * (self.pred_errors["avg_pool"] - dX)
                # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
                self.pred_errors["avg_pool"] = self.Xs["avg_pool"] - self.Us["avg_pool"]

                dX = self.avg_pool.backward(self.pred_errors["avg_pool"])

                self.Xs["conv5_x"] -= decay * self.infer_rate * (self.pred_errors["conv5_x"] - dX)
                # self.pred_errors['conv5_x'] = self.Xs['conv5_x'] - self.Us['conv5_x']
                # PCSequential class들은 backward(e) 대신 infer(x, decay, infer_rate) 사용

                self.Xs["conv4_x"] = self.conv5_x.infer(
                    self.Xs["conv5_x"], decay=decay, infer_rate=self.infer_rate
                )
                self.Xs["conv3_x"] = self.conv4_x.infer(
                    self.Xs["conv4_x"], decay=decay, infer_rate=self.infer_rate
                )
                self.Xs["conv2_x"] = self.conv3_x.infer(
                    self.Xs["conv3_x"], decay=decay, infer_rate=self.infer_rate
                )
                self.Xs["conv1"] = self.conv2_x.infer(
                    self.Xs["conv2_x"], decay=decay, infer_rate=self.infer_rate
                )

                self.pred_errors["conv1"] = self.Xs["conv1"] - self.Us["conv1"]

                self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.fc(
                    self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1)
                )

    def update_weight(self, y):
        # y는 onehot encoding을 하지 않은 상태
        dWs = []
        with torch.no_grad():
            dWs.append(self.conv1.update_weights(self.pred_errors["conv1"]))
            dWs.extend(self.conv2_x.update_weights())
            dWs.extend(self.conv3_x.update_weights())
            dWs.extend(self.conv4_x.update_weights())
            dWs.extend(self.conv5_x.update_weights())

        loss, loss_mean = self.fc.update_weights(self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1), y)

        return dWs, loss, loss_mean

    def get_lr(self):
        return self.fc.get_lr()

    def set_lr(self, new_lr):
        self.conv1.set_lr(new_lr)
        self.conv2_x.set_lr(new_lr)
        self.conv3_x.set_lr(new_lr)
        self.conv4_x.set_lr(new_lr)
        self.conv5_x.set_lr(new_lr)

    def lr_step(self, val_loss):
        self.fc.lr_step(val_loss)

        learning_rate_check = self.fc.get_lr()

        if self.learning_rate != learning_rate_check:
            self.learning_rate = learning_rate_check
            self.set_lr(self.learning_rate)

    def train_wts(self, input, y, topk=(1,)):
        label = F.one_hot(y, num_classes=self.num_classes)

        self.infer(input=input, label=label)
        with torch.no_grad():
            self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # 가중치 수정 전 마지막으로 한번더 pred_errors 계산

        dWs_batch, loss_batch, loss_batch_mean = self.update_weight(y)

        # output = self.no_grad_forward(input)
        # loss값은 weight 수정 전 값이므로
        # 정확도를 계산할 때도 수정 전 값으로 계산할 것

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=self.Us["fc"], target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch

    def val_batch(self, input, y, topk=(1,)):
        output = self.no_grad_forward(input)

        loss_batch, loss_batch_mean = self.fc.cal_loss(output, y)

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch
