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

from PCblocks import *
from PClayers import *
from PCConvLayers import *
from PCFCLayers import *
from PCfunctions import *
from PCutils import *


class PCSimpleNet(nn.Module):
    def __init__(
        self,
        learning_rate=0.1,
        n_iter_dx=25,
        infer_rate=0.05,
        beta=100,
        momentum=0.1,
        num_cnn_output_channel=256,
        num_classes=10,
        init_weights=True,
        device="cpu",
    ):
        # block은 ResBlock 또는 Bottleneck 클래스 이름을 받아서 저장
        # num_block_inlayers 는 각 층마다 residual block이 몇개씩 들어가는지 정보를 담은 list
        super().__init__()

        if num_cnn_output_channel == 256:
            self.denominator = 1
        elif num_cnn_output_channel == 128:
            self.denominator = 2

        self.in_channels = 32 // self.denominator
        # 3x3 conv 지나면 32 or 16 채널이 됨
        self.out_channels = self.in_channels * 2

        self.input_size = 32

        self.learning_rate = learning_rate
        self.n_iter_dx = n_iter_dx
        self.infer_rate = infer_rate
        self.beta = beta
        self.gamma = 1 / (1 + self.beta)

        self.momentum = momentum

        self.num_classes = num_classes

        self.device = device

        self.conv1 = ConvAG(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )

        self.conv2 = ConvAG(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )
        self.in_channels = self.out_channels
        self.out_channels = self.in_channels * 2

        self.conv3 = ConvAG(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )
        self.in_channels = self.out_channels
        self.out_channels = self.in_channels * 2

        self.conv4 = ConvAG(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )

        self.avg_pool = AdaptiveAvgPoolLayer(in_channels=self.out_channels, device=self.device)
        # filter 갯수는 유지하고 Width와 Height는 지정한 값으로 average pooling을 수행해 준다

        self.fc = FCtoSoftMax(
            in_features=self.out_channels,
            out_features=num_classes,
            learning_rate=self.learning_rate,
            bias=True,
            device=self.device,
        )
        self.softmax = nn.Softmax(dim=1)

        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
        # self._initialize_Xs(input)로 계산된 output으로 train loss를 계산할 때 사용

        self.Xs = {
            "input": [],
            "conv1": [],
            "conv2": [],
            "conv3": [],
            "conv4": [],
            "avg_pool": [],
            "fc": [],
        }
        self.Us = {
            "input": [],
            "conv1": [],
            "conv2": [],
            "conv3": [],
            "conv4": [],
            "avg_pool": [],
            "fc": [],
        }
        self.pred_errors = {
            "input": [],
            "conv1": [],
            "conv2": [],
            "conv3": [],
            "conv4": [],
            "avg_pool": [],
            "fc": [],
        }

    @torch.no_grad()
    def _initialize_Xs(self, x):
        # self.Xs = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        self.Xs["input"] = x.clone()
        self.Us["input"] = x.clone()

        self.Xs["conv1"] = self.conv1(self.Xs["input"].detach())
        self.Xs["conv2"] = self.conv2(self.Xs["conv1"].detach())
        self.Xs["conv3"] = self.conv3(self.Xs["conv2"].detach())
        self.Xs["conv4"] = self.conv4(self.Xs["conv3"].detach())

        self.Xs["avg_pool"] = self.avg_pool(self.Xs["conv4"].detach())

        self.Xs["fc"] = self.fc(self.Xs["avg_pool"].detach())

        return self.Xs["fc"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg_pool(x)

        x = self.fc(x)

        return x

    def no_grad_forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            x = self.avg_pool(x)

            x = self.fc(x)

            return x

    @torch.no_grad()
    def prediction(self):
        # self.Us = {'input':[], 'conv1':[], 'conv2':[], 'conv3':[], 'conv4':[], 'avg_pool':[], 'fc':[]}
        # self.Us['input'] = u.clone()
        # self.pred_errors['input'] = self.Xs['input'] - self.Us['input']
        # self.pred_errors['input'] = []
        # ∵ self.Xs['input'] == self.Us['input']

        self.Us["conv1"] = self.conv1(self.Xs["input"])
        self.pred_errors["conv1"] = self.Xs["conv1"] - self.Us["conv1"]

        self.Us["conv2"] = self.conv2(self.Xs["conv1"])
        self.pred_errors["conv2"] = self.Xs["conv2"] - self.Us["conv2"]

        self.Us["conv3"] = self.conv3(self.Xs["conv2"])
        self.pred_errors["conv3"] = self.Xs["conv3"] - self.Us["conv3"]

        self.Us["conv4"] = self.conv4(self.Xs["conv3"])
        self.pred_errors["conv4"] = self.Xs["conv4"] - self.Us["conv4"]

        self.Us["avg_pool"] = self.avg_pool(self.Xs["conv4"])
        self.pred_errors["avg_pool"] = self.Xs["avg_pool"] - self.Us["avg_pool"]

        self.Us["fc"] = self.fc(self.Xs["avg_pool"])

        return self.Us["fc"]

    def infer(self, input, label):
        # 여기서 label은 onehot encoding 된 상태
        output = self._initialize_Xs(input)
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

            # self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # inference loop 들어가기 전 self.Us 와 self.pred_errors 초기화
            # @@ 하지 않으면 loop i = 0 일 때 self.pred_errors["avg_pool"] = [] 상태라 에러 발생 @@

        for i in range(self.n_iter_dx):
            decay = 1 / (1 + i)
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 132번째 줄
            with torch.no_grad():
                self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
                # @@계산량 줄이기위해 변경@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # self.Us["fc"] = self.fc(self.Xs["avg_pool"].view(self.Xs["avg_pool"].size(0), -1))
                # self.pred_errors["fc"] = self.Xs["fc"] - self.Us["fc"]
                # 이제는 self.Us["fc"]를 제외하고 self.Us는 self.n_iter_dx번 도는 동안 맨처음 한번만 갱신
                # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 는 매번 갱신하고
                # https://github.com/BerenMillidge/PredictiveCodingBackprop/blob/master/cnn.py는 여기서와 같이
                # self.n_iter_dx번 마나 한번만 갱신
                # ===> 1 epoch 당 시간이 절반으로 줄어듬
                # @@계산량 줄이기위해 변경@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # self.prediction()은 self.Us['fc']이고
                # softmax 함수가 이미 적용됨

                # with torch.no_grad():
                dX = self.fc.backward(x=self.Xs["avg_pool"], e=self.pred_errors["fc"])

                self.Xs["avg_pool"] -= decay * self.infer_rate * (self.pred_errors["avg_pool"] - dX)
                # X가 수정되면 수정된 X로 pred_error를 다시 계산해서 밑 레이어로 패스
                self.pred_errors["avg_pool"] = self.Xs["avg_pool"] - self.Us["avg_pool"]

                dX = self.avg_pool.backward(self.Xs["conv4"], self.pred_errors["avg_pool"])

                self.Xs["conv4"] -= decay * self.infer_rate * (self.pred_errors["conv4"] - dX)
                self.pred_errors["conv4"] = self.Xs["conv4"] - self.Us["conv4"]

                dX = self.conv4.backward(self.Xs["conv3"], self.pred_errors["conv4"])

                self.Xs["conv3"] -= decay * self.infer_rate * (self.pred_errors["conv3"] - dX)
                self.pred_errors["conv3"] = self.Xs["conv3"] - self.Us["conv3"]

                dX = self.conv3.backward(self.Xs["conv2"], self.pred_errors["conv3"])

                self.Xs["conv2"] -= decay * self.infer_rate * (self.pred_errors["conv2"] - dX)
                self.pred_errors["conv2"] = self.Xs["conv2"] - self.Us["conv2"]

                dX = self.conv2.backward(self.Xs["conv1"], self.pred_errors["conv2"])

                self.Xs["conv1"] -= decay * self.infer_rate * (self.pred_errors["conv1"] - dX)
                self.pred_errors["conv1"] = self.Xs["conv1"] - self.Us["conv1"]

                self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.fc(self.Xs["avg_pool"])

        return output

    def update_weight(self, y):
        # y는 onehot encoding을 하지 않은 상태
        dWs = []
        # with torch.no_grad():
        dWs.append(self.conv1.update_weights(self.Xs["input"], self.Xs["conv1"]))
        dWs.append(self.conv2.update_weights(self.Xs["conv1"], self.Xs["conv2"]))
        dWs.append(self.conv3.update_weights(self.Xs["conv2"], self.Xs["conv3"]))
        dWs.append(self.conv4.update_weights(self.Xs["conv3"], self.Xs["conv4"]))

        loss, loss_mean = self.fc.update_weights(self.Xs["avg_pool"], y)

        return dWs, loss, loss_mean

    def get_lr(self):
        return self.fc.get_lr()

    def set_lr(self, new_lr):
        self.conv1.set_lr(new_lr)
        self.conv2.set_lr(new_lr)
        self.conv3.set_lr(new_lr)
        self.conv4.set_lr(new_lr)

    def lr_step(self, val_loss):
        self.fc.lr_step(val_loss)

        learning_rate_check = self.fc.get_lr()

        if self.learning_rate != learning_rate_check:
            self.learning_rate = learning_rate_check
            self.set_lr(self.learning_rate)

    def train_wts(self, input, y, topk=(1,)):
        label = F.one_hot(y, num_classes=self.num_classes)

        output = self.infer(input=input, label=label)
        # backward 함수들로 X값들을 수정하기 전에 train loss와 accuracy를 측정하기 위해
        # self._initialize_Xs(input)로 계산된 output을 사용

        with torch.no_grad():
            loss_b = self.NLL(torch.log(output), y)
            loss_batch = loss_b.item()
            # output으로 loss값 계산

            self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # 가중치 수정 전 마지막으로 한번더 pred_errors 계산하고
            # BN의 running_mean과 running_var 계산

        # dWs_batch, loss_batch, loss_batch_mean = self.update_weight(y)
        self.update_weight(y)

        # output = self.no_grad_forward(input)
        # loss값은 weight 수정 전 값이므로
        # 정확도를 계산할 때도 수정 전 값으로 계산할 것

        # (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=self.Us["fc"], target=y, topk=topk)
        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch

    def val_batch(self, input, y, topk=(1,)):
        output = self.no_grad_forward(input)

        loss_batch, loss_batch_mean = self.fc.cal_loss(output, y)

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch


# ‘fixed prediction assumption’ 미적용
# ===> 훈련 안됨
class PCResNet(nn.Module):
    def __init__(
        self,
        block_name,
        num_block_inlayers,
        learning_rate=0.1,
        n_iter_dx=25,
        infer_rate=0.05,
        beta=100,
        momentum=0.1,
        num_cnn_output_channel=256,
        num_classes=10,
        init_weights=True,
        device="cpu",
    ):
        # block은 ResBlock 또는 Bottleneck 클래스 이름을 받아서 저장
        # num_block_inlayers 는 각 층마다 residual block이 몇개씩 들어가는지 정보를 담은 list
        super().__init__()

        if num_cnn_output_channel == 256:
            self.denominator = 1
        elif num_cnn_output_channel == 128:
            self.denominator = 2

        self.in_channels = 16 // self.denominator
        # 3x3 conv 지나면 32 or 16 채널이 됨

        self.input_size = 32

        self.learning_rate = learning_rate
        self.n_iter_dx = n_iter_dx
        self.infer_rate = infer_rate
        self.beta = beta
        self.gamma = 1 / (1 + self.beta)

        self.momentum = momentum

        self.num_classes = num_classes

        self.device = device

        self.conv1 = ConvAG(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )

        self.conv2_x = self._make_layer(
            block_name=block_name,
            out_channels=(32 // self.denominator),
            num_blocks=num_block_inlayers[0],
            stride=1,
        )
        self.conv2_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]

        self.conv3_x = self._make_layer(
            block_name=block_name,
            out_channels=(64 // self.denominator),
            num_blocks=num_block_inlayers[1],
            stride=1,
        )
        self.conv3_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]

        self.conv4_x = self._make_layer(
            block_name=block_name,
            out_channels=(128 // self.denominator),
            num_blocks=num_block_inlayers[2],
            stride=2,
        )
        self.conv4_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]

        self.conv5_x = self._make_layer(
            block_name=block_name,
            out_channels=(256 // self.denominator),
            num_blocks=num_block_inlayers[3],
            stride=2,
        )
        self.conv5_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        # residual block들을 담은 4개의 층 생성

        self.avg_pool = AdaptiveAvgPoolLayer(in_channels=(256 // self.denominator), device=self.device)
        # filter 갯수는 유지하고 Width와 Height는 지정한 값으로 average pooling을 수행해 준다
        # ex: 256, 8, 8 을 256, 1, 1 로 변경

        self.fc = FCtoSoftMax(
            in_features=(256 // self.denominator) * block_name.expansion,
            out_features=num_classes,
            learning_rate=self.learning_rate,
            bias=True,
            device=self.device,
        )
        self.softmax = nn.Softmax(dim=1)

        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
        # self._initialize_Xs(input)로 계산된 output으로 train loss를 계산할 때 사용

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

        return nn.Sequential(*blocks)
        # list 인 blocks 를 * 로 unpack해서 nn.Sequential()에 입력

    @torch.no_grad()
    def _initialize_Xs(self, x):
        # self.Xs = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        self.Xs["input"] = x.clone()

        self.Xs["conv1"] = self.conv1(self.Xs["input"].detach())

        x = self.Xs["conv1"]
        for i in range(len(self.conv2_x)):
            x = self.conv2_x[i]._initialize_Xs(x.detach())
            self.conv2_xs[i + 1] = x
        self.Xs["conv2_x"] = x

        for i in range(len(self.conv3_x)):
            x = self.conv3_x[i]._initialize_Xs(x.detach())
            self.conv3_xs[i + 1] = x
        self.Xs["conv3_x"] = x

        for i in range(len(self.conv4_x)):
            x = self.conv4_x[i]._initialize_Xs(x.detach())
            self.conv4_xs[i + 1] = x
        self.Xs["conv4_x"] = x

        for i in range(len(self.conv5_x)):
            x = self.conv5_x[i]._initialize_Xs(x.detach())
            self.conv5_xs[i + 1] = x
        self.Xs["conv5_x"] = x

        self.Xs["avg_pool"] = self.avg_pool(self.Xs["conv5_x"].detach())

        self.Xs["fc"] = self.fc(self.Xs["avg_pool"].detach())

        return self.Xs["fc"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)

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

            x = self.fc(x)

            return x

    def infer(self, input, label):
        # 여기서 label은 onehot encoding 된 상태
        output = self._initialize_Xs(input)
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

            # self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # inference loop 들어가기 전 self.Us 와 self.pred_errors 초기화
            # @@ 하지 않으면 loop i = 0 일 때 self.pred_errors["avg_pool"] = [] 상태라 에러 발생 @@

        for i in range(self.n_iter_dx):
            decay = 1 / (1 + i)
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 132번째 줄
            with torch.no_grad():
                err = self.Xs["fc"] - self.fc(self.Xs["avg_pool"])

            dfdt = self.fc.backward(x=self.Xs["avg_pool"], e=err)

            with torch.no_grad():
                e_cur = self.Xs["avg_pool"] - self.avg_pool(self.Xs["conv5_x"])
                self.Xs["avg_pool"] -= decay * self.infer_rate * (e_cur - dfdt)

                err = self.Xs["avg_pool"] - self.avg_pool(self.Xs["conv5_x"])
                dfdt = self.avg_pool.backward(self.Xs["conv5_x"], err)
                e_cur = self.Xs["conv5_x"] - self.conv5_x(self.Xs["conv4_x"])
                self.Xs["conv5_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv5_x"]
                for i in reversed(range(len(self.conv5_x))):
                    if i > 1:
                        dfdt = self.conv5_x[i].backward(self.conv5_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv5_xs[i] - self.conv5_x[i - 1](self.conv5_xs[i - 1])
                        self.conv5_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv5_xs[i]
                    elif i == 1:
                        dfdt = self.conv5_x[i].backward(self.conv5_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv5_xs[i] - self.conv5_x[0](self.Xs["conv4_x"])
                        self.conv5_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv5_xs[i]
                    else:
                        dfdt = self.conv5_x[i].backward(self.Xs["conv4_x"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv4_x"] - self.conv4_x(self.Xs["conv3_x"])
                self.Xs["conv4_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv4_x"]
                for i in reversed(range(len(self.conv4_x))):
                    if i > 1:
                        dfdt = self.conv4_x[i].backward(self.conv4_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv4_xs[i] - self.conv4_x[i - 1](self.conv4_xs[i - 1])
                        self.conv4_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv4_xs[i]
                    elif i == 1:
                        dfdt = self.conv4_x[i].backward(self.conv4_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv4_xs[i] - self.conv4_x[0](self.Xs["conv3_x"])
                        self.conv4_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv4_xs[i]
                    else:
                        dfdt = self.conv4_x[i].backward(self.Xs["conv3_x"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv3_x"] - self.conv3_x(self.Xs["conv2_x"])
                self.Xs["conv3_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv3_x"]
                for i in reversed(range(len(self.conv3_x))):
                    if i > 1:
                        dfdt = self.conv3_x[i].backward(self.conv3_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv3_xs[i] - self.conv3_x[i - 1](self.conv3_xs[i - 1])
                        self.conv3_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv3_xs[i]
                    elif i == 1:
                        dfdt = self.conv3_x[i].backward(self.conv3_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv3_xs[i] - self.conv3_x[0](self.Xs["conv2_x"])
                        self.conv3_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv3_xs[i]
                    else:
                        dfdt = self.conv3_x[i].backward(self.Xs["conv2_x"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv2_x"] - self.conv2_x(self.Xs["conv1"])
                self.Xs["conv2_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv2_x"]
                for i in reversed(range(len(self.conv2_x))):
                    if i > 1:
                        dfdt = self.conv2_x[i].backward(self.conv2_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv2_xs[i] - self.conv2_x[i - 1](self.conv2_xs[i - 1])
                        self.conv2_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv2_xs[i]
                    elif i == 1:
                        dfdt = self.conv2_x[i].backward(self.conv2_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv2_xs[i] - self.conv2_x[0](self.Xs["conv1"])
                        self.conv2_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv2_xs[i]
                    else:
                        dfdt = self.conv2_x[i].backward(self.Xs["conv1"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv1"] - self.conv1(input)
                self.Xs["conv1"] -= decay * self.infer_rate * (e_cur - dfdt)

                self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.fc(self.Xs["avg_pool"])

        return output

    def update_weight(self, y):
        # y는 onehot encoding을 하지 않은 상태
        self.conv1.update_weights(self.Xs["input"], self.Xs["conv1"])

        for i in range(len(self.conv2_x)):
            if i < len(self.conv2_x) - 1 and i > 0:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.conv2_xs[i + 1])
            elif i == len(self.conv2_x) - 1:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.Xs["conv2_x"])
            else:
                self.conv2_x[i].update_weights(self.Xs["conv1"], self.conv2_xs[i + 1])

        for i in range(len(self.conv3_x)):
            if i < len(self.conv3_x) - 1 and i > 0:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.conv3_xs[i + 1])
            elif i == len(self.conv3_x) - 1:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.Xs["conv3_x"])
            else:
                self.conv3_x[i].update_weights(self.Xs["conv2_x"], self.conv3_xs[i + 1])

        for i in range(len(self.conv4_x)):
            if i < len(self.conv4_x) - 1 and i > 0:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.conv4_xs[i + 1])

            elif i == len(self.conv4_x) - 1:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.Xs["conv4_x"])
            else:
                self.conv4_x[i].update_weights(self.Xs["conv3_x"], self.conv4_xs[i + 1])

        for i in range(len(self.conv5_x)):
            if i < len(self.conv5_x) - 1 and i > 0:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.conv5_xs[i + 1])
            elif i == len(self.conv5_x) - 1:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.Xs["conv5_x"])
            else:
                self.conv5_x[i].update_weights(self.Xs["conv4_x"], self.conv5_xs[i + 1])

        self.fc.update_weights(self.Xs["avg_pool"], y)

        # return losses_layers, loss, loss_mean

    def get_lr(self):
        return self.fc.get_lr()

    def set_lr(self, new_lr):
        self.conv1.set_lr(new_lr)

        for i in range(len(self.conv2_x)):
            self.conv2_x[i].set_lr(new_lr)
        for i in range(len(self.conv3_x)):
            self.conv3_x[i].set_lr(new_lr)
        for i in range(len(self.conv4_x)):
            self.conv4_x[i].set_lr(new_lr)
        for i in range(len(self.conv5_x)):
            self.conv5_x[i].set_lr(new_lr)

    def lr_step(self, val_loss):
        self.fc.lr_step(val_loss)

        learning_rate_check = self.fc.get_lr()

        if self.learning_rate != learning_rate_check:
            self.learning_rate = learning_rate_check
            self.set_lr(self.learning_rate)

    def train_wts(self, input, y, topk=(1,)):
        label = F.one_hot(y, num_classes=self.num_classes)

        # self.infer(input=input, label=label)
        output = self.infer(input=input, label=label)

        with torch.no_grad():
            loss_b = self.NLL(torch.log(output), y)
            loss_batch = loss_b.item()
            # output으로 loss값 계산

        # dWs_batch, loss_batch, loss_batch_mean = self.update_weight(y)
        self.update_weight(y)

        # output = self.no_grad_forward(input)
        # loss값은 weight 수정 전 값이므로
        # 정확도를 계산할 때도 수정 전 값으로 계산할 것

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch

    def val_batch(self, input, y, topk=(1,)):
        output = self.no_grad_forward(input)

        loss_batch, loss_batch_mean = self.fc.cal_loss(output, y)

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch


# ‘fixed prediction assumption’ 적용
class PCResNet2(nn.Module):
    def __init__(
        self,
        block_name,
        num_block_inlayers,
        learning_rate=0.1,
        n_iter_dx=25,
        infer_rate=0.05,
        beta=100,
        momentum=0.1,
        num_cnn_output_channel=256,
        num_classes=10,
        init_weights=True,
        device="cpu",
    ):
        # block은 ResBlock 또는 Bottleneck 클래스 이름을 받아서 저장
        # num_block_inlayers 는 각 층마다 residual block이 몇개씩 들어가는지 정보를 담은 list
        super().__init__()

        if num_cnn_output_channel == 256:
            self.denominator = 1
        elif num_cnn_output_channel == 128:
            self.denominator = 2

        self.in_channels = 16 // self.denominator
        # 3x3 conv 지나면 32 or 16 채널이 됨

        self.input_size = 32

        self.learning_rate = learning_rate
        self.n_iter_dx = n_iter_dx
        self.infer_rate = infer_rate
        self.beta = beta
        self.gamma = 1 / (1 + self.beta)

        self.momentum = momentum

        self.num_classes = num_classes

        self.device = device

        self.conv1 = ConvAG(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )

        self.conv2_x = self._make_layer(
            block_name=block_name,
            out_channels=(32 // self.denominator),
            num_blocks=num_block_inlayers[0],
            stride=1,
        )
        self.conv2_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]
        self.conv2_us = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]

        self.conv3_x = self._make_layer(
            block_name=block_name,
            out_channels=(64 // self.denominator),
            num_blocks=num_block_inlayers[1],
            stride=1,
        )
        self.conv3_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]
        self.conv3_us = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]

        self.conv4_x = self._make_layer(
            block_name=block_name,
            out_channels=(128 // self.denominator),
            num_blocks=num_block_inlayers[2],
            stride=2,
        )
        self.conv4_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]
        self.conv4_us = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]

        self.conv5_x = self._make_layer(
            block_name=block_name,
            out_channels=(256 // self.denominator),
            num_blocks=num_block_inlayers[3],
            stride=2,
        )
        self.conv5_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        self.conv5_us = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        # residual block들을 담은 4개의 층 생성

        self.avg_pool = AdaptiveAvgPoolLayer(in_channels=(256 // self.denominator), device=self.device)
        # filter 갯수는 유지하고 Width와 Height는 지정한 값으로 average pooling을 수행해 준다
        # ex: 256, 8, 8 을 256, 1, 1 로 변경

        self.fc = FCtoSoftMax(
            in_features=(256 // self.denominator) * block_name.expansion,
            out_features=num_classes,
            learning_rate=self.learning_rate,
            bias=True,
            device=self.device,
        )
        self.softmax = nn.Softmax(dim=1)

        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
        # self._initialize_Xs(input)로 계산된 output으로 train loss를 계산할 때 사용

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

        return nn.Sequential(*blocks)
        # list 인 blocks 를 * 로 unpack해서 nn.Sequential()에 입력

    @torch.no_grad()
    def _initialize_Xs(self, x):
        # self.Xs = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        self.Xs["input"] = x.clone()
        self.Us["input"] = x.clone()

        self.Xs["conv1"] = self.conv1(self.Xs["input"].detach())
        self.Us["conv1"] = self.Xs["conv1"].clone()

        x = self.Xs["conv1"]
        for i in range(len(self.conv2_x)):
            x = self.conv2_x[i]._initialize_Xs(x.detach())
            self.conv2_xs[i + 1] = x
            self.conv2_us[i + 1] = self.conv2_xs[i + 1].clone()
        self.Xs["conv2_x"] = x
        self.Us["conv2_x"] = self.Xs["conv2_x"].clone()

        for i in range(len(self.conv3_x)):
            x = self.conv3_x[i]._initialize_Xs(x.detach())
            self.conv3_xs[i + 1] = x
            self.conv3_us[i + 1] = self.conv3_xs[i + 1].clone()
        self.Xs["conv3_x"] = x
        self.Us["conv3_x"] = self.Xs["conv3_x"].clone()

        for i in range(len(self.conv4_x)):
            x = self.conv4_x[i]._initialize_Xs(x.detach())
            self.conv4_xs[i + 1] = x
            self.conv4_us[i + 1] = self.conv4_xs[i + 1].clone()
        self.Xs["conv4_x"] = x
        self.Us["conv4_x"] = self.Xs["conv4_x"].clone()

        for i in range(len(self.conv5_x)):
            x = self.conv5_x[i]._initialize_Xs(x.detach())
            self.conv5_xs[i + 1] = x
            self.conv5_us[i + 1] = self.conv5_xs[i + 1].clone()
        self.Xs["conv5_x"] = x
        self.Us["conv5_x"] = self.Xs["conv5_x"].clone()

        self.Xs["avg_pool"] = self.avg_pool(self.Xs["conv5_x"].detach())
        self.Us["avg_pool"] = self.Xs["avg_pool"].clone()

        self.Xs["fc"] = self.fc(self.Xs["avg_pool"].detach())
        self.Us["fc"] = self.Xs["fc"].clone()

        return self.Xs["fc"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)

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

            x = self.fc(x)

            return x

    def infer(self, input, label):
        # 여기서 label은 onehot encoding 된 상태
        output = self._initialize_Xs(input)
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

            # self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # inference loop 들어가기 전 self.Us 와 self.pred_errors 초기화
            # @@ 하지 않으면 loop i = 0 일 때 self.pred_errors["avg_pool"] = [] 상태라 에러 발생 @@

        for i in range(self.n_iter_dx):
            decay = 1 / (1 + i)
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 132번째 줄
            with torch.no_grad():
                err = self.Xs["fc"] - self.Us["fc"]
                # PC can approximate backprop for arbitrary computation graphs (not just MLPs)
                # under the ‘fixed prediction assumption’
                # which is where during inference the predictions between layers are fixed to their feedforward pass values.
                # https://www.beren.io/2023-03-30-Thoughts-on-future-of-PC/

            dfdt = self.fc.backward(x=self.Us["avg_pool"], e=err)

            with torch.no_grad():
                e_cur = self.Xs["avg_pool"] - self.Us["avg_pool"]
                self.Xs["avg_pool"] -= decay * self.infer_rate * (e_cur - dfdt)

                err = self.Xs["avg_pool"] - self.Us["avg_pool"]
                dfdt = self.avg_pool.backward(self.Us["conv5_x"], err)
                e_cur = self.Xs["conv5_x"] - self.Us["conv5_x"]
                self.Xs["conv5_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv5_x"]
                for i in reversed(range(len(self.conv5_x))):
                    if i > 0:
                        dfdt = self.conv5_x[i].backward(self.conv5_xs[i], y, decay, self.infer_rate)
                        # block 안에서는 U를 사용하므로 여기 backward에선 self.conv5_us[i] 안 넣어도 된다.
                        e_cur = self.conv5_xs[i] - self.conv5_us[i]
                        self.conv5_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv5_xs[i]
                    else:
                        dfdt = self.conv5_x[i].backward(self.Xs["conv4_x"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv4_x"] - self.Us["conv4_x"]
                self.Xs["conv4_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv4_x"]
                for i in reversed(range(len(self.conv4_x))):
                    if i > 0:
                        dfdt = self.conv4_x[i].backward(self.conv4_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv4_xs[i] - self.conv4_us[i]
                        self.conv4_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv4_xs[i]
                    else:
                        dfdt = self.conv4_x[i].backward(self.Xs["conv3_x"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv3_x"] - self.Us["conv3_x"]
                self.Xs["conv3_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv3_x"]
                for i in reversed(range(len(self.conv3_x))):
                    if i > 0:
                        dfdt = self.conv3_x[i].backward(self.conv3_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv3_xs[i] - self.conv3_us[i]
                        self.conv3_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv3_xs[i]
                    else:
                        dfdt = self.conv3_x[i].backward(self.Xs["conv2_x"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv2_x"] - self.Us["conv2_x"]
                self.Xs["conv2_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv2_x"]
                for i in reversed(range(len(self.conv2_x))):
                    if i > 0:
                        dfdt = self.conv2_x[i].backward(self.conv2_xs[i], y, decay, self.infer_rate)
                        e_cur = self.conv2_xs[i] - self.conv2_us[i]
                        self.conv2_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv2_xs[i]
                    else:
                        dfdt = self.conv2_x[i].backward(self.Xs["conv1"], y, decay, self.infer_rate)

                e_cur = self.Xs["conv1"] - self.Us["conv1"]
                self.Xs["conv1"] -= decay * self.infer_rate * (e_cur - dfdt)

                self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.fc(self.Xs["avg_pool"])

        return output

    def update_weight(self, y):
        # y는 onehot encoding을 하지 않은 상태
        self.conv1.update_weights(self.Us["input"], self.Xs["conv1"])
        # self.Xs["input"] == self.Us["input"]은 동일

        for i in range(len(self.conv2_x)):
            if i < len(self.conv2_x) - 1 and i > 0:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.conv2_xs[i + 1])
                # block 안에서는 U를 사용하므로 여기서 self.conv2_us[i]로 바꿀 필요 없음
            elif i == len(self.conv2_x) - 1:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.Xs["conv2_x"])
            else:
                self.conv2_x[i].update_weights(self.Xs["conv1"], self.conv2_xs[i + 1])

        for i in range(len(self.conv3_x)):
            if i < len(self.conv3_x) - 1 and i > 0:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.conv3_xs[i + 1])
            elif i == len(self.conv3_x) - 1:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.Xs["conv3_x"])
            else:
                self.conv3_x[i].update_weights(self.Xs["conv2_x"], self.conv3_xs[i + 1])

        for i in range(len(self.conv4_x)):
            if i < len(self.conv4_x) - 1 and i > 0:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.conv4_xs[i + 1])

            elif i == len(self.conv4_x) - 1:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.Xs["conv4_x"])
            else:
                self.conv4_x[i].update_weights(self.Xs["conv3_x"], self.conv4_xs[i + 1])

        for i in range(len(self.conv5_x)):
            if i < len(self.conv5_x) - 1 and i > 0:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.conv5_xs[i + 1])
            elif i == len(self.conv5_x) - 1:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.Xs["conv5_x"])
            else:
                self.conv5_x[i].update_weights(self.Xs["conv4_x"], self.conv5_xs[i + 1])

        self.fc.update_weights(self.Us["avg_pool"], y)

        # return losses_layers, loss, loss_mean

    def get_lr(self):
        return self.fc.get_lr()

    def set_lr(self, new_lr):
        self.conv1.set_lr(new_lr)

        for i in range(len(self.conv2_x)):
            self.conv2_x[i].set_lr(new_lr)
        for i in range(len(self.conv3_x)):
            self.conv3_x[i].set_lr(new_lr)
        for i in range(len(self.conv4_x)):
            self.conv4_x[i].set_lr(new_lr)
        for i in range(len(self.conv5_x)):
            self.conv5_x[i].set_lr(new_lr)

    def lr_step(self, val_loss):
        self.fc.lr_step(val_loss)

        learning_rate_check = self.fc.get_lr()

        if self.learning_rate != learning_rate_check:
            self.learning_rate = learning_rate_check
            self.set_lr(self.learning_rate)

    def train_wts(self, input, y, topk=(1,)):
        label = F.one_hot(y, num_classes=self.num_classes)

        # self.infer(input=input, label=label)
        output = self.infer(input=input, label=label)

        with torch.no_grad():
            loss_b = self.NLL(torch.log(output), y)
            loss_batch = loss_b.item()
            # output으로 loss값 계산

        # dWs_batch, loss_batch, loss_batch_mean = self.update_weight(y)
        self.update_weight(y)

        # output = self.no_grad_forward(input)
        # loss값은 weight 수정 전 값이므로
        # 정확도를 계산할 때도 수정 전 값으로 계산할 것

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch

    def val_batch(self, input, y, topk=(1,)):
        output = self.no_grad_forward(input)

        loss_batch, loss_batch_mean = self.fc.cal_loss(output, y)

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch


# ‘fixed prediction assumption’ 적용
# SeqIL 미적용
class PCResNet3(nn.Module):
    def __init__(
        self,
        block_name,
        num_block_inlayers,
        learning_rate=0.1,
        n_iter_dx=25,
        infer_rate=0.05,
        beta=100,
        momentum=0.1,
        num_cnn_output_channel=256,
        num_classes=10,
        init_weights=True,
        device="cpu",
    ):
        # block은 ResBlock 또는 Bottleneck 클래스 이름을 받아서 저장
        # num_block_inlayers 는 각 층마다 residual block이 몇개씩 들어가는지 정보를 담은 list
        super().__init__()

        if num_cnn_output_channel == 256:
            self.denominator = 1
        elif num_cnn_output_channel == 128:
            self.denominator = 2

        self.in_channels = 16 // self.denominator
        # 3x3 conv 지나면 32 or 16 채널이 됨

        self.input_size = 32

        self.learning_rate = learning_rate
        self.n_iter_dx = n_iter_dx
        self.infer_rate = infer_rate
        self.beta = beta
        self.gamma = 1 / (1 + self.beta)

        self.momentum = momentum

        self.num_classes = num_classes

        self.device = device

        self.conv1 = ConvAG(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )

        self.conv2_x = self._make_layer(
            block_name=block_name,
            out_channels=(32 // self.denominator),
            num_blocks=num_block_inlayers[0],
            stride=1,
        )
        self.conv2_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]
        self.conv2_us = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]

        self.conv3_x = self._make_layer(
            block_name=block_name,
            out_channels=(64 // self.denominator),
            num_blocks=num_block_inlayers[1],
            stride=1,
        )
        self.conv3_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]
        self.conv3_us = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]

        self.conv4_x = self._make_layer(
            block_name=block_name,
            out_channels=(128 // self.denominator),
            num_blocks=num_block_inlayers[2],
            stride=2,
        )
        self.conv4_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]
        self.conv4_us = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]

        self.conv5_x = self._make_layer(
            block_name=block_name,
            out_channels=(256 // self.denominator),
            num_blocks=num_block_inlayers[3],
            stride=2,
        )
        self.conv5_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        self.conv5_us = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        # residual block들을 담은 4개의 층 생성

        self.avg_pool = AdaptiveAvgPoolLayer(in_channels=(256 // self.denominator), device=self.device)
        # filter 갯수는 유지하고 Width와 Height는 지정한 값으로 average pooling을 수행해 준다
        # ex: 256, 8, 8 을 256, 1, 1 로 변경

        self.fc = FCtoSoftMax(
            in_features=(256 // self.denominator) * block_name.expansion,
            out_features=num_classes,
            learning_rate=self.learning_rate,
            bias=True,
            device=self.device,
        )
        self.softmax = nn.Softmax(dim=1)

        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
        # self._initialize_Xs(input)로 계산된 output으로 train loss를 계산할 때 사용

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

        return nn.Sequential(*blocks)
        # list 인 blocks 를 * 로 unpack해서 nn.Sequential()에 입력

    @torch.no_grad()
    def _initialize_Xs(self, x):
        # self.Xs = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        self.Xs["input"] = x.clone()
        self.Us["input"] = x.clone()

        self.Xs["conv1"] = self.conv1(self.Xs["input"].detach())
        self.Us["conv1"] = self.Xs["conv1"].clone()

        x = self.Xs["conv1"]
        for i in range(len(self.conv2_x)):
            x = self.conv2_x[i]._initialize_Xs(x.detach())
            self.conv2_xs[i + 1] = x
            self.conv2_us[i + 1] = self.conv2_xs[i + 1].clone()
        self.Xs["conv2_x"] = x
        self.Us["conv2_x"] = self.Xs["conv2_x"].clone()

        for i in range(len(self.conv3_x)):
            x = self.conv3_x[i]._initialize_Xs(x.detach())
            self.conv3_xs[i + 1] = x
            self.conv3_us[i + 1] = self.conv3_xs[i + 1].clone()
        self.Xs["conv3_x"] = x
        self.Us["conv3_x"] = self.Xs["conv3_x"].clone()

        for i in range(len(self.conv4_x)):
            x = self.conv4_x[i]._initialize_Xs(x.detach())
            self.conv4_xs[i + 1] = x
            self.conv4_us[i + 1] = self.conv4_xs[i + 1].clone()
        self.Xs["conv4_x"] = x
        self.Us["conv4_x"] = self.Xs["conv4_x"].clone()

        for i in range(len(self.conv5_x)):
            x = self.conv5_x[i]._initialize_Xs(x.detach())
            self.conv5_xs[i + 1] = x
            self.conv5_us[i + 1] = self.conv5_xs[i + 1].clone()
        self.Xs["conv5_x"] = x
        self.Us["conv5_x"] = self.Xs["conv5_x"].clone()

        self.Xs["avg_pool"] = self.avg_pool(self.Xs["conv5_x"].detach())
        self.Us["avg_pool"] = self.Xs["avg_pool"].clone()

        self.Xs["fc"] = self.fc(self.Xs["avg_pool"].detach())
        self.Us["fc"] = self.Xs["fc"].clone()

        return self.Xs["fc"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)

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

            x = self.fc(x)

            return x

    def infer(self, input, label):
        # 여기서 label은 onehot encoding 된 상태
        output = self._initialize_Xs(input)
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

            # self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # inference loop 들어가기 전 self.Us 와 self.pred_errors 초기화
            # @@ 하지 않으면 loop i = 0 일 때 self.pred_errors["avg_pool"] = [] 상태라 에러 발생 @@

        for i in range(self.n_iter_dx):
            decay = 1 / (1 + i)
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 132번째 줄
            with torch.no_grad():
                err = self.Xs["fc"] - self.Us["fc"]
                # PC can approximate backprop for arbitrary computation graphs (not just MLPs)
                # under the ‘fixed prediction assumption’
                # which is where during inference the predictions between layers are fixed to their feedforward pass values.
                # https://www.beren.io/2023-03-30-Thoughts-on-future-of-PC/

            dfdt = self.fc.backward(x=self.Us["avg_pool"], e=err)

            with torch.no_grad():
                e_cur = self.Xs["avg_pool"] - self.Us["avg_pool"]
                self.Xs["avg_pool"] -= decay * self.infer_rate * (e_cur - dfdt)

                err = e_cur
                dfdt = self.avg_pool.backward(self.Us["conv5_x"], err)
                e_cur = self.Xs["conv5_x"] - self.Us["conv5_x"]
                self.Xs["conv5_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv5_x"]
                y_old = e_cur + self.Us["conv5_x"]
                for i in reversed(range(len(self.conv5_x))):
                    if i > 0:
                        dfdt = self.conv5_x[i].backward(self.conv5_xs[i], y, y_old, decay, self.infer_rate)
                        # block 안에서는 U를 사용하므로 여기 backward에선 self.conv5_us[i] 안 넣어도 된다.
                        e_cur = self.conv5_xs[i] - self.conv5_us[i]
                        self.conv5_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv5_xs[i]
                        y_old = e_cur + self.conv5_us[i]
                    else:
                        dfdt = self.conv5_x[i].backward(self.Xs["conv4_x"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv4_x"] - self.Us["conv4_x"]
                self.Xs["conv4_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv4_x"]
                y_old = e_cur + self.Us["conv4_x"]
                for i in reversed(range(len(self.conv4_x))):
                    if i > 0:
                        dfdt = self.conv4_x[i].backward(self.conv4_xs[i], y, y_old, decay, self.infer_rate)
                        e_cur = self.conv4_xs[i] - self.conv4_us[i]
                        self.conv4_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv4_xs[i]
                        y_old = e_cur + self.conv4_us[i]
                    else:
                        dfdt = self.conv4_x[i].backward(self.Xs["conv3_x"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv3_x"] - self.Us["conv3_x"]
                self.Xs["conv3_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv3_x"]
                y_old = e_cur + self.Us["conv3_x"]
                for i in reversed(range(len(self.conv3_x))):
                    if i > 0:
                        dfdt = self.conv3_x[i].backward(self.conv3_xs[i], y, y_old, decay, self.infer_rate)
                        e_cur = self.conv3_xs[i] - self.conv3_us[i]
                        self.conv3_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv3_xs[i]
                        y_old = e_cur + self.conv3_us[i]
                    else:
                        dfdt = self.conv3_x[i].backward(self.Xs["conv2_x"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv2_x"] - self.Us["conv2_x"]
                self.Xs["conv2_x"] -= decay * self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv2_x"]
                y_old = e_cur + self.Us["conv2_x"]
                for i in reversed(range(len(self.conv2_x))):
                    if i > 0:
                        dfdt = self.conv2_x[i].backward(self.conv2_xs[i], y, y_old, decay, self.infer_rate)
                        e_cur = self.conv2_xs[i] - self.conv2_us[i]
                        self.conv2_xs[i] -= decay * self.infer_rate * (e_cur - dfdt)
                        y = self.conv2_xs[i]
                        y_old = e_cur + self.conv2_us[i]
                    else:
                        dfdt = self.conv2_x[i].backward(self.Xs["conv1"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv1"] - self.Us["conv1"]
                self.Xs["conv1"] -= decay * self.infer_rate * (e_cur - dfdt)

                self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.fc(self.Xs["avg_pool"])

        return output

    def update_weight(self, y):
        # y는 onehot encoding을 하지 않은 상태
        self.conv1.update_weights(self.Us["input"], self.Xs["conv1"])
        # self.Xs["input"] == self.Us["input"]은 동일

        for i in range(len(self.conv2_x)):
            if i < len(self.conv2_x) - 1 and i > 0:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.conv2_xs[i + 1])
                # block 안에서는 U를 사용하므로 여기서 self.conv2_us[i]로 바꿀 필요 없음
            elif i == len(self.conv2_x) - 1:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.Xs["conv2_x"])
            else:
                self.conv2_x[i].update_weights(self.Xs["conv1"], self.conv2_xs[i + 1])

        for i in range(len(self.conv3_x)):
            if i < len(self.conv3_x) - 1 and i > 0:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.conv3_xs[i + 1])
            elif i == len(self.conv3_x) - 1:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.Xs["conv3_x"])
            else:
                self.conv3_x[i].update_weights(self.Xs["conv2_x"], self.conv3_xs[i + 1])

        for i in range(len(self.conv4_x)):
            if i < len(self.conv4_x) - 1 and i > 0:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.conv4_xs[i + 1])

            elif i == len(self.conv4_x) - 1:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.Xs["conv4_x"])
            else:
                self.conv4_x[i].update_weights(self.Xs["conv3_x"], self.conv4_xs[i + 1])

        for i in range(len(self.conv5_x)):
            if i < len(self.conv5_x) - 1 and i > 0:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.conv5_xs[i + 1])
            elif i == len(self.conv5_x) - 1:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.Xs["conv5_x"])
            else:
                self.conv5_x[i].update_weights(self.Xs["conv4_x"], self.conv5_xs[i + 1])

        self.fc.update_weights(self.Us["avg_pool"], y)

        # return losses_layers, loss, loss_mean

    def get_lr(self):
        return self.fc.get_lr()

    def set_lr(self, new_lr):
        self.conv1.set_lr(new_lr)

        for i in range(len(self.conv2_x)):
            self.conv2_x[i].set_lr(new_lr)
        for i in range(len(self.conv3_x)):
            self.conv3_x[i].set_lr(new_lr)
        for i in range(len(self.conv4_x)):
            self.conv4_x[i].set_lr(new_lr)
        for i in range(len(self.conv5_x)):
            self.conv5_x[i].set_lr(new_lr)

    def lr_step(self, val_loss):
        self.fc.lr_step(val_loss)

        learning_rate_check = self.fc.get_lr()

        if self.learning_rate != learning_rate_check:
            self.learning_rate = learning_rate_check
            self.set_lr(self.learning_rate)

    def train_wts(self, input, y, topk=(1,)):
        label = F.one_hot(y, num_classes=self.num_classes)

        # self.infer(input=input, label=label)
        output = self.infer(input=input, label=label)

        with torch.no_grad():
            loss_b = self.NLL(torch.log(output), y)
            loss_batch = loss_b.item()
            # output으로 loss값 계산

        # dWs_batch, loss_batch, loss_batch_mean = self.update_weight(y)
        self.update_weight(y)

        # output = self.no_grad_forward(input)
        # loss값은 weight 수정 전 값이므로
        # 정확도를 계산할 때도 수정 전 값으로 계산할 것

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch

    def val_batch(self, input, y, topk=(1,)):
        output = self.no_grad_forward(input)

        loss_batch, loss_batch_mean = self.fc.cal_loss(output, y)

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch


# ‘fixed prediction assumption’ 적용
# SeqIL 미적용
# decay 제거
class PCResNet4(nn.Module):
    def __init__(
        self,
        block_name,
        num_block_inlayers,
        learning_rate=0.1,
        n_iter_dx=25,
        infer_rate=0.1,
        beta=100,
        momentum=0.1,
        num_cnn_output_channel=256,
        num_classes=10,
        init_weights=True,
        device="cpu",
    ):
        # block은 ResBlock 또는 Bottleneck 클래스 이름을 받아서 저장
        # num_block_inlayers 는 각 층마다 residual block이 몇개씩 들어가는지 정보를 담은 list
        super().__init__()

        if num_cnn_output_channel == 256:
            self.denominator = 1
        elif num_cnn_output_channel == 128:
            self.denominator = 2

        self.in_channels = 16 // self.denominator
        # 3x3 conv 지나면 32 or 16 채널이 됨

        self.input_size = 32

        self.learning_rate = learning_rate
        self.n_iter_dx = n_iter_dx
        self.infer_rate = infer_rate
        self.beta = beta
        self.gamma = 1 / (1 + self.beta)

        self.momentum = momentum

        self.num_classes = num_classes

        self.device = device

        self.conv1 = ConvAG(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            device=self.device,
        )

        self.conv2_x = self._make_layer(
            block_name=block_name,
            out_channels=(32 // self.denominator),
            num_blocks=num_block_inlayers[0],
            stride=1,
        )
        self.conv2_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]
        self.conv2_us = [torch.randn(1, 1) for i in range(num_block_inlayers[0] + 1)]

        self.conv3_x = self._make_layer(
            block_name=block_name,
            out_channels=(64 // self.denominator),
            num_blocks=num_block_inlayers[1],
            stride=1,
        )
        self.conv3_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]
        self.conv3_us = [torch.randn(1, 1) for i in range(num_block_inlayers[1] + 1)]

        self.conv4_x = self._make_layer(
            block_name=block_name,
            out_channels=(128 // self.denominator),
            num_blocks=num_block_inlayers[2],
            stride=2,
        )
        self.conv4_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]
        self.conv4_us = [torch.randn(1, 1) for i in range(num_block_inlayers[2] + 1)]

        self.conv5_x = self._make_layer(
            block_name=block_name,
            out_channels=(256 // self.denominator),
            num_blocks=num_block_inlayers[3],
            stride=2,
        )
        self.conv5_xs = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        self.conv5_us = [torch.randn(1, 1) for i in range(num_block_inlayers[3] + 1)]
        # residual block들을 담은 4개의 층 생성

        self.avg_pool = AdaptiveAvgPoolLayer(in_channels=(256 // self.denominator), device=self.device)
        # filter 갯수는 유지하고 Width와 Height는 지정한 값으로 average pooling을 수행해 준다
        # ex: 256, 8, 8 을 256, 1, 1 로 변경

        self.fc = FCtoSoftMax(
            in_features=(256 // self.denominator) * block_name.expansion,
            out_features=num_classes,
            learning_rate=self.learning_rate,
            bias=True,
            device=self.device,
        )
        self.softmax = nn.Softmax(dim=1)

        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
        # self._initialize_Xs(input)로 계산된 output으로 train loss를 계산할 때 사용

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

        return nn.Sequential(*blocks)
        # list 인 blocks 를 * 로 unpack해서 nn.Sequential()에 입력

    @torch.no_grad()
    def _initialize_Xs(self, x):
        # self.Xs = {'input':[], 'conv1':[], 'conv2_x':[], 'conv3_x':[], 'conv4_x':[], 'conv5_x':[], 'avg_pool':[], 'fc':[]}
        self.Xs["input"] = x.clone()
        self.Us["input"] = x.clone()

        self.Xs["conv1"] = self.conv1(self.Xs["input"].detach())
        self.Us["conv1"] = self.Xs["conv1"].clone()

        x = self.Xs["conv1"]
        for i in range(len(self.conv2_x)):
            x = self.conv2_x[i]._initialize_Xs(x.detach())
            self.conv2_xs[i + 1] = x
            self.conv2_us[i + 1] = self.conv2_xs[i + 1].clone()
        self.Xs["conv2_x"] = x
        self.Us["conv2_x"] = self.Xs["conv2_x"].clone()

        for i in range(len(self.conv3_x)):
            x = self.conv3_x[i]._initialize_Xs(x.detach())
            self.conv3_xs[i + 1] = x
            self.conv3_us[i + 1] = self.conv3_xs[i + 1].clone()
        self.Xs["conv3_x"] = x
        self.Us["conv3_x"] = self.Xs["conv3_x"].clone()

        for i in range(len(self.conv4_x)):
            x = self.conv4_x[i]._initialize_Xs(x.detach())
            self.conv4_xs[i + 1] = x
            self.conv4_us[i + 1] = self.conv4_xs[i + 1].clone()
        self.Xs["conv4_x"] = x
        self.Us["conv4_x"] = self.Xs["conv4_x"].clone()

        for i in range(len(self.conv5_x)):
            x = self.conv5_x[i]._initialize_Xs(x.detach())
            self.conv5_xs[i + 1] = x
            self.conv5_us[i + 1] = self.conv5_xs[i + 1].clone()
        self.Xs["conv5_x"] = x
        self.Us["conv5_x"] = self.Xs["conv5_x"].clone()

        self.Xs["avg_pool"] = self.avg_pool(self.Xs["conv5_x"].detach())
        self.Us["avg_pool"] = self.Xs["avg_pool"].clone()

        self.Xs["fc"] = self.fc(self.Xs["avg_pool"].detach())
        self.Us["fc"] = self.Xs["fc"].clone()

        return self.Xs["fc"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)

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

            x = self.fc(x)

            return x

    def infer(self, input, label):
        # 여기서 label은 onehot encoding 된 상태
        output = self._initialize_Xs(input)
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
            # self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.Xs["fc"]
            self.Xs["fc"] = label.clone()
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 128번째 줄

            # self.pred_errors["fc"] = self.Xs["fc"] - self.prediction()
            # inference loop 들어가기 전 self.Us 와 self.pred_errors 초기화
            # @@ 하지 않으면 loop i = 0 일 때 self.pred_errors["avg_pool"] = [] 상태라 에러 발생 @@

        for i in range(self.n_iter_dx):
            # decay = 1 / (1 + i)
            decay = 1
            # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py 132번째 줄
            with torch.no_grad():
                err = self.Xs["fc"] - self.Us["fc"]
                # PC can approximate backprop for arbitrary computation graphs (not just MLPs)
                # under the ‘fixed prediction assumption’
                # which is where during inference the predictions between layers are fixed to their feedforward pass values.
                # https://www.beren.io/2023-03-30-Thoughts-on-future-of-PC/

            dfdt = torch.clamp(self.fc.backward(x=self.Us["avg_pool"], e=err), -50, 50)

            with torch.no_grad():
                e_cur = self.Xs["avg_pool"] - self.Us["avg_pool"]
                self.Xs["avg_pool"] -= self.infer_rate * (e_cur - dfdt)

                err = e_cur
                dfdt = torch.clamp(self.avg_pool.backward(self.Us["conv5_x"], err), -50, 50)
                e_cur = self.Xs["conv5_x"] - self.Us["conv5_x"]
                self.Xs["conv5_x"] -= self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv5_x"]
                y_old = e_cur + self.Us["conv5_x"]
                for i in reversed(range(len(self.conv5_x))):
                    if i > 0:
                        dfdt = self.conv5_x[i].backward(self.conv5_xs[i], y, y_old, decay, self.infer_rate)
                        # block 안에서는 U를 사용하므로 여기 backward에선 self.conv5_us[i] 안 넣어도 된다.
                        e_cur = self.conv5_xs[i] - self.conv5_us[i]
                        self.conv5_xs[i] -= self.infer_rate * (e_cur - dfdt)
                        y = self.conv5_xs[i]
                        y_old = e_cur + self.conv5_us[i]
                    else:
                        dfdt = self.conv5_x[i].backward(self.Xs["conv4_x"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv4_x"] - self.Us["conv4_x"]
                self.Xs["conv4_x"] -= self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv4_x"]
                y_old = e_cur + self.Us["conv4_x"]
                for i in reversed(range(len(self.conv4_x))):
                    if i > 0:
                        dfdt = self.conv4_x[i].backward(self.conv4_xs[i], y, y_old, decay, self.infer_rate)
                        e_cur = self.conv4_xs[i] - self.conv4_us[i]
                        self.conv4_xs[i] -= self.infer_rate * (e_cur - dfdt)
                        y = self.conv4_xs[i]
                        y_old = e_cur + self.conv4_us[i]
                    else:
                        dfdt = self.conv4_x[i].backward(self.Xs["conv3_x"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv3_x"] - self.Us["conv3_x"]
                self.Xs["conv3_x"] -= self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv3_x"]
                y_old = e_cur + self.Us["conv3_x"]
                for i in reversed(range(len(self.conv3_x))):
                    if i > 0:
                        dfdt = self.conv3_x[i].backward(self.conv3_xs[i], y, y_old, decay, self.infer_rate)
                        e_cur = self.conv3_xs[i] - self.conv3_us[i]
                        self.conv3_xs[i] -= self.infer_rate * (e_cur - dfdt)
                        y = self.conv3_xs[i]
                        y_old = e_cur + self.conv3_us[i]
                    else:
                        dfdt = self.conv3_x[i].backward(self.Xs["conv2_x"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv2_x"] - self.Us["conv2_x"]
                self.Xs["conv2_x"] -= self.infer_rate * (e_cur - dfdt)

                y = self.Xs["conv2_x"]
                y_old = e_cur + self.Us["conv2_x"]
                for i in reversed(range(len(self.conv2_x))):
                    if i > 0:
                        dfdt = self.conv2_x[i].backward(self.conv2_xs[i], y, y_old, decay, self.infer_rate)
                        e_cur = self.conv2_xs[i] - self.conv2_us[i]
                        self.conv2_xs[i] -= self.infer_rate * (e_cur - dfdt)
                        y = self.conv2_xs[i]
                        y_old = e_cur + self.conv2_us[i]
                    else:
                        dfdt = self.conv2_x[i].backward(self.Xs["conv1"], y, y_old, decay, self.infer_rate)

                e_cur = self.Xs["conv1"] - self.Us["conv1"]
                self.Xs["conv1"] -= self.infer_rate * (e_cur - dfdt)

                # self.Xs["fc"] = (1 - self.gamma) * label.clone() + self.gamma * self.fc(self.Xs["avg_pool"])

        return output

    def update_weight(self, y):
        # y는 onehot encoding을 하지 않은 상태
        self.conv1.update_weights(self.Us["input"], self.Xs["conv1"])
        # self.Xs["input"] == self.Us["input"]은 동일

        for i in range(len(self.conv2_x)):
            if i < len(self.conv2_x) - 1 and i > 0:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.conv2_xs[i + 1])
                # block 안에서는 U를 사용하므로 여기서 self.conv2_us[i]로 바꿀 필요 없음
            elif i == len(self.conv2_x) - 1:
                self.conv2_x[i].update_weights(self.conv2_xs[i], self.Xs["conv2_x"])
            else:
                self.conv2_x[i].update_weights(self.Xs["conv1"], self.conv2_xs[i + 1])

        for i in range(len(self.conv3_x)):
            if i < len(self.conv3_x) - 1 and i > 0:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.conv3_xs[i + 1])
            elif i == len(self.conv3_x) - 1:
                self.conv3_x[i].update_weights(self.conv3_xs[i], self.Xs["conv3_x"])
            else:
                self.conv3_x[i].update_weights(self.Xs["conv2_x"], self.conv3_xs[i + 1])

        for i in range(len(self.conv4_x)):
            if i < len(self.conv4_x) - 1 and i > 0:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.conv4_xs[i + 1])

            elif i == len(self.conv4_x) - 1:
                self.conv4_x[i].update_weights(self.conv4_xs[i], self.Xs["conv4_x"])
            else:
                self.conv4_x[i].update_weights(self.Xs["conv3_x"], self.conv4_xs[i + 1])

        for i in range(len(self.conv5_x)):
            if i < len(self.conv5_x) - 1 and i > 0:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.conv5_xs[i + 1])
            elif i == len(self.conv5_x) - 1:
                self.conv5_x[i].update_weights(self.conv5_xs[i], self.Xs["conv5_x"])
            else:
                self.conv5_x[i].update_weights(self.Xs["conv4_x"], self.conv5_xs[i + 1])

        self.fc.update_weights(self.Us["avg_pool"], y)

        # return losses_layers, loss, loss_mean

    def get_lr(self):
        return self.fc.get_lr()

    def set_lr(self, new_lr):
        self.conv1.set_lr(new_lr)

        for i in range(len(self.conv2_x)):
            self.conv2_x[i].set_lr(new_lr)
        for i in range(len(self.conv3_x)):
            self.conv3_x[i].set_lr(new_lr)
        for i in range(len(self.conv4_x)):
            self.conv4_x[i].set_lr(new_lr)
        for i in range(len(self.conv5_x)):
            self.conv5_x[i].set_lr(new_lr)

    def lr_step(self, val_loss):
        self.fc.lr_step(val_loss)

        learning_rate_check = self.fc.get_lr()

        if self.learning_rate != learning_rate_check:
            self.learning_rate = learning_rate_check
            self.set_lr(self.learning_rate)

    def train_wts(self, input, y, topk=(1,)):
        label = F.one_hot(y, num_classes=self.num_classes)

        # self.infer(input=input, label=label)
        output = self.infer(input=input, label=label)

        with torch.no_grad():
            loss_b = self.NLL(torch.log(output), y)
            loss_batch = loss_b.item()
            # output으로 loss값 계산

        # dWs_batch, loss_batch, loss_batch_mean = self.update_weight(y)
        self.update_weight(y)

        # output = self.no_grad_forward(input)
        # loss값은 weight 수정 전 값이므로
        # 정확도를 계산할 때도 수정 전 값으로 계산할 것

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch

    def val_batch(self, input, y, topk=(1,)):
        output = self.no_grad_forward(input)

        loss_batch, loss_batch_mean = self.fc.cal_loss(output, y)

        (top1_count_batch, top5_count_batch), acc_batch = accuracy(output=output, target=y, topk=topk)

        return loss_batch, (top1_count_batch, top5_count_batch), acc_batch
