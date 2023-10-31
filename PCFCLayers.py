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


class FCtoSoftMax(nn.Module):
    def __init__(self, in_features, out_features, learning_rate, bias=False, device="cpu", init_weights=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias

        self.device = device

        self.linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias, device=self.device
        )
        self.weights = self.linear.weight

        if init_weights:
            self._initialize_weights()

        self.lin_and_smax = nn.Sequential(self.linear, nn.Softmax(dim=1))

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.linear.parameters(), lr=self.learning_rate, weight_decay=0)
        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
        # self.MSE = nn.MSELoss(reduction="sum").to(self.device)
        self.lr_scheduler = ReduceLROnPlateau(self.optim, mode="min", factor=0.1, patience=5)

    def forward(self, x):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # self.inp = x.clone()
        # x.clone()을 하게 되면 x가 leaf node가 아니게 된다
        # FCtoSoftMax class에서는 x->p->loss 계산한 후 backprop을 하기때문에
        # x가 leaf node여야 한다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # self.inp = x

        x = self.lin_and_smax(x)

        # self.pre_activation = x

        return x
        # 반환값도 self.pre_activation이 아니라 x = self.linear(x) 에서 나온 x를 바로 반환할것

    def backward(self, x, e):
        # 복잡한 softmax의 derivative를 정의하는 대신 torch.autograd.functional.vjp 사용
        # x는 이 fc 레이어에 들어오는 input, e는 이 레이어의 output으로 계산된 pred_error
        _, dfdt = torch.autograd.functional.vjp(self.lin_and_smax, x, e)

        return dfdt

    def update_weights(self, x, y):
        # y는 onehot encoding을 하지 않은 상태
        # x = x.detach().clone()
        # print(f"==>> x.requires_grad: {x.requires_grad}")
        # print(f"==>> y.requires_grad: {y.requires_grad}")
        x = x.detach().clone().requires_grad_(requires_grad=True)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # update_weights 함수에 x를 넣을 때 .detach()를 해주게 바꾸면 이 함수 안에서 detach()를 할 필요는 없다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # y = y.detach().clone().requires_grad_(requires_grad=True)
        # print(f"==>> x.requires_grad: {x.requires_grad}")

        p = self.forward(x)
        # p = self.forward(x).requires_grad_(requires_grad=True)
        # print(f"==>> p.grad_fn: {p.grad_fn}")
        # loss_real = self.MSE(p, F.one_hot(y,num_classes=10).float())
        loss = self.NLL(torch.log(p), y)
        # print(f"==>> loss.grad_fn: {loss.grad_fn}")
        # loss.requires_grad = True
        # loss_mean = loss / p.size(0)
        # print(f"==>> loss.grad_fn: {loss.grad_fn}")
        # p.size(0) == batch_size

        self.optim.zero_grad()
        loss.backward()
        # loss_real.backward()
        self.optim.step()

        # return loss.item(), loss_mean.item()

    def cal_loss(self, p, y):
        loss = self.NLL(torch.log(p), y)
        loss_mean = loss / p.size(0)

        return loss.item(), loss_mean.item()

    def _initialize_weights(self):
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)
        # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    # 현재 lr값을 알려주는 함수 -> 훈련 중 print에 사용
    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group["lr"]

    def lr_step(self, val_loss):
        self.lr_scheduler.step(val_loss)
