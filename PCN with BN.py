# 참조한 코드들
# https://deep-learning-study.tistory.com/534
# https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/imagenet/main.py#L171
# https://ai.dreamkkt.com/54
# https://github.com/BerenMillidge/PredictiveCodingBackprop
# https://github.com/nalonso2/PredictiveCoding-MQSeqIL/tree/main
# https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py


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
from PCNet import *
from PCfunctions import *


def testpcnet34(learning_rate):
    return PCNet(PCBlockTest, [3, 4, 6, 3], learning_rate=learning_rate, device=device)


def pcnet34(learning_rate):
    return PCNet(PCBlockBN, [3, 4, 6, 3], learning_rate=learning_rate, device=device, n_iter_dx=25)


def pcnet34_10(learning_rate):
    return PCNet(PCBlockBN, [3, 4, 6, 3], learning_rate=learning_rate, device=device, n_iter_dx=10)


if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    # GPU 사용가능하면 True 반환

    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"==>> device: {device}")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.262]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.262]),
        ]
    )

    # normalize 값 계산 : https://github.com/kuangliu/pytorch-cifar/issues/19

    batch_size = 32

    train_set = dsets.CIFAR10(root="../CIFAR10", train=True, download=True, transform=transform)
    # train_set.data는 (50000, 32, 32, 3)꼴
    # train_set.targets는 (50000,) 꼴

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = dsets.CIFAR10(root="../CIFAR10", train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    learning_rate = 0.01

    # model = testpcnet34(learning_rate)
    # model = pcnet34(learning_rate)
    model = pcnet34_10(learning_rate)
    # torchsummary.summary(model, (3,32,32), batch_size)

    # load_state(model=model, load_path='../CIFAR10/data/2023-09-13-12;18;53_epoch_7.pth')
    # date_load_file = '2023-09-13-12;18;53_history'
    # load_path = os.path.join("../CIFAR10/data/", f'{date_load_file}.json')
    # # load_path = '../CIFAR10/data/2023-09-13-12;18;53_history.json'
    # loss_history, acc_history, total_num_epochs, time_history = load_history(load_path=load_path)

    loss_history = {"train": [], "val": []}
    acc_history = {"train_top1": [], "val_top1": [], "train_top5": [], "val_top5": []}
    total_num_epochs = 0
    time_history = []

    # params_train = {
    #     "num_epochs": 5,
    #     "train_loader": train_loader,
    #     "val_loader": val_loader,
    #     "sanity_check": True,
    #     # 모델 오류 확인 떄 sanity_check True로 두면 빠르게 확인 가능
    #     "save_path": "../CIFAR10/data/",
    #     "loss_history": loss_history,
    #     "acc_history": acc_history,
    #     "total_num_epochs": total_num_epochs,
    #     "time_history": time_history
    # }

    params_train = {
        "num_epochs": 20,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "sanity_check": False,
        # 모델 오류 확인 떄 sanity_check True로 두면 빠르게 확인 가능
        "save_path": "../CIFAR10/data/",
        "loss_history": loss_history,
        "acc_history": acc_history,
        "total_num_epochs": total_num_epochs,
        "time_history": time_history,
    }

    trained_model, loss_hist, acc_hist, total_num_epochs, time_hist = train_and_val(
        model=model, params=params_train
    )

    # plot loss progress
    plt.title("Train-Val Loss")
    plt.plot(range(1, total_num_epochs + 1), loss_hist["train"], label="train")
    plt.plot(range(1, total_num_epochs + 1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    # plot top1 accuracy progress
    plt.title("Train-Val Top1 Accuracy")
    plt.plot(range(1, total_num_epochs + 1), acc_hist["train_top1"], label="train")
    plt.plot(range(1, total_num_epochs + 1), acc_hist["val_top1"], label="val")
    plt.ylabel("Top1 Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    # plot top5 accuracy progress
    plt.title("Train-Val Top5 Accuracy")
    plt.plot(range(1, total_num_epochs + 1), acc_hist["train_top5"], label="train")
    plt.plot(range(1, total_num_epochs + 1), acc_hist["val_top5"], label="val")
    plt.ylabel("Top5 Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()
