import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

import json

# n01443537 같이 되어있는 클래스 이름들을 goldfish 와 같이 쉽게 바꿔줄 때 사용할 파일이 JSON파일
import os

# os.path.join(save_path, filename) 으로 파일 경로 합칠 때 사용
import shutil

# shutil.copyfile(path_a, path_b) a 경로의 파일을 b 경로에 복사

import time
import datetime

# 시간 측정에 사용

from PCfunctions import *


USE_CUDA = torch.cuda.is_available()
# GPU 사용가능하면 True 반환

device = torch.device("cuda" if USE_CUDA else "cpu")


# 훈련 weight 등 저장 함수
def save_state_small(model, is_best, save_path, filename, timestamp=""):
    filename = os.path.join(save_path, filename)

    states = {
        "model_state_dict": model.state_dict(),
        "conv1_optim": model.conv1.optim.state_dict(),
        "fc_optim": model.fc.optim.state_dict(),
        "fc_lr_scheduler": model.fc.lr_scheduler.state_dict(),
    }

    module_number = 0
    DP1_str = "_DP1_optim"
    PW_str = "_PW_optim"
    DP2_str = "_DP2_optim"
    for block in model.conv2_x.blocks:
        states[str(module_number)] = block.state_dict()
        states[str(module_number) + DP1_str] = block.DP1.optim.state_dict()
        states[str(module_number) + PW_str] = block.PW.optim.state_dict()
        states[str(module_number) + DP2_str] = block.DP2.optim.state_dict()
        module_number += 1
    for block in model.conv3_x.blocks:
        states[str(module_number)] = block.state_dict()
        states[str(module_number) + DP1_str] = block.DP1.optim.state_dict()
        states[str(module_number) + PW_str] = block.PW.optim.state_dict()
        states[str(module_number) + DP2_str] = block.DP2.optim.state_dict()
        module_number += 1

    torch.save(states, filename)

    if is_best:
        bestname = os.path.join(save_path, f"model_best_{timestamp}.pth")
        shutil.copyfile(filename, bestname)
        # filename 경로에 있는 파일을 bestname 경로에 복사 저장


# 훈련 weight 등 불러오기 함수
def load_state_small(model: nn.Module, load_path):
    load_dict = torch.load(load_path)

    model.load_state_dict(load_dict["model_state_dict"])
    model.conv1.optim.load_state_dict(load_dict["conv1_optim"])
    model.fc.optim.load_state_dict(load_dict["fc_optim"])
    model.fc.lr_scheduler.load_state_dict(load_dict["fc_lr_scheduler"])

    module_number = 0
    DP1_str = "_DP1_optim"
    PW_str = "_PW_optim"
    DP2_str = "_DP2_optim"
    for block in model.conv2_x.blocks:
        block.load_state_dict(load_dict[str(module_number)])
        block.DP1.optim.load_state_dict(load_dict[str(module_number) + DP1_str])
        block.PW.optim.load_state_dict(load_dict[str(module_number) + PW_str])
        block.DP2.optim.load_state_dict(load_dict[str(module_number) + DP2_str])
        module_number += 1
    for block in model.conv3_x.blocks:
        block.load_state_dict(load_dict[str(module_number)])
        block.DP1.optim.load_state_dict(load_dict[str(module_number) + DP1_str])
        block.PW.optim.load_state_dict(load_dict[str(module_number) + PW_str])
        block.DP2.optim.load_state_dict(load_dict[str(module_number) + DP2_str])
        module_number += 1


def train_and_val_small(model, params):
    print(str(datetime.datetime.now()).split(".")[0])
    # 시작시간

    num_epochs = params["num_epochs"]
    t_loader = params["train_loader"]
    v_loader = params["val_loader"]
    sanity_check = params["sanity_check"]
    save_path = params["save_path"]

    loss_history = params["loss_history"]
    acc_history = params["acc_history"]
    total_num_epochs = params["total_num_epochs"]
    time_history = params["time_history"]

    # save, load를 위해 수정할 때는 함수 인수로 load된 loss_history와 acc_history를 받기
    # loss_history = {'train':[], 'val':[]}
    # acc_history = {'train_top1':[], 'val_top1':[], 'train_top5':[], 'val_top5':[]}

    best_loss = float("inf")
    # float('inf') 는 +∞, float('-inf') 는 -∞ (@@int('inf')는 안된다)

    for epoch in range(num_epochs):
        start_time = time.time()

        current_lr = model.get_lr()
        # print에 쓸 현재 learning rate 값 불러오기

        print("".center(50, "-"))
        print(str(datetime.datetime.now()).split(".")[0])
        # 이번 epoch 시작시간
        print(f"Epoch {epoch}/{num_epochs-1}, current lr = {current_lr}")

        # 훈련
        model.train()
        for block in model.conv2_x.blocks:
            block.train()
        for block in model.conv3_x.blocks:
            block.train()

        train_loss, train_acc1, train_acc5 = loss_epoch(
            model=model,
            data_loader=t_loader,
            sanity_check=sanity_check,
            is_training=True,
        )
        # loss_epoch 함수에서 이번 epoch forward pass backward pass 둘다 진행하고 loss값, acc값 다 계산

        loss_history["train"].append(train_loss)
        acc_history["train_top1"].append(train_acc1)
        acc_history["train_top5"].append(train_acc5)

        train_time = time.time()
        train_elapsed_time = datetime.timedelta(seconds=(train_time - start_time))
        train_elapsed_time_ = str(train_elapsed_time).split(".")[0]

        print(
            f"train loss: {train_loss:>.9}, train accuracy: (top1: {train_acc1:3.2f}%, top5: {train_acc5:3.2f}%)"
        )
        print(f"elapsed time: {train_elapsed_time_}")

        # 검증
        model.eval()
        # model.eval()만 하면 block안의 모듈들은
        # self.training값이 안변한다
        for block in model.conv2_x.blocks:
            block.eval()
        for block in model.conv3_x.blocks:
            block.eval()

        with torch.no_grad():
            val_loss, val_acc1, val_acc5 = loss_epoch(
                model=model,
                data_loader=v_loader,
                sanity_check=sanity_check,
                is_training=False,
            )

        loss_history["val"].append(val_loss)
        acc_history["val_top1"].append(val_acc1)
        acc_history["val_top5"].append(val_acc5)

        is_best = False
        if val_loss < best_loss:
            best_loss = val_loss
            is_best = True

        model.lr_step(val_loss)
        # lr 값을 변경할지 말지 lr_scheduler가 판단하도록 val_loss 넘기기

        val_time = time.time()
        val_elapsed_time = datetime.timedelta(seconds=(val_time - train_time))
        val_elapsed_time_ = str(val_elapsed_time).split(".")[0]

        print(f"val loss: {val_loss:>.9}, val accuracy: (top1: {val_acc1:3.2f}%, top5: {val_acc5:3.2f}%)")
        print(f"elapsed time: {val_elapsed_time_}")

        epoch_elapsed_time = train_elapsed_time + val_elapsed_time
        epoch_elapsed_time_ = str(epoch_elapsed_time).split(".")[0]
        print(f" epoch elapsed time = {epoch_elapsed_time_}")
        time_history.append(epoch_elapsed_time_)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 이번 epoch weight, opt, lr_scheduler 저장
        # timestamp = str(datetime.datetime.now()).split(" ")[0]
        timestamp = str(datetime.datetime.now()).split(" ")
        timestamp = timestamp[0] + "-" + ";".join(timestamp[1].split(".")[0].split(":"))
        # 2023-08-08 또는 2023-08-08_13;52;06
        # @@@@@@@ 파일 이름에 : 사용 불가능 ==> ; 로 변경

        if sanity_check is False:
            save_state_small(
                model=model,
                is_best=is_best,
                save_path=save_path,
                filename=f"{timestamp}_epoch_{total_num_epochs}.pth",
                timestamp=timestamp,
            )

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        total_num_epochs += 1
        print(f"==>> total_num_epochs: {total_num_epochs}")
        save_history(
            loss_history=loss_history,
            acc_history=acc_history,
            time_history=time_history,
            total_num_epochs=total_num_epochs,
            save_path=save_path,
            filename=f"{timestamp}_history.json",
        )

    return model, loss_history, acc_history, time_history, total_num_epochs
