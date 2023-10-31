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

from PCutils import *

USE_CUDA = torch.cuda.is_available()
# GPU 사용가능하면 True 반환

device = torch.device("cuda" if USE_CUDA else "cpu")


# 훈련 weight 등 저장 함수
def save_state_simple(model, is_best, save_path, filename, timestamp=""):
    filename = os.path.join(save_path, filename)

    states = {
        "model_state_dict": model.state_dict(),
        "conv1_optim": model.conv1.optim.state_dict(),
        "conv2_optim": model.conv2.optim.state_dict(),
        "conv3_optim": model.conv3.optim.state_dict(),
        "conv4_optim": model.conv4.optim.state_dict(),
        "fc_optim": model.fc.optim.state_dict(),
        "fc_lr_scheduler": model.fc.lr_scheduler.state_dict(),
    }

    torch.save(states, filename)

    if is_best:
        bestname = os.path.join(save_path, f"model_best_{timestamp}.pth")
        shutil.copyfile(filename, bestname)
        # filename 경로에 있는 파일을 bestname 경로에 복사 저장


def save_state_resnet(model, is_best, save_path, filename, timestamp=""):
    filename = os.path.join(save_path, filename)

    states = {
        "model_state_dict": model.state_dict(),
        "conv1_optim": model.conv1.optim.state_dict(),
        "fc_optim": model.fc.optim.state_dict(),
        "fc_lr_scheduler": model.fc.lr_scheduler.state_dict(),
    }

    module_number = 0
    optim_str = "_optim"
    for block in model.conv2_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()
        module_number += 1
    for block in model.conv3_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()
        module_number += 1
    for block in model.conv4_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()
        module_number += 1
    for block in model.conv5_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()
        module_number += 1

    torch.save(states, filename)

    if is_best:
        bestname = os.path.join(save_path, f"model_best_{timestamp}.pth")
        shutil.copyfile(filename, bestname)
        # filename 경로에 있는 파일을 bestname 경로에 복사 저장


def save_state_shufflenet(model, is_best, save_path, filename, timestamp=""):
    filename = os.path.join(save_path, filename)

    states = {
        "model_state_dict": model.state_dict(),
        "conv1_optim": model.conv1.optim.state_dict(),
        "fc_optim": model.fc.optim.state_dict(),
        "fc_lr_scheduler": model.fc.lr_scheduler.state_dict(),
    }

    module_number = 0
    optim_str = "_optim"
    optim_short_str = "_optim_short"
    for block in model.conv2_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()

        if not block.split:
            for i in range(len(block.optims_short)):
                states[str(module_number) + f"_{i}" + optim_short_str] = block.optims_short[i].state_dict()

        module_number += 1
    for block in model.conv3_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()

        if not block.split:
            for i in range(len(block.optims_short)):
                states[str(module_number) + f"_{i}" + optim_short_str] = block.optims_short[i].state_dict()

        module_number += 1
    for block in model.conv4_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()

        if not block.split:
            for i in range(len(block.optims_short)):
                states[str(module_number) + f"_{i}" + optim_short_str] = block.optims_short[i].state_dict()

        module_number += 1
    for block in model.conv5_x:
        states[str(module_number)] = block.state_dict()
        for i in range(len(block.optims)):
            states[str(module_number) + f"_{i}" + optim_str] = block.optims[i].state_dict()

        if not block.split:
            for i in range(len(block.optims_short)):
                states[str(module_number) + f"_{i}" + optim_short_str] = block.optims_short[i].state_dict()

        module_number += 1

    torch.save(states, filename)

    if is_best:
        bestname = os.path.join(save_path, f"model_best_{timestamp}.pth")
        shutil.copyfile(filename, bestname)
        # filename 경로에 있는 파일을 bestname 경로에 복사 저장


save_functions_table = {
    "simple": save_state_simple,
    "resnet": save_state_resnet,
    "shufflenet": save_state_shufflenet,
}
# model 이름따라 다른 save_state_모델이름 함수를 부르고 싶을 때 사용


# 훈련 weight 등 불러오기 함수
def load_state_simple(model: nn.Module, load_path):
    load_dict = torch.load(load_path)

    model.load_state_dict(load_dict["model_state_dict"])
    model.conv1.optim.load_state_dict(load_dict["conv1_optim"])
    model.conv2.optim.load_state_dict(load_dict["conv2_optim"])
    model.conv3.optim.load_state_dict(load_dict["conv3_optim"])
    model.conv4.optim.load_state_dict(load_dict["conv4_optim"])
    model.fc.optim.load_state_dict(load_dict["fc_optim"])
    model.fc.lr_scheduler.load_state_dict(load_dict["fc_lr_scheduler"])


def load_state_resnet(model: nn.Module, load_path):
    load_dict = torch.load(load_path)

    model.load_state_dict(load_dict["model_state_dict"])
    model.conv1.optim.load_state_dict(load_dict["conv1_optim"])

    module_number = 0
    optim_str = "_optim"
    for block in model.conv2_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])
        module_number += 1
    for block in model.conv3_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])
        module_number += 1
    for block in model.conv4_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])
        module_number += 1
    for block in model.conv5_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])
        module_number += 1

    model.fc.optim.load_state_dict(load_dict["fc_optim"])
    model.fc.lr_scheduler.load_state_dict(load_dict["fc_lr_scheduler"])


def load_state_shufflenet(model: nn.Module, load_path):
    load_dict = torch.load(load_path)

    model.load_state_dict(load_dict["model_state_dict"])
    model.conv1.optim.load_state_dict(load_dict["conv1_optim"])

    module_number = 0
    optim_str = "_optim"
    optim_short_str = "_optim_short"
    for block in model.conv2_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])

        if not block.split:
            for i in range(len(block.optims_short)):
                block.optims_short[i].load_state_dict(
                    load_dict[str(module_number) + f"_{i}" + optim_short_str]
                )

        module_number += 1
    for block in model.conv3_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])

        if not block.split:
            for i in range(len(block.optims_short)):
                block.optims_short[i].load_state_dict(
                    load_dict[str(module_number) + f"_{i}" + optim_short_str]
                )

        module_number += 1
    for block in model.conv4_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])

        if not block.split:
            for i in range(len(block.optims_short)):
                block.optims_short[i].load_state_dict(
                    load_dict[str(module_number) + f"_{i}" + optim_short_str]
                )

        module_number += 1
    for block in model.conv5_x:
        block.load_state_dict(load_dict[str(module_number)])
        for i in range(len(block.optims)):
            block.optims[i].load_state_dict(load_dict[str(module_number) + f"_{i}" + optim_str])

        if not block.split:
            for i in range(len(block.optims_short)):
                block.optims_short[i].load_state_dict(
                    load_dict[str(module_number) + f"_{i}" + optim_short_str]
                )

        module_number += 1

    model.fc.optim.load_state_dict(load_dict["fc_optim"])
    model.fc.lr_scheduler.load_state_dict(load_dict["fc_lr_scheduler"])


load_functions_table = {
    "simple": load_state_simple,
    "resnet": load_state_resnet,
    "shufflenet": load_state_shufflenet,
}


# loss history, acc history 저장 함수
def save_history(loss_history, acc_history, time_history, total_num_epochs, save_path, filename):
    filepath = os.path.join(save_path, filename)
    with open(filepath, "w") as f:
        json.dump([loss_history, acc_history, time_history, total_num_epochs], f)


def load_history(load_path):
    with open(load_path, "r") as f:
        contents = f.read()
        loss_history, acc_history, time_history, total_num_epochs = json.loads(contents)

    return loss_history, acc_history, time_history, total_num_epochs


# https://blog.naver.com/PostView.naver?blogId=mmmy2513&logNo=222299816049&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
# sanity check: 긴시간이 들어가는 본 학습에 들어가기전
# 전체 데이터 중 매우 작은 일부에만 학습을 반복 진행 해보는 것
# 제대로된 모델이라면 overfitting이 일어나 loss값이 빠르게 0에 수렴하는 것을 볼 수 있음


def loss_epoch(model, data_loader, sanity_check=False, is_training=True):
    running_loss = 0.0
    running_top1_count = 0.0
    running_top5_count = 0.0
    len_data = len(data_loader.dataset)
    print(f"==>> len_data: {len_data}")
    # trainloader.dataset의 길이는 50000,
    # testloader.dataset은 10000

    # sanity check 때 사용
    # if sanity_check:
    #     temp = 0

    for x_b, y_b in data_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        if is_training:
            loss_b, (top1_count_b, top5_count_b), acc_b = model.train_wts(x_b, y_b, topk=(1, 5))
        else:
            loss_b, (top1_count_b, top5_count_b), acc_b = model.val_batch(x_b, y_b, topk=(1, 5))

        running_loss += loss_b

        running_top1_count += float(top1_count_b[0])
        # top1_count_b 는 크기가 1인 numpy array이므로 [0]으로 접근해야 값이 나옴
        running_top5_count += float(top5_count_b[0])

        if sanity_check:
            # temp += 1

            # if temp == 5:
            #     len_data = 160
            #     print(f"==>> len_data: {len_data}")
            #     break

            len_data = x_b.size(0)
            print(f"==>> len_data: {len_data}")
            break

            # 모델 설계 간단체크때는 첫배치만 하고 break해서
            # 매 epoch마다 똑같은 첫번째 배치만 사용하는 것 아닌가?
            # 지금 코드 방식으로는 dataloader가 iterator이기때문에
            # 매 epoch마다 전체를 도는 것은 아니지만
            # 배치 한개 하고나서 다음 epoch에서는 다른 배치를 사용
            # 또 dataloader가 shuffle도 하고 있기때문에 매번 dataloader를 초기화해도 문제?

    loss = running_loss / len_data

    acc1 = running_top1_count * 100.0 / len_data
    acc5 = running_top5_count * 100.0 / len_data

    return loss, acc1, acc5


def train_and_val(model, params, m_name="resnet"):
    print(str(datetime.datetime.now()).split(".")[0])
    # 시작시간

    global save_functions_table

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
        # for block in model.conv2_x.blocks:
        #     block.train()
        # for block in model.conv3_x.blocks:
        #     block.train()
        # for block in model.conv4_x.blocks:
        #     block.train()
        # for block in model.conv5_x.blocks:
        #     block.train()
        # https://mindee.com/blog/batch-normalization/
        # "after training, we freeze all the weights of the model
        # and run one epoch in to estimate the moving average on the whole dataset."
        # 새로운 epoch으로 넘어가면 running estimate들을 다 초기화 해주기

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

        # for i, block in enumerate(model.conv5_x.blocks):
        #     if i == len(model.conv5_x.blocks) - 1:
        #         print(f"==>> block.DP2.BN_weight: {block.DP2.BN_weight}")
        #         print(f"==>> block.DP2.BN_bias: {block.DP2.BN_bias}")
        #         print(f"==>> block.DP2.running_mean: {block.DP2.running_mean}")
        #         print(f"==>> block.DP2.running_var: {block.DP2.running_var}")

        # 검증
        model.eval()
        # model.eval()만 하면 block안의 모듈들은
        # self.training값이 안변한다
        # for block in model.conv2_x.blocks:
        #     block.eval()
        # for block in model.conv3_x.blocks:
        #     block.eval()
        # for block in model.conv4_x.blocks:
        #     block.eval()
        # for block in model.conv5_x.blocks:
        #     block.eval()

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
            save_functions_table[m_name](
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
