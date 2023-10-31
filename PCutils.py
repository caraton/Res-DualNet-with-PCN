import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# import copy

# import json

# n01443537 같이 되어있는 클래스 이름들을 goldfish 와 같이 쉽게 바꿔줄 때 사용할 파일이 JSON파일
# import os

# os.path.join(save_path, filename) 으로 파일 경로 합칠 때 사용
# import shutil

# shutil.copyfile(path_a, path_b) a 경로의 파일을 b 경로에 복사

# import time
# import datetime

# 시간 측정에 사용


USE_CUDA = torch.cuda.is_available()
# GPU 사용가능하면 True 반환

device = torch.device("cuda" if USE_CUDA else "cpu")


def linear(x):
    return x


def d_linear(x):
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # return torch.ones_like(x, dtype=float, device=device)
    # dtype=float 으로 두면 float64가 되어서 계산에 문제 생김
    return torch.ones_like(x, device=device).float()
    # return torch.ones_like(x, dtype=torch.float32, device=device)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def relu(x):
    return torch.clamp(x, min=0)


def d_relu(x):
    rel = relu(x)
    rel[rel > 0] = 1
    return rel


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        # pred는 [batch_size, maxk] 꼴
        # largest=True 이므로 각 행 1열이 최댓값 top1의 index이다.
        pred = pred.t()
        # transpose 해서 [maxk, batch_size] 로 변경
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # target을 view로 [1, batchsize] 꼴로 바꾸고
        # expand_as(pred) 로 [maxk, batch_size] 꼴이 되어 pred랑 동일한 size가 된다.
        # pred 가 각 열마다 top1, top2, top3 순으로 되어 있기때문에
        # correct는 각 열마다 top1 == target, top2 == target, top3 == target 순으로 되어 있음

        res = []
        respercent = []
        for k in topk:
            # topk = (1, maxk) 꼴
            correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
            # k가 1일 때는 correct[:1]을 하면 [1, batch_size] 형태로 top1 == target 만 남고
            # k가 maxk 일때는 correct[:k] 는 [maxk, batch_size] 형태로 전부 다 들어 있음
            # 또 하나의 열은 전부 다 False 이거나 (top 1 ~ top k 다 틀림), 하나만 True (top 1 ~ top k 중 하나가 정답이랑 일치)
            # ==> 따라서 reshape(-1) 로 [maxk * batch_size] 꼴이 되어도 True 갯수는 batch_size를 넘지 않음
            # float() 로 변환한 후 sum함수로 다 더하면 정답 갯수가 나옴
            # keepdim=True를 두면 sum 값이 13.0 이렇게 float 하나만 반환하지 않고
            # [13.0] 과 같이 (1,) 꼴로 반환해줌

            percent_k = correct_k.mul(100.0 / batch_size)
            # .mul_을 쓰면 원본도 변경되므로 .mul

            correct_k = correct_k.detach().to("cpu").numpy()
            percent_k = percent_k.detach().to("cpu").numpy()
            # https://byeongjo-kim.tistory.com/32

            res.append(correct_k)
            respercent.append(percent_k)

        return res, respercent
        # res = [ndarray(top1 개수,), ndarray(top5 개수,)] 반환
        # respercent = [ndarray(top1 %,), ndarray(top5 %,)] 반환

        # res = [tensor([top1 개수]), tensor([topk 개수])] 반환
        # respercent = [tensor([top1 %]), tensor([topk %])] 반환
