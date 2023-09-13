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


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        learning_rate,
        stride=1,
        padding=1,
        bias=False,
        f=linear,
        df=d_linear,
        device="cpu",
        init_weights=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride
        self.padding = padding
        self.output_size = (
            math.floor((self.input_size + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        )
        self.bias = bias

        self.f = f
        self.df = df

        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            device=self.device,
        )
        self.flat_weights = self.conv.weight.view(self.out_channels, -1)
        # # self.flat_weights의 size는 (output_channel_number, input_channel_number*kernel_size*kernel_size)

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.conv.parameters(), lr=self.learning_rate)

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride
        ).to(self.device)
        self.fold = nn.Fold(
            output_size=(self.input_size, self.input_size),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride,
        ).to(self.device)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.X_col = self.unfold(x.clone())
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # unfold는 deepcopy가 아니기때문에 x가 변해도 X_col이 안변하도록 x.clone() 입력
        # .clone()은 deepcopy를 해서 새로운 메모리에 할당하면서 grad_fn 히스토리는 유지한다
        # 그러나 leaf node를 leaf node가 아니게 바꾸기때문에
        # in-place operation이 금지된 leaf node와 다르게 in-place operation이 가능해진다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # input의 size는 (batch_size, input_channel_number, self.input_size, self.input_size)
        # nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride) 적용 후
        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # @@ sliding_local_block_number == self.output_size * self.output_size

        self.pre_activation = self.conv(x)

        return self.f(self.pre_activation)

    def backward(self, e):
        dfx = self.df(self.pre_activation)
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # μ' = f'(self.pre_activation)
        e = e * dfx
        self.e = e.reshape(-1, self.out_channels, self.output_size * self.output_size)
        # self.e의 size는 (batch_size, output_channel_number, self.output_size * self.output_size)

        dX_col = self.flat_weights.T @ self.e
        # self.flat_weights의 size는 (output_channel_number, input_channel_number*kernel_size*kernel_size)
        # transpose 하면 (input_channel_number*kernel_size*kernel_size, output_channel_number)
        # batch_size 갯수만큼 broadcasting 되어 self.dout에 연산
        # =>(batch_size, input_channel_number*kernel_size*kernel_size, output_channel_number) @ (batch_size, output_channel_number, self.output_size * self.output_size)
        # ==> (batch_size, input_channel_number*kernel_size*kernel_size, self.output_size * self.output_size)

        dX = self.fold(dX_col)
        # (batch_size, input_channel_number*kernel_size*kernel_size, self.output_size * self.output_size)에
        # nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride)를
        # (batch_size, input_channel_number, self.input_size, self.input_size)로 변경
        # @@@ fold는 unfold의 반대방향. 겹치는 부분들은 더한다. @@@

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # M = input_channel_number, N = output_channel_number, k = kernel_size, in = self.input_size, out = self.output_size
        #
        # (input_channel_number*kernel_size*kernel_size, output_channel_number) @ (output_channel_number, self.output_size * self.output_size) 행렬곱
        # => (M*k*k , N) @ (N, out*out) 행렬곱
        #
        # 이 행렬곱을 outer product들의 합으로 보면 (1열ⓧ1행 + 2열ⓧ2행 + .... + N열ⓧN행)
        # @@@  n번째필터e맵 := (e * fn_deriv)[n] == [ε1, ε2, ......, ε(out*out)]  @@@
        # @@@  n번째필터weight벡터 := (M, k, k) tensor인 n번째 필터를 M*k*k 차원 column vector로 변환한 것  @@@
        # ==> (1번째필터weight벡터 ⓧ 1번째필터e맵 + 2번째필터weight벡터 ⓧ 2번째필터e맵 + ..... + N번째필터weight벡터 ⓧ N번째필터e맵)
        #
        # n번째 항 n번째필터weight벡터 ⓧ n번째필터e맵 만 보면
        # column 하나하나는 n번째필터weight벡터에 ε1, ε2, ...., ε(out*out)를 scalar 곱 해놓은 것과 같다.
        # nn.Fold는 column 하나하나를 sliding block으로 다시 접어주기때문에 각 열들을 하나 하나 sliding block으로 바라보면
        # 각 sliding block들은 모두 동일한 weight를 가진 block에 ε1, ε2, ...., ε(out*out) 만 다르게 곱해진 것과 같다.
        # stride = 1 이고 in == out 일 때
        # (i,j) 자리에는 ε(i,j) * weight block 이 자리하게 된다.
        # ===> nn.Fold는 겹치는 부분은 더하기에
        # x(m, i, j) 자리에는 ∑∑{ε(i+k,j+l) * nthW(m, i-k, j-l)} 값이 들어간다.
        #
        # 이제 다시 1~N 번째 모든 항을 다시 고려하면
        # x(m, i, j) == ₁∑ⁿ∑∑{nthε(i+k,j+l) * nthW(m, i-k, j-l)} 이고
        # 이 값은 구하고자 하는 ^e(l) 값이다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return dX

    def update_weights(self, e):
        dfx = self.df(self.pre_activation)
        e = e * dfx
        # 샘플 1개의 output channel 1개( (1, 1, output_size, output_size) 꼴)에서만 볼 때, (i = 1~output_size, j = 1~output_size)
        # e(i,j) = x(l+1)(i,j) - μ(l+1)(i,j) 이고 μ(l+1)(i,j) = relu(합성곱(w,x(l))) 이면
        # dF/dw = -∑( e(i,j) * relu'( 합성곱(w,x(l)) ) * x(l) ) 꼴이 된다

        self.e = e.reshape(-1, self.out_channels, self.output_size * self.output_size)
        # self.dout의 size는 (batch_size, output_channel_number, self.output_size * self.output_size)
        # @@ self.output_size * self.output_size == sliding_local_block_number
        dW = self.e @ self.X_col.permute(0, 2, 1)
        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # .permute(0번째dim, 2번째dim, 1번째 dim)을 하면
        # size는 (batch_size, sliding_local_block_number, input_channel_number*kernel_size*kernel_size)로 변경

        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # .permute(0번째dim, 2번째dim, 1번째 dim)을 하면
        # size는 (batch_size, sliding_local_block_number, input_channel_number*kernel_size*kernel_size)로 변경
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # ==> (input_channel_number*kernel_size*kernel_size, sliding_local_block_number)꼴의 2차행렬이 batch_size 개 있던 걸
        # transpose해서 (sliding_local_block_number, input_channel_number*kernel_size*kernel_size)꼴의 2차행렬이 batch_size 개 있는 걸로 변경
        # @@ tensor a(i,j,k)를 permute 해서 만든 b(i,k,j)가 있을 때 a(i,j,k) == b(i,k,j) @@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # (batch_size, output_channel_number, sliding_local_block_number) @ (batch_size, sliding_local_block_number, input_channel_number*kernel_size*kernel_size)는
        # 동일한 batch 넘버를 가지는 (output_channel_number, sliding_local_block_number) 행렬과
        # (sliding_local_block_number, input_channel_number*kernel_size*kernel_size) 행렬 사이의 행렬 곱을 수행
        # 따라서 (batch_size, output_channel_number, input_channel_number*kernel_size*kernel_size) 꼴이 된다.

        # dF/dw = -∑( e(i,j) * relu'( 합성곱(w,x(l)) ) * x(l) ) 인데 -를 곱하지 않았기 때문에
        # 여기서의 dW는 -dF/dw

        dW = torch.sum(dW, dim=0)
        # (batch_size, output_channel_number, input_channel_number*kernel_size*kernel_size) 꼴을 dim=0 batch 차원으로 sum하면
        # (output_channel_number, input_channel_number*kernel_size*kernel_size) 꼴이 된다.
        # batch안의 input 하나 하나 마다 계산된 dW를 다 더하는 것과 동일
        # ∵ 샘플 하나의 F값을 Fi로 두고 F = ∑Fi 라고 하면 (Fi는 ln(확률i)로 정의되어 있으므로 F = ∑Fi로 두어야 F = ln(π확률i) 꼴이 된다.)
        # dF/dW == ∑(dFi/dW)

        dW = dW.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        # (output_channel_number, input_channel_number*kernel_size*kernel_size) 꼴을
        # (output_channel_number, input_channel_number, kernel_size, kernel_size) 꼴로 변경

        self.conv.weight.grad = -dW

        self.optim.step()

        return dW

    def _initialize_weights(self):
        if self.f is linear:
            # self.conv.weight.normal_(mean=0,std=0.05)
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        elif self.f is relu:
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.conv.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            nn.init.constant_(self.conv.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    def set_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group["lr"] = new_lr

    # def get_true_weight_grad(self):
    #     return self.kernel.grad

    # def set_weight_parameters(self):
    #     self.kernel = nn.Parameter(self.kernel)

    # def save_layer(self,logdir,i):
    # np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.kernel.detach().cpu().numpy())

    # def load_layer(self,logdir,i):
    # kernel = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    # self.kernel = set_tensor(torch.from_numpy(kernel))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# https://pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html
# https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py
# 나중에 가능하면
# torch.autograd.functional.vjp(self.wts[layer], targ[layer], err) 이용해서 편하게 ε(l+1)*f'(~l)*θ(l) 계산하도록 구현해보기
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class PGConv(nn.Module):  # 미완성
    def __init__(self, groups, in_channels, out_channels, channel_shuffle=True):
        super().__init__()
        self.channel_shuffle = channel_shuffle
        self.groups = groups

        self.out_channels = out_channels
        # Channel Shuffle할때 사용

        self.PG = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            groups=self.groups,
            kernel_size=1,
            stride=1,
            padding=0,
        ).to(device)

    def forward(self, x):
        x = self.PG(x)

        if self.channel_shuffle:
            shape = x.shape
            # PG 후에 (batch_size, out_channels, output_size, output_size) 꼴

            x = x.reshape(shape[0], self.groups, self.out_channels // self.groups, shape[-2], -1)
            # (batch_size, groups, out_channels//groups, output_size, output_size) 꼴

            x = torch.transpose(x, dim0=1, dim1=2)
            # (batch_size, out_channels//groups, groups, output_size, output_size) 꼴로 변경

            x = torch.flatten(x, start_dim=1, end_dim=2)
            # dim=1과 dim=2 부분을 flatten하면
            # (batch_size, out_channels//groups * groups, output_size, output_size) 꼴로 변경
            # 즉 (batch_size, out_channels, output_size, output_size)로 다시 돌아오지만
            # 채널 셔플 되어 있음

        return x

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # transpose 하기 전 x가
        # [group0 [g0_fil0, g0_fil1, ....],
        #  group1 [g1_fil0, g1_fil1, ....],
        #  .............................. ,
        #  group-1 [g-1_fil0, g-1_fil1, ....]] 형태로 되어 있던 것을
        #
        # transpose를 하면
        # [[g0_fil0, g1_fil0, ....., g-1_fil0],
        #  [g0_fil1, g1_fil1, ....., g-1_fil1],
        #  .................................. ,
        #  [g0_fil-1, g1_fil-1,..., g-1_fil-1]] 꼴로 변경된다
        #
        # 이것을 flatten하면
        # [g0_fil0, g1_fil0, ....., g-1_fil0, g0_fil1, g1_fil1, ....., g-1_fil1, ....., g0_fil-1, g1_fil-1,..., g-1_fil-1] 꼴이 되어
        # CS 완료
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class DWConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        learning_rate,
        stride=1,
        padding=1,
        bias=False,
        f=linear,
        df=d_linear,
        device="cpu",
        init_weights=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size

        self.stride = stride
        self.padding = padding
        self.output_size = (
            math.floor((self.input_size + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        )

        self.bias = bias

        self.f = f
        self.df = df

        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            groups=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            device=self.device,
        )
        self.flat_weights = self.conv.weight.view(self.out_channels, -1, 1)
        # # self.flat_weights의 size는 (output_channel_number, kernel_size*kernel_size, 1)

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.conv.parameters(), lr=self.learning_rate)

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride
        ).to(self.device)
        self.fold = nn.Fold(
            output_size=(self.input_size, self.input_size),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride,
        ).to(self.device)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.X_col = self.unfold(x.clone())
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # unfold는 deepcopy가 아니기때문에 x가 변해도 X_col이 안변하도록 x.clone() 입력
        # .clone()은 deepcopy를 해서 새로운 메모리에 할당하면서 grad_fn 히스토리는 유지한다
        # 그러나 leaf node를 leaf node가 아니게 바꾸기때문에
        # in-place operation이 금지된 leaf node와 다르게 in-place operation이 가능해진다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # input의 size는 (batch_size, input_channel_number, self.input_size, self.input_size)
        # nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride) 적용 후
        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # @@ sliding_local_block_number == self.output_size * self.output_size

        self.pre_activation = self.conv(x)

        return self.f(self.pre_activation)

    def backward(self, e):
        dfx = self.df(self.pre_activation)
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # μ' = f'(self.pre_activation)
        e = e * dfx

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 여기서 바로
        # _, dX = torch.autograd.functional.vjp(self.conv, X, e)를
        # 바로 계산해도 똑같은 dX 값 나옴
        # 심지어 convolution은 linear 함수이기때문에 X가 무슨 값이던
        # 자코비안 안의 미분값들이 항상 상수값으로 동일
        # ===> X는 이 레이어의 실제 input을 딱히 넣을 필요 없음
        # https://darkpgmr.tistory.com/132 : Jacobian 함수 설명
        # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.e = e.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
        # self.e의 size는 (batch_size, output_channel_number, 1, self.output_size * self.output_size)

        dX_col = self.flat_weights @ self.e
        # self.flat_weights의 size는 (output_channel_number, kernel_size*kernel_size, 1)
        # batch_size 갯수만큼 broadcasting 되어 self.e에 연산
        # =>(output_channel_number, kernel_size*kernel_size, 1) @ (batch_size, output_channel_number, 1, self.output_size * self.output_size)
        # ==> (batch_size, output_channel_number, kernel_size*kernel_size, self.output_size * self.output_size)

        dX_col = dX_col.reshape(
            -1, self.out_channels * self.kernel_size * self.kernel_size, self.output_size * self.output_size
        )
        # (batch_size, output_channel_number * kernel_size*kernel_size, self.output_size * self.output_size) 형태로 변경

        dX = self.fold(dX_col)
        # (batch_size, output_channel_number * kernel_size*kernel_size, self.output_size * self.output_size)에
        # nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride)를 적용하여
        # (batch_size, input_channel_number, self.input_size, self.input_size)로 변경
        # DWConv이므로 output_size == input_size, output_channel_number == input_channel_number
        # @@@ fold는 unfold의 반대방향. 겹치는 부분들은 더한다. @@@

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # M = input_channel_number, N = output_channel_number, k = kernel_size, in = self.input_size, out = self.output_size
        #
        # (output_channel_number, kernel_size*kernel_size, 1) @ (batch_size, output_channel_number, 1, self.output_size * self.output_size) 하면
        # (batch_size, output_channel_number, kernel_size*kernel_size, self.output_size * self.output_size) 형태
        # output_channel n번째 dim=2, dim=3 행렬곱 부분만 보면
        # => (k*k , 1) @ (1, out*out) 행렬곱
        # ==> (k*k , out*out)
        # 이 행렬의 각열은 동일한 weight벡터에 ε(n,1), ε(n,2), ...., ε(n,s),.... ε(n,out*out)를 scalar 곱 해놓은 것과 같다. (s = 1, 2, ...., out*out)
        # 이제 이 N개의 행렬들을 vertical stack으로 weight벡터들을 길게 이어 붙이면
        # s 열만 보면 DWConv의 weight벡터의 첫 k*k개 원소에 ε(1,s)를 곱하고, 그다음 k*k개 원소에 ε(2,s)를 곱하고,
        # ....., ε(n,s), ...., 마지막 k*k개 원소에 ε(N,s)을 곱한 형태가 된다.
        # ===> nn.Fold를 적용하면 각각의 열들이 하나의 sliding block으로 바뀌어 겹치는 부분은 더하면서
        # 원래의 X와 동일한 size를 가지게 변한다.
        # 그리고 이 값은 구하고자 하는 ^e(l) 값이 된다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return dX

    def update_weights(self, e):
        dfx = self.df(self.pre_activation)
        # ε(l+1,n,i,j) == x(l+1,n,i,j) - μ(l+1,n,i,j)
        # μ(l+1,n,i,j) == f(합성곱(x(l))(n,i,j))
        # 합성곱(x(l))(n,i,j) == self.pre_activation(n,i,j)
        # dfx = f'(self.pre_activation(n,i,j))
        e = e * dfx
        # ==> ε(l+1,n,i,j) * f'(self.pre_activation(n,i,j))

        self.e = e.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
        # self.e의 size는 (batch_size, output_channel_number, 1, self.output_size * self.output_size)
        # @@ self.output_size * self.output_size == sliding_local_block_number

        X_col_s = self.X_col.reshape(
            -1, self.in_channels, self.kernel_size * self.kernel_size, self.output_size * self.output_size
        )
        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # reshape로 (batch_size, input_channel_number, kernel_size*kernel_size, sliding_local_block_number) 형태로 변경

        # .permute(0번째dim, 1번째 dim, 3번째dim, 2번째dim)을 하면
        # X_col_s의 size는 (batch_size, input_channel_number, sliding_local_block_number, kernel_size*kernel_size)로 변경
        dW = self.e @ X_col_s.permute(0, 1, 3, 2)
        # (batch_size, output_channel_number, 1, self.output_size * self.output_size)와
        # (batch_size, input_channel_number, sliding_local_block_number, kernel_size*kernel_size)의 행렬곱은
        # 동일한 (batch, channel)인 (output_channel_number, 1, sliding_local_block_number) 행렬과
        # (sliding_local_block_number, kernel_size*kernel_size) 행렬 사이의 행렬 곱을 수행
        # 따라서 (batch_size, output_channel_number, 1, kernel_size*kernel_size) 꼴이 된다.

        # dF/dw(n,k,l) = -∑i∑j(ε(l+1,n,i,j) * f'(self.pre_activation(n,i,j)) * x(l, n, i+k, j+l) ) 인데 -를 곱하지 않았기 때문에
        # 여기서의 dW는 -dF/dw

        dW = torch.sum(dW, dim=0)
        # (batch_size, output_channel_number, 1, kernel_size*kernel_size) 꼴을 dim=0 batch 차원으로 sum하면
        # (output_channel_number, 1, kernel_size*kernel_size) 꼴이 된다.
        # batch별로 계산된 dW를 다 더하는 것과 동일
        # ∵ 샘플 하나의 F값을 Fi로 두고 F = ∑Fi 라고 하면 (Fi는 ln(확률i)로 정의되어 있으므로 F = ∑Fi로 두어야 F = ln(π확률i) 꼴이 된다.)
        # dF/dW == ∑(dFi/dW)

        dW = dW.reshape((self.out_channels, 1, self.kernel_size, self.kernel_size))
        # (output_channel_number, 1, kernel_size*kernel_size) 꼴을
        # (output_channel_number, 1, kernel_size, kernel_size) 꼴로 변경

        self.conv.weight.grad = -dW

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 여기서 dW를 복잡하게 직접 구하는 대신
        # local loss를 정의해서 bp처럼 하는 것도 가능
        # (ex:  함수 인자로 x(l+1)을 추가로 받고
        #       mse = torch.nn.MSELoss(reduction='sum')
        #       p = self.f(self.pre_activation)  <-- μ(l+1) 계산
        #       loss = 0.5 * mse(p, x(l+1))) / p.size(0)     <-- p.size(0) == batch_size
        #       self.optim.zero_grad()
        #       loss.backward()
        #       self.optim.step()                                                         )
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.optim.step()

        return dW

    def _initialize_weights(self):
        if self.f is linear:
            # self.conv.weight.normal_(mean=0,std=0.05)
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        elif self.f is relu:
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.conv.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            nn.init.constant_(self.conv.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    def set_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group["lr"] = new_lr


class DualPathConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        learning_rate,
        stride=1,
        padding=1,
        bias=False,
        f=linear,
        df=d_linear,
        device="cpu",
        init_weights=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.input_size = input_size

        self.stride = stride
        self.padding = padding
        self.output_size = (
            math.floor((self.input_size + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        )

        self.bias = bias

        self.f = f
        self.df = df

        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=2 * self.out_channels,
            groups=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            device=self.device,
        )

        # self.conv.weight의 size는 (output_channel_number * 2, 1, kernel_size, kernel_size)
        self.flat_weights = self.conv.weight.view(self.out_channels, 2, -1, 1)
        # # self.flat_weights의 size는 (output_channel_number, 2, kernel_size*kernel_size, 1)

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.conv.parameters(), lr=self.learning_rate)

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride
        ).to(self.device)
        self.fold = nn.Fold(
            output_size=(self.input_size, self.input_size),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride,
        ).to(self.device)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.X_col = self.unfold(x.clone())
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # unfold는 deepcopy가 아니기때문에 x가 변해도 X_col이 안변하도록 x.clone() 입력
        # .clone()은 deepcopy를 해서 새로운 메모리에 할당하면서 grad_fn 히스토리는 유지한다
        # 그러나 leaf node를 leaf node가 아니게 바꾸기때문에
        # in-place operation이 금지된 leaf node와 다르게 in-place operation이 가능해진다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # input의 size는 (batch_size, input_channel_number, self.input_size, self.input_size)
        # nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride) 적용 후
        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # @@ sliding_local_block_number == self.output_size * self.output_size

        self.pre_activation = self.conv(x).reshape(
            -1, self.out_channels, 2, self.output_size, self.output_size
        )
        # self.conv(x)의 size는 (batch_size, 2 * self.out_channels, output_size, output_size)
        # input channel 하나 마다 1x3x3 필터를 2개 적용해서 output channel 2개씩 생성

        # reshape로 (batch_size, self.out_channels, 2, output_size, output_size) 형태로 변경

        self.pre_activation = torch.sum(self.pre_activation, dim=2)
        # 각 input channel 별 2개씩 있던 output을 합쳐서
        # 각 input channel 마다 output channel 한개로 변경
        # (batch_size, self.out_channels, output_size, output_size) 형태

        return self.f(self.pre_activation)

    def backward(self, e):
        dfx = self.df(self.pre_activation)
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # μ' = f'(self.pre_activation)
        e = e * dfx

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 여기서 바로
        # _, dX = torch.autograd.functional.vjp(self.conv, X, e)를
        # 바로 계산해도 똑같은 dX 값 나옴
        # 심지어 convolution은 linear 함수이기때문에 X가 무슨 값이던
        # 자코비안 안의 미분값들이 항상 상수값으로 동일
        # ===> X는 이 레이어의 실제 input을 딱히 넣을 필요 없음
        # https://darkpgmr.tistory.com/132 : Jacobian 함수 설명
        # https://github.com/nalonso2/PredictiveCoding-MQSeqIL/blob/main/IL_Conv.py
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.e = e.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
        # self.e의 size는 (batch_size, output_channel_number, 1, self.output_size * self.output_size)

        weights = torch.sum(self.flat_weights, dim=1)
        # self.pre_activations 은 동일한 input을 가지는 두개의 DWConv를 더한 것이므로
        # self.pre_activations(n, i, j) = ∑k∑l{θ1(k,l) + θ2(k,l)}x(n,i+k, j+l)
        # 두개의 DWConv의 weight가 선형결합 ===> x로 미분하면 {θ1(k,l) + θ2(k,l)} 꼴
        # ====> 따라서 동일한 output channel로 가는 두개의 weight들을 하나로 합친다
        # self.flat_weights의 size는 (output_channel_number, 2, kernel_size*kernel_size, 1)
        # dim=1 을 합치면 (output_channel_number, kernel_size*kernel_size, 1)꼴

        dX_col = weights @ self.e
        # batch_size 갯수만큼 broadcasting 되어 self.e에 연산
        # =>(output_channel_number, kernel_size*kernel_size, 1) @ (batch_size, output_channel_number, 1, self.output_size * self.output_size)
        # ==> (batch_size, output_channel_number, kernel_size*kernel_size, self.output_size * self.output_size)

        dX_col = dX_col.reshape(
            -1, self.out_channels * self.kernel_size * self.kernel_size, self.output_size * self.output_size
        )
        # (batch_size, output_channel_number * kernel_size*kernel_size, self.output_size * self.output_size) 형태로 변경

        dX = self.fold(dX_col)
        # (batch_size, output_channel_number * kernel_size*kernel_size, self.output_size * self.output_size)에
        # nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride)를 적용하여
        # (batch_size, input_channel_number, self.input_size, self.input_size)로 변경
        # DWConv이므로 output_size == input_size, output_channel_number == input_channel_number
        # @@@ fold는 unfold의 반대방향. 겹치는 부분들은 더한다. @@@

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # M = input_channel_number, N = output_channel_number, k = kernel_size, in = self.input_size, out = self.output_size
        #
        # (output_channel_number, kernel_size*kernel_size, 1) @ (batch_size, output_channel_number, 1, self.output_size * self.output_size) 하면
        # (batch_size, output_channel_number, kernel_size*kernel_size, self.output_size * self.output_size) 형태
        # output_channel n번째 dim=2, dim=3 행렬곱 부분만 보면
        # => (k*k , 1) @ (1, out*out) 행렬곱
        # ==> (k*k , out*out)
        # 이 행렬의 각열은 동일한 weight벡터에 ε(n,1), ε(n,2), ...., ε(n,s),.... ε(n,out*out)를 scalar 곱 해놓은 것과 같다. (s = 1, 2, ...., out*out)
        # 이제 이 N개의 행렬들을 vertical stack으로 weight벡터들을 길게 이어 붙이면
        # s 열만 보면 DWConv의 weight벡터의 첫 k*k개 원소에 ε(1,s)를 곱하고, 그다음 k*k개 원소에 ε(2,s)를 곱하고,
        # ....., ε(n,s), ...., 마지막 k*k개 원소에 ε(N,s)을 곱한 형태가 된다.
        # ===> nn.Fold를 적용하면 각각의 열들이 하나의 sliding block으로 바뀌어 겹치는 부분은 더하면서
        # 원래의 X와 동일한 size를 가지게 변한다.
        # 그리고 이 값은 구하고자 하는 ^e(l) 값이 된다.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return dX

    def update_weights(self, e):
        dfx = self.df(self.pre_activation)
        # ε(l+1,n,i,j) == x(l+1,n,i,j) - μ(l+1,n,i,j)
        # μ(l+1,n,i,j) == f(합성곱(x(l))(n,i,j))
        # 합성곱(x(l))(n,i,j) == self.pre_activation(n,i,j)
        # dfx = f'(self.pre_activation(n,i,j))
        e = e * dfx
        # ==> ε(l+1,n,i,j) * f'(self.pre_activation(n,i,j))

        self.e = e.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
        # self.e의 size는 (batch_size, output_channel_number, 1, self.output_size * self.output_size)
        # @@ self.output_size * self.output_size == sliding_local_block_number

        X_col_s = self.X_col.reshape(
            -1, self.in_channels, self.kernel_size * self.kernel_size, self.output_size * self.output_size
        )
        # self.X_col의 size는 (batch_size, input_channel_number*kernel_size*kernel_size, sliding_local_block_number)
        # reshape로 (batch_size, input_channel_number, kernel_size*kernel_size, sliding_local_block_number) 형태로 변경

        # .permute(0번째dim, 1번째 dim, 3번째dim, 2번째dim)을 하면
        # X_col_s의 size는 (batch_size, input_channel_number, sliding_local_block_number, kernel_size*kernel_size)로 변경
        dW = self.e @ X_col_s.permute(0, 1, 3, 2)
        # (batch_size, output_channel_number, 1, self.output_size * self.output_size)와
        # (batch_size, input_channel_number, sliding_local_block_number, kernel_size*kernel_size)의 행렬곱은
        # 동일한 (batch, channel)인 (output_channel_number, 1, sliding_local_block_number) 행렬과
        # (sliding_local_block_number, kernel_size*kernel_size) 행렬 사이의 행렬 곱을 수행
        # 따라서 (batch_size, output_channel_number, 1, kernel_size*kernel_size) 꼴이 된다.

        # dF/dw(n,k,l) = -∑i∑j(ε(l+1,n,i,j) * f'(self.pre_activation(n,i,j)) * x(l, n, i+k, j+l) ) 인데 -를 곱하지 않았기 때문에
        # 여기서의 dW는 -dF/dw

        dW = torch.sum(dW, dim=0)
        # (batch_size, output_channel_number, 1, kernel_size*kernel_size) 꼴을 dim=0 batch 차원으로 sum하면
        # (output_channel_number, 1, kernel_size*kernel_size) 꼴이 된다.
        # batch별로 계산된 dW를 다 더하는 것과 동일
        # ∵ 샘플 하나의 F값을 Fi로 두고 F = ∑Fi 라고 하면 (Fi는 ln(확률i)로 정의되어 있으므로 F = ∑Fi로 두어야 F = ln(π확률i) 꼴이 된다.)
        # dF/dW == ∑(dFi/dW)

        dW = torch.cat([dW, dW], dim=-1).reshape(
            2 * self.out_channels, 1, self.kernel_size * self.kernel_size
        )
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # self.pre_activations 은 동일한 input을 가지는 두개의 DWConv를 더한 것이므로
        # self.pre_activations(n, i, j) = ∑k∑l{θ1(k,l) + θ2(k,l)}x(n,i+k, j+l)
        # θ1이나 θ2나 미분하면 동일하게 x(n,i+k, j+l)이 남는다. ===> dF/θ1(k,l) == dF/θ2(k,l)
        # 두 DWConv는 동일한 dW를 가진다
        # @@ self.conv의 weight는 두 DWConv의 weight가 1 channel씩 교차해서 들어가 있다. @@
        # https://stackoverflow.com/questions/61026393/pytorch-concatenate-rows-in-alternate-order
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        dW = dW.reshape((2 * self.out_channels, 1, self.kernel_size, self.kernel_size))
        # (output_channel_number, 1, kernel_size*kernel_size) 꼴을
        # (output_channel_number, 1, kernel_size, kernel_size) 꼴로 변경

        self.conv.weight.grad = -dW

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 여기서 dW를 복잡하게 직접 구하는 대신
        # local loss를 정의해서 bp처럼 하는 것도 가능
        # (ex:  함수 인자로 x(l+1)을 추가로 받고
        #       mse = torch.nn.MSELoss(reduction='sum')
        #       p = self.f(self.pre_activation)  <-- μ(l+1) 계산
        #       loss = 0.5 * mse(p, x(l+1))) / p.size(0)     <-- p.size(0) == batch_size
        #       self.optim.zero_grad()
        #       loss.backward()
        #       self.optim.step()                                                         )
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.optim.step()

        return dW

    def _initialize_weights(self):
        if self.f is linear:
            # self.conv.weight.normal_(mean=0,std=0.05)
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        elif self.f is relu:
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.conv.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            nn.init.constant_(self.conv.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    def set_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group["lr"] = new_lr


class FCLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        learning_rate,
        bias=False,
        f=linear,
        df=d_linear,
        device="cpu",
        init_weights=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.f = f
        self.df = df

        self.device = device

        self.linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=self.bias, device=self.device
        )
        self.weights = self.linear.weight

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.linear.parameters(), lr=self.learning_rate)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.inp = x.clone()

        self.pre_activation = self.linear(self.inp)

        return self.f(self.pre_activation)

    def backward(self, e):
        # FC layer의 output 뉴런의 개수를 N이라고 두면
        # self.activations과 ε = x - μ 모두 N차원 벡터
        self.fn_deriv = self.df(self.pre_activation)
        out = torch.matmul(e * self.fn_deriv, self.weights.T)
        # ε * f'(self.activations) 는 N차원끼리 element-wise 곱 => (1xN)
        # weight는 원래 MxN 이므로 transpose해서 NxM
        # ==> out은 1xM 꼴
        return out

    def update_weights(self, e):
        self.fn_deriv = self.df(self.pre_activation)
        dW = torch.matmul(self.inp.T, e * self.fn_deriv)
        # dF/dw = - x(l).T ⓧ (ε(l+1) * f'(pre)) 이므로
        # dw == -dF/dw
        # BP일땐 dL/dy(l+1) == δ(l+1)을 받아서
        # dL/dw = x(l).T ⓧ (δ(l+1) * f'(pre))를 계산
        # dw == dL/dw

        self.linear.weight.grad = -dW

        self.optim.step()

        return dW

    def _initialize_weights(self):
        nn.init.normal_(self.linear.weight, 0, 0.01)

        if self.linear.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            nn.init.constant_(self.linear.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    def set_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group["lr"] = new_lr


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
        self.optim = optim.Adam(self.linear.parameters(), lr=self.learning_rate)
        self.NLL = nn.NLLLoss(reduction="sum").to(self.device)
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
        x = x.detach().clone().requires_grad_(requires_grad=True)
        # y = y.detach().clone().requires_grad_(requires_grad=True)

        p = self.forward(x)
        # p = self.forward(x).requires_grad_(requires_grad=True)
        loss = self.NLL(torch.log(p), y)
        # loss.requires_grad = True
        loss_mean = loss / p.size(0)
        # p.size(0) == batch_size

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), loss_mean.item()

    def cal_loss(self, p, y):
        loss = self.NLL(torch.log(p), y)
        loss_mean = loss / p.size(0)

        return loss.item(), loss_mean.item()

    def _initialize_weights(self):
        nn.init.normal_(self.linear.weight, 0, 0.01)

        if self.linear.bias is not None:
            # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
            nn.init.constant_(self.linear.bias, 0)
            # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

    # 현재 lr값을 알려주는 함수 -> 훈련 중 print에 사용
    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group["lr"]

    def lr_step(self, val_loss):
        self.lr_scheduler.step(val_loss)


class BNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # def get_lr(self):
    #     for param_group in self.optim.param_groups:
    #         return param_group['lr']


class AvgPoolLayer(nn.Module):
    def __init__(self, in_channels, input_size, stride, device="cpu"):
        super().__init__()
        self.in_channels = in_channels
        # self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride
        self.output_size = math.floor((self.input_size + (2 * 1) - 3) / self.stride) + 1
        self.device = device

        self.avg = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            groups=self.in_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=False,
            device=self.device,
        )
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.avg.weight = nn.Parameter(torch.full([self.in_channels, 1, 3, 3], 1 / (3 * 3)).to(self.device))
        # weight에 새 값을 대입할 때 대입하는 tensor가 반드시 동일한 device에 있도록 해야함
        # nn.Parameter(torch.full([self.in_channels,1,3,3], 1/(3*3))).to(self.device) 하면 에러발생
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.flat_weight = self.avg.weight.view(self.in_channels, -1, 1)
        # self.flat_weight의 size는 (self.in_channels, 9, 1)

        self.fold = nn.Fold(
            output_size=(self.input_size, self.input_size), kernel_size=(3, 3), padding=1, stride=self.stride
        ).to(self.device)

    def forward(self, x):
        x = self.avg(x)
        # x의 size는 (batch_size, input_channel_number, input_size, input_size)

        return x

    def backward(self, e):
        # e의 size는 (batch_size, output_channel_number, output_size, output_size)
        # output_channel_number == input_channel_number
        self.e = e.reshape(-1, self.in_channels, 1, self.output_size * self.output_size)

        dX_col = self.flat_weight @ self.e
        # self.flat_weight의 size는 (self.in_channels, 9, 1)
        # self.e의 size는 (batch_size, self.in_channels, 1, self.output_size * self.output_size)
        # ==> dX_col의 size는 (batch_size, self.in_channels, 9, self.output_size * self.output_size)

        dX_col = dX_col.reshape(-1, self.in_channels * 3 * 3, self.output_size * self.output_size)
        # (batch_size, self.in_channels * 3 * 3, self.output_size * self.output_size) 형태로 변경

        dX = self.fold(dX_col)
        # (batch_size, self.in_channels * 3 * 3, self.output_size * self.output_size)에
        # nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(3,3),padding=1,stride=self.stride)를 하여
        # (batch_size, input_channel_number, self.input_size, self.input_size)로 변경
        # DWConv이므로 output_size == input_size, output_channel_number == input_channel_number
        # @@@ fold는 unfold의 반대방향. 겹치는 부분들은 더한다. @@@

        return dX

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, e):
        return 0

    def _initialize_weights(self):
        pass


class ShortcutPath(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x

    def backward(self, e):
        # e의 size는 (batch_size, output_channel_number, output_size, output_size)
        # output_channel_number == input_channel_number
        self.e = e

        dX = self.e

        return dX

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, e):
        return 0

    def _initialize_weights(self):
        pass


class AddLayer(nn.Module):
    def __init__(self, concat):
        super().__init__()
        self.concat = concat

    def forward(self, x1, x2):
        if self.concat:
            self.pre_activation = torch.cat((x1, x2), dim=1)
        else:
            self.pre_activation = x1 + x2

        return relu(self.pre_activation)

    def backward(self, e):
        dfx = d_relu(self.pre_activation)
        # e의 size는 (batch_size, output_channel_number, output_size, output_size)
        # output_channel_number == input_channel_number
        self.e = e * dfx
        # μ = f(pre_activation) 이고,
        # dF/dμ * dμ/dx == ε * f'(pre_activation) * d(pre_activation)/dx
        # concat일 경우 μ = f(torch.cat((x1, x2), dim=1)) 이므로
        # dμ/dx = f'(torch.cat((x1, x2), dim=1)) * d(torch.cat((x1, x2), dim=1))/dx
        # concat이 아닐 경우 μ = f(x1 + x2) 이므로
        # dμ/dx = f'(x1 + x2) * d(x1 + x2)/dx

        if self.concat:
            dX = torch.split(self.e, self.e.shape[1] // 2, dim=1)
            dX1 = dX[0]
            dX2 = dX[1]
        else:
            dX1 = self.e
            dX2 = self.e

        return dX1, dX2

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, e):
        return 0

    def _initialize_weights(self):
        pass


class RELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.pre_activation = x
        return relu(x)

    def backward(self, e):
        dfx = d_relu(self.pre_activation)
        # e의 size는 (batch_size, output_channel_number, output_size, output_size)
        # output_channel_number == input_channel_number
        self.e = e * dfx

        dX = self.e

        return dX

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, e):
        return 0

    def _initialize_weights(self):
        pass


# weight의 모든 원소가 1/(kernel_size*kernel_size)이고
# kernel_size == input_size 이고 padding == 0 인 DWConv로 취급?
class AdaptiveAvgPoolLayer(nn.Module):
    def __init__(self, in_channels, input_size, device="cpu"):
        super().__init__()
        self.in_channels = in_channels
        # self.kernel_size = kernel_size
        self.input_size = input_size
        self.device = device

        self.avg = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            groups=self.in_channels,
            kernel_size=self.input_size,
            stride=1,
            padding=0,
            bias=False,
            device=self.device,
        )
        self.avg.weight = nn.Parameter(
            torch.full(
                [self.in_channels, 1, self.input_size, self.input_size],
                1 / (self.input_size * self.input_size),
            ).to(self.device)
        )
        # self.flat_weight = self.avg.weight.view(self.in_channels, -1, 1)

        # self.fold = nn.Fold()

    def forward(self, x):
        x = self.avg(x)
        # x의 size는 (batch_size, input_channel_number, input_size, input_size)

        return x

    def backward(self, e):
        # e의 size는 (batch_size, output_channel_number, 1, 1)
        # output_channel_number == input_channel_number
        self.e = e

        dX = self.avg.weight.reshape(self.in_channels, self.input_size, self.input_size) * self.e
        # (self.in_channels, self.input_size, self.input_size) * (batch_size, output_channel_number, 1, 1)
        # == (batch_size, self.in_channels, self.input_size, self.input_size)

        return dX

    # ==> kernel은 고정이므로 dW = 0
    def update_weights(self, e):
        return 0

    def _initialize_weights(self):
        pass
