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


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        learning_rate,
        stride=1,
        padding=1,
        # bias=False,
        # BN 할때는 conv bias 무조건 False
        momentum=0.01,
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
        # self.bias = bias

        self.f = f
        self.df = df

        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
            device=self.device,
        )
        self.flat_weights = self.conv.weight.view(self.out_channels, -1)
        # # self.flat_weights의 size는 (output_channel_number, input_channel_number*kernel_size*kernel_size)

        self.BN_weight = nn.Parameter(torch.full([self.out_channels], 1).float()).to(self.device)
        self.BN_bias = nn.Parameter(torch.full([self.out_channels], 0).float()).to(self.device)
        # BN weight와 bias
        # nn.Parameter 클래스를 이 모듈 ConvBN의 attribute로 추가해놓으면
        # self.parameters() iterator로 순회할 때 포함된다.\

        self.eps = 0.00001
        # self.batch_count = 0
        # cumulative moving average 일때 사용
        self.momentum = momentum
        # self.running_mean = nn.Module.re(torch.full([self.out_channels], 0).float()).to(self.device)
        self.register_buffer(name="running_mean", tensor=torch.zeros([self.out_channels]).float())
        # self.running_var = nn.Parameter(torch.full([self.out_channels], 0).float()).to(self.device)
        # self.register_buffer(name="running_var", tensor=torch.zeros([self.out_channels]).float())
        self.register_buffer(name="running_var", tensor=torch.ones([self.out_channels]).float())
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # https://stackoverflow.com/questions/72899079/what-do-batchnorm2ds-running-mean-running-var-mean-in-pytorch
        # The running mean and variance are initialized to zeros and ones, respectively.
        # 평균 0, 표준편차 1인 분포로 normalize 하므로 초기값도 그에 맞게 설정
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # batch normalization의 running estimates 도 state_dict()로 저장되고 불러들어야할 값들이지만
        # optimizer로 학습하지 않는다 ==> parameter가 아니라 buffer로 만들어야 한다.
        # buffer는 leaf-node인 nn.Parameter와 다르게 inplace operation도 가능하지만,
        # BP 과정에서는 빠진다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # buffer는 생성될 때 Module의 device를 따른다 ==> Module.to('cuda') 로 생성한 경우에는
        # register_buffer로 바로 buffer를 GPU에 올릴 수 있다.
        self.running_mean = self.running_mean.to(self.device)
        self.running_var = self.running_var.to(self.device)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)

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

    def forward(self, x, init=False):
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

        self.pre_BN = self.conv(x)

        delta = 0.0

        if self.training:
            # print(f"==>> self.training: {self.training}")
            # self.batch_count += 1
            # cumulative moving average 일 경우 delta값
            # delta = 1.0 / float(self.batch_count)
            delta = self.momentum

            self.mean = self.pre_BN.mean([0, 2, 3])
            # (batch_size, output_channel, output_size, output_size) 에서 dim=0,2,3 평균내면
            # (output_channel) 꼴로 바뀜

            self.var = self.pre_BN.var([0, 2, 3], unbiased=False)
            # 분산 계산

            self.n = self.pre_BN.numel() / self.pre_BN.size(1)
            # .numel() 는 총 원소 개수 반환, .size(1) 는 output_channel 개수 반환하므로
            # n은 평균 계산의 분모 값

            if init:
                # _initialize_Xs 함수로 실행될 때만 running_mean이랑 running_var 계산
                with torch.no_grad():
                    # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L63
                    self.running_mean = delta * self.mean + (1 - delta) * self.running_mean
                    self.running_var = (
                        delta * self.var * (self.n / (self.n - 1)) + (1 - delta) * self.running_var
                    )
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # forward pass 마다 self.running_mean과 self.running_var를 갱신해야하므로
                    # 이 둘은 Parameter가 아니라 buffer로 만들어야 한다.
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                    # update running_var with unbiased var
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # ==> Implementation follows batch normalization paper,
                    # https://arxiv.org/abs/1502.03167 that calls for unbiased variance when computing moving averages (page 4),
                    # and biased variance during training (page 3).
                    # It is expected that batch norm results are different during training and evaluation,
                    # because during training the batch is normalized with its own mean and (biased) variance,
                    # and during evaluation accumulated running variance and mean are used, thus making evaluation batch-independent.
                    # https://github.com/pytorch/pytorch/issues/3122#issuecomment-336685514
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        else:
            # print(f"==>> self.training: {self.training}")
            self.mean = self.running_mean
            self.var = self.running_var

        self.deviation = self.pre_BN - self.mean[None, :, None, None]
        self.std = torch.sqrt(self.var[None, :, None, None] + self.eps)
        self.pre_gamma = self.deviation / self.std
        self.pre_activation = (
            self.pre_gamma * self.BN_weight[None, :, None, None] + self.BN_bias[None, :, None, None]
        )
        # (output_channel) 꼴 tensor에 [None, : , None, None]를 붙이면 (1, output_channel, 1, 1) 꼴로 변경된다

        return self.f(self.pre_activation)

    def backward(self, e):
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # self.pre_activation = γ(self.deviation / self.std) + β
        # self.deviation = self.pre_BN - self.mean[None, :, None, None]
        dfx = self.df(self.pre_activation)
        # (dμ/d(pre_activation)) 계산
        e = e * dfx
        # e = (df/dμ) * (dμ/d(pre_activation))
        e = e * self.BN_weight[None, :, None, None]
        # e = (df/dμ) * (dμ/d(pre_activation)) * (d(pre_activation)/d(self.deviation / self.std)

        # @@ dvar = df/d(var) 먼저 구하기 @@
        dvar = e * self.deviation * (-1 / 2) / (self.std**3)
        dvar = dvar.sum([0, 2, 3])
        # dvar는 (output_channel_number) 꼴

        # @@ dmean = df/d(mean) @@
        dmean1 = e * -1 / self.std
        dmean1 = dmean1.sum([0, 2, 3])
        # dmean1 = df/d(self.deviation / self.std)  * d(self.deviation / self.std)/dmean
        dmean2 = self.deviation.sum([0, 2, 3])
        dmean2 = dmean2 * dvar * -2 / self.n
        # dmean2 = df/dvar * dvar/dmean
        dmean = dmean1 + dmean2

        # @@ df/d(self.pre_BN) @@
        dpre1 = e / self.std
        # df/d(self.deviation / self.std)  * d(self.deviation / self.std)/d(self.pre_BN)
        dpre2 = dvar * 2 / self.n
        dpre2 = dpre2[None, :, None, None] * self.deviation
        # df/dvar * dvar/d(self.pre_BN)
        dpre3 = dmean / self.n
        dpre3 = dpre3[None, :, None, None]
        # df/dmean * dmean/d(self.pre_BN)
        dpre = dpre1 + dpre2 + dpre3
        # dpre는 (batch_size, output_channel_number, self.output_size, self.output_size) 꼴

        self.e = dpre.reshape(-1, self.out_channels, self.output_size * self.output_size)
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
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # self.pre_activation = γ(self.deviation / self.std) + β
        # self.deviation = self.pre_BN - self.mean[None, :, None, None]
        dfx = self.df(self.pre_activation)
        # (dμ/d(pre_activation)) 계산
        e = e * dfx
        # e = (df/dμ) * (dμ/d(pre_activation))

        # @@ df/dγ @@
        dgamma = e * self.pre_gamma
        dgamma = dgamma.sum([0, 2, 3])
        # df/d(pre_activation) * d(pre_activation)/dγ
        self.BN_weight.grad = -dgamma
        # @@ df/dβ @@
        dbeta = e.sum([0, 2, 3])
        # d(pre_activation)/dβ == 1
        # ==> df/d(pre_activation) * d(pre_activation)/dβ == df/d(pre_activation)
        self.BN_bias.grad = -dbeta

        e = e * self.BN_weight[None, :, None, None]
        # e = (df/dμ) * (dμ/d(pre_activation)) * (d(pre_activation)/d(self.deviation / self.std)

        # @@ dvar = df/d(var) 먼저 구하기 @@
        dvar = e * self.deviation * (-1 / 2) / (self.std**3)
        dvar = dvar.sum([0, 2, 3])
        # dvar는 (output_channel_number) 꼴

        # @@ dmean = df/d(mean) @@
        dmean1 = e * -1 / self.std
        dmean1 = dmean1.sum([0, 2, 3])
        # dmean1 = df/d(self.deviation / self.std)  * d(self.deviation / self.std)/dmean
        dmean2 = self.deviation.sum([0, 2, 3])
        dmean2 = dmean2 * dvar * -2 / self.n
        # dmean2 = df/dvar * dvar/dmean
        dmean = dmean1 + dmean2

        # @@ df/d(self.pre_BN) @@
        dpre1 = e / self.std
        # df/d(self.deviation / self.std)  * d(self.deviation / self.std)/d(self.pre_BN)
        dpre2 = dvar * 2 / self.n
        dpre2 = dpre2[None, :, None, None] * self.deviation
        # df/dvar * dvar/d(self.pre_BN)
        dpre3 = dmean / self.n
        dpre3 = dpre3[None, :, None, None]
        # df/dmean * dmean/d(self.pre_BN)
        dpre = dpre1 + dpre2 + dpre3
        # dpre는 (batch_size, output_channel_number, self.output_size, self.output_size) 꼴

        self.e = dpre.reshape(-1, self.out_channels, self.output_size * self.output_size)
        # self.e의 size는 (batch_size, output_channel_number, self.output_size * self.output_size)

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


class DWConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        learning_rate,
        stride=1,
        padding=1,
        # bias=False,
        momentum=0.01,
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

        # self.bias = False

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
            bias=False,
            device=self.device,
        )
        self.flat_weights = self.conv.weight.view(self.out_channels, -1, 1)
        # # self.flat_weights의 size는 (output_channel_number, kernel_size*kernel_size, 1)

        self.BN_weight = nn.Parameter(torch.full([self.out_channels], 1).float()).to(self.device)
        self.BN_bias = nn.Parameter(torch.full([self.out_channels], 0).float()).to(self.device)
        # BN weight와 bias
        # nn.Parameter 클래스를 이 모듈의 attribute로 추가해놓으면
        # self.parameters() iterator로 순회할 때 포함된다.

        self.eps = 0.00001
        # self.batch_count = 0
        # cumulative moving average 일때 사용
        self.momentum = momentum
        self.register_buffer(name="running_mean", tensor=torch.zeros([self.out_channels]).float())
        # self.register_buffer(name="running_var", tensor=torch.zeros([self.out_channels]).float())
        self.register_buffer(name="running_var", tensor=torch.ones([self.out_channels]).float())
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # buffer는 생성될 때 Module의 device를 따른다 ==> Module.to('cuda') 로 생성한 경우에는
        # register_buffer로 바로 buffer를 GPU에 올릴 수 있다.
        self.running_mean = self.running_mean.to(self.device)
        self.running_var = self.running_var.to(self.device)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)

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

    def forward(self, x, init=False):
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

        self.pre_BN = self.conv(x)

        delta = 0.0

        if self.training:
            # self.batch_count += 1
            # cumulative moving average 일 경우 delta값
            # delta = 1.0 / float(self.batch_count)
            delta = self.momentum

            self.mean = self.pre_BN.mean([0, 2, 3])
            # (batch_size, output_channel, output_size, output_size) 에서 dim=0,2,3 평균내면
            # (output_channel) 꼴로 바뀜

            self.var = self.pre_BN.var([0, 2, 3], unbiased=False)
            # 분산 계산

            self.n = self.pre_BN.numel() / self.pre_BN.size(1)
            # .numel() 는 총 원소 개수 반환, .size(1) 는 output_channel 개수 반환하므로
            # n은 평균 계산의 분모 값
            if init:
                # _initialize_Xs 함수로 실행될 때만 running_mean이랑 running_var 계산
                with torch.no_grad():
                    # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L63
                    self.running_mean = delta * self.mean + (1 - delta) * self.running_mean
                    self.running_var = (
                        delta * self.var * (self.n / (self.n - 1)) + (1 - delta) * self.running_var
                    )
                    # update running_var with unbiased var
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # ==> Implementation follows batch normalization paper,
                    # https://arxiv.org/abs/1502.03167 that calls for unbiased variance when computing moving averages (page 4),
                    # and biased variance during training (page 3).
                    # It is expected that batch norm results are different during training and evaluation,
                    # because during training the batch is normalized with its own mean and (biased) variance,
                    # and during evaluation accumulated running variance and mean are used, thus making evaluation batch-independent.
                    # https://github.com/pytorch/pytorch/issues/3122#issuecomment-336685514
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.deviation = self.pre_BN - self.mean[None, :, None, None]
        self.std = torch.sqrt(self.var[None, :, None, None] + self.eps)
        self.pre_gamma = self.deviation / self.std
        self.pre_activation = (
            self.pre_gamma * self.BN_weight[None, :, None, None] + self.BN_bias[None, :, None, None]
        )
        # (output_channel) 꼴 tensor에 [None, : , None, None]를 붙이면 (1, output_channel, 1, 1) 꼴로 변경된다

        return self.f(self.pre_activation)

    def backward(self, e):
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # self.pre_activation = γ(self.deviation / self.std) + β
        # self.deviation = self.pre_BN - self.mean[None, :, None, None]
        dfx = self.df(self.pre_activation)
        # (dμ/d(pre_activation)) 계산
        e = e * dfx
        # e = (df/dμ) * (dμ/d(pre_activation))
        e = e * self.BN_weight[None, :, None, None]
        # e = (df/dμ) * (dμ/d(pre_activation)) * (d(pre_activation)/d(self.deviation / self.std)

        # @@ dvar = df/d(var) 먼저 구하기 @@
        dvar = e * self.deviation * (-1 / 2) / (self.std**3)
        dvar = dvar.sum([0, 2, 3])
        # dvar는 (output_channel_number) 꼴

        # @@ dmean = df/d(mean) @@
        dmean1 = e * -1 / self.std
        dmean1 = dmean1.sum([0, 2, 3])
        # dmean1 = df/d(self.deviation / self.std)  * d(self.deviation / self.std)/dmean
        dmean2 = self.deviation.sum([0, 2, 3])
        dmean2 = dmean2 * dvar * -2 / self.n
        # dmean2 = df/dvar * dvar/dmean
        dmean = dmean1 + dmean2

        # @@ df/d(self.pre_BN) @@
        dpre1 = e / self.std
        # df/d(self.deviation / self.std)  * d(self.deviation / self.std)/d(self.pre_BN)
        dpre2 = dvar * 2 / self.n
        dpre2 = dpre2[None, :, None, None] * self.deviation
        # df/dvar * dvar/d(self.pre_BN)
        dpre3 = dmean / self.n
        dpre3 = dpre3[None, :, None, None]
        # df/dmean * dmean/d(self.pre_BN)
        dpre = dpre1 + dpre2 + dpre3
        # dpre는 (batch_size, output_channel_number, self.output_size, self.output_size) 꼴

        self.e = dpre.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
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
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # self.pre_activation = γ(self.deviation / self.std) + β
        # self.deviation = self.pre_BN - self.mean[None, :, None, None]
        dfx = self.df(self.pre_activation)
        # (dμ/d(pre_activation)) 계산
        e = e * dfx
        # e = (df/dμ) * (dμ/d(pre_activation))

        # @@ df/dγ @@
        dgamma = e * self.pre_gamma
        dgamma = dgamma.sum([0, 2, 3])
        # df/d(pre_activation) * d(pre_activation)/dγ
        self.BN_weight.grad = -dgamma
        # @@ df/dβ @@
        dbeta = e.sum([0, 2, 3])
        # d(pre_activation)/dβ == 1
        # ==> df/d(pre_activation) * d(pre_activation)/dβ == df/d(pre_activation)
        self.BN_bias.grad = -dbeta

        e = e * self.BN_weight[None, :, None, None]
        # e = (df/dμ) * (dμ/d(pre_activation)) * (d(pre_activation)/d(self.deviation / self.std)

        # @@ dvar = df/d(var) 먼저 구하기 @@
        dvar = e * self.deviation * (-1 / 2) / (self.std**3)
        dvar = dvar.sum([0, 2, 3])
        # dvar는 (output_channel_number) 꼴

        # @@ dmean = df/d(mean) @@
        dmean1 = e * -1 / self.std
        dmean1 = dmean1.sum([0, 2, 3])
        # dmean1 = df/d(self.deviation / self.std)  * d(self.deviation / self.std)/dmean
        dmean2 = self.deviation.sum([0, 2, 3])
        dmean2 = dmean2 * dvar * -2 / self.n
        # dmean2 = df/dvar * dvar/dmean
        dmean = dmean1 + dmean2

        # @@ df/d(self.pre_BN) @@
        dpre1 = e / self.std
        # df/d(self.deviation / self.std)  * d(self.deviation / self.std)/d(self.pre_BN)
        dpre2 = dvar * 2 / self.n
        dpre2 = dpre2[None, :, None, None] * self.deviation
        # df/dvar * dvar/d(self.pre_BN)
        dpre3 = dmean / self.n
        dpre3 = dpre3[None, :, None, None]
        # df/dmean * dmean/d(self.pre_BN)
        dpre = dpre1 + dpre2 + dpre3
        # dpre는 (batch_size, output_channel_number, self.output_size, self.output_size) 꼴
        self.e = dpre.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
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


class DualPathConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        learning_rate,
        stride=1,
        padding=1,
        # bias=False,
        momentum=0.01,
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

        # self.bias = False

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
            bias=False,
            device=self.device,
        )

        # self.conv.weight의 size는 (output_channel_number * 2, 1, kernel_size, kernel_size)
        self.flat_weights = self.conv.weight.view(self.out_channels, 2, -1, 1)
        # # self.flat_weights의 size는 (output_channel_number, 2, kernel_size*kernel_size, 1)

        self.BN_weight = nn.Parameter(torch.full([self.out_channels], 1).float()).to(self.device)
        self.BN_bias = nn.Parameter(torch.full([self.out_channels], 0).float()).to(self.device)
        # BN weight와 bias
        # nn.Parameter 클래스를 이 모듈의 attribute로 추가해놓으면
        # self.parameters() iterator로 순회할 때 포함된다.

        self.eps = 0.00001
        # self.batch_count = 0
        # cumulative moving average 일때 사용
        self.momentum = momentum
        self.register_buffer(name="running_mean", tensor=torch.zeros([self.out_channels]).float())
        # self.register_buffer(name="running_var", tensor=torch.zeros([self.out_channels]).float())
        self.register_buffer(name="running_var", tensor=torch.ones([self.out_channels]).float())
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # buffer는 생성될 때 Module의 device를 따른다 ==> Module.to('cuda') 로 생성한 경우에는
        # register_buffer로 바로 buffer를 GPU에 올릴 수 있다.
        self.running_mean = self.running_mean.to(self.device)
        self.running_var = self.running_var.to(self.device)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)

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

    def forward(self, x, init=False):
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

        self.pre_BN = torch.sum(self.pre_activation, dim=2)
        # 각 input channel 별 2개씩 있던 output을 합쳐서
        # 각 input channel 마다 output channel 한개로 변경
        # (batch_size, self.out_channels, output_size, output_size) 형태

        delta = 0.0

        if self.training:
            # print(f"==>> self.training: {self.training}")
            # self.batch_count += 1
            # cumulative moving average 일 경우 delta값
            # delta = 1.0 / float(self.batch_count)
            delta = self.momentum

            self.mean = self.pre_BN.mean([0, 2, 3])
            # (batch_size, output_channel, output_size, output_size) 에서 dim=0,2,3 평균내면
            # (output_channel) 꼴로 바뀜

            self.var = self.pre_BN.var([0, 2, 3], unbiased=False)
            # 분산 계산

            self.n = self.pre_BN.numel() / self.pre_BN.size(1)
            # .numel() 는 총 원소 개수 반환, .size(1) 는 output_channel 개수 반환하므로
            # n은 평균 계산의 분모 값
            if init:
                # _initialize_Xs 함수로 실행될 때만 running_mean이랑 running_var 계산
                with torch.no_grad():
                    # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L63
                    self.running_mean = delta * self.mean + (1 - delta) * self.running_mean
                    self.running_var = (
                        delta * self.var * (self.n / (self.n - 1)) + (1 - delta) * self.running_var
                    )
                    # update running_var with unbiased var
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # ==> Implementation follows batch normalization paper,
                    # https://arxiv.org/abs/1502.03167 that calls for unbiased variance when computing moving averages (page 4),
                    # and biased variance during training (page 3).
                    # It is expected that batch norm results are different during training and evaluation,
                    # because during training the batch is normalized with its own mean and (biased) variance,
                    # and during evaluation accumulated running variance and mean are used, thus making evaluation batch-independent.
                    # https://github.com/pytorch/pytorch/issues/3122#issuecomment-336685514
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        else:
            # print(f"==>> self.training: {self.training}")
            self.mean = self.running_mean
            self.var = self.running_var

        self.deviation = self.pre_BN - self.mean[None, :, None, None]
        self.std = torch.sqrt(self.var[None, :, None, None] + self.eps)
        self.pre_gamma = self.deviation / self.std
        self.pre_activation = (
            self.pre_gamma * self.BN_weight[None, :, None, None] + self.BN_bias[None, :, None, None]
        )
        # (output_channel) 꼴 tensor에 [None, : , None, None]를 붙이면 (1, output_channel, 1, 1) 꼴로 변경된다

        return self.f(self.pre_activation)

    def backward(self, e):
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # self.pre_activation = γ(self.deviation / self.std) + β
        # self.deviation = self.pre_BN - self.mean[None, :, None, None]
        dfx = self.df(self.pre_activation)
        # (dμ/d(pre_activation)) 계산
        e = e * dfx
        # e = (df/dμ) * (dμ/d(pre_activation))
        e = e * self.BN_weight[None, :, None, None]
        # e = (df/dμ) * (dμ/d(pre_activation)) * (d(pre_activation)/d(self.deviation / self.std)

        # @@ dvar = df/d(var) 먼저 구하기 @@
        dvar = e * self.deviation * (-1 / 2) / (self.std**3)
        dvar = dvar.sum([0, 2, 3])
        # dvar는 (output_channel_number) 꼴

        # @@ dmean = df/d(mean) @@
        dmean1 = e * -1 / self.std
        dmean1 = dmean1.sum([0, 2, 3])
        # dmean1 = df/d(self.deviation / self.std)  * d(self.deviation / self.std)/dmean
        dmean2 = self.deviation.sum([0, 2, 3])
        dmean2 = dmean2 * dvar * -2 / self.n
        # dmean2 = df/dvar * dvar/dmean
        dmean = dmean1 + dmean2

        # @@ df/d(self.pre_BN) @@
        dpre1 = e / self.std
        # df/d(self.deviation / self.std)  * d(self.deviation / self.std)/d(self.pre_BN)
        dpre2 = dvar * 2 / self.n
        dpre2 = dpre2[None, :, None, None] * self.deviation
        # df/dvar * dvar/d(self.pre_BN)
        dpre3 = dmean / self.n
        dpre3 = dpre3[None, :, None, None]
        # df/dmean * dmean/d(self.pre_BN)
        dpre = dpre1 + dpre2 + dpre3
        # dpre는 (batch_size, output_channel_number, self.output_size, self.output_size) 꼴

        self.e = dpre.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
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
        # μ = f(self.pre_activation), self.pre_activation의 size는 (batch_size, output_channel_number, self.output_size, self.output_size)
        # self.pre_activation = γ(self.deviation / self.std) + β
        # self.deviation = self.pre_BN - self.mean[None, :, None, None]
        dfx = self.df(self.pre_activation)
        # (dμ/d(pre_activation)) 계산
        e = e * dfx
        # e = (df/dμ) * (dμ/d(pre_activation))

        # @@ df/dγ @@
        dgamma = e * self.pre_gamma
        dgamma = dgamma.sum([0, 2, 3])
        # df/d(pre_activation) * d(pre_activation)/dγ
        self.BN_weight.grad = -dgamma
        # @@ df/dβ @@
        dbeta = e.sum([0, 2, 3])
        # d(pre_activation)/dβ == 1
        # ==> df/d(pre_activation) * d(pre_activation)/dβ == df/d(pre_activation)
        self.BN_bias.grad = -dbeta

        e = e * self.BN_weight[None, :, None, None]
        # e = (df/dμ) * (dμ/d(pre_activation)) * (d(pre_activation)/d(self.deviation / self.std)

        # @@ dvar = df/d(var) 먼저 구하기 @@
        dvar = e * self.deviation * (-1 / 2) / (self.std**3)
        dvar = dvar.sum([0, 2, 3])
        # dvar는 (output_channel_number) 꼴

        # @@ dmean = df/d(mean) @@
        dmean1 = e * -1 / self.std
        dmean1 = dmean1.sum([0, 2, 3])
        # dmean1 = df/d(self.deviation / self.std)  * d(self.deviation / self.std)/dmean
        dmean2 = self.deviation.sum([0, 2, 3])
        dmean2 = dmean2 * dvar * -2 / self.n
        # dmean2 = df/dvar * dvar/dmean
        dmean = dmean1 + dmean2

        # @@ df/d(self.pre_BN) @@
        dpre1 = e / self.std
        # df/d(self.deviation / self.std)  * d(self.deviation / self.std)/d(self.pre_BN)
        dpre2 = dvar * 2 / self.n
        dpre2 = dpre2[None, :, None, None] * self.deviation
        # df/dvar * dvar/d(self.pre_BN)
        dpre3 = dmean / self.n
        dpre3 = dpre3[None, :, None, None]
        # df/dmean * dmean/d(self.pre_BN)
        dpre = dpre1 + dpre2 + dpre3
        # dpre는 (batch_size, output_channel_number, self.output_size, self.output_size) 꼴
        self.e = dpre.reshape(-1, self.out_channels, 1, self.output_size * self.output_size)
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
