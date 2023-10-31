# Res-DualNet-with-PCN
수십층의 깊은 합성곱 신경망인 ResNet 계열에 Predictive Coding Network(PCN) 구조를 적용해보았다.   

Predictive Coding Network를 사용하면 각 layer들의 가중치를 수정할 때, 

BP(역전파) 알고리즘과는 다르게 그 layer의 이전 층과 다음 층의 데이터만 필요하다.

그러나 가중치 수정을 하기 전에 infer 과정을 반복하면서 layer 예측치를 실제 활성값으로 수렴하는 과정이 반드시 포함되어야 하기 때문에

학습에 필요한 연산량이 증가한다. 

CIFAR-10 데이터셋에 대해 BP 알고리즘을 사용하는 경우 1 epoch 학습에 30초~1분정도 걸리지만

PCN 에서 infer 과정을 100번 반복하는 경우 1 epoch 학습에 34~35분 정도 걸린다. 

![블록 구조4](https://github.com/paokimsiwoong/Res-DualNet-with-PCN/assets/37607763/783af46b-2b65-459b-b5b9-f4a0f47050b1)
기본 블록 구조는 Res-DualNet과 같이 ShuffleNet의 Depthwise Conv를 Dualpath Conv로 변경하고 두번 수행하는 대신, 1x1 Pointwise Conv를 한번만 수행한다.

Res-DualNet의 구조에 ShuffleNet의 Pointwise Group Conv까지 사용하면 성능이 현저하게 떨어져서 사용하지 않았고   

PGConv와 같이 쓰이는 Channel Shuffle도 사용하지 않았다.

output channel의 갯수가 2배로 늘어날때는 ShuffleNet과 같이

shortcut path 에서 3x3 평균 풀링을 수행하고 residual path와 channel concatenation으로 합친다.

또 input의 가로세로 크기가 2분의 1로 줄어들 때는 

residual path의 첫 Dualpath Conv와 shortcut path의 3x3 평균 풀링의 stride를 2로 두었다.   


## 실험방법   
PCN test.ipynb 파일로 학습을 진행한다.  

BP 모델의 경우   

BPNetwork.ipynb 파일로 학습을 진행한다.   



## 실험결과     

![실험 loss 그래프2](https://github.com/paokimsiwoong/Res-DualNet-with-PCN/assets/37607763/e1e02192-bb46-402a-8189-4ae8c907b450)
![실험 acc 그래프3](https://github.com/paokimsiwoong/Res-DualNet-with-PCN/assets/37607763/2162c7ac-7208-4378-a367-cd614b8680d7)     

#1: ResNet구조, #2:BP모델, #3:PC모델

   

## TO-DO list   
~~Validation 할때 Batch Normalization 문제 해결~~   

~~-> train 정확도 90.59%, val 정확도 41.52%로 차이가 심함~~   

(train loss 계산을 마지막 FC층의 예측값과 target의 예측 오차가 최소화되는 수렴과정을 거치는 활성값 수정 이후에 진행해서
실제 모델의 loss와 정확도보다 훨씬 좋게 측정되는 문제였음.)
-> BN문제가 아니라 fixed prediction assumption을 지키지 않아 모델의 학습이 제대로 진행되고 있지 않았으며 
수정 후에는 정상적으로 학습이 진행

~~ConvAG, DualPathConvAG, FCtoSoftMax에 적용한 것과 같이      
나머지 모든 layer들도 backward dx값 수정은 torch.autograd.functional.vjp(),      
가중치 수정은 각 층마다 분리된 그래프를 만들어 역전파를 수행하는 방식으로 변경하기~~
   
## 참고문헌   

[An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf)

[Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://arxiv.org/abs/2006.04182)

[Understanding and Improving Optimization in Predictive Coding Networks](https://arxiv.org/abs/2305.13562)

[ShuffleNet : An Extremely Efficient Convolutional Neural Network for Mobile Devices ](https://arxiv.org/abs/1707.01083)

[Res-DualNet Dual-Path Depthwise 컨볼루션 기반 ResNet 네트워크 경량화 연구](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11035735&nodeId=NODE11035735&medaTypeCode=185005&language=ko_KR&hasTopBanner=true)

   
## 참조한 코드들   

https://deep-learning-study.tistory.com/534

https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/imagenet/main.py#L171

https://ai.dreamkkt.com/54

https://github.com/BerenMillidge/PredictiveCodingBackprop

https://github.com/nalonso2/PredictiveCoding-MQSeqIL/tree/main

https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
