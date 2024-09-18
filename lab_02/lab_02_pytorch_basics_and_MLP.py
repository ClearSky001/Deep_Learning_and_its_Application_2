#!/usr/bin/env python
# coding: utf-8

# PyTorch
# ================
# 
# 파이토치는 파이썬 기반의 scientific computing package로 아래 독자층을 타겟하고 있다
# 
# -  GPU를 이용할 수 있는 NumPy 대체 패키지
# -  최고의 유연성과 속도를 제공하는 딥러닝 연구 플랫폼

# In[2]:


import numpy as np
import torch


# 파이토치 시작하기
# ---------------
# 
# ### 텐서(Tensors)
# 
# 텐서는 NumPy의 ndarray와 비슷하며 GPU를 활용한 계산을 부가적으로 제공한다

# ### 텐서 초기화하기

# 초기화되지 않은 5x3 tensor 만들기

# 랜덤하게 초기화된 텐서 만들기

# 값이 0이고 dtype이 long인 텐서 만들기

# 값이 1이고 dtype이 float16인 텐서 만들기

# 이미 존재하는 데이터로 부터 텐서 만들기

# 이미 존재하는 tensor를 이용하여 새로운 텐서 생성하기
# 
# 이 함수들은 tensor의 새로운 속성 (e.g. dtype, size)을 전달받지 않는한, input tensor의 속성을 그대로 재사용한다.

# numpy로 부터 tensor 생성하기

# Torch Tensor와 NumPy array는 메모리 위치를 공유한다 (Torch Tensor가 CPU 위에 있는 경우)
# 
# 따라서 둘중 하나의 값을 변경하면 다른것의 값도 변경된다

# tensor로 부터 numpy array 가져오기
# 
# CPU위에 있는 모든 텐서는 Numpy로 변환거다 역변환하는것을 지원한다 (CharTensor 제외)

# ### 텐서의 속성 (Attributes of tensor)
# - size : 데이터 shape
# - dtype : 개별 데이터의 자료형
# - device : 어느 device (CPU/GPU)위에 있는지

# ## 텐서 연산 (Operations on tensors)

# ### 연산의 위치 (Operation locations)

# 텐서는 ``.to`` 메서드를 이용해 다른 디바이스로 옮길 수 있다

# GPU 위에 있는 텐서끼리는 연산을 할 수 있다

# 서로 다른 디바이스에 있는 텐서끼리는 연산할 수 없다

# 
# ### 산술 연산 (Arithmatic Operations)
# PyTorch에는 다양한 연산들이 미리 정의 되어 있으며, 하나의 연산에도 여러 syntax가 존재한다

# 덧셈: syntax 1

# 덧셈: syntax 2

# 결과를 저장할 텐서를 argument로 전달하기

# in-place 연산

# <div class="alert alert-info"><h4>Note</h4><p>모든 in-place연산의 함수 이름은 ``_``로 끝난다.
#     예: ``x.copy_(y)``, ``x.t_()`` 등의 연산은 ``x``의 값을 변경할 것이다.</p></div>

# 스칼라 곱

# 행렬 곱

# element-wise 곱

# 집계 함수 (aggregation functions)
# 다양한 집계 함수들이 이미 정의 되어 있으며 필요할 경우 documentation을 찾아보고 이용한다
# 
# 집계 후의 텐서 값을 가져오기 위해서는 ``.item()``함수를 이용한다

# ### 인덱싱과 슬라이싱 (indexing and slicing)
# Numpy에서 이용하였던 방법을 그대로 모두 이용할 수 있다

# ### joining

# ### Resizing

# ### Reshaping

# <mark>실습</mark>
# 
# 아래 값을 가지는 `torch.tensor` 를 생성하라
# 
# $\begin{bmatrix} 1 & 2.2 & 9.6 \\ 4 & -7.2 & 6.3 \end{bmatrix}$
# 
# `.mean()` 함수를 이용해 각 행과 열의 평균을 계산하라.
# 
# 각 결과의 shape을 출력하라
# 

# Autograd: Automatic Differentiation
# ===================================
# 
# PyTorch의 가장 중요한 패키기중 하나는 ``autograd`` 패키지 이다.
# 
# ``autograd`` 패키지는 Tensor의 모든 연산에 대응하는 미분을 자동으로 계산한다.
# 연산 방법은 런타임에 자동으로 정의되며, 모든 backprop은 당신의 코드가 실행되면서 결정된다. 즉 매 스텝마다 미분 계산이 달라질 수 있다.
# 
# ``torch.Tensor`` 의 속성 중 ``.requires_grad`` 를 ``True``로 설정하게 되면 ``autograd``패키지는 이 텐서에 수행되는 모든 연산을 추적한다.
# 연산이 끝난뒤에는 ``.backward()`` 함수를 호출하여 gradient를 모두 자동으로 계산할 수 있다.
# 이 텐서에 해당하는 gradient는 ``.grad`` 속성에 저장되며 점차 쌓이게 된다.
# 

# ``requires_grad=True``인 tensor를 하나 생성하여 이 변수의 연산을 tracking해보자

# tensor 연산을 수행

# ``y``는 ``x``를 이용한 연산의 결과로 생겼으므로 ``grad_fn`` 속성을 갖고 있다

# ``y``에 다른 연산을 더 적용해보자

# backprop을 위해서는 target 값에 ``backward()``함수를 호출한다.
# 
# 즉 아래는 
# $\frac{d(out)}{dx}$
# 미분값을 출력한다

# 맞는지 확인해보자
# 
# $out = \frac{1}{4}\sum_i 3(x_i+2)^2$
# 
# $\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$
# 
# hence, 
# $\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$.
# 
# 

# 
# Forward pass에서 autograd는 아래 두가지 일을 수행한다:
# * 결과 텐서를 계산하기 위해 연산을 수행한다.
# * 자동미분을 위해 연산에 대응하는 gradient function를 DAG (directed acyclic graph)형태로 저장한다.
# 
# backward pass는 ``backward()`` 함수가 DAG root에서 호출될 경우 진행된다:
# * 각 ``.grad_fn`` 에 대한 미분을 계산한다
# * 그 결과값을 대응하는 텐서의 ``.grad`` 속성에 쌓는다 (즉 계산된 gradient는 ``.grad``속성에 저장된다)
# * 연쇄법칙(chain rule)을 통해 leaf tensor까지 전달한다.

# ``.requires_grad=True``인 텐서에 대해서 autograd 추적을 원하지 않을 경우 
# 
# ``with torch.no_grad():`` 문에 코드를 둘러싼다

# 또는 ``.detach()``를 사용하여 값은 동일하지만 requires_grad=False인 새로운 텐서를 얻는다
# 
# 이렇게 gradient tracking을 끄고 싶은 경우는 경우는 아래와 같은 경우가 있다.
# 1. Neural Network의 특정 파라미터들을 frozen 시키고 싶을 경우. 예: finetuning a pretrained network
# 
# 2. forward pass만 필요하기 떄문에 계산을 빨리하고 싶은 경우 (예: 모델 평가)

# ``.requires_grad_``를 사용해 텐서의 ``requires_grad``속성을 in-place로 변경ㄴ할 수도 있다
# 

# 이제 아래와 같은 함수를 생각해보자 $$f = x^2 + y^2 + z^2$$

# 두번째 ``backward()``호출은 에러를 일으킨다. 즉 하나의 computational graph에 한번의 backward만 가능하다.

# 만약 ``retain_graph``= True로 두면 gradient를 accumulation할수 있다.

# Training a Neural Network
# ===============
# 
# Neural networks 는 레이어/모듈로 구성되어 있으며 ``torch.nn``에 필요한 모든 building block이 정의되어 있다.
# 
# 모든 PyTorch 모듈은 ``nn.Module``을 상속(subclass)하며 당신이 구현하는 neural network도 그 자체로 모듈이며 다른 모듈(혹은 레어어)를 포함할 수 있다
# 
# 이렇게 모듈이 중첩된 구조(nested structure)로 당신의 Neural network를 구성함으로써 매우 복잡한 아키텍쳐도 쉽게 관리할 수 있다.
# 
# ``nn`` 패키지는 ``autograd``를 이용하여 모델 파라미터를 미분한다.
# 
# 신경망 학습과정은 보통 다음의 과정을 통해 이루어진다:
# - 신경망과 learnable parameter (weight)들을 정의한다 
# - 데이터를 순회하며 신경망의 forward propagation을 수행한다.
# - loss를 계산한다.
# - backward를 수행한다.
# - gradient descent에 따라 weight를 업데이트 한다:
# 
#   ``weight = weight - learning_rate * gradient``

# ### 간단한 neural network를 정의해보자.

# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 10) ## an linear operation: y = Wx + b

    def forward(self, x):
        x = self.linear(x)
        return x


# ``nn.Module``는 다음과 같은 요소들로 구성되어 있다.
# 
# - `__init__` 함수에서는 신경망에서 사용할 모듈(module)과 레이어(layer)를 정의하며 이를 통해 학습 가능한 파라미터(learnable parameters)가 정의되고 초기화된다.
# 
# - `forward` 함수는 신경망의 순전파(forward propagation) 과정에서 수행될 텐서 연산을 정의한다. PyTorch에서 제공하는 다양한 텐서 연산뿐만 아니라, Python의 기본 연산도 자유롭게 사용할 수 있다. 순전파의 계산 결과를 리턴한다.
# 
# - `backward` 함수는 역전파(backpropagation)를 위해 `autograd`에 의해 자동으로 정의된다. 사용자는 `backward` 함수를 직접 구현할 필요가 없으며, PyTorch가 자동으로 기울기를 계산하고 가중치를 업데이트한다.
# 

# ``.parameters()`` 메서드를 통해 learnable parameter를 얻을 수 있다

# ### Forward pass
# 1x100 크기의 random input을 전달해보자

# 참고: ``torch.nn``는 mini-batch만 지원한다.
# 
# 예를들어 ``nn.Linear`` 레이어는 (batch_size, input_size)의 2차원 텐서를 입력받는다.

# ### Loss Function
# loss function은 입력 데이터에 대응하는 모델의 출력과, target(label) 쌍을 입력받아 출력과 target의 차이를 계산함으로써 모델이 얼마나 틀린 결과를 내고 있는지에 측정값을 제공한다. 우리는 학습과정에서 loss를 최소화함으로써 모델의 성능을 개선할 수 있다.
# 
# 자주 사용되는 loss 함수는 회귀를 위한 nn.MSELoss (Mean Square Error)와 분류를 위한 nn.NLLLoss (Negative Log Likelihood) 등이 있으며, [PyTorch documentation](<https://pytorch.org/docs/stable/nn.html#loss-functions>)을 참고하기 바람.

# 이제 역전파 수행을 위해 ``loss.backward()``를 호출할 수 있다.
# 
# loss 계산을 위한 사용된 모든 computational graph에 대하여 미분을 수행하며,
# graph상의 tensor들 중 ``requires_grad=True``인 것들은 ``.grad``에 계산된 gradient가 축적된다.
# 
# 즉, $\frac{\partial L}{\partial W}$ 가 계산된다
# 

# 예를 들어, 몇 단계의 backward를 따라가 보자.

# ### Backprop
# ``loss.backward()``를 이용하여 Backpropagation을 수행하자.
# 
# backward 수행 후 parameter의 gradient가 계산된 것을 확인할 수 있다.

# ### Optimization (Update the weights)
# Optimization은 학습과정에서 model parameters를 조정하여 모델의 에러를 줄이는 과정이다. 
# 
# Optimization 알고리즘은 어떠한 방식으로 모델 파라미터를 업데이트할지를 정의한다.
# 
# 가장 간단한 파라미터 업데이트 방법은 Stochastic Gradient Descent (SGD)이다
# 
# ```
# weight = weight - learning_rate * gradient
# ```
# 
# SGD의 파라미터 업데이트는 다음과 같이 수행할 수 있다.
# ``` python
# learning_rate = 0.01
# for p in net.parameters():
#     p.data.sub_(p.grad.data * learning_rate)
# ```
# 
# 하지만 우리는 이 방법 외에도 매우 다양한 방법으로 파라미터를 업데이트 하고자 하며, Adam, RMSProc을 포함하는 다양한 optimization 방법들이 ``torch.optim``에 구현되어 있다.
# 
# 따라서 우리는 optimizer라는 별도의 객체에 파라미터를 업데이트를 위임하며,
# optimizer를 초기화할 때 모델 파라미터를 전달하여 optimizer가 학습과정에서 파라미터를 대신 업데이트할 수 있도록 한다

# zero_grad()가 필요한 이유는 한 computational graph에 한번의 back_prop만이 가능하기 때문이다. 
# 
# (혹은 ``retain_graph`` = True일 경우 gradient가 쌓임)
# 

# 축하합니다! 당신은 지금까지 neural networks를 정의하고, loss를 계산하고, weights를 업데이트 하는 방법에 대해 모두 배웠습니다.
# 이제는 당신만의 신경망을 학습할 준비가 되었습니다.
# 
# 
# **참고:**
#   -  ``torch.Tensor`` - *multi-dimensional array*로서 autograd와 GPU연산을 지원함. gradient값도 보관한다.
#   -  ``nn.Module`` - Neural network 모듈. 모델 파라미터를 은닉(encapsulate)하고 편리하게 관리할 수 있도록 해주며, GPU로의 이동, export, loading등 다양한 편의성도 제공한다.
#   -  ``nn.Parameter`` - 텐서의 일종으로 nn.Module의 속성에 할당될 경우 자동으로 파라미터로 등록됨.

# Training a Multi-layer perceptron
# --------------

# ### 1. 데이터셋 가져오기
# 학습의 시작은 데이터를 읽어오는 것으로, 먼저 원하는 데이터를 읽어와 torch.tensor로 변환을 수행한다.
# 
# 아래와 같은 패키지들이 유용하다.
# 
# -  For images, packages such as Pillow, OpenCV 
# -  For audio, packages such as scipy and librosa
# -  For text, either raw Python or Cython based loading, or NLTK and
#    SpaCy
# 
# 
# ``torchvision`` 패키지에는 vision분야에서 자주 사용되는 몇몇 유명한 데이터셋을 읽어오는 함수를 제공한다.
# 
# 우리는 이번에 MNIST데이터셋을 분류하는 Multi-layter Perceptron모델을 구현해볼 것이다.
# 
# MNIST 데이터셋은 handwritten digits 분류하는, 머신러닝 분야에서 매우 유명한 데이터셋이다.
# 
# 이미지 분류모델을 아래 과정을 통해 학습된다.
# 
# 1. ``torchvision``을 이용하여 데이터셋을 읽어온다.
# 2. Neural Network를 정의한다
# 3. loss 함수를 정의한다.
# 4. training data를 이용하여 학습한다.
# 5. test data를 이용하여 모델을 평가한다.
# 
# 

# In[24]:


import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


# 데이터 전처리를 하는 코드들은 지저분하기 마련이며 관리하기도 어렵다.
# 
# 이상적으로는 데이터셋을 관리하는 코드와 모델을 학습하는 코드가 완전히 분리되는것이 좋다 (for better readability, resuability, and modularity).
# 
# PyTorch에서는 데이터를 관리하기 위한 두가지 모듈을 제공한다: ``torch.utils.data.Dataset``, ``torch.utils.data.DataLoader``
# 
# ``Dataset``는 샘플 데이터와 라벨을 저장하고 있으며, iterator가 감싸져있어 쉽게 데이터에 접근할 수 있다.
# 
# ``torchvision.datasets``는 ``torch.utils.data.Dataset``를 상속하는 다양한 데이터셋을 제공한다.

# In[25]:


def load_MNIST_datasets(data_root_dir):
    train_dataset = datasets.MNIST(
        root=data_root_dir, train=True, download=True, 
        transform=ToTensor() # convert PILImage images of range [0, 1] to tensors of normalized range [-1, 1].
    )
    test_dataset = datasets.MNIST(
        root=data_root_dir, train=False, download=True, 
        transform=ToTensor()
    )

    return train_dataset, test_dataset


# In[27]:


def visualize_few_samples(dataset, cols=8, rows=5):
    figure = plt.figure(figsize=(6, 4))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# ``Dataset``은 데이터 feature와 라벨을 한번에 하나씩 가져오는 기능을 제공한다. 하지만 보통 학습에서는 샘플들을 “minibatches”로 가져온고, 매 epoch마다 랜덤하게 섞어주며, multiprocessing을 사용해 데이터 획득을 빠르게 하고자 한다.
# 
# DataLoader이 복잡한 과정을 쉽도록 도와주는 iterable이다.
# 
# - 데이터로더는 데이터셋을 배치 단위로 묶어준다.
# - ``shuffle``를 통해 매 epoch마다 랜덤하게 섞어주는 기능을 제공한다
# - ``num_workers``를 통해 데이터 전처리를 multiprocessing으로 수행할 수 있다.

# ### 2. Network 정의하기

# [PyTorch documentation](https://pytorch.org/docs/stable/nn.html)을 참고하여 3개의 레이어로 Multi-layer Perceptron모듈을 작성하라

# In[30]:


class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.flatten = nn.Flatten()
        
        ##### YOUR CODE START #####  
        # 1. First Layer
        #    - Linear layer with input size `in_dim` and output size `hidden_dim`
        #    - ReLU (Rectified Linear Unit) activation
        # 2. Second Layer:
        #    - Linear layer with input size `hidden_dim` and output size `hidden_dim`
        #    - ReLU (Rectified Linear Unit) activation
        # 3. Output Layer:
        #    - Linear layer with input size `hidden_dim` and output size `out_dim`
        #    - `out_dim` units represent the number of classes in a classification task
        # Use nn.Sequential() to stack these layers together.
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        ##### YOUR CODE END #####

    def forward(self, x):
        x = self.flatten(x)

        ##### YOUR CODE START #####
        # Write a forward pass of your model
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        ##### YOUR CODE END #####

        return logits


# model을 사용하기 위해 input data를 () 연산을 통해 전달하면, 이를 통해 ``forward`` 함수와 더불어 여러 작업들이 자동으로 수행된다.
# 
# 직접 ``model.forward()``를 호출하지 말것!

# 우리 모델은 아래와 같이 구성된다:
# - nn.Flatten: 이차원 28x28 이미지를 연속된 784 픽셀값을 가지는 1차원 텐서로 변환한다 (minibatch dimension (at dim=0)은 그대로 유지됨).
# - nn.Linear: linear레이어는 입력에 대하여 learnable weights와 biases를 이용하여 선형 변환을 수행하는 모듈이다 (Fully connected layer)
# - nn.ReLU: Non-linear activations. 선형 변환 후에 적용하여 비선형성을 추가하며, 모델이 다양한 분포를 학습할수 있도록 해준다.
# - nn.Sequential: 순서가 있는 모듈의 컨테이너(container). 데이터는 컨테이너 내의 모든 모듈들을 정의된것과 같은 순서로 통과한다.
# - nn.Softmax: MLP의 마지막 레이어는 [-infty, infty]에 분포하는 logit 값을 리턴한다. logit값은 nn.Softmax모듈에 전달되거 [0, 1]사이의 확률값으로 변환된다. dim 인자를 통해 어느 축의 값이 더해서 1이 될지 정한다.

# 정의한 모델은 그 내부에 parameter(weights and biases)를 갖고 있다.
# 
# Network를 정의할때 ``nn.Module``를 상속하게 되면 모델에 정의된 모든 파라미터를 자동으로 추적하며 ``parameters()`` 또는 ``named_parameters()`` 메서드를 통해 접근 가능해지게 된다.

# ### 3. Optimizing your model parameters

# 아래 코드를 통해 backward()를 수행시 모델 파라미터의 gradient가 계산되어 저장되는 것을 확인할 수 있다.

# 이제 parameter 업데이트를 통해 학습을 수행하자.
# 
# 학습은 다음의 과정으로 구성된다.
# 
# for  each iteration (called an **epoch**):
# - forward pass를 통하여 output (guess) 계산
# - output과 target의 차이를 통해 error(loss)를 계산
# - 파라미터에 대한 loss의 값의 미분 계산
# - 경사하강법(gradient descent)를 통하여 파라미터 optimize
# 
# 각 epoch은 다음의 과정으로 구성된다
# - 학습 루프(Train Loop) - 학습 데이터를 수행하여 최적 파라미터를 학습한다
# - Evaluation(Validation/Test) Loop - 평가 데이터셋을 순회하며 모델의 성능이 좋아지고 있는지 평가한다.
# 
# 아직 이 과정이 익숙하지 않은 학생들은 [영상](https://www.youtube.com/watch?v=tIeHLnjs5U8)을 참고할것

# <mark>과제</mark>
# ``train_loop``와 ``evaluation_loop``를 완성하라

# In[35]:


def train_loop(model, device, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() #Switch to train mode

    running_loss = 0.0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        ##### YOUR CODE START #####
        # Forward propagation and compute loss
        logits = model(X)
        loss = loss_fn(logits, y)
        # Backpropagation and update parameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ##### YOUR CODE END #####
    
        running_loss += loss.item()

        if batch_idx % 300 == 0:
            print(f"Train loss: {loss.item():>7f}  [{(batch_idx + 1) * len(X):>5d}/{size:>5d}]")

    avg_train_loss = running_loss / len(dataloader)
    return(avg_train_loss)


# In[36]:


def evaluation_loop(model, device, dataloader, loss_fn):    
    model.eval() # Set the model to evaluation mode. important for batch normalization and dropout layers

    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            ##### YOUR CODE START #####
            # accumulate test_loss : sum of loss over mini-batches
            # accumulate correct : the number of correct prediction (hint: use argmax)
            logits = model(X)
            loss = loss_fn(logits, y)

            test_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            ##### YOUR CODE END #####

    avg_test_loss = test_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    print(f"\nTest Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_test_loss:>8f} \n")

    return(avg_test_loss, accuracy)


# ### 4. 학습 수행
# 
# 실제 학습 수행을 위해서는 hyperparameter를 지정해주어야 한다.
# 하이퍼 파라미터는 모델 학습 과정을 컨트롤 할수 있는 파라미터로, 모델 학습 및 수렴속도를 결정짓는 중요한 값들입니다.
# 
# - Number of Epochs: 데이터 전체를 반복할 횟수
# - Batch Size: mini-batch 사이즈. 
# - Learning Rate: 각 iteration마다 얼마나 모델 파라미터를 업데이트 할지.
# 
# ``nn.CrossEntropyLoss``는 ``nn.LogSoftmax``와 ``nn.NLLLoss``를 합친 loss function이다
# 
# 현재까지 배운것을 모두 합쳐 학습을 수행해보자.

# In[37]:


def main():
    # Hyper parameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 40

    # training setting
    in_dim, hidden_dim, out_dim = 28*28, 512, 10
    save_model = True
    
    seed = 1
    torch.manual_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")


    train_dataset, test_dataset = load_MNIST_datasets("/datasets")
    
    train_dataloader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=False)


    model = MultiLayerPerceptron(in_dim, hidden_dim, out_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    train_loss_list, test_loss_list, test_accuracy_list = [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(model, device, train_dataloader, loss_fn, optimizer)
        test_loss, test_accuracy = evaluation_loop(model, device, test_dataloader, loss_fn)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

    if save_model:
        torch.save(model.state_dict(), "model.pth")
        print("Saved Model State to model.pth")

    return train_loss_list, test_loss_list, test_accuracy_list


# In[ ]:




