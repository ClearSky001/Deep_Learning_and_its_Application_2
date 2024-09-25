#!/usr/bin/env python
# coding: utf-8

# Convolutional neural Network
# ============================
# 
# 이번 실습에서는 지난 시간에 작성한 코드의 사용성을 더 개선하고 간단한 CNN모델을 구현해본다.
# 1. custom datasets 작성하기
# 2. W&B를 이용하여 학습및 평가 트래킹하기
# 3. CNN모델 구현하기

# ### Custom datasets 작성하기
# ``torch.utils.data.Dataset``는 dataset을 위한 추상클래스이다. custom dataset은 이 클래스를 상속하고 아래 두가지 메서드를 override하여야 한다:
# 
# 
# 
# * `__len__`을 구현하여 len(dataset)이 데이터셋 크기를 리턴한다
# 
# * `__getitem__` 을 구현하여 dataset[i] 와 같은 인덱싱에서 i번째 샘플을 리턴한다.

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


class CustomDatasetExample(Dataset):
    def __init__ (self , X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]


# 이제 아래 csv파일로 부터 이미지 경로와 라벨 정보를 읽어오는 사용자정의 dataset을 구현해보자

# <mark>과제</mark>: root directory의 경로 "resources/cat_dog_images"와 이미지의 경로 및 label를 저장하고 있는 메타데이터 meta.csv파일을 입력으로 받는 custom datsets을 구현하라
# 
# * 이미지를 읽는데에는 [PIL.Image.open](https://pillow.readthedocs.io/en/stable/reference/Image.html) 함수를 사용할 것.

# In[25]:


from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, metadata_filename, transform=None, target_transform=None):
        self.metadata_df = pd.read_csv(os.path.join(root_dir, metadata_filename))
        self.root_dir = root_dir

        classes = self.metadata_df["label"].drop_duplicates().sort_values().tolist()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        ##### YOUR CODE START #####
        return len(self.metadata_df)
        ##### YOUR CODE END #####

    def __getitem__(self, idx):
        ##### YOUR CODE START #####
        img_path = os.path.join(self.root_dir, self.metadata_df.iloc[idx, 0]) # 첫 번째 컬럼에 이미지 경로가 있다고 가정
        label_str = self.metadata_df.iloc[idx, 1] # 두 번째 컬럼에 레이블이 있다고 가정

        # 이미지 파일을 읽음
        image = Image.open(img_path).convert("RGB")
        ##### YOUR CODE END #####

        label = self.class_to_idx[label_str] # encode label to integer

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


# In[27]:


def visualize_samples(dataset, cols=8, rows=5):
    figure = plt.figure(figsize=(12, 6))
    for i in range(len(dataset)):
        img, label = dataset[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()


# 하지만 모든 샘플의 이미지 크기가 같지 않기 때문에 아래와 같이 dataloader를 이용해 mini-batch로 쌓으려고 하면 오류가 생긴다.

# 따라서 transforms를 이용한 전처리가 필요하다.
# ``torchvision.transforms``에 다양한 이미지 전처리 함수가 이미 구현되어있다. 자세히는 [documentation](https://pytorch.org/vision/main/transforms.html)을 참고하기 바란다. 
# 
# 여기서는 그 중 자주 사용되는 몇 가지를 소개한다.

# 데이터를 뽑을때 마다 랜덤하게 다른 augmented image가 출력되는것을 확인할 수 있다.
# 데이터로더에서도 (3, 224, 224) 크기의 이미지가 잘 출력됨을 확인할 수 있다

# ### custom Sampler로 weighted sampling하기
# 
# Sampler는 dataloader가 data를 추출하는 순서를 정해주는 객체로 datasets의 index를 순차적으로 리턴하는 iterator이다.
# 
# ``__iter__``에서는 데이터셋의 순서를 정하는 로직을 통과한 뒤 그 인덱스들을 순회할수 있는 iterator를 반환한다.
# 
# 아래 예시는 label imbalance를 해결하기 위해 각 class로 부터 같은 숫자의 샘플들을 추출하는 샘플러이다.

# In[34]:


from torch.utils.data import Sampler
import random

class BalancedSampler(Sampler): 
    def __init__(self, dataset, class_column_name='label'):
        """
        Args:
            dataset: The dataset to sample from.
            class_column_name: The column name in the dataset that contains class labels.
        """
        self.dataset = dataset
        self.class_column_name = class_column_name
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(self.dataset.metadata_df[class_column_name]):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.min_class_length = min(len(indices) for indices in self.class_indices.values())
        
    def __iter__(self):
        selected_indices = []

        for class_name, indices in self.class_indices.items():
            sampled_indices = random.sample(indices, self.min_class_length)
            selected_indices.extend(sampled_indices)
        
        random.shuffle(selected_indices)
        return iter(selected_indices)
    
    def __len__(self):
        # The length of the sampler will be twice the minimum class length since we are balancing
        return self.min_class_length * len(self.class_indices)


# ### 학습 코드 개선하기
# 
# 아래는 지난 실습에서 작성한 학습코드를 다음의 측면에서 개선한 코드이다. 코드를 리뷰하며 다시 복습해보자.
# 
# - tqdm을 이용학 학습 시간 추적
# - AverageMeter를 사용하여 학습 metric을 추적
# - save_checkpoint와 load_checkpoint를 이용하여 모델을 저장하고 불러오는 기능 추가

# In[43]:


import os, time, shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb


# In[44]:


def load_MNIST_datasets(data_root_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root=data_root_dir, train=True, download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root_dir, train=False, download=True, 
        transform=transform
    )

    return train_dataset, test_dataset


# In[45]:


def create_dataloaders(train_dataset, test_dataset, device, batch_size, num_worker):
    kwargs = {}
    if device.startswith("cuda"):
        kwargs.update({
            'pin_memory': True,
        })

    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_worker, **kwargs)
    
    return train_dataloader, test_dataloader


# In[46]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {avg' + self.fmt + '} (n={count}))'
        return fmtstr.format(**self.__dict__)

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def save_checkpoint(filepath, model, optimizer, epoch, best_acc1, is_best, best_model_path):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'best_acc1': best_acc1,
    }
    
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_model_path)


def load_checkpoint(filepath, model, optimizer, device):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        print(f"=> loaded checkpoint '{filepath}' (epoch {start_epoch})")
        return start_epoch, best_acc1
    else:
        print(f"=> no checkpoint found at '{filepath}'")
        return 0, 0


# In[47]:


def train_loop(model, device, dataloader, criterion, optimizer, epoch):
    # train for one epoch
    losses = AverageMeter('Loss', ':.4e')
    acc_top1 = AverageMeter('Acc@1', ':6.2f')
    data_time = AverageMeter('Data_Time', ':6.3f') # Time for data loading
    batch_time = AverageMeter('Batch_Time', ':6.3f') # time for mini-batch train
    metrics_list = [losses, acc_top1, data_time, batch_time, ]
    
    model.train() # switch to train mode

    end = time.time()

    tqdm_epoch = tqdm(dataloader, desc=f'Training Epoch {epoch + 1}', total=len(dataloader))
    for images, target in tqdm_epoch:
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, = calculate_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        acc_top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        tqdm_epoch.set_postfix(avg_metrics = ", ".join([str(x) for x in metrics_list]))

        end = time.time()
    tqdm_epoch.close()

    ##### YOUR CODE START #####
    # wandb log following variables: losses.avg, acc_top1.avg
    wandb.log({
        "Train Loss": losses.avg,
        "Train Accuracy": acc_top1.avg,
        "Epoch": epoch + 1
    })
    ##### YOUR CODE END #####


# In[48]:


def evaluation_loop(model, device, dataloader, criterion, epoch = 0):
    losses = AverageMeter('Loss', ':.4e')
    acc_top1 = AverageMeter('Acc@1', ':6.2f')
    metrics_list = [losses, acc_top1]

    model.eval() # switch to evaluate mode

    with torch.no_grad():
        tqdm_val = tqdm(dataloader, desc='Validation/Test', total=len(dataloader))
        for images, target in tqdm_val:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, = calculate_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            acc_top1.update(acc1[0], images.size(0))

            tqdm_val.set_postfix(avg_metrics = ", ".join([str(x) for x in metrics_list]))

        tqdm_val.close()

    ##### YOUR CODE START #####
    wandb.log({
        "Validation Loss": losses.avg,
        "Validation Accuracy": acc_top1.avg,
        "Epoch": epoch + 1
    })
    ##### YOUR CODE END #####

    return acc_top1.avg


# In[49]:


class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.flatten(x)

        logits = self.linear_relu_stack(x)
        return logits


# In[50]:


def train_main_MLP():
    ## data and preprocessing settings
    data_root_dir = '/datasets' 
    num_worker = 4

    ## Hyper parameters
    batch_size = 128
    learning_rate = 1e-3
    start_epoch = 0
    num_epochs = 50

    ## checkpoint setting
    checkpoint_save_interval = 10
    checkpoint_path = "checkpoints/checkpoint.pth"
    best_model_path = "checkpoints/best_model.pth"
    load_from_checkpoint = None # Options: "latest", "best", or None

    ## variables
    best_acc1 = 0

    ## set learning deterministic
    # torch.manual_seed(1)

    ##### YOUR CODE START #####
    wandb.init(project="MNIST-MLP", config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    })
    ##### YOUR CODE END #####


    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataset, test_dataset = load_MNIST_datasets(data_root_dir)
    num_classes = len(train_dataset.classes)
    
    train_dataloader, test_dataloader = create_dataloaders(train_dataset, test_dataset, device, 
                                                           batch_size = batch_size, num_worker = num_worker)


    model = MultiLayerPerceptron(in_dim = 28*28, hidden_dim = 512, out_dim = num_classes).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    if load_from_checkpoint:
        load_checkpoint_path = (best_model_path if load_from_checkpoint == "best" else checkpoint_path)
        start_epoch, best_acc1 = load_checkpoint(load_checkpoint_path, model, optimizer, device)

    for epoch in range(start_epoch, num_epochs):
        train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
        acc1 = evaluation_loop(model, device, test_dataloader,criterion, epoch)


        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if (epoch + 1) % checkpoint_save_interval == 0 or is_best:
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_acc1, is_best, best_model_path)

    ##### YOUR CODE START #####
    wandb.finish()
    ##### YOUR CODE END #####


# <mark>과제</mark> [wandb](https://kr.wandb.ai/) 회원가입을 하고 메뉴얼에 따라 Train Loss, Train Accuracy, Validation Loss, Validation Accuracy를 추적하도록 위 함수들을 적절히 수정하고 학습해보라

# <mark>과제</mark> wandb 로그 폴더와 checkpoint를 git에 올리지 않기 위에 `.gitignore`를 수정하라

# (참고) shell 환경변수를 설정하여 wandb를 끌수있다.

# ### Training a small Convolutional Neural Network

# CNN은 보통 아래 4개 레이어로 구성된다
# 
# - Convolutional layer
# - Activation 
# - Pooling layer
# - FC layer
# 
# PyTorch에는 다양한 레이어가 있지만 우리는 그중 Conv2d, ReLU, Maxpool, Linear(FC) 레이어를 이용할 예정이다.

# #### nn.Conv2d
# 
# parameter
# 
# - in_channels(int) : Number of input images channel
# - out_channels(int) : Number of filters
# - kernel_size(int or tuple) : 필터의 크기는 kernel_size * kernel_size이다
# - stride(int or tuple) : Default = 1
# - padding(int or tuple) : zero padding. Default = 0

# #### nn.MaxPool2d
# 
# parameter
# - kernel_size – the size of the window to take a max over
# - stride – the stride of the window. (Default = kernel_size)
# - padding – implicit zero padding to be added on both sides (default = 0)

# 이제 CNN 모델을 구현할 준비가 완료되었다.
# 
# <mark>과제</mark> 아래 구조의 CNN모델을 완성하라
# 
# * 5x5 Convolutional layer with 8 filters, strides of 1, no zero padding, and ReLU activation
# * 2x2 Max pooling layer with strides of 2
# * 5x5 Convolutional layer with 16 filters, strides of 1, no zero padding, and ReLU activation
# * 2x2 Max pooling layer with strides of 2
# * Fully connected layer with 128 output units and ReLU activation
# * Fully connected layer with 64 output units and ReLU activation
# * Fully connected layer with 10 output units 

# In[59]:


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##### YOUR CODE START #####
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)  # Fully connected layer, input size depends on image size after conv/pool
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        ##### YOUR CODE END #####

    def forward(self, x):
        ##### YOUR CODE START #####
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 4 * 4)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        ##### YOUR CODE END #####

        return x


# 파라미터 수를 스스로 계산해보고 맞게 구현됐는지 검증하라
# 
# - Conv layer = filter number x filter size x filter size + bias(filter number)
# - FC layer = input size x output size + bias(output size)

# <mark>과제</mark> 학습을 수행하여 MLP 모델과의 결과를 비교하라.

# In[61]:


def train_main_CNN():
    ##### YOUR CODE START #####
    # wandb 설정 및 초기화
    wandb.init(project="MNIST-CNN", config={
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 128
    })
    config = wandb.config

    # 디바이스 설정 추가
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device = device

    # 데이터셋 및 데이터 로더 설정
    transform = transforms.Compose([transforms.ToTensor()])
    data_root_dir = './datasets'
    
    train_dataset = datasets.MNIST(root=data_root_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_root_dir, train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    # 모델 및 손실 함수, 옵티마이저 설정
    model = SimpleCNN().to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # 학습 루프
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Train Accuracy 계산
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 100 == 99:  # 매 100 미니배치마다 로그 기록
                wandb.log({"Train Loss": running_loss / 100})
                running_loss = 0.0

        train_acc = 100 * correct_train / total_train

        # 검증 루프
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss = running_val_loss / len(test_loader)

        # wandb 로그 기록 (Epoch, Train Loss, Train Accuracy, Validation Loss, Validation Accuracy)
        wandb.log({
            "Epoch": epoch + 1,
            "Train Accuracy": train_acc,
            "Train Loss": running_loss,
            "Validation Accuracy": val_acc,
            "Validation Loss": val_loss
        })

        print(f'Epoch {epoch + 1}, Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

    # wandb 종료
    wandb.finish()
    ##### YOUR CODE END #####


# In[ ]:




