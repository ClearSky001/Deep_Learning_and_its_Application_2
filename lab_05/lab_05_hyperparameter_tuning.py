#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import wandb

from training_utilities import train_loop, evaluation_loop, save_checkpoint, load_checkpoint


# # wandb 업데이트

# In[2]:


get_ipython().system(' pip install wandb --upgrade')


# In[4]:


print("wandb version : ",wandb.__version__)


# 이번 실습시간에는 다양한 학습 전략과 hyperparameter tuning을 통해 CIFAR-10 테스트셋에서 높은 분류 성능을 얻는 것이 목표이다.
# 
# <mark>과제</mark> 다양한 조건에서 CIFAR-10 데이터셋 학습을 실험해보고 test 데이터셋에서 80% 이상의 accuracy를 달성하라.
# 
# * 제출물1 : <u>5개 이상의 학습 커브</u>를 포함하는 wandb 화면 캡처 (wandb 웹페이지의 본인 이름 포함하여 캡처)
# * 제출물2 : 실험 결과에 대한 분석과 논의 (아래에 markdown으로 기입)
# 
# 참고: 코드에 대한 pytest가 따로 없으므로 자유롭게 코드를 변경하여도 무방함.
# 
# 단, <U>Transfer learning 혹은 Batch size는 변경은 수행하지 말것</U>
# 
# 실험 조건 예시
# - [Network architectures](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
# - input normalization
# - [Weight initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_)
# - [Optimizers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) (Adam, SGD with momentum, ... )
# - Regularizations (weight decay, dropout, [Data augmentation](https://pytorch.org/vision/0.9/transforms.html), ensembles, ...)
# - learning rate & [learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
# 
# 스스로 neural network를 구축할 경우 아래 사항들을 고려하라
# - Filter size
# - Number of filters
# - Pooling vs Strided Convolution
# - Activation functions

# # 첫번째 모델

# In[9]:


# Modify the configuration to experiment with different hyperparameters
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,  # As per the assignment, batch size should not be changed
    'learning_rate': 5e-4,  # Adjusting learning rate for experimentation
    'num_epochs': 100,  # Reducing the number of epochs for quicker experimentation
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_hyperparameter_tuning_modified',

    # Using Adam optimizer in this configuration
    "checkpoint_save_interval": 10,
    "checkpoint_path": "checkpoints/checkpoint_modified.pth",
    "best_model_path": "checkpoints/best_model_modified.pth",
    "load_from_checkpoint": None,  # Start from scratch for this experiment
}

# I will adjust the training function to use Adam optimizer and modify the learning rate scheduler.
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("Model not supported: {}".format(model_name))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")

    return model

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42

    normalize = transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
    
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)

    # Split train dataset into train and validataion dataset
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), 
                                                  test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # DataLoader
    kwargs = {}
    if device.startswith("cuda"):
        kwargs.update({
            'pin_memory': True,
        })

    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_worker, **kwargs)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def train_loop(model, device, train_dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    print(f"Training Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

def evaluation_loop(model, device, dataloader, criterion, epoch=None, phase="validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    if epoch is not None:
        print(f"{phase.capitalize()} Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return accuracy, avg_loss

def train_main_modified(config):
    ## data and preprocessing settings
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)

    ## Hyper parameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']

    ## checkpoint setting
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)

    ## variables
    best_acc1 = 0

    wandb.init(
        project=config["wandb_project_name"],
        config=config
    )

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_cifar10_dataloaders(
        data_root_dir, device, batch_size=batch_size, num_worker=num_worker)

    model = get_model(model_name=config["model_name"], num_classes=num_classes, config=config).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Using Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Using CosineAnnealingLR scheduler for better learning rate adaptation
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    if load_from_checkpoint:
        load_checkpoint_path = best_model_path if load_from_checkpoint == "best" else checkpoint_path
        start_epoch, best_acc1 = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    if config.get('test_mode', False):
        # Only evaluate on the test dataset
        print("Running test evaluation...")
        test_acc, test_loss = evaluation_loop(model, device, test_dataloader, criterion, phase="test")
        print(f"Test Accuracy: {test_acc}")

    else:
        # Train and validate using train/val datasets
        for epoch in range(start_epoch, num_epochs):
            # Training phase
            train_loss, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
            
            # Validation phase
            val_acc1, val_loss = evaluation_loop(model, device, val_dataloader, criterion, epoch=epoch, phase="validation")
            scheduler.step()

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc1
            })

            if (epoch + 1) % checkpoint_save_interval == 0 or (epoch + 1) == num_epochs:
                is_best = val_acc1 > best_acc1
                best_acc1 = max(val_acc1, best_acc1)
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_acc1, is_best, best_model_path)

    wandb.finish()

# Run the modified training function to perform the experiment
train_main_modified(config_modified)


# ## 모델의 가중치(weights) 추출하기

# In[11]:


# 모델 학습 후 최종 상태의 가중치를 불러오고 추출하기
model = get_model("resnet50", num_classes=10, config=config_modified)
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 학습된 체크포인트 불러오기
checkpoint_path = config_modified["best_model_path"]  # 학습된 최적의 모델 경로
optimizer = optim.Adam(model.parameters(), lr=config_modified['learning_rate'], weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 저장된 가중치 로드
start_epoch, best_acc1 = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

# 모델에서 가중치만 추출하기
weights = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        weights[name] = param.data.cpu().numpy()  # 가중치를 추출하고 numpy 배열로 변환
        print(f"Layer: {name} | Weight Shape: {param.shape} | Number of Weights: {param.numel()}")


# ## 성능 평가

# In[10]:


# Evaluate the best model on the test set
config_testmode = {
    **config_modified,
    'test_mode': True,  # True if evaluating only on the test set
    'load_from_checkpoint': 'best'
}

train_main_modified(config_testmode)


# 실험이 모두 끝나면 best model에 대해 test set성능을 평가한다. 

# In[ ]:


config_testmode = {
    **config, 
    'test_mode': True, # True if evaluating only test set
    'load_from_checkpoint': 'best'
}

train_main(config_testmode)


# <mark>제출물</mark>
# 
# 1. 본인 이름이 나오도록 wandb 결과 화면을 캡처하여 `YOUR_PRIVATE_REPOSITORY_NAME/lab_05/wandb_results.png`에 저장한다. (5 points)
# 2. 결과를 table로 정리한 뒤 그 아래에 분석 및 논의를 작성 한다. (15 points)
# 
# -----

# #### wandb 결과
# 
# * **fearless-armadillo-4**
# 
# <center><img src="wandb_results.png" width="1000px"></img></center>
# 
# #### 5개 이상의 실험 결과
# 
# | 모델     | 실험 조건                   | Train Accuracy (%) | Train Loss | Validation Accuracy (%) | Validation Loss | 설명           |
# |----------|----------------------------|--------------------|------------|-------------------------|-----------------|----------------|
# | ResNet50 | learning_rate=5e-4, Adam   | **81.03**          | **0.5378** | **75.29**               | **0.6942**      | 기본 설정 실험 |
# 
# **best model test_set accuracy:** **75.29%**
# 
# #### 분석 및 논의
# 
# ##### 첫번째 모델(fearless-armadillo-4)
# * **모델 학습 과정:** 학습 초기에는 모델의 손실 값이 매우 높았으나, 에포크가 진행됨에 따라 점진적으로 감소하며 정확도가 증가하는 것을 관찰할 수 있었다. 특히, 학습이 50 에포크 부근에 도달했을 때 모델의 성능이 눈에 띄게 향상되었다. 하지만, 대략 60 에포크 이후부터는 모델의 학습 성능이 소폭 감소되는 양상을 보였다.
# 
# * **Validation 성능:** 검증 정확도 또한 학습이 진행됨에 따라 향상되는 흐름을 보였다. 그러나, 학습 정확도와 검증 정확도 간의 차이가 점차 벌어지는 경향이 보였으며 이는 과적합의 가능성을 나타낸다.
# 
# * **Adam 옵티마이저와 학습률 스케줄러:** Adam 옵티마이저를 사용하여 학습이 빠르게 진행되었으며, CosineAnnealingLR 스케줄러를 사용하여 학습률을 점차 줄여 안정적인 학습을 유도했다. 이를 통해 초기에는 높은 학습률로 빠르게 학습하고, 후반에는 낮은 학습률로 더 안정된 학습을 진행할 수 있었다.
# 
# * **결과 분석:** 최종 테스트 정확도는 75.29%로, 목표였던 80% 정확도에는 도달하지 못했지만, 전반적인 학습 곡선의 형태와 성능 개선을 확인할 수 있었다.
# 
# * ***개선 방안:*** 데이터셋의 다양성 및 모델 구조의 한계로 인해 추가적인 데이터 전처리나, 모델 학습 시에 다른 기법(예: Dropout)이 추가로 필요할 것으로 보인다. 또한, 옵티마이저의 종류나 학습 스케줄러의 종류를 변경하여 추가적인 실험이 필요해 보인다.
# 
# ##### 두번째 모델
# 

# -----
# 
# #### Lab을 마무리 짓기 전 저장된 checkpoint를 모두 지워 저장공간을 확보한다

# In[1]:


import shutil, os
if os.path.exists('checkpoints/'):
    shutil.rmtree('checkpoints/')

