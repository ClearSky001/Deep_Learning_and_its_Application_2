#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import wandb

from training_utilities import train_loop, evaluation_loop, save_checkpoint, load_checkpoint


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

# # 첫번째 모델(ResNet50_first_train)

# In[8]:


# Modify the configuration to experiment with different hyperparameters
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,  # As per the assignment, batch size should not be changed
    'learning_rate': 5e-4,  # Adjusting learning rate for experimentation
    'num_epochs': 100,  # Reducing the number of epochs for quicker experimentation
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_training_with_various_models',

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
        config=config,
        name="ResNet50_first_train"
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


# ## 성능 평가

# In[9]:


# Evaluate the best model on the test set
config_testmode = {
    **config_modified,
    'test_mode': True,  # True if evaluating only on the test set
    'load_from_checkpoint': 'best'
}

train_main_modified(config_testmode)


# # 두번째 모델(ResNet50_model_second_train)

# In[4]:


# Configuration for improved experiment
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,  # Fixed as per assignment
    'learning_rate': 1e-3,  # Higher initial learning rate
    'num_epochs': 100,
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_training_with_various_models',
    "checkpoint_save_interval": 10,
    "checkpoint_path": "checkpoints/checkpoint_modified.pth",
    "best_model_path": "checkpoints/best_model_modified.pth",
    "load_from_checkpoint": None,
}

def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise Exception("Model not supported: {}".format(model_name))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")
    return model

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Data Augmentation for training set
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'pin_memory': True} if device.startswith("cuda") else {}
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, **kwargs)
    
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
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)
    best_acc1 = 0

    wandb.finish()
    wandb.init(
        project=config["wandb_project_name"],
        config=config,
        name="ResNet50_model_second_train"
    )
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_cifar10_dataloaders(
        data_root_dir, device, batch_size=batch_size, num_worker=num_worker)
    model = get_model(model_name=config["model_name"], num_classes=num_classes, config=config).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Using SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    if load_from_checkpoint:
        load_checkpoint_path = best_model_path if load_from_checkpoint == "best" else checkpoint_path
        start_epoch, best_acc1 = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    if config.get('test_mode', False):
        print("Running test evaluation...")
        test_acc, test_loss = evaluation_loop(model, device, test_dataloader, criterion, phase="test")
        print(f"Test Accuracy: {test_acc}")
    else:
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
            val_acc1, val_loss = evaluation_loop(model, device, val_dataloader, criterion, epoch=epoch, phase="validation")
            scheduler.step(val_loss)

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


# ## 성능 평가

# In[5]:


# Evaluate the best model on the test set
config_testmode = {
    **config_modified,
    'test_mode': True,  # True if evaluating only on the test set
    'load_from_checkpoint': 'best'
}

train_main_modified(config_testmode)


# # 세 번째 모델

# In[6]:


# Configuration for improved experiment
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,  # Fixed as per assignment
    'learning_rate': 1e-3,  # Higher initial learning rate
    'num_epochs': 200,
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_training_with_various_models',
    "checkpoint_save_interval": 10,
    "checkpoint_path": "checkpoints/checkpoint_modified.pth",
    "best_model_path": "checkpoints/best_model_modified.pth",
    "load_from_checkpoint": None,
}

def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise Exception("Model not supported: {}".format(model_name))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")
    return model

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Data Augmentation for training set
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'pin_memory': True} if device.startswith("cuda") else {}
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, **kwargs)
    
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
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)
    best_acc1 = 0

    wandb.finish()
    wandb.init(
        project=config["wandb_project_name"],
        config=config,
        name="ResNet50_model_3rd_train"
    )
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_cifar10_dataloaders(
        data_root_dir, device, batch_size=batch_size, num_worker=num_worker)
    model = get_model(model_name=config["model_name"], num_classes=num_classes, config=config).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Using SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    if load_from_checkpoint:
        load_checkpoint_path = best_model_path if load_from_checkpoint == "best" else checkpoint_path
        start_epoch, best_acc1 = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    if config.get('test_mode', False):
        print("Running test evaluation...")
        test_acc, test_loss = evaluation_loop(model, device, test_dataloader, criterion, phase="test")
        print(f"Test Accuracy: {test_acc}")
    else:
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
            val_acc1, val_loss = evaluation_loop(model, device, val_dataloader, criterion, epoch=epoch, phase="validation")
            scheduler.step(val_loss)

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


# ## 성능 평가

# In[7]:


# Evaluate the best model on the test set
config_testmode = {
    **config_modified,
    'test_mode': True,  # True if evaluating only on the test set
    'load_from_checkpoint': 'best'
}

train_main_modified(config_testmode)


# # 네 번째 모델

# In[2]:


import torch
from torch.optim import Optimizer

# SAM Optimizer Implementation
class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "SAM requires non-negative rho."
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        scale = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (scale + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * (torch.abs(p) if group["adaptive"] else 1.0) * scale.to(p)
                p.add_(e_w)  # Ascent step
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # Descent step
        self.base_optimizer.step()  # Perform actual update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # Get loss for first step
        self.first_step(zero_grad=True)
        closure()
        self.second_step()


# In[3]:


# Configuration for improved experiment with GELU, BatchNorm, and SAM optimizer
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,  # Fixed as per assignment
    'learning_rate': 1e-3,  # Higher initial learning rate
    'num_epochs': 150,  # Updated to 150 epochs
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_training_with_various_models',
    "checkpoint_save_interval": 10,
    "checkpoint_path": "checkpoints/checkpoint_modified.pth",
    "best_model_path": "checkpoints/best_model_modified.pth",
    "load_from_checkpoint": None,
}

def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization 추가
            nn.GELU(),            # GELU 활성화 함수로 변경
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise Exception("Model not supported: {}".format(model_name))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")
    return model

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Data Augmentation for training set
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'pin_memory': True} if device.startswith("cuda") else {}
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, **kwargs)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def train_loop(model, device, train_dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # SAM optimizer requires two steps: forward-backward for sharpness-aware gradient adjustment
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(closure)  # SAM의 두 단계 업데이트

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
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)
    best_acc1 = 0

    wandb.finish()
    wandb.init(
        project=config["wandb_project_name"],
        config=config,
        name="ResNet50_model_4th_train_with_SAM"
    )
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_cifar10_dataloaders(
        data_root_dir, device, batch_size=batch_size, num_worker=num_worker)
    model = get_model(model_name=config["model_name"], num_classes=num_classes, config=config).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Using SAM with 'SGD with momentum'
    optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    if load_from_checkpoint:
        load_checkpoint_path = best_model_path if load_from_checkpoint == "best" else checkpoint_path
        start_epoch, best_acc1 = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    if config.get('test_mode', False):
        print("Running test evaluation...")
        test_acc, test_loss = evaluation_loop(model, device, test_dataloader, criterion, phase="test")
        print(f"Test Accuracy: {test_acc}")
    else:
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
            val_acc1, val_loss = evaluation_loop(model, device, val_dataloader, criterion, epoch=epoch, phase="validation")
            scheduler.step(val_loss)

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


# ## 성능 평가

# In[4]:


# Evaluate the best model on the test set
config_testmode = {
    **config_modified,
    'test_mode': True,  # True if evaluating only on the test set
    'load_from_checkpoint': 'best'
}

train_main_modified(config_testmode)


# # 다섯번째 모델

# In[6]:


import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
from torchvision import models, transforms
from torchvision.transforms import AutoAugmentPolicy
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# SAM Optimizer Implementation (이미 작성된 SAM 코드 그대로 사용)
class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "SAM requires non-negative rho."
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        scale = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (scale + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * (torch.abs(p) if group["adaptive"] else 1.0) * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()


# In[7]:


# Configuration with AutoAugment, increased weight_decay, and other techniques
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 250,
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_training_with_various_models',
    "checkpoint_save_interval": 10,
    "checkpoint_path": "checkpoints/checkpoint_modified.pth",
    "best_model_path": "checkpoints/best_model_modified.pth",
    "load_from_checkpoint": None,
    'test_mode': False  # 기본적으로 학습 모드
}

# Model with extended hidden dimensions, additional layer, and stochastic depth
def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 1024),  # 확장된 hidden dimension
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),                   # 추가된 hidden layer
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
    else:
        raise Exception("Model not supported: {}".format(model_name))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")
    return model

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Data Augmentation for training set with AutoAugment
    train_transforms = transforms.Compose([
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'pin_memory': True} if device.startswith("cuda") else {}
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, **kwargs)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

# Training loop without Gradient Clipping
def train_loop(model, device, train_dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(closure) 

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    print(f"Training Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def evaluation_loop(model, device, dataloader, criterion, phase="validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    print(f"{phase.capitalize()} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    if phase == "test":
        print(f"Test Accuracy: {accuracy:.2f}")  # Add summary test accuracy output
    return accuracy, avg_loss

def train_main_modified(config):
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)
    test_mode = config.get('test_mode', False)
    best_acc = 0

    # Initialize WandB
    wandb.finish()
    wandb.init(
        project=config["wandb_project_name"],
        config=config,
        name="ResNet50_5th_train_with_extra_tuning"
    )

    # Set Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model Setup
    model = get_model(config["model_name"], 10, config).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Optimizer and Scheduler Setup
    optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Load Data
    train_dataloader, val_dataloader, test_dataloader, _ = load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker)

    # Load checkpoint if needed
    if load_from_checkpoint:
        load_checkpoint_path = best_model_path if load_from_checkpoint == "best" else checkpoint_path
        start_epoch, best_acc = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    # Test mode: evaluate only on the test set and skip training
    if test_mode:
        print("Running in test mode...")
        test_acc, test_loss = evaluation_loop(model, device, test_dataloader, criterion, phase="test")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        wandb.finish()
        return

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
        val_acc, val_loss = evaluation_loop(model, device, val_dataloader, criterion, phase="validation")
        scheduler.step(val_loss)


        # Log metrics to WandB
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc
        })

        # Save checkpoint periodically or if it's the final epoch
        if (epoch + 1) % checkpoint_save_interval == 0 or (epoch + 1) == num_epochs:
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_acc, is_best, best_model_path)

    wandb.finish()

# Run the modified training function to perform the experiment with AutoAugment
train_main_modified(config_modified)


# ## 성능 평가

# In[8]:


# Configuration for Test Mode
config_testmode = {
    **config_modified,
    'test_mode': True,  # 평가 모드 설정
    'load_from_checkpoint': 'best'
}

# Run the model in test mode for evaluation
train_main_modified(config_testmode)


# # 여섯 번째 모델

# In[10]:


import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
from torchvision import models, transforms
from torchvision.transforms import AutoAugmentPolicy
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# SAM Optimizer Implementation (이미 작성된 SAM 코드 그대로 사용)
class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "SAM requires non-negative rho."
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        scale = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (scale + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * (torch.abs(p) if group["adaptive"] else 1.0) * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()


# In[11]:


# Configuration with AutoAugment, increased weight_decay, and other techniques
config_modified = {
    'data_root_dir': '/datasets',
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 250,
    'model_name': 'resnet50',
    'wandb_project_name': 'CIFAR10_training_with_various_models',
    "checkpoint_save_interval": 10,
    "checkpoint_path": "checkpoints/checkpoint_modified.pth",
    "best_model_path": "checkpoints/best_model_modified.pth",
    "load_from_checkpoint": None,
    'test_mode': False  # 기본적으로 학습 모드
}

# Model with extended hidden dimensions, additional layer, and stochastic depth
def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 1024),  # 확장된 hidden dimension
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),                   # 추가된 hidden layer
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )
    else:
        raise Exception("Model not supported: {}".format(model_name))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")
    return model

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Data Augmentation for training set with AutoAugment
    train_transforms = transforms.Compose([
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'pin_memory': True} if device.startswith("cuda") else {}
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, **kwargs)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

# Training loop without Gradient Clipping
def train_loop(model, device, train_dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(closure) 

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    print(f"Training Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def evaluation_loop(model, device, dataloader, criterion, phase="validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    if phase == "test":
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")  # Update output to show only once for test mode
    else:
        print(f"{phase.capitalize()} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss

def train_main_modified(config):
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)
    test_mode = config.get('test_mode', False)
    best_acc = 0

    # Initialize WandB
    wandb.finish()
    wandb.init(
        project=config["wandb_project_name"],
        config=config,
        name="ResNet50_6th_train"
    )

    # Set Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model Setup
    model = get_model(config["model_name"], 10, config).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Optimizer and Scheduler Setup
    optimizer = SAM(model.parameters(), base_optimizer=torch.optim.AdamW, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Load Data
    train_dataloader, val_dataloader, test_dataloader, _ = load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker)

    # Load checkpoint if needed
    if load_from_checkpoint:
        load_checkpoint_path = best_model_path if load_from_checkpoint == "best" else checkpoint_path
        start_epoch, best_acc = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    # Test mode: evaluate only on the test set and skip training
    if test_mode:
        print("Running in test mode...")
        test_acc, test_loss = evaluation_loop(model, device, test_dataloader, criterion, phase="test")
        wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        wandb.finish()
        return

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
        val_acc, val_loss = evaluation_loop(model, device, val_dataloader, criterion, phase="validation")
        scheduler.step()

        # Log metrics to WandB
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc
        })

        # Save checkpoint periodically or if it's the final epoch
        if (epoch + 1) % checkpoint_save_interval == 0 or (epoch + 1) == num_epochs:
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_acc, is_best, best_model_path)

    wandb.finish()

# Run the modified training function to perform the experiment with AutoAugment
train_main_modified(config_modified)


# ## 성능 평가

# In[12]:


# Configuration for Test Mode
config_testmode = {
    **config_modified,
    'test_mode': True,  # 평가 모드 설정
    'load_from_checkpoint': 'best'
}

# Run the model in test mode for evaluation
train_main_modified(config_testmode)


# <mark>제출물</mark>
# 
# 1. 본인 이름이 나오도록 wandb 결과 화면을 캡처하여 `YOUR_PRIVATE_REPOSITORY_NAME/lab_05/wandb_results.png`에 저장한다. (5 points)
# 2. 결과를 table로 정리한 뒤 그 아래에 분석 및 논의를 작성 한다. (15 points)
# 
# -----

# # wandb 결과
# 
# <center><img src="wandb_results_updated.png" width="1000px"></img></center>
# 
# # 5개 이상의 실험 결과
# 
# |모델|실험 조건|test_accuracy|설명|
# |----|---------------------------------------------------|------|---------------------|
# |ResNet50|Activation Function: RELU, Learning Rate: 5e-4, Epochs: 100, Optimizer: Adam, Learning Rate Scheduler: CosineAnnealingLR|74.93%|      |
# |ResNet50|Activation Function: RELU, Learning Rate: 1e-3, Epochs: 100, Optimizer: SGD with Momentum(momentum=0.9, weight_decay=1e-4를 사용), Learning Rate Scheduler: ReduceLROnPlateau, 데이터 증강(Data Augmentation) 적용~>Random Crop: 이미지 크기를 32로 유지하면서 padding=4를 적용한 Random Crop을 사용, Random Horizontal Flip 시용, 0.5의 Dropout 확률 적용|76.18%|마지막 레이어를 확장하여, 512 유닛의 레이어와 ReLU 활성화 함수, Dropout(0.5)을 추가|
# |ResNet50|두번째 모델과 모두 동일하게 설정하였고, epochs만 200으로 변경.|77.19%||
# |ResNet50|Activation Function: GELU, Learning Rate: 1e-3, Epochs: 150, Optimizer: SAM(Sharpness-Aware Minimization)에 SGD with Momentum(momentum=0.9, weight_decay=1e-4를 사용)을 적용, Learning Rate Scheduler: ReduceLROnPlateau, 데이터 증강(Data Augmentation) 적용~>Random Crop: 이미지 크기를 32로 유지하면서 padding=4를 적용한 Random Crop을 사용, Random Horizontal Flip 시용, 0.5의 Dropout 확률 적용|80.79%|최종 FC 층을 재정의함. 2048 → 512로 변환, 배치 정규화 및 GELU 활성화 함수 적용, 드롭아웃을 통한 과적합 방지.|
# |ResNet50|Activation Function: GELU, Learning Rate: 1e-3, Epochs: 250, Optimizer: SAM (base_optimizer: AdamW, weight_decay=1e-4), Learning Rate Scheduler: CosineAnnealingLR (T_max=250, eta_min=1e-6), 데이터 증강 (Data Augmentation): AutoAugment, Random Crop (padding=4), Random Horizontal Flip|83.83%|Hidden Layer를 1024 유닛으로 확장하고, 추가적인 512 유닛의 레이어를 더함., Batch Normalization과 GELU 활성화 함수 적용., Dropout(0.5)를 두 번 적용하여 과적합 방지., AutoAugment를 활용하여 다양한 데이터 증강을 수행함으로써 모델의 일반화 성능을 높임.|
# |ResNet50|Activation Function: GELU, Learning Rate: 1e-3, Epochs: 250, Optimizer: SAM (base_optimizer: AdamW, weight_decay=1e-4), Learning Rate Scheduler: CosineAnnealingLR (T_max=250, eta_min=1e-6), 데이터 증강 (Data Augmentation): AutoAugment, Random Crop (padding=4), Random Horizontal Flip|89.64%|Dropout 비율을 0.3으로 낮춤.|
# 
# **best model test_set accuracy**: **89.64%**
# 
# # 분석 및 논의
# * **모델 학습 과정:** 모든 모델이 초기 학습 시 높은 손실값을 보이다가 학습이 진행됨에 따라 손실값이 감소하고 정확도가 증가하는 경향을 보였다. 첫 번째 모델부터 여섯 번째 모델까지 각기 다른 학습 조건과 최적화 기법을 적용하였고 모델 성능의 지속적인 향상이 관찰되었다. 네 번째 모델 이후부터는 현재까지 알려진 optimizer로 알려진 SAM 옵티마이저를 기본적으로 사용하였다. 그래서인지 모델의 일반화 성능이 네번째 모델 이후부터는 80%를 넘는 등 이전과 비교할때 개선된 것을 확인할 수 있었다.
# 
# * **Validation 성능:** 학습이 진행됨에 따라 검증 정확도 또한 모든 모델에서 지속적으로 향상되는 것을 확인할 수 있었다. 첫 번째와 두 번째 모델에서는 검증 성능이 상대적으로 낮고 학습 정확도와의 차이가 나타나면서 과적합의 징후가 보였다. 반면, 다섯 번째와 여섯 번째 모델에서는 학습 정확도와 검증 정확도 사이의 차이가 거의 없어, 과적합이 줄어들고 모델의 일반화 능력이 개선된 것을 알 수 있었다.
# 
# * **Adam 옵티마이저와 학습률 스케줄러:** 첫 번째 모델에서는 Adam 옵티마이저와 CosineAnnealingLR 학습률 스케줄러를 사용하여 학습이 빠르게 진행되었고, 초기에는 높은 학습률로 빠르게 학습한 후 후반부에는 낮은 학습률로 안정적인 학습을 유도했지만 과적합의 징후가 보이며 안정적이지 못한 모습이 관찰되었다. 두 번째와 세 번째 모델에서는 SGD 옵티마이저를 사용하면서도 ReduceLROnPlateau를 활용하여 학습률을 동적으로 감소시키고자 하였다. 네 번째 모델부터는 SAM(Sharpness-Aware Minimization)과 AutoAugment를 적용하여 test 데이터셋에서 더욱 향상된 일반화 성능을 달성할 수 있었다.
# 
# * **데이터 증강 및 모델 구조 변경:** 앞선 네 가지 모델들에서는 데이터 증강 기법으로 Random Crop과 Random Horizontal Flip 기법만 적용하여 학습하였다. 다섯 번째 모델부터는 여기에 AutoAugment를 적용하였다. 또한 다섯 번째와 여섯 번째 모델에서는 기본 ResNet의 마지막 Fully Connected 층의 구조를 변경하여 Hidden Layer를 확장하고 Dropout을 사용해 과적합을 방지하고자 하여 모델의 일반화 성능을 높이고자 하였다.
# 
# * **결과 분석:** 여섯 번째 모델에서 최종 테스트 정확도는 89.64%로, 이전 모델들과 비교할때 가장 높은 일반화 성능 수치를 관찰하였다. 이 모델의 경우 SAM 옵티마이저와 AutoAugment를 사용하여 모델이 다양한 데이터에 대해 더욱 일반화된 성능을 보일 수 있었고, AdamW 옵티마이저와 CosineAnnealingLR을 통해 학습을 더 효율적으로 진행할 수 있었다. 또한, Dropout의 수치를 0.3으로 설정하여, 모델의 뉴런들이 지나치게 적어지지 않도록 하였다.

# -----
# 
# #### Lab을 마무리 짓기 전 저장된 checkpoint를 모두 지워 저장공간을 확보한다

# In[ ]:


import shutil, os
if os.path.exists('checkpoints/'):
    shutil.rmtree('checkpoints/')

