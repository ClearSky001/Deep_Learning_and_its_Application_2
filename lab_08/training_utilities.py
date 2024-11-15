import os, time, shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42

    normalize = transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
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
    
def save_checkpoint(filepath, model, optimizer, scheduler, epoch, best_metric, is_best, best_model_path):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'epoch': epoch + 1,
        'best_metric': best_metric,
    }
    
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_model_path)


def load_checkpoint(filepath, model, optimizer, scheduler, device):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        print(f"=> loaded checkpoint '{filepath}' (epoch {start_epoch})")
        return start_epoch, best_metric
    else:
        print(f"=> no checkpoint found at '{filepath}'")
        return 0, 0

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

def train_loop(model, device, dataloader, criterion, optimizer, epoch):
    # train for one epoch
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    data_time = AverageMeter('Data_Time', ':6.3f') # Time for data loading
    batch_time = AverageMeter('Batch_Time', ':6.3f') # time for mini-batch train
    metrics_list = [loss_meter, acc1_meter, data_time, batch_time, ]
    
    model.train() # switch to train mode

    end = time.time()

    tqdm_epoch = tqdm(dataloader, desc=f'Training Epoch {epoch + 1}', total=len(dataloader))
    for images, target in tqdm_epoch:
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        loss_meter.update(loss.item(), images.size(0))
        acc1, = calculate_accuracy(output, target, topk=(1,))
        acc1_meter.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        tqdm_epoch.set_postfix(avg_metrics = ", ".join([str(x) for x in metrics_list]))

        end = time.time()
    tqdm_epoch.close()

    wandb.log({
        "epoch" : epoch,
        "Train Loss": loss_meter.avg, 
        "Train Acc": acc1_meter.avg
    })

def evaluation_loop(model, device, dataloader, criterion, epoch = 0, phase = "validation"):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    metrics_list = [loss_meter, acc1_meter]

    model.eval() # switch to evaluate mode

    with torch.no_grad():
        tqdm_val = tqdm(dataloader, desc='Validation/Test', total=len(dataloader))
        for images, target in tqdm_val:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, = calculate_accuracy(output, target, topk=(1,))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1[0], images.size(0))

            tqdm_val.set_postfix(avg_metrics = ", ".join([str(x) for x in metrics_list]))

        tqdm_val.close()

    wandb.log({
        "epoch" : epoch,
        f"{phase.capitalize()} Loss": loss_meter.avg, 
        f"{phase.capitalize()} Acc": acc1_meter.avg
    })

    return acc1_meter.avg