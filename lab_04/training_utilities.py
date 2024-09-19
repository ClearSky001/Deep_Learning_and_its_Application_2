import os, time, shutil

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import wandb

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

    wandb.log({"Train Loss": losses.avg, "Train Accuracy@1": acc_top1.avg})

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

    wandb.log({"Validation Loss": losses.avg, "Validation Accuracy@1": acc_top1.avg})

    return acc_top1.avg