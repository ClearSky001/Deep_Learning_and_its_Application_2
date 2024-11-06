import os, time, shutil

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

def create_dataloaders(train_dataset, test_dataset, device, batch_size, num_worker):
    kwargs = {}
    if device.startswith("cuda"):
        kwargs.update({
            'pin_memory': True,
        })

    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_worker, **kwargs)
    
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

def calculate_pixel_accuracy(output, target):
    _, preds = torch.max(output, dim=1)
    
    correct = (preds == target).sum().item()
    total = target.numel()

    return correct / total

def train_loop(model, device, dataloader, criterion, optimizer, epoch):
    # train for one epoch
    loss_meter = AverageMeter('Loss', ':.4e')
    pixel_acc_meter = AverageMeter('Pixel_Acc', ':6.2f')
    data_time = AverageMeter('Data_Time', ':6.3f') # Time for data loading
    batch_time = AverageMeter('Batch_Time', ':6.3f') # time for mini-batch train
    metrics_list = [loss_meter, pixel_acc_meter, data_time, batch_time, ]
    
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
        pixel_acc = calculate_pixel_accuracy(output, target)
        pixel_acc_meter.update(pixel_acc, images.size(0))

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
        "Train Pixel Acc": pixel_acc_meter.avg
    })

