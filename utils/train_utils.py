import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from tensorboardX import SummaryWriter
import pandas as pd

from typing import Generator, Any
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_acc(output: torch.Tensor, target: torch.Tensor, ks: tuple[int] = (1,)) -> torch.Tensor:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in ks:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Set whether model parameters need requires gradient"""
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def trainables(model: nn.Module) -> Generator[nn.Parameter, None, None]:
    """Generate all trainable parameters of the model"""
    for param in model.parameters():
        if param.requires_grad:
            yield param

def count_params(model: nn.Module) -> int:
    num = 0
    for param in model.parameters():
        num += param.numel()
    return num

def save_model(model: nn.Module, model_name: str, normalize: bool,
               optimizer: optim.Optimizer, lr_scheduler: _LRScheduler, 
               epoch: int, path: str) -> None:
    info = {
        'name': model_name,
        'normalize': normalize,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    print(f'Saving model at epoch {epoch} to {path} ...')
    torch.save(info, path)


def train_epoch(loader: DataLoader, model: nn.Module, criterion, optimizer: optim.Optimizer, epoch: int, 
                device: torch.device, writer: SummaryWriter = None, print_freq: int = 10):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    requires_grad(model, True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs.size(0)

        inputs, targets = inputs.to(device), targets.to(device)        
        logits = model(inputs)

        loss: torch.Tensor = criterion(logits, targets)

        acc1, acc5 = topk_acc(logits, targets, ks=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader) - 1, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, top1.avg)


def train(loader: DataLoader, model: nn.Module, criterion, optimizer: optim.Optimizer, 
                lr_scheduler: _LRScheduler, epochs: int, device: torch.device, 
                writer: SummaryWriter = None, print_freq: int = 10, test_freq: int = 1, 
                test_print_freq: int = 20, target_acc: float = None, 
                save_params: dict[str, Any] = None) -> pd.DataFrame:
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best = 0
    for epoch in range(epochs):
        loss, acc = train_epoch(loader, model, criterion, optimizer, epoch, device, 
                                writer, print_freq)
        train_losses.append(loss)
        train_accs.append(acc)

        lr_scheduler.step()
        writer.add_scalar('learning-rate', lr_scheduler.get_last_lr())

        if epoch % test_freq == 0 or epoch == epochs - 1:
            loss, acc = test(loader, model, criterion, epoch, device, writer, 
                             test_print_freq)
            test_losses.append(loss)
            test_accs.append(acc)

            if acc > best:
                # best test acc so far
                save_model(model, **save_params, optimizer=optimizer,
                           lr_scheduler=lr_scheduler, epoch=epoch)
                best = acc

            if abs(acc - target_acc) <= 1e-3:
                # reached vicinity of target accuracy
                break        
        else:
            test_losses.append(test_losses[-1])
            test_accs.append(test_accs[-1])


    df = pd.DataFrame({
        "train loss": train_losses,
        "train acc": train_accs,
        "test loss": test_losses,
        "test acc": test_accs,
    })
    return df, epoch


def test(loader: DataLoader, model: nn.Module, criterion, epoch: int, device: torch.device, 
         writer: SummaryWriter = None, print_freq: int = 10):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = topk_acc(outputs, targets, ks=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, top1.avg)