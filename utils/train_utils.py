import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from tensorboardX import SummaryWriter
import pandas as pd

from typing import Generator, Any
from argparse import Namespace, ArgumentParser
import time
import os

from datasets import get_dataset
from architectures import (
    get_architecture, get_kan_architecture
)

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


def prologue(args: Namespace):
    args.log_dir = os.path.join(args.log_dir, 'train')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    args.model_dir = model_filepath(args.model_dir, args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    writer = SummaryWriter(args.log_dir)

    trainset = get_dataset(args.dataset, 'train', args.data_dir, not args.normalize)
    testset = get_dataset(args.dataset, 'test', args.data_dir, not args.normalize)
    trainloader = DataLoader(trainset, args.batch, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(testset, args.test_batch, shuffle=False, num_workers=args.workers)

    if args.kan:
        model = get_kan_architecture(args.arch, args.dataset, args.normalize,
                                     args.spline_order, args.grid_size,
                                     args.l1_decay)
    else:
        model = get_architecture(args.arch, args.dataset, args.normalize)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, 
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr, args.betas, args.eps, 
                               weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'{args.optimizer} optimizer not supported.')
    
    if args.scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    elif args.scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min)
    else:
        raise NotImplementedError(f'{args.scheduler} scheduler is not supported.')
    
    starting_epoch = 0
    if args.resume_path:
        ckpt = torch.load(args.resume_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])

        starting_epoch = ckpt['epoch']
    
    device = torch.device('cuda') if (args.use_gpu) and torch.cuda.is_available() \
        else torch.device('cpu')
    criterion = nn.CrossEntropyLoss().to(device)

    return trainloader, testloader, writer, model, optimizer, lr_scheduler, starting_epoch,\
          device, criterion


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

def save_model(model: nn.Module, model_name: str, normalize: bool, kan: bool,
               optimizer: optim.Optimizer, lr_scheduler: _LRScheduler, 
               epoch: int, path: str) -> None:
    info = {
        'name': model_name,
        'normalize': normalize,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'time': time.strftime("%Y-%m-%d %H:%M:%S", \
                              time.gmtime(time.time()))
    }
    if kan:
        info['spline_order'] = model.spline_order
        info['grid_size'] = model.grid_size
        info['l1_decay'] = model.l1_decay
    
    path = os.path.join(path, f'checkpoint@{epoch}.pth')
    print(f'Saving model at epoch {epoch} to {path} ...')
    torch.save(info, path)

def construct_fpath(base_dir: str, keys: list[str], args: Namespace) -> str:
    s = base_dir
    for key in keys:
        value = getattr(args, key, None)
        if isinstance(value, bool):
            name = ('' if value else 'non-') + value
        else:
            name = f'{key}={value}'
        s = os.path.join(s, name)
    return s


def model_filepath(base_dir: str, args: Namespace) -> str:
    keys = ['dataset', 'arch', 'normliaze', 'kan']
    if args.kan:
        keys.extend(['spline_order', 'grid_size', 'l1_decay'])
    
    keys.extend(['epochs', 'lr', 'weight-_decay', 'optimizer'])
    if args.optimizer == 'SGD':
        keys.append('momentum')
    else:
        keys.extend(['betas', 'adam_eps'])
    
    keys.append('scheduler')
    if args.scheduler == 'step':
        keys.extend(['lr_step_size', 'lr_gamma'])
    else:
        keys.append(['lr_min'])
    
    return construct_fpath(base_dir, keys, args)



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


def train(trainloader: DataLoader, testloader: DataLoader, model: nn.Module, criterion, 
          optimizer: optim.Optimizer, lr_scheduler: _LRScheduler, epochs: int,
          device: torch.device, writer: SummaryWriter = None, print_freq: int = 10, 
          test_freq: int = 1, test_print_freq: int = 20, target_acc: float = None, 
          starting_epoch: int = 0, save_params: dict[str, Any] = None) -> pd.DataFrame:
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best = 0
    for epoch in range(starting_epoch, epochs):
        loss, acc = train_epoch(trainloader, model, criterion, optimizer, epoch, device, 
                                writer, print_freq)
        train_losses.append(loss)
        train_accs.append(acc)

        lr_scheduler.step()
        writer.add_scalar('learning-rate', lr_scheduler.get_last_lr())

        if epoch % test_freq == 0 or epoch == epochs - 1:
            loss, acc = test(testloader, model, criterion, epoch, device, writer, 
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