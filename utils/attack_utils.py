import foolbox as fb
from tensorboardX import SummaryWriter
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, Any
from argparse import Namespace
import time
import os

from .train_utils import AverageMeter
from datasets import get_dataset, get_num_classes
from architectures import load_ckpt

ATTACKERS = ['FastGrad', 'PGD', 'AdamPGD', 'CW', 'DDN', 'DeepFool']
NORMS = ['L1', 'L2', 'Linf']
TARGETS_GENERATORS = ['random', 'paired', 'fixed']

def random_targets(labels: torch.Tensor, nclass: int = 10) -> torch.Tensor:
    targets = []
    for i in range(labels.size(0)):
        y = labels[i].item()
        t = list(range(nclass)).remove(y)

        idx = torch.randint(0, nclass - 1)
        targets.append(t[idx])
    return torch.tensor(targets, device=labels.device)

def paired_targets(labels: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return targets[labels]

def fixed_targets(labels: torch.Tensor, nclass: int = 10) -> torch.Tensor:
    return (labels + 2024) * 3 // 7 % nclass

def prologue(args: Namespace):
    args.log_dir = os.path.join(args.log_dir, 'attack_eval')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(args.log_dir)

    ckpt = torch.load(args.model_path)
    dataset = get_dataset(args.dataset, 'test', args.data_dir, not ckpt['normalize'])
    loader = DataLoader(dataset, args.batch, shuffle=False, num_workers=args.workers)

    model = load_ckpt(ckpt, args.dataset)
    attack = get_attack(args.attack, args.norm, args)

    device = torch.device('cuda') if (args.use_gpu) and torch.cuda.is_available() \
        else torch.device('cpu')
    
    nclass = get_num_classes(args.dataset)
    if args.targets_gen == 'random':
        targets_gen = lambda labels: random_targets(labels, nclass)
    elif args.targets_gen == 'paired':
        targets = torch.tensor(args.targets, device=device)
        targets_gen = lambda labels: paired_targets(labels, targets)
    elif args.targets_gen == 'fixed':
        targets_gen = lambda labels: fixed_targets(labels, nclass)
    else:
        targets_gen = None

    return loader, model, attack, targets_gen, device, writer


def attack_call(model: nn.Module, 
                attack: fb.attacks.Attack, 
                loader: DataLoader, 
                epsilons: float | list[float],
                bounds: tuple[float],
                dist_p: int = 2,
                save_freq: int = 20,
                print_freq: int = 10,
                preprocessing: Any = None, 
                device: torch.device = torch.device('cuda'),
                targets_gen: Callable = None,
                writer: SummaryWriter = None
                ) -> pd.DataFrame:
    
    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device, preprocessing=preprocessing)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    if isinstance(epsilons, float):
        epsilons = [epsilons]
    else:
        successes = [AverageMeter() for _ in range(len(epsilons))]
        distances = [AverageMeter() for _ in range(len(epsilons))]
    
    end = time.time()

    for idx, (inputs, labels) in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        data_time.update(time.time() - end)

        if targets_gen is None:
            # untargeted attack
            criterion = fb.criteria.Misclassification(labels)

            # only consider those inputs that were originally correct
            logits = fmodel(inputs)
            preds = torch.argmax(logits, -1)
            is_correct = (preds == labels)
            inputs, labels = inputs[is_correct], labels[is_correct]
        else:
            criterion = fb.criteria.TargetedMisclassification(
                target_classes=targets_gen(labels)
            )
        
        raw, clipped, is_adv = attack(fmodel, inputs, criterion=criterion,
                                      epsilons=epsilons)        

        for i in range(len(epsilons)):
            mask = is_adv[i]
            successes[i].update(mask.mean().item(), inputs.size(0))
            distance = (raw[i][mask] - inputs[i][mask]).flatten(1).norm(dist_p, -1).item()
            distances[i].update(distance, inputs.size(0))

        if idx % save_freq == 0 and writer:
            for i, imgs in enumerate(clipped):
                writer.add_images(f'clipped@{epsilons[i]}', imgs.detach().cpu())               
        
        if idx % print_freq == 0 or idx == len(loader - 1):            
            info = f'Attack {idx}/{len(loader - 1)}\t'
            f'Time {batch_time.avg:.3f}\t'
            f'Data {data_time.avg:.3f}\t'

            print_step_size = max(len(epsilons) // 4, 1)
            _epsilons = epsilons[:print_step_size * 4:print_step_size]
            for i, eps in enumerate(_epsilons):
                idx = i * print_step_size
                info += f'Success@{eps} {successes[idx].avg:.3f} ({successes[idx].sum}/{successes[idx].count})\t'
            for i, eps in enumerate(_epsilons):
                info += f'L{dist_p}@{eps} {distances[idx].avg:.3f}\t'
            print(info)

        
        batch_time.update(time.time() - end)
        end = time.time()
    
    df = pd.DataFrame({
        'epsilons': epsilons,
        'successes': [s.avg for s in successes],
        'distances': [d.avg for d in distances]
    })
    return df
        

def get_attack(approach: str, norm: str, args: Namespace) -> fb.attacks.Attack:
    if approach == 'FastGrad':
        if norm == 'L2':
            return fb.attacks.FGM(random_start=args.random_start)
        elif norm == 'Linf':
            return fb.attacks.FGSM(random_start=args.random_start)
        
    elif approach == 'PGD':
        params = {
            'abs_stepsize': args.alpha,
            'steps': args.steps,
            'random_start': args.random_start
        }

        if norm == 'L1':
            return fb.attacks.L1PGD(**params)
        elif norm == 'L2':
            return fb.attacks.L2PGD(**params)
        elif norm == 'Linf':
            return fb.attacks.LinfPGD(**params)
        
    elif approach == 'AdamPGD':
        params = {
            'abs_stepsize': args.alpha,
            'steps': args.steps,
            'random_start': args.random_start,
            'adam_beta1': args.adam_betas[0],
            'adam_beta2': args.adam_betas[1],
            'adam_epsilon': args.adam_eps
        }

        if norm == 'L1':
            return fb.attacks.L1AdamPGD(**params)
        elif norm == 'L2':
            return fb.attacks.L2AdamPGD(**params)
        elif norm == 'Linf':
            return fb.attacks.LinfAdamPGD(**params)
    
    elif approach == 'CW':
        if norm == 'L2':
            return fb.attacks.carlini_wagner.L2CarliniWagnerAttack(
                binary_search_steps=args.binary_steps,
                steps=args.steps,
                stepsize=args.alpha,
                confidence=args.conf,
                initial_const=args.init_const,
                abort_early=args.abort_early
            )
    
    elif approach =='DDN':
        if norm == 'L2':
            return fb.attacks.ddn.DDNAttack(
                init_epsilon=args.init_eps,
                steps=args.steps,
                gamma=args.gamma
            )
    
    elif approach == 'DeepFool':
        params = {
            'steps': args.steps,
            'candidates': args.candidates,
            'overshoot': args.overshoot,
            'loss': args.loss
        }
        if norm == 'L2':
            return fb.attacks.L2DeepFoolAttack(**params)
        elif norm == 'Linf':
            return fb.attacks.LinfDeepFoolAttack(**params)
    
    raise ValueError(f'<{norm} norm, {approach} attack> is not supported.')

def norm2p(norm: str) -> float:
    if norm == 'L1':
        return 1
    elif norm == 'L2':
        return 2
    elif norm == 'Linf':
        return torch.inf
    else:
        raise NotImplementedError(f'{norm} norm is not supported.')