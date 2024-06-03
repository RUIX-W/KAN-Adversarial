import foolbox as fb
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, Any
import argparse
import time

from .train_utils import AverageMeter

def attack(model: nn.Module, 
           attack: fb.attacks.Attack, 
           loader: DataLoader, 
           epsilons: float | list[float],
           bounds: tuple[float],
           dist_p: int = 2,
           save_freq: int = 20,
           print_freq: int = 10,
           preprocessing: Any = None, 
           device: torch.device = torch.device('cuda:0'),
           targets_gen: Callable = None,
           writer: SummaryWriter = None
           ) -> tuple[int]:
    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device, preprocessing=preprocessing)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    if isinstance(epsilons, float):
        success = AverageMeter()
        distances = AverageMeter()
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
        if isinstance(epsilons, float):
            success.update(is_adv.mean().item(), inputs.size(0))
            distance = (raw[is_adv] - inputs[is_adv]).flatten(1).norm(dist_p, -1).item()
            distances.update(distance, inputs.size(0))
        else:
            for i in range(len(epsilons)):
                mask = is_adv[i]
                successes[i].update(mask.mean().item(), inputs.size(0))
                distance = (raw[i][mask] - inputs[i][mask]).flatten(1).norm(dist_p, -1).item()
                distances[i].update(distance, inputs.size(0))

        if idx % save_freq == 0 and writer:
            if isinstance(epsilons, float):
                writer.add_images(f'clipped@{epsilons}', clipped.detach().cpu())
            else:
                for i, imgs in enumerate(clipped):
                    writer.add_images(f'clipped@{epsilons[i]}', imgs.detach().cpu())
        
        if idx % print_freq == 0 or idx == len(loader - 1):
            if isinstance(epsilons, float):
                print(f'Attack {idx}/{len(loader - 1)}\t'
                      f'Time {batch_time.avg:.3f}\t'
                      f'Data {data_time.avg:.3f}\t'
                      f'Success@{epsilons} {success.avg:.3f} ({success.sum}/{success.count})\t'
                      f'L{dist_p}@{epsilons} {distances.avg:.3f}')
            else:
                info = f'Attack {idx}/{len(loader - 1)}\t'
                f'Time {batch_time.avg:.3f}\t'
                f'Data {data_time.avg:.3f}\t'
                for i, eps in enumerate(epsilons):
                    info += f'Success@{eps} {successes[i].avg:.3f} ({successes[i].sum}/{successes[i].count})\t'
                for i, eps in enumerate(epsilons):
                    info += f'L{dist_p}@{eps} {distances[i].avg:.3f}'
                print(info)

        
        batch_time.update(time.time() - end)
        end = time.time()
    
    if isinstance(epsilons, float):
        return success.avg, distances.avg
    else:
        return tuple([s.avg for s in successes] + [d.avg for d in distances])
        

def get_attacker(approach: str, norm: str, args: argparse.Namespace) -> fb.attacks.Attack:
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
            'adam_beta1': args.adam_beta1,
            'adam_beta2': args.adam_beta2,
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