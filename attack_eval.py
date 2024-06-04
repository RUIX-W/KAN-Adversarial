import torch
import torch.nn as nn

from argparse import ArgumentParser
import os

from utils import (
    attack_prologue, attack_call, norm2p, ATTACKERS, 
    NORMS, TARGETS_GENERATORS
)
from datasets import DATASETS
from architectures import get_architecture

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='PyTorch KAN Adversarial, Attack Evaluation'
    )

    #############################################
    # Positional arguments
    parser.add_argument('dataset', type=str, choices=DATASETS,
                        help='The dataset used in this experiment')
    parser.add_argument('arch-path', type=str,
                        help='Path to the archive storing trained models.')
    parser.add_argument('attack', type=str, choices=ATTACKERS,
                        help='The attack approach.')
    parser.add_argument('dist-norm', type=str, choices=NORMS,
                        help='Distance norm for attacker.')
    
    #############################################
    # Data configurations
    parser.add_argument('--data-dir', type=str, default='~/DATA/',
                        help='Path to the dataset directory')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size during attack.')
    
    #############################################
    # Attack configurations
    parser.add_argument('--targets-gen', type=str, choices=TARGETS_GENERATORS,
                        default=None, help='Approach to generate targets.')
    parser.add_argument('--epsilons', type=float, nargs='+', default=1.0,
                        help='Epsilons used to find adversarial samples.')
    parser.add_argument('--bounds', type=float, nargs=2, default=(0, 1),
                        help='Bounds of inputs for attacked models.')
    
    #############################################
    # Logging configurations
    parser.add_argument('--save-freq', type=int, default=20,
                        help='Frequency to save adversarial examples.')
    parser.add_argument('--test-freq', type=int, default=10,
                        help='Frequency to print results.')
    
    # PGD arguments (also applied to other attackers)
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='Step size in finding adversarial samples')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for attacks.')
    parser.add_argument('--random-start', action='store_true',
                        help='Whether the perturbation is initialized randomly or starts at zero.')
    
    # AdamPGD arguments
    parser.add_argument('--adam-betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Betas used in AdamPGD attack.')
    parser.add_argument('--adam-eps', type=float, default=1e-8,
                        help='Epsilon used in AdamPGD attack.')
    
    # CW arguments
    parser.add_argument('--binary-steps', type=int, default=9,
                        help='Binary step numbers to find best constant for each sample in CW attack.')
    parser.add_argument('--conf', type=float, default=0.0,
                        help='Confidence in CW attack.')
    parser.add_argument('--init-const', type=float, default=1e-3,
                        help='Initial constant in CW attack.')
    parser.add_argument('--abort-early', action='store_true',
                        help='Whther to abort early in CW attack.')
    
    # DDN arguments
    parser.add_argument('--init-eps', type=float, default=1.0,
                        help='Initial epsilon in DDN attack.')
    
    # DeepFool arguments
    parser.add_argument('--candidates', type=int, default=10,
                        help='Limit on the number of the most likely classes that should be considered in DeepFool attack.')
    parser.add_argument('--overshoot', type=float, default=2e-2,
                        help='How much to overshoot the boundary in DeepFool attack.')
    parser.add_argument('--loss', type=str, choices=['crossentropy', 'logits'], default='logits',
                        help='Loss function to use in DeepFool attack.')
    
    return parser
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    loader, model, attack, targets_gen, device, writer = attack_prologue(args)
    df = attack_call(model, attack, loader, args.epsilons, args.bounds, 
                     norm2p(args.dist_norm), args.save_freq, args.print_freq,
                     None, device, targets_gen, writer)
    df.to_csv(os.path.join(args.log_dir, 'results.csv'), index=False)