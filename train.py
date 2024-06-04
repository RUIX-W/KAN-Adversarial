from argparse import ArgumentParser
import os

from architectures import ARCHITECTURES
from datasets import DATASETS
from utils import train_prologue, train, model_filepath

def get_parser() -> ArgumentParser:

    parser = ArgumentParser(
        description='PyTorch KAN Adversarial, Training'
    )

    #############################################
    # Positional arguments
    parser.add_argument('dataset', type=str, choices=DATASETS,
                        help='The dataset used in this experiment')
    parser.add_argument('arch', type=str, choices=ARCHITECTURES,
                        help='Architecture of the model used in the experiment')
    

    #############################################
    # Data configurations
    parser.add_argument('--data-dir', type=str, default='~/DATA/',
                        help='Path to the dataset directory')
    

    #############################################
    # Model configurations
    parser.add_argument('--normalize', action='store_true',
                        help='whether to normalize input in model.')
    
    # KAN arguments
    parser.add_argument('--kan', action='store_true', help='Whether to use KAN.')
    parser.add_argument('--spline-order', type=int, default=3,
                        help='Spline order in KAN.')
    parser.add_argument('--grid-size', type=int, default=5,
                        help='Grid size in KAN.')
    parser.add_argument('--l1-decay', type=float, default=5e-5,
                        help='L1 weight decay of KAN.')
    

    #############################################
    # Training configurations
    parser.add_argument('--epochs', type=int, default=90, help='Training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD',
                        help='Optimizer choice during training.')
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine'], default='cosine',
                        help='Learning rate scheduler choice during training.')
    parser.add_argument('--batch', type=int, default=64, help='Batch size during training.')
    parser.add_argument('--use-gpu', action='store_true', help='Whether to use GPU during training.')
    
    # SGD arguments
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum in SGD optimizer.')
    
    # Adam arguments
    parser.add_argument('--betas', type=int, nargs=2, default=(0.9, 0.999),
                        help='Betas in Adam optimizer.')
    parser.add_argument('--adam-eps', type=float, default=1e-8,
                        help='Epsilon in Adam optimizer.')
    
    # Step scheduler arguments
    parser.add_argument('--lr-step-size', type=int, default=30, 
                        help='Step size for StepLR scheduler.')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma in StepLR scheduler.')
    
    # Cosine scheduler arguments
    parser.add_argument('--lr-min', type=float, default=0.0,
                        help='Minimum lr in CosineAnnealingLR scheduler.')
    
    #############################################
    # Evaluation configurations
    parser.add_argument('--test-batch', type=int, default=128,
                        help='Batch size during evaluation.')
    parser.add_argument('--test-freq', type=int, default=1,
                        help='Frequency to evaluate model on test set.')
    

    #############################################
    # Logging configurations
    parser.add_argument('--log-dir', type=str, default='./logs/',
                        help='Path to the directory that stores logging')
    parser.add_argument('--model-dir', type=str, default='~/Models/',
                        help='Path to the directory that stores model checkpoint.')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='Frequency to print in one training epoch.')
    parser.add_argument('--test-print-freq', type=int, default=10,
                        help='Frequency to print in one testing epoch.')
    
    #############################################
    # other configurations
    parser.add_argument('--resume-path', type=str, default=None,
                        help='Path to the ckechpoint from which to resume, None means not resuming.')
    parser.add_argument('--target-acc', type=float, default=1.0,
                        help='Target test accuracy to achieve.')

    return parser
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    trainloader, testloader, writer, model, optimizer, lr_scheduler, starting_epoch, \
        device, criterion = train_prologue(args)
    
    save_params = {
        'model_name': args.arch,
        'normalize': args.normalize,
        'kan': args.kan,
        'path': args.model_dir
    }
    train_df = train(trainloader, testloader, model, criterion, optimizer, lr_scheduler,
                     args.epochs, device, writer, args.print_freq, args.test_freq,
                     args.test_print_freq, args.target_acc, starting_epoch, save_params)
    
    train_df.to_csv(os.path.join(args.log_dir, 'train_stats.csv'), index=False)