# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import (
   resnet18, resnet50, resnet101
)
import torch.backends.cudnn as cudnn

from datasets import get_normalize_layer
from models import LeNet, LeKANet, reskanet_18x32p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lenet - the classic LeNet for MNIST
# resnet18 - the classic ResNet-18, sized for ImageNet
# resnet50 - the classic ResNet-50, sized for ImageNet
# resnet101 - the classic ResNet-101, sized for ImageNet
ARCHITECTURES = ["lenet", "resnet18", "resnet50", "resnet101"]

def get_architecture(model_name: str, dataset: str, normalize: bool) -> nn.Module:
    """ Return a neural network (with random weights)

    :param model_name: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset not in ['mnist', 'cifar10']:
       raise NotImplementedError(f'{dataset} dataset is not supported.')
       
    if model_name == 'lenet':
         model = LeNet()
    elif model_name == 'resnet18':
       model = resnet18(pretrained=False)
       model.fc = nn.Linear(512, 10)
    elif model_name == 'resnet50':
       model = resnet50(pretrained=False)
       model.fc = nn.Linear(2048, 10)
    elif model_name == 'resnet101':
       model = resnet101(pretrained=False)
       model.fc = nn.Linear(2048, 10)
    else:
         raise NotImplementedError(f'{model_name} model is not supported.')
    
    if normalize:
       model = nn.Sequential(get_normalize_layer(dataset), model)
    
    return model.to(device)


def get_kan_architecture(model_name: str, dataset: str, normalize: bool,
                         spline_order: int = 3, grid_size: int = 5,
                         l1_decay: float = 5e-5) -> nn.Module:
    """ Return a neural network (with random weights)

    :param model_name: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """

    if model_name == 'lenet':
       assert dataset == 'mnist'
       model = LeKANet(spline_order=spline_order, grid_size=grid_size,
                       l1_decay=l1_decay)
    elif model_name == 'resnet18':
       model = reskanet_18x32p(3, 10, spline_order=spline_order,
                               grid_size=grid_size, l1_decay=l1_decay)
    else:
       raise NotImplementedError(f'{model_name} model is not supported.')
    
    if normalize:
       model = nn.Sequential(get_normalize_layer(dataset), model)
    
    return model.to(device)


def load_ckpt(ckpt: dict, dataset: str) -> nn.Module:
   if ckpt['kan']:
      model = get_kan_architecture(ckpt['model_name'], dataset,
                                   ckpt['normalize'], ckpt['spline_order'],
                                   ckpt['grid_size'], ckpt['l1_decay'])
   else:
      model = get_architecture(ckpt['name'], dataset,
                              ckpt['normalize'])
   model.load_state_dict(ckpt['state_dict'])
   return model