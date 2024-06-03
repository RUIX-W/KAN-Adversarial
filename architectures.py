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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
  def __init__(self, num_classes=10):
    super(LeNet, self).__init__()
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, num_classes)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# lenet - the classic LeNet for MNIST
# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["lenet", "resnet18", "resnet50", "resnet101"]

def get_architecture(model_name: str, dataset: str, normalize: bool) -> nn.Module:
    """ Return a neural network (with random weights)

    :param model_name: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if model_name == 'lenet':
       assert dataset == 'mnist'
       model = LeNet()
    elif model_name == 'resnet18':
       model = resnet18(pretrained=(dataset == 'imagenet'))
       if dataset in ['mnist', 'cifar10']:
          model.fc = nn.Linear(512, 10)
    elif model_name == 'resnet50':
       model = resnet50(pretrained=(dataset == 'imagenet'))
       if dataset in ['mnist', 'cifar10']:
          model.fc = nn.Linear(2048, 10)
    elif model_name == 'resnet101':
       model = resnet101(pretrained=(dataset == 'imagenet'))
       if dataset in ['mnist', 'cifar10']:
          model.fc = nn.Linear(2048, 10)
    else:
       raise NotImplementedError(f'{model_name} model is not supported.')
    
    if normalize:
       model = nn.Sequential(get_normalize_layer(dataset), model)
    
    return model.to(device)

def load_ckpt(ckpt: dict, dataset: str) -> nn.Module:
   model = get_architecture(ckpt['name'], dataset,
                            ckpt['normalize'])
   model.load_state_dict(ckpt['state_dict'])
   return model