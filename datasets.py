# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import os
from typing import *

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# list of all datasets
DATASETS = ["cifar10", "mnist"]


class MNIST_IND(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(MNIST_IND, self).__getitem__(index)
        return img, target, index


def get_dataset(dataset: str, split: str, root: str, 
                normalize: bool = False) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "cifar10":
        return _cifar10(split, root, normalize)
    elif dataset == "mnist":
        return _mnist(split, root)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return torch.nn.Identity()


# _IMAGENET_MEAN = [0.485, 0.456, 0.406]
# _IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _mnist(split: str, root: str) -> Dataset:
    if split == "train":
        return datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST(root, train=False, transform=transforms.ToTensor())
    if split == "train_i":
        return MNIST_IND(root, train=True, download=True, transform=transforms.ToTensor())


def _cifar10(split: str, root: str, normalize: bool) -> Dataset:
    transforms_lst = [transforms.ToTensor()]
    if normalize:
        transforms_lst.append(transforms.Normalize(_CIFAR10_MEAN, 
                                                   _CIFAR10_STDDEV))
    if split == 'train':
        transforms_lst = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ] + transforms_lst
    
    return datasets.CIFAR10(root, train=(split == 'train'), download=True, 
                            transform=transforms.Compose(transforms_lst))


# def _imagenet(split: str, root: str, normalize: bool) -> Dataset:
#     dir = root
#     if split == "train":
#         subdir = os.path.join(dir, "train")
#         transforms_lst = [
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()
#         ]
#     elif split == "test":
#         subdir = os.path.join(dir, "val")
#         transforms_lst = [
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor()
#         ]
    
#     if normalize:
#         transforms_lst.append(transforms.Normalize(_IMAGENET_MEAN, 
#                                                    _IMAGENET_STDDEV))
    
#     return datasets.ImageFolder(subdir, transforms.Compose(transforms_lst))


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
