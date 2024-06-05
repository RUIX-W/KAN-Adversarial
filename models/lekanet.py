import torch
import torch.nn as nn
import torch.nn.functional as F

from kans import mlp_kan, mlp_fastkan, mlp_kacn, mlp_kagn, mlp_kaln
from kan_convs import (
    KANConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer, KALNConv2DLayer,
    FastKANConv2DLayer
)
from regularization import L1

from typing import Optional, Callable, List

def kan_conv5x5(in_planes: int, out_planes: int, spline_order: int = 3, groups: int = 1, stride: int = 1,
                dilation: int = 1, grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                grid_range: List = [-1, 1], l1_decay: float = 0.0, dropout: float = 0.0) -> KANConv2DLayer:
    """5x5 convolution with padding"""

    conv = KANConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=5,
        spline_order=spline_order,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def kaln_conv5x5(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                 l1_decay: float = 0.0) -> KALNConv2DLayer:
    """5x5 convolution with padding"""
    conv = KALNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=5,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def kagn_conv5x5(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                 l1_decay: float = 0.0) -> KAGNConv2DLayer:
    """5x5 convolution with padding"""
    conv = KAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=5,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def kacn_conv5x5(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                 dilation: int = 1, l1_decay: float = 0.0, dropout: float = 0.0) -> KACNConv2DLayer:
    """5x5 convolution with padding"""
    conv = KACNConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=5,
        degree=degree,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def fast_kan_conv5x5(in_planes: int, out_planes: int, groups: int = 1, stride: int = 1,
                     dilation: int = 1, grid_size=8, base_activation=nn.SiLU,
                     grid_range=[-2, 2], l1_decay: float = 0.0, dropout: float = 0.0) -> FastKANConv2DLayer:
    """5x5 convolution with padding"""
    conv = FastKANConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

class LeKANet(nn.Module):
    def __init__(self, num_classes: int = 10, spline_order: int = 3, grid_size: int = 5,
                 l1_decay: float = 5e-5):
        super(LeKANet, self).__init__()
        self.num_classes = num_classes
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.l1_decay = l1_decay

        self.conv1 = kan_conv5x5(1, 20, spline_order=spline_order,
                                 grid_size=grid_size, l1_decay=l1_decay)
        self.conv2 = kan_conv5x5(20, 50, spline_order=spline_order,
                                 grid_size=grid_size, l1_decay=l1_decay)
        
        self.fc = mlp_kan([5 * 5 * 50, 500, num_classes], spline_order=spline_order,
                          grid_size=grid_size, l1_decay=l1_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Fast_LeKANet(nn.Module):
    def __init__(self, num_classes: int = 10, grid_size: int = 5, l1_decay: float = 5e-5):
        super(Fast_LeKANet, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.l1_decay = l1_decay

        self.conv1 = fast_kan_conv5x5(1, 20, grid_size=grid_size)
        self.conv2 = fast_kan_conv5x5(20, 50, grid_size=grid_size)
        
        self.fc = mlp_fastkan([4 * 4 * 50, 500, num_classes], grid_size=grid_size,
                              l1_decay=l1_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc(x)
        return x
    
class LeKALNet(nn.Module):
    def __init__(self, num_classes: int = 10, degree: int = 3, l1_decay: float = 5e-5):
        super(LeKALNet, self).__init__()
        self.num_classes = num_classes
        self.degree = degree
        self.l1_decay = l1_decay

        self.conv1 = kaln_conv5x5(1, 20, degree=degree, l1_decay=l1_decay)
        self.conv2 = kaln_conv5x5(20, 50, degree=degree, l1_decay=l1_decay)
        
        self.fc = mlp_kaln([4 * 4 * 50, 500, num_classes], degree=degree, 
                           l1_decay=l1_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc(x)
        return x

class LeKAGNet(nn.Module):
    def __init__(self, num_classes: int = 10, degree: int = 3, l1_decay: float = 5e-5):
        super(LeKAGNet, self).__init__()
        self.num_classes = num_classes
        self.degree = degree
        self.l1_decay = l1_decay

        self.conv1 = kagn_conv5x5(1, 20, degree=degree, l1_decay=l1_decay)
        self.conv2 = kagn_conv5x5(20, 50, degree=degree, l1_decay=l1_decay)
        
        self.fc = mlp_kagn([4 * 4 * 50, 500, num_classes], degree=degree, 
                           l1_decay=l1_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc(x)
        return x
    
class LeKACNet(nn.Module):
    def __init__(self, num_classes: int = 10, degree: int = 3, l1_decay: float = 5e-5):
        super(LeKACNet, self).__init__()
        self.num_classes = num_classes
        self.degree = degree
        self.l1_decay = l1_decay

        self.conv1 = kacn_conv5x5(1, 20, degree=degree, l1_decay=l1_decay)
        self.conv2 = kacn_conv5x5(20, 50, degree=degree, l1_decay=l1_decay)
        
        self.fc = mlp_kacn([4 * 4 * 50, 500, num_classes], degree=degree, 
                           l1_decay=l1_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc(x)
        return x