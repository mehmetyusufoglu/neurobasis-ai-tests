"""ResNet model factory helpers for benchmarking.

Purpose:
    Provide a thin wrapper around torchvision ResNet variants (18/34/50) with an
    adjustable classification head (`num_classes`). Used by benchmarks to swap
    backbone depth or enable pretrained weights when desired.

Key behavior:
    * Loads specified architecture via torchvision.
    * If `pretrained=True`, uses the default torchvision weights (ImageNet) and
        then replaces the final fully-connected layer to match `num_classes`.
    * Returns a standard `nn.Module` ready for `.to(device)` and `.eval()`.

Example:
    from models.resnet import get_resnet
    model = get_resnet('resnet18', pretrained=False, num_classes=10)
    # For inference: model.eval()

Note:
    Changing `num_classes` after loading pretrained weights discards the original
    classifier weights; fine-tune if you want transfer learning performance.

---------------------------------------------------------------------
ResNet18 (CIFAR-10 3×32×32 input) – Layer-by-Layer
Input: 3x32x32
    Conv1: 3 -> 64, 7x7, stride 2, pad 3        => 64x16x16
    MaxPool: 3x3, stride 2, pad 1               => 64x8x8
    Layer1: 2 BasicBlocks (each 2×(3x3,64))     => 64x8x8
    Layer2: BasicBlock (stride2 64->128) + block=> 128x4x4
    Layer3: BasicBlock (stride2 128->256)+block => 256x2x2
    Layer4: BasicBlock (stride2 256->512)+block => 512x1x1
    Global AvgPool (1x1)                        => 512
    FC: 512 -> num_classes (e.g. 10 CIFAR-10)
Activation: ReLU after each conv; BatchNorm after each conv; residual add then ReLU.
Projection (1x1 conv) used in first block of layers 2–4 for stride/channel change.

SHAPES Summary (CIFAR-10 path)
    32 ->16 (Conv1) ->8 (Pool) ->8 (Layer1) ->4 (Layer2) ->2 (Layer3) ->1 (Layer4) -> FC
    Channels: 3 ->64 ->64 ->64 ->128 ->256 ->512 ->512 -> num_classes

ImageNet (224x224) variant (reference only): 224 ->112 ->56 ->56 ->28 ->14 ->7 ->1.

What is a BasicBlock?

A BasicBlock is a building block of the ResNet architecture, consisting of two or more convolutional layers with a skip connection (or shortcut) that bypasses one or more layers. This design helps to mitigate the vanishing gradient problem in deep networks by allowing gradients to flow more easily through the network during training.

No-downsample BasicBlock:

Save input (skip)
Conv1
BatchNorm1
ReLU1
Conv2
BatchNorm2
Add skip
ReLU2
Downsample BasicBlock (stride2 or channel change):

Save input
Conv1 (stride2)
BatchNorm1
ReLU1
Conv2
BatchNorm2
Skip projection (1x1 conv + BN) (two micro-steps; often counted as one “projection” step)
Add
ReLU2

---------------------------------------------------------------------
"""
from torchvision import models
from torch import nn
from typing import Literal

def get_resnet(name: Literal['resnet18','resnet34','resnet50']='resnet18', pretrained: bool=False, num_classes: int=10) -> nn.Module:
    """Return a torchvision ResNet with adjustable num_classes."""
    fn = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50
    }[name]
    model = fn(weights='DEFAULT' if pretrained else None)
    # Replace classifier
    if hasattr(model, 'fc'):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    return model
