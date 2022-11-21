"""Model definitions for ResNet/S for CIFAR-10 as described in [1].

To make all models accessible to the training script, add the model identification and model definition to
`model_registry` in `models/__init__.py`.

Reference:
[1] He et al (2015), Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import Callable, Type, Union, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PaddedResidual(nn.Module):
    def __init__(self, lambda_fn: Callable):
        super().__init__()
        self.lambda_fn = lambda_fn
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.lambda_fn(x)
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, option: str = 'A'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                """
                He et al. (2015) use option A for CIFAR-10 ResNet.
                """
                self.shortcut = PaddedResidual(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels // 4, out_channels // 4), "constant",
                                    0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * out_channels),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return out


class ResNetS(nn.Module):
    in_channels = 16

    def __init__(self, block: Type[Union[BasicBlock]], layers: List[int], num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels: int, blocks: int, stride) -> nn.Sequential:
        strides = [stride] + [1]*(blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNetS20():
    return ResNetS(BasicBlock, [3, 3, 3])


def ResNetS38():
    return ResNetS(BasicBlock, [6, 6, 6])
