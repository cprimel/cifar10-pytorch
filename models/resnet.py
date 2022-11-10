from typing import Union, Type, List

import torch
import torch.nn as nn
from torch import Tensor

"""
    See https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html.
"""


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=groups,
                     bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, base_width: int = 64,
                 padding: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride) # TODO: define directly rather than by function
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        """
            Inclusion of ReLU after addition has a minor negative effect on test performance. 
            
            See http://torch.ch/blog/2016/02/04/resnets.html
        """
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, base_width: int = 64, groups: int = 1,
                 padding: int = 1):
        super().__init__()
        width = int(out_channels * base_width / 64.0) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, padding)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels * self.expansion, stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        """
            Inclusion of ReLU after addition in a standard ResBlock has a minor negative effect on test performance. 

            TODO: Test if above is also true for Bottleneck block.
            
            See http://torch.ch/blog/2016/02/04/resnets.html
        """
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
        TODO: Weight initialization
        TODO: Dilation - not necessary but might be useful for iterating over different architectures?
    """

    def __init__(self, block: Type[Union[ResBlock, Bottleneck]], layers: List[int], num_classes: int = 10,
                 groups: int = 1, width_per_group: int = 64):
        super().__init__()
        assert len(layers) == 4, f"layers requires a list of 4 integers, got {len(layers)}"

        self.in_channels = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[ResBlock, Bottleneck]], out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.base_width, self.groups))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride, base_width=self.base_width, groups=self.groups))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # def forward(self, x: Tensor) -> Tensor:
    #     return self._forward_impl(x)


def ResNet18():
    return ResNet(ResBlock, [2, 2, 2, 2])


def ResNet10():
    return ResNet(ResBlock, [1, 1, 1, 1])


def ResNeXt10_32_2d():
    return ResNet(Bottleneck, [1, 1, 1, 1], groups=32, width_per_group=2)


def ResNet26_2_32d():
    return ResNet(Bottleneck, [1, 1, 1, 1], groups=1, width_per_group=128)
