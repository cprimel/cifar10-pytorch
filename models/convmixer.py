from typing import Callable

import torch.nn as nn
from torch import Tensor

"""
    See https://github.com/locuslab/convmixer-cifar10/blob/main/train.py#L36
"""


class Residual(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


def ConvMixer(dim: int = 256, depth: int = 8, kernel_size: int = 9, patch_size: int = 1,
              n_classes: int = 10) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(Residual(nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


def ConvMixer256_8_k5():
    return ConvMixer(depth=8, kernel_size=5)


def ConvMixer256_8_k5_p2():
    return ConvMixer(depth=8, kernel_size=5, patch_size=2)


def ConvMixer256_8_k9():
    return ConvMixer(depth=8, kernel_size=9)


def ConvMixer256_16_k9():
    return ConvMixer(depth=16, kernel_size=9)
