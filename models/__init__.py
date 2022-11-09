from .convmixer import ConvMixer256_8
from .resnet import ResNet10, ResNeXt10_32_2d

"""
    [key] = cmd line arg: [value] = func
"""
model_registry = {
    "resnet10": ResNet10,
    "resnext10_32_2d": ResNeXt10_32_2d,
    "convmixer256_8": ConvMixer256_8
}
