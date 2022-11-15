from .convmixer import ConvMixer256_8_k5, ConvMixer256_8_k9, ConvMixer256_16_k9, ConvMixer256_8_k5_p2
from .resnet import ResNet10, ResNeXt10_32_2d

"""
    [key] = cmd line arg: [value] = func
"""
model_registry = {
    "resnet10": ResNet10,
    "resnext10_32_2d": ResNeXt10_32_2d,
    "convmixer256_8_k5": ConvMixer256_8_k5,
    "convmixer256_8_k5_p2": ConvMixer256_8_k5_p2,
    "convmixer256_8_k9": ConvMixer256_8_k9,
    "convmixer256_16_k9": ConvMixer256_16_k9

}
