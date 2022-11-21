"""Model registry.

Key:value pairs for calling models via command line arguments in the training script. Key must match the model
identifier passed via the experiment config file or command line argument. Value must match the name of the function
where the model is defined.
"""

from .convmixer import ConvMixer256_8_k5_p1, ConvMixer256_8_k9_p1, ConvMixer256_16_k9_p2, ConvMixer256_8_k5_p2, \
    ConvMixer256_8_k5_p1, ConvMixer256_8_k9_p2
from .resnet import ResNet10, ResNeXt10_32_2d
from .resnet_s import ResNetS20, ResNetS38

model_registry = {
    "resnet_s20": ResNetS20,
    "resnet_s38": ResNetS38,
    "resnet10": ResNet10,
    "resnext10_32_2d": ResNeXt10_32_2d,
    "convmixer256_8_k5_p1": ConvMixer256_8_k5_p1,
    "convmixer256_8_k5_p2": ConvMixer256_8_k5_p2,
    "convmixer256_8_k9_p1": ConvMixer256_8_k9_p1,
    "convmixer256_8_k9_p2": ConvMixer256_8_k9_p2,
    "convmixer256_16_k9_p2": ConvMixer256_16_k9_p2

}
