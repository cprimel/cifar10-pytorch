from typing import Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def create_transform(input_size, mean: Tuple[float, float, float], std: Tuple[float, float, float],
                     is_training: bool = False, no_aug: bool = False, hflip: float = 0.5, vflip: float = 0.0,
                     crop_pct: float = 0.0, rand_aug: bool = True, jitter: bool = True):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    t = []
    if is_training and no_aug:
        t += [transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR), transforms.CenterCrop(img_size)]
    elif is_training:
        t = [transforms.RandomResizedCrop(img_size)]
        # TODO: Add RandomRotation default 30
        if hflip > 0.0:
            t += [transforms.RandomHorizontalFlip(p=hflip)]
        if vflip > 0.0:
            t += [transforms.RandomVerticalFlip(p=vflip)]
        if rand_aug:
            t += [transforms.RandAugment(num_ops=1, magnitude=8)]
        if jitter:
            t += [transforms.ColorJitter(0.1, 0.1, 0.1)]
    t += [transforms.ToTensor(), transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std))]
    return transforms.Compose(t)
