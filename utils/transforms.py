from typing import Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def create_transform(input_size, mean: Tuple[float, float, float], std: Tuple[float, float, float],
                     is_training: bool = False, no_aug: bool = False, hflip: float = 0.5, vflip: float = 0.0,
                     crop_pct: float = 0.0, rand_aug: bool = True, jitter: float = 0.1, scale: float = 0.9,
                     prob_erase: float = 0.0):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    t = []
    if is_training and no_aug:
        t += [transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR), transforms.CenterCrop(img_size)]
    elif is_training:
        t = [transforms.RandomResizedCrop(img_size, scale=(scale, 1.0), ratio=(1.0, 1.0))]
        # TODO: Add RandomRotation default 30?
        if hflip > 0.0:
            t += [transforms.RandomHorizontalFlip(p=hflip)]
        if vflip > 0.0:
            t += [transforms.RandomVerticalFlip(p=vflip)]
        if rand_aug:
            t += [transforms.RandAugment(num_ops=2, magnitude=12)]
        if jitter > 0.0:
            t += [transforms.ColorJitter(jitter, jitter, jitter)]
    t += [transforms.ToTensor(), transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std))]
    if is_training and not no_aug:
        t += [transforms.RandomErasing(p=prob_erase)]

    return transforms.Compose(t)
