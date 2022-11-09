import torch.utils.data
from torch import Tensor

from .transforms import create_transform


def create_loader(dataset, input_size, mean: Tensor, std: Tensor, batch_size: int = 128, is_training: bool = False,
                  no_aug: bool = False, hflip: float = 0.5, vflip: float = 0.0,
                  crop_pct: float = 0.0) -> torch.utils.data.DataLoader:
    dataset.transform = create_transform(input_size=input_size, mean=mean, std=std, is_training=is_training,
                                         no_aug=no_aug, hflip=hflip, vflip=vflip, crop_pct=crop_pct)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
    return loader
