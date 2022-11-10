import torch.utils.data
from torch import Tensor

from .transforms import create_transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def create_loader(dataset, input_size, mean: Tensor, std: Tensor, batch_size: int = 128, is_training: bool = False,
                  no_aug: bool = False, hflip: float = 0.5, vflip: float = 0.0,
                  crop_pct: float = 0.0) -> torch.utils.data.DataLoader:
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = Dataset(dataset)
    dataset.transform = create_transform(input_size=input_size, mean=mean, std=std, is_training=is_training,
                                         no_aug=no_aug, hflip=hflip, vflip=vflip, crop_pct=crop_pct)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
    return loader
