from typing import Callable

import numpy as np
import torch.optim

"""
    Wrapper for scheduler so that it has a step_update funciton.
"""


def oneCycleLR(num_epochs: int, lr: float) -> Callable:
    return lambda t: np.interp([t], [0, num_epochs * 2 // 5, num_epochs * 4 // 5, num_epochs],
                               [0, lr, lr / 20.0, 0])[0]


def create_scheduler(optimizer: torch.optim.Optimizer, lr: float, sched: str = 'cosine_warm', num_epochs: int = 300,
                     min_lr: float = 0.0, T_0: int = 200, T_mult: int = 1, plateau_mode: str = 'min',
                     patience: int = 10):
    lr_scheduler = None
    if sched == 'cosine_warm':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                            T_0=T_0,
                                                                            T_mult=T_mult,
                                                                            eta_min=min_lr)
    elif sched == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=plateau_mode,
                                                                  patience=patience)
    elif sched == 'onecycle':
        lr_scheduler = oneCycleLR(num_epochs, lr)
    return lr_scheduler
