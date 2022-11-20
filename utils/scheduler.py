"""Learning rate schedulers.

Currently available schedulers:
    * CosineAnnealingWarmRestarts
    * ReduceLROnPlateau
    * OneCycleLR
"""

import torch.optim


def create_scheduler(optimizer: torch.optim.Optimizer, lr: float, sched: str = 'cosine_warm', num_epochs: int = 300,
                     steps_per_epoch: int = 10, min_lr: float = 0.0, T_0: int = 200, T_mult: int = 1,
                     plateau_mode: str = 'min', patience: int = 10):
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
        # lr_scheduler = oneCycleLR(num_epochs, lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs,
                                                           steps_per_epoch=steps_per_epoch)
    return lr_scheduler
