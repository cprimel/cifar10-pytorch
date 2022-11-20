"""Optimizers.

Currently available optimizers:
    * SGD
    * Adam
    * AdamW
"""

import torch


def create_optimizer(params, opt_name: str = "sgd", lr: float = 0.01, weight_decay: float = 0.0,
                     eps: float = 0.0) -> torch.optim.Optimizer:
    """Creates optimizer.

    Returns:
         Optimizer: a learning optimizer, subclassing torch.optim.Optimizer
    """
    optimizer = None
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(params=params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(params=params, lr=lr, eps=eps,
                                     weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    return optimizer
