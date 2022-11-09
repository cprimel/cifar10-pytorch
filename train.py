import argparse
import logging
from typing import Tuple, Callable

import torch
import torch.utils.data
import torchvision as torchvision
import yaml
from torch import Tensor

import utils
from models import model_registry
from utils import create_optimizer

"""
    See https://github.com/rwightman/pytorch-image-models/blob/main/train.py
"""

_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet10', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet10")')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=0.01, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--epochs', type=int, default=50, metavar='N',
                   help='number of epochs to train (default: 50)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
        TODO: Revise model creation to take parameters as kwargs:
            
            model_registry[args.model] -> model_registry[args.model](**kwargs)
            
        Requires adding direct call to class definition, e.g., { "resnet": ResNet }   
    """

    model = model_registry[args.model]()
    model = model.to(device)
    _logger.info(f"{args.model} loaded to {device}.")
    optimizer = create_optimizer(params=model.parameters(), opt_name=args.opt, lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    validate_loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # TODO: Resume from checkpoint

    # Create train and test datasets
    ROOT = '.data'
    train_data = torchvision.datasets.CIFAR10(ROOT,
                                              train=True,
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(ROOT,
                                             train=False,
                                             download=True)

    # OPTIONAL: mixup / cutmix

    # TODO: Create dataloaders w/augmentation pipeline
    mean = train_data.data.mean(axis=(0, 1, 2)) / 255
    std = train_data.data.std(axis=(0, 1, 2)) / 255
    input_size = (3, 32, 32)

    train_loader = utils.create_loader(train_data, input_size=input_size, mean=mean, std=std,
                                       batch_size=args.batch_size, is_training=True)
    test_loader = utils.create_loader(test_data, input_size=input_size, mean=mean, std=std, batch_size=args.batch_size,
                                      is_training=False)

    # TODO: Setup checkpoint saver and metric tracking

    # TODO: Setup learning rate schedule and starting epoch
    start_epoch = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, train_loss_fn, args, device)
            print(train_metrics)
    except KeyboardInterrupt:
        pass


def calculate_accuracy(y_pred: Tensor, y: Tensor):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train_one_epoch(epoch: int, model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, train_loss_fn: Callable, args, device=torch.device('cuda')
                    ) -> Tuple[float, float]:
    num_batches = len(loader)
    last_idx = num_batches - 1
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = train_loss_fn(outputs, targets)

        acc = calculate_accuracy(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if batch_idx % args.log_interval == 0:
            _logger.info(
                f"Epoch: {epoch} [{batch_idx}/{num_batches} ({100 * batch_idx / last_idx:.0f}%)]     "
                f"Loss: {loss:.3f} ({epoch_loss / (batch_idx + 1):.3f})    "
                f"Acc: {acc:.3f} ({epoch_acc / (batch_idx + 1):.3f})"
            )

    return epoch_loss / num_batches, epoch_acc / num_batches


if __name__ == '__main__':
    main()
