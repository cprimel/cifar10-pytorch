import argparse
import json
import logging
import os.path
import time
from typing import Tuple, Callable

import torch
import torch.utils.data
import torchvision as torchvision
import yaml
from torch import Tensor

import utils
from models import model_registry
from utils import create_optimizer, create_scheduler

"""
    See https://github.com/rwightman/pytorch-image-models/blob/main/train.py.
"""

_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Training", add_help=False)
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
group.add_argument('--sched', type=str, default='cosine_warm', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine_warm")')
group.add_argument('--lr', type=float, default=0.01, metavar='LR',
                   help='learning rate (default: 0.01`)')
group.add_argument('--min-lr', type=float, default=0.0, metavar='MINLR',
                   help='minimum learning rate (default: 0.0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 50)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')
group.add_argument('--t-initial', type=int, default=200, metavar='T_0',
                   help='T_0 for cosine annealing (default: 200)')
group.add_argument('--t-mult', type=int, default=1, metavar='T_M',
                   help='T_mult for cosine annealing (default: 1)')
group.add_argument('--plateau-mode', type=str, default='min', metavar='P_M',
                   help='plateau mode for LR reduction on plateau (default: "min")')
group.add_argument('--patience', type=int, default=10, metavar='PAT',
                   help='Number of updates to wait before reducing LR (default: 10)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--val-ratio', type=float, default=0.9, help='ratio for train-validation split')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--scale', type=float, default=0.9, help='scale for random resizing')
group.add_argument('--rand_aug', type=float, default=0.0, help='prob for random augmentation')
group.add_argument('--ra-n', type=float, default=0.0, help='number of random augmentation ops')
group.add_argument('--ra-m', type=float, default=0.0, help='magnitude of random augmentation ops')
group.add_argument('--erase', type=float, default=0.25, help='prob for random erasing')
group.add_argument('--jitter', type=float, default=0.1, help='prob for color jitter')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('--checkpoint-dir', default='checkpoints', type=str, metavar='PATH',
                   help='path to checkpoints (default: checkpoints/)')
group.add_argument('--log-dir', default='logs', type=str, metavar='PATH', help='path to training logs')
group.add_argument('--output', default='results', type=str, metavar='PATH', help='path to output')
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
    args, args_text = _parse_args()

    _logger.info(f"Preparing experiment {args.experiment}...")

    ckpt_path = os.path.join(args.checkpoint_dir, args.experiment)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    log_path = os.path.join(args.log_dir, args.experiment)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, f"{args.experiment}_config.yml"), "w") as f:
        f.write(args_text)
        f.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #        TODO: Revise model creation to take parameters as kwargs?
    #           model_registry[args.model] -> model_registry[args.model](**kwargs)
    #           Requires adding direct call to class definition, e.g., { "resnet": ResNet }

    model = model_registry[args.model]()
    model = model.to(device)

    _logger.info(f"{args.model} created, # of params: {sum([m.numel() for m in model.parameters()]):,d}.")
    optimizer = create_optimizer(params=model.parameters(), opt_name=args.opt, lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    validate_loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Create train and test datasets
    ROOT = '.data'
    train_data = torchvision.datasets.CIFAR10(ROOT,
                                              train=True,
                                              download=True)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    n_train = int(len(train_data) * args.val_ratio)
    n_val = len(train_data) - n_train
    train_data, val_data = torch.utils.data.random_split(train_data, [n_train, n_val])

    # TODO: Add optional mixup / cutmix augmentation

    # Create dataloaders w/augmentation pipeline
    input_size = (3, 32, 32)

    train_loader = utils.create_loader(train_data, input_size=input_size, mean=mean, std=std,
                                       batch_size=args.batch_size, is_training=True, rand_aug=args.rand_aug,
                                       ra_n=args.ra_n, ra_m=args.ra_m, jitter=args.jitter, scale=args.scale,
                                       prob_erase=args.erase)
    val_loader = utils.create_loader(val_data, input_size=input_size, mean=mean, std=std, batch_size=args.batch_size,
                                     is_training=False)

    # Resume from checkpoint
    start_epoch = 0
    best_acc = None
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt['acc']

    # Create scheduler
    lr_scheduler = create_scheduler(optimizer=optimizer, lr=args.lr, sched=args.sched, num_epochs=args.epochs,
                                    steps_per_epoch=len(train_loader), min_lr=args.min_lr,
                                    T_0=args.t_initial, T_mult=args.t_mult, plateau_mode=args.plateau_mode,
                                    patience=args.patience)
    if (lr_scheduler is not None or lr_scheduler != 'custom') and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    metrics = {}
    try:
        for epoch in range(start_epoch, args.epochs):
            start = time.time()

            (train_loss, train_acc, lr) = train_one_epoch(epoch, model, train_loader, optimizer, lr_scheduler,
                                                          train_loss_fn, args,
                                                          device)
            (val_loss, val_acc) = validate(model, val_loader, validate_loss_fn, args, device)
            if args.sched == 'plateau':
                lr_scheduler.step(val_loss)

            t_epoch = time.time() - start
            _logger.info(
                f"Epoch {epoch + 1} complete:\n\tTrain Acc: {train_acc:.2f}\n\tTest Acc: {val_acc:.2f}\n\t"
                f"lr: {lr:.5f}\n\tTime: {t_epoch:.1f}s, ")
            # TODO:
            #  Find better solution:
            #       val_loss and val_acc are returned as tensors--they shouldn't be!
            metrics[epoch] = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss.item(),
                              'val_acc': val_acc.item(), "lr": lr, "t_epoch": t_epoch}

            if best_acc is None or val_acc > best_acc:
                if best_acc is not None:
                    _logger.info(
                        f"Accuracy increased ({0.00 if None else best_acc:.2f} -> {val_acc:.2f}). Saving model...")
                torch.save({
                    'epoch': epoch,
                    'loss': val_loss,
                    'acc': val_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(args.checkpoint_dir, args.experiment, f"{args.model}_{epoch}_{time.time()}.pt"))
                best_acc = val_acc


    except KeyboardInterrupt:
        pass

    # dump loss and accuracy metrics to json
    if metrics:
        data_dump = json.dumps(metrics)
        f = open(os.path.join(log_path, f"train_{time.time()}"), "w")
        f.write(data_dump)
        f.close()


def accuracy(y_pred: Tensor, y: Tensor):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train_one_epoch(epoch: int, model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, lr_scheduler: Callable, train_loss_fn: Callable, args,
                    device=torch.device('cuda')
                    ) -> Tuple[float, float, float]:
    num_batches = len(loader)
    last_idx = num_batches - 1
    num_updates = epoch * num_batches
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()
    lr = None

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        lr = lr_scheduler.get_last_lr()[0]
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = train_loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)

        loss.backward()
        optimizer.step()
        if args.sched == 'onecycle':
            lr_scheduler.step()
        elif args.sched == 'cosine_warm':
            lr_scheduler.step(epoch + batch_idx / num_batches)
        num_updates += 1

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if (batch_idx + 1) % args.log_interval == 0:
            _logger.info(
                f"Epoch: {epoch + 1} [{batch_idx + 1}/{num_batches} ({100 * batch_idx / last_idx:.0f}%)]     "
                f"Loss: {loss:.3f} ({epoch_loss / (batch_idx + 1):.3f})    "
                f"Acc: {acc:.3f} ({epoch_acc / (batch_idx + 1):.3f})    "
                f"lr: {lr:.6f}"
            )

    return epoch_loss / num_batches, epoch_acc / num_batches, lr


def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn: Callable, args,
             device=torch.device('cuda')) -> Tuple[float, float]:
    model.eval()
    num_batches = len(loader)
    last_idx = len(loader) - 1
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            acc = accuracy(outputs, targets)

            val_loss += loss
            val_acc += acc

    return val_loss / num_batches, val_acc / num_batches


if __name__ == '__main__':
    main()
