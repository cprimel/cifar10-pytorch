# TODO: Script for testing model
import argparse
import logging

import torch
import torchvision
from torch import Tensor

import utils
from models import model_registry

_logger = logging.getLogger('test')

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Testing")
parser.add_argument('--model', '-m', metavar='NAME', default='resnet10',
                    help='model architecture (default: resnet10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')

def accuracy(y_pred: Tensor, y: Tensor):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# device
def validate(args):
    device = torch.device(args.device)

    # Load model
    model = model_registry[args.model]()

    # Load checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    ROOT = ".data"
    test_data = torchvision.datasets.CIFAR10(ROOT,
                                             train=False,
                                             download=True)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    input_size = (3, 32, 32)

    test_loader = utils.create_loader(test_data, input_size=input_size, mean=mean, std=std, batch_size=args.batch_size,
                                      is_training=False)

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    results = {}
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            acc = accuracy(outputs.detach(), targets)
            test_loss += loss
            test_acc += acc

            results[batch_idx] = {'test_loss': loss, 'test_acc': acc}
            if (batch_idx + 1) % args.log_interval == 0:
                _logger.info(
                    f"Test: [{batch_idx}/{num_batches}     "
                    f"Loss: {loss:.3f} ({test_loss / (batch_idx + 1):.3f})    "
                    f"Acc: {acc:.3f} ({test_acc / (batch_idx + 1):.3f})"
                )
        return test_loss / num_batches, test_acc / num_batches


def main():
    args = parser.parse_args()

    test_loss, test_acc = validate(args)

    _logger.info(f"Results:\n\tLoss: {test_loss:.2f}\tAcc: {test_acc:.2f}")

if __name__ == '__main__':
    main()
