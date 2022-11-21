"""PyTorch NN testing script for CIFAR-10 classification models.


Typical usage:
    $python test.py --model=<model_name> --checkpoint=<path-to-checkpoint> --logs=<path-to-save-logs>
"""
import argparse
import json
import logging
import os
import time

import torch
import torchvision
from torch import Tensor

import utils
from models import model_registry

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Testing")
parser.add_argument('--model', '-m', metavar='NAME', default='resnet10',
                    help='Model identifier (default: resnet10)')
parser.add_argument('--experiment', default='', help='Name of experiment (default: None')
parser.add_argument('--checkpoint', default='', type=str, metavar='CKPT_PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--device', default='cuda', type=str, metavar="DEV",
                    help='Device to use (default: "cuda".')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='Batch size (default: 256)')
parser.add_argument('--log-interval', default=10, type=int,
                    metavar='LOG_I', help='Batch logging frequency (default: 10)')
parser.add_argument('--logs', default='', type=str, metavar="LOG_PATH",
                    help='Path to logs (default: None)')


def accuracy(y_pred: Tensor, y: Tensor):
    """Calculates accuracy."""
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# device
def validate(args):
    """

    Returns:
        tuple: loss, accuracy, time
    """
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
    results = {}
    test_acc = 0
    m = 0
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            start = time.time()
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            end = time.time() - start

            criterion(outputs, targets)
            test_acc += (outputs.max(1)[1] == targets).sum().item()
            m += targets.size(0)

            results[batch_idx] = {'test_acc': test_acc / m, 'predicted_labels': outputs.tolist()[0],
                                  'true_labels': targets.tolist()[0]}

            if (batch_idx + 1) % args.log_interval == 0:
                logging.info(
                    f"Test: [{batch_idx + 1}/{num_batches}     "
                    f"Acc:  {test_acc / m:.3f}     "
                    f"Time: {end:.4f}"
                )

        if results:
            data_dump = json.dumps(results)
            f = open(os.path.join(args.logs, args.experiment, f"test_{time.time()}"), "w")
            f.write(data_dump)
            f.close()
        return test_acc / m


def main():
    args = parser.parse_args()

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)

    test_acc = validate(args)
    logging.info(f"Results:\n\tTest Acc: {test_acc:.3f}")


if __name__ == '__main__':
    main()
