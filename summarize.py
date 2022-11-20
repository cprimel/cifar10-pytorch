"""Summarize PyTorch NN models.

Quickly output model summary. Optional, save graph for viewing in TensorBoard.

Typical usage:
    $python summarize.py --config=<model-config-file>
"""

import argparse
import os

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models import model_registry
from utils import create_loader

parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Model Summary Script", add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser.add_argument('--model', default='resnet10', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet10")')
parser.add_argument('--logs', default='', type=str, metavar="LOG_PATH",
                    help='Path to logs (default: None)')
parser.add_argument('--save-graph', default=False, type=bool, metavar="GRAPH",
                    help="Save tensorboard graph (default: False)")

if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Summary for {args.model}")

    model = model_registry[args.model]()

    summary(model, (3, 32, 32), device="cpu")

    if args.save_graph:
        print(f"Saving graph to {args.logs}/{args.model}")
        ROOT = ".data"
        test_data = torchvision.datasets.CIFAR10(ROOT, train=False, download=True)

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        input_size = (3, 32, 32)

        loader = create_loader(test_data, input_size=input_size, mean=mean, std=std, batch_size=64,
                               is_training=False)

        writer = SummaryWriter(os.path.join(args.logs, args.model))

        images, labels = next(iter(loader))
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)
        writer.close()
