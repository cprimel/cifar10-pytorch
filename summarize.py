import argparse
from torchsummary import summary

from models import model_registry

config_parser = parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Model Summary Script", add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--model', default='resnet10', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet10")')

if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Summary for {args.model}")

    net = model = model_registry[args.model]()

    summary(net, (3, 32, 32), device="cpu")
