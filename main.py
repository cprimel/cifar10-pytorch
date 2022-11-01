from torchsummary import summary

from models.resnet import ResNet18v

if __name__ == "__main__":
    net = ResNet18v()
    summary(net, (3,32,32), device="cpu")
