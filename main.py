from torchsummary import summary

from models.resnet import ResNet18

if __name__ == "__main__":
    net = ResNet18()
    summary(net, (3,32,32), device="cpu")
