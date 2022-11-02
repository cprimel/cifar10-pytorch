from torchsummary import summary

from models.resnet import ResNet10

if __name__ == "__main__":
    net = ResNet10()
    summary(net, (3,32,32), device="cpu")
