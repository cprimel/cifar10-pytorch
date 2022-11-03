from torchsummary import summary

from models.resnet import ResNeXt10_32_2d

if __name__ == "__main__":
    net = ResNeXt10_32_2d()
    summary(net, (3,32,32), device="cpu")
