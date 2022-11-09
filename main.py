from torchsummary import summary

from models.convmixer import ConvMixer
from models.resnet import ResNet26_2_32d

if __name__ == "__main__":
    net = ResNet26_2_32d()
    summary(net, (3, 32, 32), device="cpu")
