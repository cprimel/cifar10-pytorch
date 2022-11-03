from torchsummary import summary

from models.convmixer import ConvMixer

if __name__ == "__main__":
    net = ConvMixer()
    summary(net, (3, 32, 32), device="cpu")
