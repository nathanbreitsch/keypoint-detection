import torch
import torch.nn as nn

def KeypointDetector96():
    return nn.Sequential(
        ConvUnit(1, 32), # 48 x 32
        ConvUnit(32, 64), # 24 x 64
        ConvUnit(64, 128), # 12 x 128
        ConvUnit(128, 256), # 6 x 256
        Flatten(6 * 6 * 256),
        nn.Linear(in_features=6 * 6 * 256, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=136),
        #nn.Tanh()
    )

def Shallow96():
    return nn.Sequential(
        ConvUnit(1, 32), # 48 * 32
        ConvUnit(32, 64), # 24 * 64
        Flatten(24 * 24 * 64),
        nn.Linear(in_features=24 * 24 * 64, out_features=2048),
        nn.ReLU(),
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=136)
    )

def ConvUnit(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Flatten(nn.Module):

    def __init__(self, num_features):
        super(Flatten, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

