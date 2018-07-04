import torch
import torch.nn as nn

def KeypointDetector512():
    return nn.Sequential(
        DoubleConv(3, 32), # 256 x 32
        DoubleConv(32, 64), # 128 x 64
        DoubleConv(64, 128), # 64 x 128
        DoubleConv(128, 64), # 32 x 64
        Flatten(32 * 32 * 64),
        nn.Linear(in_features=32 * 32 * 64, out_features=2048),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=2048, out_features=136),
        nn.Sigmoid()
    )

def ConvUnit(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def DoubleConv(in_channels, out_channels):
    return nn.Sequential(
        ConvUnit(in_channels, out_channels),
        ConvUnit(out_channels, out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Flatten(nn.Module):

    def __init__(self, num_features):
        super(Flatten, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

