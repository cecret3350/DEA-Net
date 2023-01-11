from torch import nn


class DownSample(nn.Module):
    def __init__(self, in_channels, padding_mode):
        super(DownSample, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        self.act = nn.ReLU(inplace=True) 

    def forward(self, x):
        res = self.act(self.down_conv(x))
        return res