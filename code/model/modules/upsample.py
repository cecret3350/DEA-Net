from torch import nn


class UpSample(nn.Module):
    def __init__(self, in_channels, padding_mode):
        super(UpSample, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode=padding_mode) 
        self.act = nn.ReLU(inplace=True) 

    def forward(self, x):
        res = self.act(self.up_conv(x))
        return res