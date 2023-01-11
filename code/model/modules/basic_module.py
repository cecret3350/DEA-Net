import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction=8, mode='FA'):
        super(SpatialAttention, self).__init__()
        self.mode = mode
        if mode == 'FA':
            self.sa = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif mode == 'CGA':
            self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        if self.mode == 'FA':
            sattn = self.sa(x)
        elif self.mode == 'CGA':
            x_avg = torch.mean(x, dim=1, keepdim=True)
            x_max, _ = torch.max(x, dim=1, keepdim=True)
            x2 = torch.concat([x_avg, x_max], dim=1)
            sattn = self.sa(x2)
            return sattn
            
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8, mode='FA'):
        super(ChannelAttention, self).__init__()
        self.mode = mode
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
        )
        if mode == 'FA':
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        if self.mode == 'FA':
            cattn = self.sigmoid(cattn)
        return cattn


class PixelAttention(nn.Module):

    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * channels, channels, 7, padding=3, padding_mode='reflect' ,groups=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding_mode):
        super(ResBlock, self).__init__()
        print('Basic Module: ResBlock')
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, bias=True, padding_mode=padding_mode) 
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, bias=True, padding_mode=padding_mode) 

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = self.conv2(res)
        res += x
        return res


class FABlock(nn.Module):
    def __init__(self, channels, kernel_size, padding_mode):
        super(FABlock, self).__init__()
        print('Basic Module: FABlock')
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, bias=True, padding_mode=padding_mode) 
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, bias=True, padding_mode=padding_mode) 
        self.sa = SpatialAttention(channels, reduction=8, mode='FA')
        self.ca = ChannelAttention(channels, reduction=8, mode='FA')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        res = res * cattn
        sattn = self.sa(res)
        res = res * sattn
        res = res + x
        return res


class CGABlock(nn.Module):
    def __init__(self, channels, kernel_size, padding_mode):
        super(CGABlock, self).__init__()
        print('Basic Module: CGABlock')
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, bias=True, padding_mode=padding_mode) 
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, bias=True, padding_mode=padding_mode) 
        self.sa = SpatialAttention(channels, reduction=8, mode='CGA')
        self.ca = ChannelAttention(channels, reduction=8, mode='CGA')
        self.pa = PixelAttention(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


def resblock_k3(channels, padding_mode):
    return ResBlock(channels, 3, padding_mode)


def fablock_k3(channels, padding_mode):
    return FABlock(channels, 3, padding_mode)


def cgablock_k3(channels, padding_mode):
    return CGABlock(channels, 3, padding_mode)