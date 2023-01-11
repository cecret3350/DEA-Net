import torch
from torch import nn


class MixUp(nn.Module):
    def __init__(self, m=-0.80):
        super(MixUp, self).__init__()
        w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, cur_feat, prev_feat):
        mix_factor = self.mix_block(self.w)
        out = cur_feat * mix_factor.expand_as(cur_feat) + prev_feat * (1 - mix_factor.expand_as(prev_feat))
        return out


class FusionIdentity(nn.Module):
    def __init__(self):
        super(FusionIdentity, self).__init__()

    def forward(self, cur_feat: torch.Tensor, prev_feat) -> torch.Tensor:
        return cur_feat 


def mixup_mn1(channels, padding_mode):
    return MixUp(m=-1)

def mixup_mn0p6(channels, padding_mode):
    return MixUp(m=-0.6)