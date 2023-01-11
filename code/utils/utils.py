import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def norm_zero_to_one(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def save_heat_image(x, save_path, norm=False):
    if norm:
        x = norm_zero_to_one(x)
    x = x.squeeze(dim=0)
    C, H, W = x.shape
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if C == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)[:, :, ::-1]
    x = Image.fromarray(x)
    x.save(save_path)