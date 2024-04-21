import torch
from einops.layers.torch import Rearrange

def convert_cdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
    conv_weight_cd[:, :, :] = conv_weight[:, :, :]
    conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
    conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
    return conv_weight_cd

def convert_hdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
    conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
    conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
    conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
    return conv_weight_hd

def convert_vdc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
    conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
    conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
    conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
    return conv_weight_vd

def convert_adc(w):
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_ad = conv_weight - conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
    conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
    return conv_weight_ad

saved_model_path = '../trained_models/ITS/PSNR4020_SSIM9934.pk'
ckp = torch.load(saved_model_path)
ckp = ckp['model']
simplified_ckp = {}

for key in ckp.keys():
    if 'conv1_1' in key:
        if 'weight' in key:
            w_cdc = convert_cdc(ckp[key])
        elif 'bias' in key:
            b_cdc = ckp[key]
    elif 'conv1_2' in key:
        if 'weight' in key:
            w_hdc = convert_hdc(ckp[key])
        elif 'bias' in key:
            b_hdc = ckp[key]
    elif 'conv1_3' in key:
        if 'weight' in key:
            w_vdc = convert_vdc(ckp[key])
        elif 'bias' in key:
            b_vdc = ckp[key]
    elif 'conv1_4' in key:
        if 'weight' in key:
            w_adc = convert_adc(ckp[key])
        elif 'bias' in key:
            b_adc = ckp[key]
    elif 'conv1_5' in key:
        if 'weight' in key:
            w_vc = ckp[key]
        elif 'bias' in key:
            b_vc = ckp[key]
            w = w_cdc + w_hdc + w_vdc + w_adc + w_vc
            b = b_cdc + b_hdc + b_vdc + b_adc + b_vc
            simplified_ckp[key.split('conv1_5')[0] + 'weight'] = w
            simplified_ckp[key.split('conv1_5')[0] + 'bias'] = b
    else:
        simplified_ckp[key] = ckp[key]
    torch.save(simplified_ckp, saved_model_path.split('.pk')[0] + '_simplified.pk')