import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import * 


class Backbone(nn.Module):
    
    def __init__(self, model_config):
        super(Backbone, self).__init__()

        base_channel = model_config['base_channel']
        padding_mode = model_config['padding_mode']
        self.level_depths = [int(x) for x in model_config['level_depths'].split('-')]
        stage_init_infos = model_config['stage_init_infos'].split('-')
        stage_layer_infos = model_config['stage_layer_infos'].split('-')
        stage_fusion_infos = model_config['stage_fusion_infos'].split('-')
        stage_share_weights = model_config['stage_share_weights'].split('-')
        self.stage_num = len(self.level_depths)
        self.level_num = self.stage_num // 2 + 1

        self.fusion_cur_pos = ['_' for i in range(self.stage_num)]
        self.fusion_prev_pos = ['_' for i in range(self.stage_num)]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.inits = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.layers = nn.ModuleList() 

        self.init_conv = nn.Conv2d(3, base_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.output_conv = nn.Conv2d(base_channel, 3, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        for i in range(self.stage_num):
            level = i + 1 if i < self.level_num else self.stage_num - i
            level_channel = base_channel * 2 ** (level - 1)

            init_module_info = stage_init_infos[i]
            layer_module_info = stage_layer_infos[i]
            fusion_module_info = stage_fusion_infos[i]
            is_sw = True if stage_share_weights[i] != 'None' else False
           
            if init_module_info != 'None':
                self.inits.append(nn.Conv2d(level_channel, level_channel, 3, 1, padding=1,
                                        bias=True, padding_mode=padding_mode))
            else:
                self.inits.append(nn.Identity()) 

            if layer_module_info != 'None':
                tmp = BasicLayer(self.level_depths[i], is_sw)
                if is_sw:
                    tmp.append(eval(layer_module_info.lower())(level_channel, padding_mode))
                    assert tmp.get_depth() == 1, 'error initializing basic layer! (you are sharing weights)'
                else:
                    for j in range(self.level_depths[i]):
                        tmp.append(eval(layer_module_info.lower())(level_channel, padding_mode))
                    assert tmp.get_depth() == self.level_depths[i], 'error initializing basic layer!'
                self.layers.append(tmp)
            else:
                self.layers.append(nn.Identity()) 

            if fusion_module_info != 'None':
                self.fusions.append(eval(fusion_module_info.split('|')[0].lower())(level_channel, padding_mode))
                tmp = fusion_module_info.split('|')[1]
                self.fusion_cur_pos[i] = tmp.split('_')[0]
                self.fusion_prev_pos[self.stage_num - 1 - i] = tmp.split('_')[0]
            else:
                self.fusions.append(FusionIdentity())

            if i < self.level_num - 1:
                self.downs.append(DownSample(level_channel, padding_mode))
            else:
                self.downs.append(nn.Identity())

            if i in range(self.level_num - 1, self.stage_num - 1):
                self.ups.append(UpSample(level_channel, padding_mode))
            else:
                self.ups.append(nn.Identity())
                

    def forward(self, x):
        feat_cache = {}
        x = self.init_conv(x)

        for i in range(self.stage_num):
            cur_pos = self.fusion_cur_pos[i]
            prev_pos = self.fusion_prev_pos[i]

            if prev_pos == 'front':
                feat_cache['stage_{}_front'.format(i + 1)] = x
            if cur_pos == 'front':
                x = self.fusions[i](x, feat_cache['stage_{}_front'.format(self.stage_num - i)])

            x = self.inits[i](x)

            if prev_pos == 'middle':
                feat_cache['stage_{}_middle'.format(i + 1)] = x
            if cur_pos == 'middle':
                x = self.fusions[i](x, feat_cache['stage_{}_middle'.format(self.stage_num - i)])

            x = self.layers[i](x)

            if prev_pos == 'rear':
                feat_cache['stage_{}_rear'.format(i + 1)] = x
            if cur_pos == 'rear':
                x = self.fusions[i](x, feat_cache['stage_{}_{}'.format(self.stage_num - i,
                                    self.fusion_prev_pos[self.stage_num - i - 1])])
            x = self.downs[i](x)
            x = self.ups[i](x)
        
        res = self.output_conv(x)
        return res

# if __name__ == '__main__':
#     with open(os.path.join('../configs', 'ITS', 'baseline.json'), 'r') as f:
#         model_config = json.load(f)
#     x = torch.rand((16, 3, 256, 256)).cuda()
#     network = Backbone(model_config).cuda()
#     y = network(x)