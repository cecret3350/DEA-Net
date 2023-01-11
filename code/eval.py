import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils import AverageMeter, pad_img, val_psnr, val_ssim, save_heat_image
from data import ValDataset
from option import opt
from model import Backbone


# training environment
if opt.use_ddp:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if local_rank == 0: print('==> Using DDP.')
else:
    world_size = 1


# model config
with open(os.path.join('configs', opt.dataset, opt.model + '.json'), 'r') as f:
    model_config = json.load(f)


def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in tqdm(val_loader, desc='validation'):
        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cuda()

        with torch.no_grad():
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 4)
            output = network(hazy_img)
            output = output.clamp(0, 1)
            output = output[:, :, :H, :W]
            if opt.save_infer_results:
                save_image(output, os.path.join(opt.saved_infer_dir, batch['filename'][0]))

        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img).item()
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    # define network, and use DDP for faster training
    network = Backbone(model_config) 
    network.cuda()

    if opt.use_ddp:
        network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
        if model_config['batch_size'] // world_size < 16:
            if local_rank == 0: print('==> Using SyncBN because of too small norm-batch-size.')
            nn.SyncBatchNorm.convert_sync_batchnorm(network)
    else:
        network = DataParallel(network)
        # if m_setup['batch_size'] // torch.cuda.device_count() < 16:
        # 	print('==> Using SyncBN because of too small norm-batch-size.')
        # 	convert_model(network)

    # define dataset
    val_dataset = ValDataset(os.path.join(opt.val_dataset_dir, 'hazy'), os.path.join(opt.val_dataset_dir, 'clear'))
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)
    val_loader.num_workers = 12

    # load saved model
    model_info = torch.load(opt.pre_trained_model, map_location='cpu')
    network.load_state_dict(model_info['state_dict'])
    best_psnr = model_info['best_psnr'] 
    print('best PSNR:{}'.format(best_psnr))

    # start validation

    avg_psnr, avg_ssim = valid(val_loader, network) 
    print('valid PSNR:{} SSIM:{}'.format(avg_psnr, avg_ssim))