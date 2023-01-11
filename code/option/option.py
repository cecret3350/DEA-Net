import torch,os,sys,torchvision,argparse
import torch,warnings
import json

# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='ITS')
parser.add_argument('--train_dataset_dir', type=str, default='/home/lab/dataset/RESIDE/ITS')
parser.add_argument('--val_dataset_dir', type=str, default='/home/lab/dataset/RESIDE/SOTS/indoor')
parser.add_argument('--model', type=str, default='baseline', help='model name')
parser.add_argument('--exp_name', type=str, default='baseline', help='experiment name')
parser.add_argument('--saved_model_dir', type=str, default='saved_model', help='path to save model')
parser.add_argument('--saved_log_dir', type=str, default='saved_log', help='path to save log')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')
parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')

# only need for resume
parser.add_argument('--resume', action='store_true', default=False, help='resume training from the pre-trained model')
parser.add_argument('--pre_trained_model', type=str, default='null', help='path of pre trained model for resume training')
# only need for valid
parser.add_argument('--valid', action='store_true', default=False, help='valid using pre-trained model')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
opt=parser.parse_args()

exp_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
exp_model_dir = os.path.join(exp_dataset_dir, opt.exp_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)

if not os.path.exists(exp_dataset_dir):
    os.mkdir(exp_dataset_dir)

opt.saved_model_dir = os.path.join(exp_model_dir, 'saved_model')
opt.saved_log_dir = os.path.join(exp_model_dir, 'saved_log')
opt.saved_infer_dir = os.path.join(exp_model_dir, 'saved_infer')
if not os.path.exists(exp_model_dir):
    os.mkdir(exp_model_dir)
    os.mkdir(opt.saved_model_dir)
    os.mkdir(opt.saved_log_dir)
    os.mkdir(opt.saved_infer_dir)
elif opt.resume:
    tmp = os.path.join(opt.saved_model_dir, opt.pre_trained_model)
    if os.path.exists(tmp):
        opt.pre_trained_model = tmp
    else:
        print(f'path {tmp} is not existed!')
        exit()
elif opt.valid:
    tmp = os.path.join(opt.saved_model_dir, opt.pre_trained_model)
    if os.path.exists(tmp):
        if opt.save_infer_results:
            sid = os.path.join(opt.saved_infer_dir, opt.pre_trained_model.split('.')[0])
            os.makedirs(sid, exist_ok=True)
            opt.saved_infer_dir = sid
        opt.pre_trained_model = tmp
    else:
        print(f'path {tmp} is not existed!')
        exit()
else:
    print(f'{exp_model_dir} has already existed!')
    exit()

with open(os.path.join(exp_model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)