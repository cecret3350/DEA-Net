import os,argparse
import json


parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='ITS')
parser.add_argument('--val_dataset_dir', type=str, default='/home/lab/dataset/RESIDE/SOTS/indoor')
parser.add_argument('--model', type=str, default='baseline', help='model name')
parser.add_argument('--exp_name', type=str, default='DEA-Net', help='experiment name')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

# only need for evaluation
parser.add_argument('--pre_trained_model', type=str, default='null', help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
opt=parser.parse_args()

exp_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
exp_model_dir = os.path.join(exp_dataset_dir, opt.exp_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)

if not os.path.exists(exp_dataset_dir):
    os.mkdir(exp_dataset_dir)

opt.saved_infer_dir = os.path.join(exp_model_dir, opt.pre_trained_model.split('.pth')[0])
if not os.path.exists(exp_model_dir):
    os.mkdir(exp_model_dir)
    os.mkdir(opt.saved_infer_dir)

with open(os.path.join(exp_model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)