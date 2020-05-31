import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.models import DenseNet121,DenseNet161
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from dataloaders import chest_xray_14
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics
from config import CLASS_NAMES_DICTS,CLASS_NUM_DICTS,RESIZE_DICTS,CREATE_MODEL_DICTS
from trainer import train_semi_model,train_full_model

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../dataset/skin/training_data/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='../dataset/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='../dataset/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='../dataset/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--bnm_loss', type=int, default=1, help='whether use bnm loss to train model')
parser.add_argument('--labeled_rate', type=float, default=0.2, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=22000, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--backbone', type=str,  default='densenet121', help='backbone network')
parser.add_argument('--supervise_level', type=str,  default='semi', help='full or semi supervised')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
#add by liupeng
parser.add_argument('--task', type=str,  default='skin', help='which task')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.task + "_" + args.exp+"_"+args.backbone+"_"+args.supervise_level + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)




if __name__ == "__main__":
    class_names = CLASS_NAMES_DICTS[args.task]
    class_num = CLASS_NUM_DICTS[args.task]
    resize = RESIZE_DICTS[args.task]
    args.batch_size = args.batch_size * len(args.gpu.split(','))
    base_lr = args.base_lr
    args.labeled_bs = args.labeled_bs * len(args.gpu.split(','))
    args.resize = resize
    args.class_names = class_names
    args.class_num = class_num
    args.root_path = args.root_path.replace("/skin/","/"+args.task+"/")
    args.csv_file_train = args.csv_file_train.replace("/skin/","/"+args.task+"/")
    args.csv_file_val = args.csv_file_val.replace("/skin/","/"+args.task+"/")
    args.csv_file_test = args.csv_file_test.replace("/skin/","/"+args.task+"/")
    if args.task == 'chest':
        dataset = chest_xray_14
    print("class name:",class_names)
    print("class num:",class_num)
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))
    if args.supervise_level == 'semi':
        train_semi_model(args,snapshot_path)
    else:
        train_full_model(args,snapshot_path)