import os 
import sys

import argparse
import random

import torch
import wandb
import logging
from trainer import Trainer
from config import CLASS_NAMES_DICTS,CLASS_NUM_DICTS

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='model_test', help='the name of this run')
parser.add_argument('--task', type=str, default='skin', help='train which task')
# Datasets
parser.add_argument('--root_path', type=str, default='../../dataset/chest/training_data/',
                    help='dataset root dir')
parser.add_argument('--train_file', type=str, default='./dataSplit/train_0_shuffle.txt')
parser.add_argument('--test_file', type=str, default='./dataSplit/test_0.txt')
parser.add_argument('--val_file', type=str, default='./dataSplit/val_0.txt')
parser.add_argument('--resize', type=int, default=256, help='reize image')
parser.add_argument('--crop_size', type=int, default=224, help='image crop')
#semi-supervised setting
parser.add_argument('--labeled_num', type=int, default=500, help='number of labeled samples')
parser.add_argument('--labeled_bs', type=int, default=32, help='num of labeled samples in each batch')
# Optimization options
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--checkpoint', type=str, help='whether load checkpoint')
parser.add_argument('--train_bs', type=int, default=646, help='train batchsize')
parser.add_argument('--test_bs', type=int, default=64, help='test batchsize')
parser.add_argument('--lr','--learning_rate', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, help='Dropout ratio')
# network architecture
parser.add_argument('--backbone', type=str, default='densenet121', help='backbone of network')
# loss
parser.add_argument('--vat_loss_weight', type=float, default=0.0,help='weight of vat_loss')
parser.add_argument('--vat_start_epoch', type=int, default=0, help='vat loss start epoch')
parser.add_argument('--vat_filter_batch', type=bool, default=False, help='whether vat loss do filter')
parser.add_argument('--vat_filter_num', type=int, default=16, help='vat loss filter num')
parser.add_argument('--bnm_loss_weight', type=float, default=0.0,help='weight of vat_loss')
parser.add_argument('--bnm_start_epoch', type=int, default=0, help='vat loss start epoch')
parser.add_argument('--entropy_loss_weight', type=float, default=0.0,help='weight of entropy_loss')
parser.add_argument('--entropy_start_epoch', type=int, default=0, help='entropy loss start epoch')
#Miscs
parser.add_argument('--manualSeed', type=int, default=1337, help='manual seed')
parser.add_argument('--pretrained', type=bool, default=True, help='is use pretrained model')
#device options
parser.add_argument('--gpu_ids', default='0,1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

#Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
use_cuda = torch.cuda.is_available()

#Random seed
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    class_names = CLASS_NAMES_DICTS[args.task]
    args.class_num_dict = CLASS_NUM_DICTS[args.task]
    class_num = len(class_names)
    args.class_names = class_names
    args.class_num = class_num
    args.root_path = '../../dataset/'+args.task+'/training_data/'
    args.train_file = '../../dataset/'+args.task+'/training.txt'
    args.test_file = '../../dataset/'+args.task+'/testing.txt'
    args.val_file = '../../dataset/'+args.task+'/validation.txt'
    snap_name = args.task + "_" + args.exp + "_" + args.backbone+"_labeledNum"+str(args.labeled_num)
    logging.basicConfig(filename="../../log/"+snap_name+"_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    wandb.init(project=args.task + "-semi-supervised-newcode", name=snap_name)
    wandb.config.update(args)
    Trainer.train(args,wandb,logging)


if __name__ == '__main__':
    main()
