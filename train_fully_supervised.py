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
from torch import nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pretrainedmodels

from networks.models import DenseNet121,DenseNet161
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from dataloaders import chest_xray_14
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../dataset/skin/training_data/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='../dataset/skin/training_fold1_frac02.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='../dataset/skin/testing_fold1.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='../dataset/skin/testing_fold1.csv', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_rate', type=float, default=0.2, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=22000, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--backbone', type=str,  default='xception', help='backbone network')
parser.add_argument('--supervise_level', type=str,  default='full', help='full or semi supervised')

# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
parser.add_argument('--resize', type=int,  default=256, help='image resize')
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
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



if __name__ == "__main__":
    resize = args.resize
    CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
     'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
    if args.task == 'chest':
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax','Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
        'Pleural_Thickening', 'Hernia']
        dataset = chest_xray_14
        resize = 384
        args.root_path = args.root_path.replace("/skin/","/chest/")
        args.csv_file_train = args.csv_file_train.replace("/skin/","/chest/")
        args.csv_file_val = args.csv_file_val.replace("/skin/","/chest/")
        args.csv_file_test = args.csv_file_test.replace("/skin/","/chest/")
    if args.task == 'hip':
        CLASS_NAMES = ['Normal','ONFH_I','ONFH_II']
        args.root_path = args.root_path.replace("/skin/","/hip/")
        args.csv_file_train = args.csv_file_train.replace("/skin/","/hip/")
        args.csv_file_val = args.csv_file_val.replace("/skin/","/hip/")
        args.csv_file_test = args.csv_file_test.replace("/skin/","/hip/")
    if args.task == 'hip_3cls':
        CLASS_NAMES = ['Normal','OA','ONFH']
        args.root_path = args.root_path.replace("/skin/","/hip_3cls/")
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    def create_model():
        # Network definition
        num_class = 7
        if args.task == 'hip' or args.task == 'hip_3cls':
            num_class = 3
        
        net = pretrainedmodels.__dict__[args.backbone](num_classes=1000,
                                                      pretrained='imagenet')
        if args.backbone == 'xception':
            num_fc = net.last_linear.in_features
            net.last_linear = nn.Linear(num_fc, num_class)
        if args.backbone == 'densenet121':
            num_fc = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_fc, num_class)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        return model

    model = create_model()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
    #                              betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    normalize = transforms.Normalize([0.605, 0.605, 0.605],
                                     [0.156, 0.156, 0.156])

    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=transforms.Compose([
                                                #             transforms.RandomCrop(480),
                                                # transforms.RandomRotation(15),
                                                # transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
                                                transforms.Resize((resize, resize)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                # transforms.RandomRotation(10),
                                                # transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ]))

    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((resize, resize)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((resize, resize)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    print("train_dataset len:",len(train_dataset))
    #labeled_idxs = list(range(labeled_num))
    #unlabeled_idxs = list(range(labeled_num, train_dataset_num))
    #batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size = batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)#, worker_init_fn=worker_init_fn)
    
    model.train()

    loss_fn = losses.cross_entropy_loss(args)

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()

    #train
    class_weight = torch.FloatTensor([1.0,6.0,1.5]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    for epoch in range(args.start_epoch, args.epochs):
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_consistency_relation = MetricLogger(delimiter="  ")
        time1 = time.time()
        iter_max = len(train_dataloader)  
        for i, (_, _, image_batch, label_batch) in enumerate(train_dataloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, label_batch = image_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cuda()
            inputs = image_batch #+ noise1

            outputs = model(inputs)

            ## calculate the loss
            #label_batch = torch.max(label_batch, 1)[1]
            loss_classification = loss_fn(outputs, label_batch)
            loss = loss_classification

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # outputs_soft = F.softmax(outputs, dim=1)
            meters_loss.update(loss=loss)
            meters_loss_classification.update(loss=loss_classification)

            iter_num = iter_num + 1
            # write tensorboard
            if i % 100 == 0:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_classification', loss_classification, iter_num)

                logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, lr: {}"
                            .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg,optimizer.param_groups[0]['lr']))

                image = inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('raw/Image', grid_image, iter_num)

        timestamp = get_timestamp()

        # validate student
        # 

        AUROCs, Accus, Senss, Specs = epochVal_metrics(model, val_dataloader, args)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        logging.info("\nVAL Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nVAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(Specs)]))

        # test student
        # 
        AUROCs, Accus, Senss, Specs = epochVal_metrics(model, test_dataloader, args)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(CLASS_NAMES[i], v) for i,v in enumerate(Specs)]))

        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({    'epoch': epoch + 1,
                        'global_step': iter_num,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epochs'    : epoch,
                        # 'AUROC'     : AUROC_best,
                   }
                   , save_mode_path
        )
        logging.info("save model to {}".format(save_mode_path))

        # update learning rate
        lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
