import sys
import os
import argparse
import numpy as np
import wandb
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from validation import epochVal, epochVal_metrics,epochTest_metrics
from dataloaders import dataset
from config import CLASS_NAMES_DICTS,CLASS_NUM_DICTS,RESIZE_DICTS,CREATE_MODEL_DICTS
from networks.create_model import create_semi_model

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
parser.add_argument('--bnm_loss', type=int, default=0, help='whether use bnm loss to train model')
parser.add_argument('--bnm_loss_improve', type=int, default=0, help='whether use imporved bnm loss')
parser.add_argument('--bnm_loss_weight', type=float, default=1.0, help='weight of bnm_loss')
parser.add_argument('--vat_loss', type=int, default=0, help='whether use vat loss')
parser.add_argument('--vat_loss_weight', type=float, default=1.0, help='weight of vat_loss')
parser.add_argument('--vat_dis_type', type=str,  default="kl", help='vat loss distance type')
parser.add_argument('--vat_filter_batch', type=bool,  default=False, help='whether vat loss do filter ')
parser.add_argument('--vat_filter_num', type=int,  default=8, help='vat losss filter num ')
parser.add_argument('--wcp_loss', type=int, default=0, help='whether use wcp loss')
parser.add_argument('--wcp_loss_weight', type=float, default=1.0, help='weight of wcp_loss')
parser.add_argument('--supCon_loss', type=int, default=0, help='whether use supervised contrastive loss')
parser.add_argument('--supCon_loss_weight', type=float, default=1.0, help='whether use supervised contrastive loss')
parser.add_argument('--entropy_loss', type=int, default=0, help='whether use entropy loss')
parser.add_argument('--entropy_loss_weight', type=float, default=1.0, help='entropy loss weight')
parser.add_argument('--labeled_rate', type=float, default=0.2, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=22000, help='random seed')
parser.add_argument('--gpu', type=str,  default='10,11', help='GPU to use')
parser.add_argument('--baseline', type=int, default=0, help='whether train baseline model')

### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--class_names', type=str,  default=None, help='class name list')
parser.add_argument('--class_num', type=str,  default=None, help='class num list')
parser.add_argument('--resize', type=int,  default=224, help='resize the image')
parser.add_argument('--backbone', type=str,  default='densenet121', help='backbone network')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--consistency_relation_weight', type=float,  default=1.0, help='consistency relation weight')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
parser.add_argument('--consistency_began_epoch', type=int,  default=20, help='consistency loss began epoch')
#add by liupeng
parser.add_argument('--task', type=str,  default='hip_4cls', help='which task')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
class_names = CLASS_NAMES_DICTS[args.task]
class_num = CLASS_NUM_DICTS[args.task]
resize = RESIZE_DICTS[args.task]
args.resize = resize
args.class_names = class_names
args.class_num = class_num
args.csv_file_test = '../dataset/'+ args. task +'/testing.csv'
model = create_semi_model(args)
model_list = glob("../model/hip_4cls_0722*/")
for model_path in model_list:
    #model_path = '../model/skin_0722_frac01_BNM_w01_vat_batch32_densenet121/'
    wandb.init(project=args.task+"-semi-supervised-test",name=model_path.split('/')[-2],reinit=True)
    logging.basicConfig(filename=model_path+"/log_test.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    for epoch in range(1,101):
        checkpoint_path = model_path + 'checkpoint/epoch_'+str(epoch)+'.pth'
        if not os.path.exists(checkpoint_path):
            continue
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                                csv_file=args.csv_file_test,
                                                transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.TenCrop(224),
                                                transforms.Lambda
                                                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                                transforms.Lambda
                                                (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                                ]))
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=64,
                                        shuffle=False, num_workers=8, pin_memory=True)#, worker_init_fn=worker_init_fn)

        #AUROCs, Accus, Senss, Specs = epochVal_metrics_test(model, test_dataloader, args)  
        AUROCs, Accus, Senss, Specs, Pre, F1 = epochTest_metrics(model, test_dataloader, args)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        Pre_avg = np.array(Pre).mean()
        F1_avg = np.array(F1).mean()

        logging.info("\nTEST Student: Epoch: {}".format(epoch))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST Pre: {:6f}, TEST F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, Pre_avg, F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))
        wandb.log({'epoch': epoch,'TEST AUROC': AUROC_avg,'TEST Accus':Accus_avg,'TEST Senss':Senss_avg,
        'TEST Specs':Specs_avg,'TEST Pre':Pre_avg,'TEST F1':F1_avg})