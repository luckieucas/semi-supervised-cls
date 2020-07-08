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
import wandb

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
from dataloaders import  dataset as dt
from dataloaders import chest_xray_14
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics
from wcp_loss import wcp_loss_torch

from networks.create_model import create_semi_model,create_full_model



def get_current_consistency_weight(args,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def train_semi_model(args,snapshot_path):
    resize = args.resize
    batch_size = args.batch_size
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))



    model = create_semi_model(args)
    ema_model = create_semi_model(args,ema=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
                                betas=(0.9, 0.999), weight_decay=5e-4)
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    dataset = dt
    if args.task == 'chest':
        dataset = chest_xray_14
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((resize, resize)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                # transforms.RandomRotation(10),
                                                # transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))

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
    train_dataset_num = len(train_dataset)
    labeled_num = int(train_dataset_num*args.labeled_rate)
    print("labeled_num:",labeled_num)
    labeled_idxs = list(range(labeled_num))
    unlabeled_idxs = list(range(labeled_num, train_dataset_num))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    #batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, batch_size, labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)#, worker_init_fn=worker_init_fn)

    model.train()
    loss_fn = losses.cross_entropy_loss(args)
    loss_supCon_fn = losses.SupConLoss()
    vat_loss_fn = losses.VATLoss(task=args.task)
    if args.task == 'chest':
        loss_fn = losses.Loss_Ones()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()

    #train
    for epoch in range(args.start_epoch, args.epochs):
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_consistency_relation = MetricLogger(delimiter="  ")
        meters_loss_bnm = MetricLogger(delimiter="  ")
        meters_loss_bnm_improve = MetricLogger(delimiter="  ")
        meters_loss_supCon = MetricLogger(delimiter="  ")
        meters_loss_vat = MetricLogger(delimiter="  ")
        meters_loss_wcp = MetricLogger(delimiter="  ")
        meters_loss_entropy = MetricLogger(delimiter="  ")
        time1 = time.time()
        iter_max = len(train_dataloader)
        #label_count = torch.LongTensor([-1])
        for i, (_,_, (image_batch, ema_image_batch), label_batch) in enumerate(train_dataloader):
            time2 = time.time()
            #label_count = torch.cat((label_count, torch.argmax(label_batch[:labeled_bs], dim=1)), 0)
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
            # unlabeled_image_batch = ema_image_batch[labeled_bs:]

            # noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
            # noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)
            ema_inputs = ema_image_batch #+ noise2
            inputs = image_batch #+ noise1

            activations, outputs = model(inputs)
            with torch.no_grad():
                ema_activations, ema_output = ema_model(ema_inputs)

            ## calculate the loss
            loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
            loss = loss_classification
            ## MT loss (have no effect in the beginneing)
            if args.ema_consistency == 1 and epoch > args.consistency_began_epoch:
                consistency_weight = get_current_consistency_weight(args,epoch)
                consistency_dist = torch.sum(losses.softmax_mse_loss(outputs, ema_output, args)) / batch_size #/ dataset.N_CLASSES
                consistency_loss = consistency_weight * consistency_dist  

                # consistency_relation_dist = torch.sum(losses.relation_mse_loss_cam(activations, ema_activations, model, label_batch)) / batch_size
                if args.multi_scale_densenet == 1:
                    consistency_relation_dist0 = torch.sum(losses.relation_mse_loss(activations[0], ema_activations[0])) / batch_size
                    consistency_relation_dist1 = torch.sum(losses.relation_mse_loss(activations[1], ema_activations[1])) / batch_size
                    consistency_relation_dist2 = torch.sum(losses.relation_mse_loss(activations[2], ema_activations[2])) / batch_size
                    consistency_relation_dist3 = torch.sum(losses.relation_mse_loss(activations[3], ema_activations[3])) / batch_size
                    consistency_relation_dist = args.scale1_weight * consistency_relation_dist0 + \
                        args.scale2_weight * consistency_relation_dist1 + args.scale3_weight * consistency_relation_dist2+ \
                            args.scale4_weight * consistency_relation_dist3
                    #consistency_relation_dist = consistency_relation_dist2
                else:
                    consistency_relation_dist = torch.sum(losses.relation_mse_loss(activations, ema_activations)) / batch_size

                consistency_relation_loss = consistency_weight * consistency_relation_dist * args.consistency_relation_weight
            else:
                consistency_loss = 0.0
                consistency_relation_loss = 0.0
                consistency_weight = 0.0
                consistency_dist = 0.0
             #+ consistency_loss

             # bnm loss
            if args.bnm_loss == 1 and epoch > args.consistency_began_epoch:
                 bnm_loss = args.bnm_loss_weight * (losses.bnm_loss(outputs[labeled_bs:]))
            else:
                bnm_loss = 0.0
            
            # improved bnm loss
            if args.bnm_loss_improve == 1:
                bnm_loss_improve = args.bnm_loss_weight * losses.bnm_loss_improve(outputs[labeled_bs:])
            else:
                bnm_loss_improve = 0.0
            
            # supervised Contrastive Learning
            if args.supCon_loss == 1 and epoch > args.consistency_began_epoch:
                supCon_fea = torch.cat([F.normalize(activations,dim=1).unsqueeze(1),F.normalize(ema_activations,dim=1).unsqueeze(1)],dim=1)
                supCon_loss = args.supCon_loss_weight * loss_supCon_fn(supCon_fea[:labeled_bs],
                                                                       torch.argmax(label_batch[:labeled_bs], dim=1))
            else:
                supCon_loss = 0.0
            
            # use VAT loss
            if args.vat_loss ==1 and epoch > args.consistency_began_epoch:
                #consistency_weight = get_current_consistency_weight(args,epoch)
                consistency_weight = 1.0
                vat_loss = consistency_weight * args.vat_loss_weight * vat_loss_fn(model,image_batch[labeled_bs:])
            else:
                vat_loss = 0.0

            if args.wcp_loss ==1:
                start = time.clock()
                wcp_loss = wcp_loss_torch(model,image_batch[labeled_bs:])
                elapsed = (time.clock()-start)
                print("WCP time used:",elapsed)
            else:
                wcp_loss = 0.0
            #loss += bnm_loss

            # use entropy mini loss
            if args.entropy_loss == 1 and epoch > args.consistency_began_epoch:
                entropy_loss = args.entropy_loss_weight * losses.entropy_y_x(outputs[labeled_bs:])
            else:
                entropy_loss = 0.0
            if epoch > args.consistency_began_epoch and args.baseline == 0:
                loss = loss_classification + consistency_loss + consistency_relation_loss + \
                    bnm_loss + bnm_loss_improve + supCon_loss + vat_loss + entropy_loss + wcp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # outputs_soft = F.softmax(outputs, dim=1)
            meters_loss.update(loss=loss)
            meters_loss_classification.update(loss=loss_classification)
            meters_loss_bnm.update(loss=bnm_loss)
            meters_loss_bnm_improve.update(loss=bnm_loss_improve)
            meters_loss_supCon.update(loss=supCon_loss)
            meters_loss_vat.update(loss=vat_loss)
            meters_loss_wcp.update(loss=wcp_loss)
            meters_loss_entropy.update(loss=entropy_loss)
            meters_loss_consistency.update(loss=consistency_loss)
            meters_loss_consistency_relation.update(loss=consistency_relation_loss)

            iter_num = iter_num + 1
            # write tensorboard
            if i % 100 == 0:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_classification', loss_classification, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/bnm_loss', bnm_loss, iter_num)
                writer.add_scalar('train/bnm_loss_improve', bnm_loss_improve, iter_num)
                writer.add_scalar('train/supCon_loss', supCon_loss, iter_num)
                writer.add_scalar('train/vat_loss', vat_loss, iter_num)
                writer.add_scalar('train/wcp_loss', wcp_loss, iter_num)
                writer.add_scalar('train/entropy_loss', entropy_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

                logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f},\
                     consistency loss: {:.6f}, consistency relation loss: {:.6f}, bnm loss: {:.6f},bnm loss improve: {:.6f},\
                         supCon loss: {:.6f},vat loss: {:.6f},wcp loss: {:.6f},entropy loss: {:.6f},consistency weight: {:.6f}, lr: {}"
                            .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg,
                                 meters_loss_consistency.loss.avg, meters_loss_consistency_relation.loss.avg,
                                 meters_loss_bnm.loss.avg, meters_loss_bnm_improve.loss.avg, 
                                 meters_loss_supCon.loss.avg,meters_loss_vat.loss.avg,meters_loss_wcp.loss.avg,
                                 meters_loss_entropy.loss.avg, consistency_weight,
                                  optimizer.param_groups[0]['lr']))

                image = inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('raw/Image', grid_image, iter_num)
                image = ema_inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('noise/Image', grid_image, iter_num)

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
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))

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
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))

        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({    'epoch': epoch + 1,
                        'global_step': iter_num,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
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



def train_full_model(args,snapshot_path):
    resize = args.resize
    batch_size = args.batch_size
    base_lr = args.base_lr
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = create_full_model(args)
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
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))

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
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))

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