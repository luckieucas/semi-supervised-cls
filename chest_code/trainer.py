import os
import sys
import time

import numpy as np
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn.metrics.ranking import roc_auc_score

from models import DenseNet121,DenseNet169,DenseNet201
from DatasetGenerator import DatasetGenerator, TwoStreamBatchSampler
from losses import VATLoss,bnm_loss,bnm_loss_improve
from utils import AverageMeter
from metrics import compute_metrics_test
from losses import cross_entropy_loss,entropy_y_x
from mixup import mixup_data_sup


class Trainer():
    # train the classification model

    def train(args,wandb,logging):
        """
        train model
        """
        #-------------------SETTINGs: NETWORK ARCHITECTURE
        model = DenseNet121(args.class_num,True)
        model = torch.nn.DataParallel(model).cuda()

        #--------------------training and validation data SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # train and validation data transform
        trainval_transforms_list = []
        trainval_transforms_list.append(transforms.RandomResizedCrop(args.crop_size))
        trainval_transforms_list.append(transforms.RandomHorizontalFlip())
        trainval_transforms_list.append(transforms.ToTensor())
        trainval_transforms_list.append(normalize)
        trainval_transform_sequence = transforms.Compose(trainval_transforms_list)

        #test data transfrom
        test_transforms_list = []
        test_transforms_list.append(transforms.Resize(args.resize))
        test_transforms_list.append(transforms.TenCrop(args.crop_size))
        test_transforms_list.append(transforms.Lambda(lambda crops: \
            torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms_list.append(transforms.Lambda(lambda crops: \
            torch.stack([normalize(crop) for crop in crops])))
        test_transform_sequence = transforms.Compose(test_transforms_list)


        #--------------------SETTINGS: BUILD DATASET
        datasetTrain = DatasetGenerator(pathImageDirectory=args.root_path, pathDatasetFile=args.train_file,
         transform=trainval_transform_sequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=args.root_path, pathDatasetFile=args.val_file,
         transform=trainval_transform_sequence)
        datasetTest = DatasetGenerator(pathImageDirectory=args.root_path, pathDatasetFile=args.test_file,
         transform=test_transform_sequence)

        #--------------------SETTINGS: SEMI-SUPERVISED
        train_dataset_num = len(datasetTrain)
        #labeled_num = int(train_dataset_num*args.labeled_rate)
        print("labeled_num:",args.labeled_num)
        labeled_idxs = list(range(args.labeled_num))
        unlabeled_idxs = list(range(args.labeled_num, train_dataset_num))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.train_bs,
                                            args.train_bs-args.labeled_bs)
        dataLoaderTrain = DataLoader(dataset=datasetTrain,batch_sampler=batch_sampler, num_workers=8,
                                    pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=64, shuffle=False, num_workers=8,
                                    pin_memory=True)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=args.test_bs, num_workers=8, 
                                    shuffle=False, pin_memory=True)

        #--------------------SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode='min')

        #--------------------SETTINGS: LOSS
        class_weight = torch.Tensor([sum(args.class_num_dict)/i for i in args.class_num_dict]).cuda()
        loss_fn = cross_entropy_loss(args)
        if args.task == 'chest':
            loss_fn = torch.nn.BCELoss(size_average=True)

        #------Load checkpoint
        if args.checkpoint != None:
            modelCheckpoint = torch.load(args.checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
        
        #-----TRAIN MODEL
        for epoch in range(args.start_epoch, args.epochs):
            Trainer.epochTrain(args, wandb, model, epoch, dataLoaderTrain, optimizer, scheduler,loss_fn)
            Trainer.epochVal(args, wandb, model, epoch, dataLoaderVal,loss_fn)
            #Trainer.epochTest_new(args,wandb, logging, model, epoch, args.root_path)
            Trainer.epochTest(args, wandb, logging, model, epoch, dataLoaderTest)

    def epochTrain(args, wandb, model, epoch, dataloader, optimzer, scheduler,loss_fn):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()
        cls_loss = AverageMeter()
        end = time.time()
        vat_filter_num = 8 + int((args.train_bs-args.labeled_bs-8) * epoch / 75)

        wandb.log({'vat_filter_num':vat_filter_num})
        for step, (input, target) in enumerate(dataloader):
            #measure data loading time
            data_time.update(time.time() - end)
            target = target.cuda()
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target).long()
            varOutput = model(varInput)
            if args.mixup:
                mixed_input,lam = mixup_data_sup(varInput[args.labeled_bs:])
            cls_loss = loss_fn(varOutput[:args.labeled_bs], varTarget[:args.labeled_bs])
            #use vat
            vat_loss_fn = VATLoss(filter_batch=args.vat_filter_batch,filter_num=vat_filter_num)
            if epoch >= args.vat_start_epoch and args.vat_loss_weight > 0.0:
                vat_input = varInput[args.labeled_bs:]
                if args.mixup:
                    vat_input = mixed_input
                vat_loss = args.vat_loss_weight * vat_loss_fn(model, vat_input)
            else:
                vat_loss = 0.0

            if epoch >= args.bnm_start_epoch and args.bnm_loss_weight > 0.0:
                loss_bnm = args.bnm_loss_weight * bnm_loss_improve(varOutput[args.labeled_bs:],vat_filter_num)
            else:
                loss_bnm = 0.0

            if epoch >= args.entropy_start_epoch and args.entropy_loss_weight > 0.0:
                loss_entropy = args.entropy_loss_weight * entropy_y_x(varOutput[args.labeled_bs:])
            else:
                loss_entropy = 0.0

            lossvalue = cls_loss + vat_loss + loss_bnm + loss_entropy
            if step % 200 == 0:
                wandb.log({'train cls loss':cls_loss, 'train vat loss':vat_loss,'train bnm loss':loss_bnm,
                'train loss entropy':loss_entropy, 
                'train total loss':lossvalue})
            
            optimzer.zero_grad()
            lossvalue.backward()
            optimzer.step()
    
    def epochVal(args, wandb, model, epoch, dataloader, loss_fn):
        model.eval()
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        
        for i, (input, target) in enumerate(dataloader):
            varOutput = model(torch.autograd.Variable(input.cuda()))
            varTarget = torch.autograd.Variable(target.cuda())

            losstensor = loss_fn(varOutput, varTarget)
            losstensorMean += losstensor.data 
            lossVal += losstensor.data 
            lossValNorm += 1
        outLoss = lossVal / lossValNorm 
        losstensorMean = losstensorMean / lossValNorm 

        return outLoss, losstensorMean

    def epochTest(args, wandb, logging, model, epoch, dataloader):
        model.eval()
        cudnn.benchmark = True
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()


        for i, (input,target) in enumerate(dataloader):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            bs, n_crops, c, h, w = input.size()
            with torch.no_grad():
                out = model(input.view(-1, c, h, w).cuda())
            outMean = out.view(bs, n_crops, -1).mean(1)
            outPRED = torch.cat((outPRED, outMean.data), 0)

        AUROCs, Accus, Senss, Specs, Pre, F1 = compute_metrics_test(outGT, outPRED, args, competition=True)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        Pre_avg = np.array(Pre).mean()
        F1_avg = np.array(F1).mean()
        #aurocMean = np.array(aurocIndividual).mean()
        logging.info("\nTEST Student: Epoch: {}".format(epoch))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST Pre: {:6f}, TEST F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, Pre_avg, F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))
        wandb.log({'epoch':epoch,'TEST AUROC': AUROC_avg,'TEST Accus':Accus_avg,'TEST Senss':Senss_avg,
        'TEST Specs':Specs_avg,'TEST Pre':Pre_avg,'TEST F1':F1_avg})
        
     
        return


    def epochTest_new(args, wandb, logging, model, epoch, pathDirData):  
        nnClassCount = args.class_num
        trBatchSize = 64
        transResize = 256
        transCrop = 224
        pathFileTest = args.test_file
        timestampLaunch = ''      
        
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transforms_list = []
        transforms_list.append(transforms.RandomResizedCrop(args.crop_size))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform_sequence = transforms.Compose(transforms_list)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transform_sequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (input, target) in enumerate(dataLoaderTest):


            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            #varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
            with torch.no_grad():
                out = model(input.cuda())
            outPRED = torch.cat((outPRED, out.data), 0)

        #aurocIndividual = Trainer.computeAUROC(outGT, outPRED, nnClassCount)
        AUROCs, Accus, Senss, Specs, Pre, F1 = compute_metrics_test(outGT, outPRED, args, competition=True)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        Pre_avg = np.array(Pre).mean()
        F1_avg = np.array(F1).mean()
        #aurocMean = np.array(aurocIndividual).mean()
        logging.info("\nTEST Student without aug: Epoch: {}".format(epoch))
        logging.info("\nTEST AUROC without aug: {:6f}, TEST Accus without aug: {:6f}, TEST Senss without aug: {:6f}, TEST Specs: {:6f}, TEST Pre: {:6f}, TEST F1 without aug: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, Pre_avg, F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(AUROCs)]))
        logging.info("Accus: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Accus)]))
        logging.info("Senss: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Senss)]))
        logging.info("Specs: " + " ".join(["{}:{:.6f}".format(args.class_names[i], v) for i,v in enumerate(Specs)]))
        wandb.log({'TEST AUROC without aug': AUROC_avg,'TEST Accus without aug':Accus_avg,'TEST Senss without aug':Senss_avg,
        'TEST Specs without aug':Specs_avg,'TEST Pre without aug':Pre_avg,'TEST F1 without aug':F1_avg})
        # print ('AUROC mean ', aurocMean)
        # wandb.log({'without test aug AUROC': aurocMean})
        # for i in range (0, len(aurocIndividual)):
        #     print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return
    
    def computeAUROC(dataGT, dataPRED, classCount):
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC





