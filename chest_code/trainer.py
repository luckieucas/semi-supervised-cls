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
from sklearn.metrics.ranking import roc_auc_score

from models import DenseNet121,DenseNet169,DenseNet201
from DatasetGenerator import DatasetGenerator, TwoStreamBatchSampler
from losses import VATLoss
from utils import AverageMeter


class Trainer():
    # train the classification model

    def train(args,wandb):
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
        loss = torch.nn.BCELoss(size_average=True)

        #------Load checkpoint
        if args.checkpoint != None:
            modelCheckpoint = torch.load(args.checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
        
        #-----TRAIN MODEL
        for epoch in range(args.start_epoch, args.epochs):
            Trainer.epochTrain(args, wandb, model, epoch, dataLoaderTrain, optimizer, scheduler)
            Trainer.epochVal(args, wandb, model, epoch, dataLoaderVal)
            Trainer.epochTest(args, wandb, model, epoch, dataLoaderTest)

    def epochTrain(args, wandb, model, epoch, dataloader, optimzer, scheduler):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()
        cls_loss = AverageMeter()
        end = time.time()


        for step, (input, target) in enumerate(dataloader):
            #measure data loading time
            data_time.update(time.time() - end)
            target = target.cuda()
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(taraget)
            varOutput = model(varInput)

            cls_loss = loss(varOutput[:args.labeled_bs], varTarget[:args.labeled_s])
            #use vat
            vat_loss_fn = VATLoss()
            if epoch >= args.vat_start_epoch and args.vat_loss_weight > 0.0:
                vat_loss = vat_loss_fn(model, varInput[args.labeled_bs:])
            else:
                vat_loss = 0.0
            lossvalue = cls_loss + vat_loss
            if step % 200 == 0:
                wandb.log({'train cls loss':cls_loss, 'train vat loss':vat_loss, 'train total loss':lossvalue})
            
            optimizer.zero_grad()
            lossvalue.backward()
            optimzer.step()
    
    def epochVal(args, wandb, model, epoch, dataloader):
        model.eval()
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        
        for i, (input, target) in enumerate(dataloader):
            varOutput = model(torch.autograd.Variable(input.cuda()))
            varTarget = torch.autograd.Variable(target.cuda())

            losstensor = loss(varOutput, varTarget)
            lossstensorMean += losstensor.data 
            lossVal += losstensor.data 
            lossValNorm += 1
        outLoss = lossVal / lossValNorm 
        losstensorMean = losstensorMean / lossValNorm 

        return outLoss, losstensorMean

    def epochTest(args, wandb, model, epoch, dataloader):
        model.eval()
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        cudnn.benchmark = True
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        model.eval()

        for i, (input,target) in enumerate(dataloader):
            target = target.cuda()
            outGT = toorch.cat((outGT, target), 0)
            bs, n_crops, c, h, w = input.size()
            with torch.no_grad():
                out = model(input.view(-1, c, h, w).cuda())
            outMean = out.view(bs, n_crops, -1).mean(1)
            outPRED = torch.cat((outPRED, outMean.data), 0)

        aurocIndividual = Trainer.computeAUROC(outGT, outPRED, args.class_num)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        wandb.log({'AUROC': aurocMean})
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return
    
    def computeAUROC(dataGT, dataPRED, classCount):
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC





