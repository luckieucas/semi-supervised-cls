import os
import sys
import time

import numpy as np
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import DenseNet121,DenseNet169,DenseNet201


class Trainer():
    # train the classification model

    def train(args,class_num):
        """
        train model
        """
        #-------------------SETTINGs: NETWORK ARCHITECTURE
        model = DenseNet121(class_num,True)
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
        test_transforms_list.append(transforms.Resize(transResize))
        test_transforms_list.append(transforms.TenCrop(transCrop))
        test_transforms_list.append(transforms.Lambda(lambda crops: \
            torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms_list.append(transforms.Lambda(lambda crops: \
            torch.stack([normalize(crop) for crop in crops])))
        test_transform_sequence = transforms.Compose(test_transforms_list)


        #--------------------SETTINGS: BUILD DATASET
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
         transform=trainval_transform_sequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal,
         transform=trainval_transform_sequence)
         datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal,
         transform=test_transform_sequence)

        #--------------------SETTINGS: SEMI-SUPERVISED
        train_dataset_num = len(datasetTrain)
        #labeled_num = int(train_dataset_num*args.labeled_rate)
        print("labeled_num:",labeledNum)
        labeled_idxs = list(range(labeledNum))
        unlabeled_idxs = list(range(labeledNum, train_dataset_num))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, trBatchSize,
                                            trBatchSize-labeledBatchSize)
        dataLoaderTrain = DataLoader(dataset=datasetTrain,batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)


