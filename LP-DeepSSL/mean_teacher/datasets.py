# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import torchvision.transforms as transforms

from . import data
from .utils import export
import torch

import os
import pdb

@export
def cifar10(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))


    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]

    data_dir = 'data-local/images/cifar/cifar10/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def cifar100(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616]) # should we use different stats - do this
    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/cifar/cifar100/by-image'

    print("Using CIFAR-100 from", data_dir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }



@export
def miniimagenet(isTwice=True):
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))


    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/miniimagenet'
    

    print("Using mini-imagenet from", data_dir)


    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }

@export
def skin(isTwice=True):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    trainval_transforms_list = []
    trainval_transforms_list.append(transforms.RandomResizedCrop(224))
    trainval_transforms_list.append(transforms.RandomHorizontalFlip())
    trainval_transforms_list.append(transforms.ToTensor())
    trainval_transforms_list.append(normalize)
    trainval_transform_sequence = transforms.Compose(trainval_transforms_list)
    if isTwice:
        train_transformation = data.TransformTwice(trainval_transform_sequence)
    else:
        train_transformation = data.TransformOnce(trainval_transform_sequence)

    test_transforms_list = []
    test_transforms_list.append(transforms.Resize(256))
    test_transforms_list.append(transforms.TenCrop(224))
    test_transforms_list.append(transforms.Lambda(lambda crops: \
        torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    test_transforms_list.append(transforms.Lambda(lambda crops: \
        torch.stack([normalize(crop) for crop in crops])))
    test_transform_sequence = transforms.Compose(test_transforms_list)
    eval_transformation = test_transform_sequence

    myhost = os.uname()[1]

    data_dir = 'data-local/images/cifar/cifar10/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 7
    }