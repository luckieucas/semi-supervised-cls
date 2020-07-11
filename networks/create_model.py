import sys
import torch
import torchvision

import pretrainedmodels
import torch.utils.model_zoo as model_zoo


from .models import DenseNet121,DenseNet161,DenseNet121MultiScale



def create_semi_model(args, ema=False):
    """
        create semi supervised model
    """
    # Network definition
    print("create semi supervised model")
    num_class = len(args.class_names)
    net = DenseNet121(out_size=num_class, drop_rate=args.drop_rate)
    if args.task == 'chest':
        net = DenseNet161(out_size=14, drop_rate=args.drop_rate)
    # if args.multi_scale_densenet == 1:
    #     net = DenseNet121MultiScale(drop_rate = args.drop_rate, out_size=num_class)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_full_model(args, ema=False):
    """
    create fully supervised model
    """
    print("create full supervised model")
    num_class = len(args.class_names)
    if args.backbone == 'xception':
        net = pretrainedmodels.__dict__['xception'](num_classes=1000,
                                                    pretrained='imagenet')
        num_fc = net.last_linear.in_features
        net.last_linear = nn.Linear(num_fc, num_class)
    if args.backbone == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
        num_fc = net.classifier.in_features
        net.classifier = torch.nn.Linear(num_fc, num_class)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    return model