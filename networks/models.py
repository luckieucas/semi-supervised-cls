# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from . import densenet


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, drop_rate=0):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            #nn.Sigmoid()
        )
            
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        out = self.densenet121.classifier(out)
            
        return self.activations, out

class DenseNet161(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size,drop_rate=0):
        super(DenseNet161, self).__init__()
        self.densenet161 = densenet.densenet161(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            #nn.Sigmoid()
        )

        
        # Official init from torch repo.
        for m in self.densenet161.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet161.features(x)
        out = F.relu(features, inplace=True)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        
        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        out = self.densenet161.classifier(out)
            
        return self.activations, out

class DenseNet121MultiScale(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size,drop_rate=0):
        super(DenseNet121MultiScale, self).__init__()
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            #nn.Sigmoid()
        )

        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.densenet121.features.conv0(x)
        x = self.densenet121.features.norm0(x)
        x = self.densenet121.features.relu0(x)
        x = self.densenet121.features.pool0(x)
        x = self.densenet121.features.denseblock1(x)
        fea1 = self.densenet121.features.transition1(x)
        fea2 = self.densenet121.features.denseblock2(fea1)
        fea2 = self.densenet121.features.transition2(fea2)
        fea3 = self.densenet121.features.denseblock3(fea2)
        fea3 = self.densenet121.features.transition3(fea3)
        features = self.densenet121.features.denseblock4(fea3)
        features = self.densenet121.features.norm5(features)
        out = F.relu(features, inplace=True) 
        fea_out1 = F.adaptive_avg_pool2d(fea1, (1, 1)).view(fea2.size(0), -1)
        fea_out2 = F.adaptive_avg_pool2d(fea2, (1, 1)).view(fea2.size(0), -1)
        fea_out3 = F.adaptive_avg_pool2d(fea3, (1, 1)).view(fea3.size(0), -1)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        out = self.densenet121.classifier(out)
            
        return [fea_out1,fea_out2,fea_out3,self.activations], out
