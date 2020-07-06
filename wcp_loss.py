import re
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cupy as xp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from networks import densenet
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo


def delta_forward(model,logit, x, x_d, w_d, size_list):
    #h = x + x_d
    h = x
    # densenet
    k=1
    h = F.conv2d(h,model.state_dict()['module.densenet121.features.conv0.weight'] + 
                 w_d[size_list[0]:size_list[k]].reshape(model.state_dict()['module.densenet121.features.conv0.weight'].shape),
                 stride=2,padding=3)
    h = F.max_pool2d(F.relu(F.batch_norm(h,model.state_dict()['module.densenet121.features.norm0.running_mean'],
        model.state_dict()['module.densenet121.features.norm0.running_var'],
        model.state_dict()['module.densenet121.features.norm0.weight'],
        model.state_dict()['module.densenet121.features.norm0.bias'],training=True)),kernel_size=3, stride=2, padding=1)
    #denseblock 1
    denselayer_num = [6,12,24,16]
    for i in range(1,5):
        for j in range(1,denselayer_num[i-1]+1):
        #denselayer 1
            h_temp = h
            dense_pre = 'module.densenet121.features.denseblock'+str(i)+'.denselayer'+str(j)
            h = F.relu(F.batch_norm(h,model.state_dict()[dense_pre+'.norm1.running_mean'],
                model.state_dict()[dense_pre+'.norm1.running_var'],
                model.state_dict()[dense_pre+'.norm1.weight'],
                model.state_dict()[dense_pre+'.norm1.bias'],training=True))
            h = F.conv2d(h,model.state_dict()[dense_pre+'.conv1.weight'] + 
                w_d[size_list[k]:size_list[k+1]].reshape(model.state_dict()[dense_pre+'.conv1.weight'].shape))
            k+=1
            h = F.relu(F.batch_norm(h,model.state_dict()[dense_pre+'.norm2.running_mean'],
                model.state_dict()[dense_pre+'.norm2.running_var'],
                model.state_dict()[dense_pre+'.norm2.weight'],
                model.state_dict()[dense_pre+'.norm2.bias'],training=True))
            h = F.conv2d(h,model.state_dict()[dense_pre+'.conv2.weight'] + 
                w_d[size_list[k]:size_list[k+1]].reshape(model.state_dict()[dense_pre+'.conv2.weight'].shape),padding=1)
            h = torch.cat([h_temp,h], 1)
            k+=1
        # transition1
        if i < 4:
            transi_pre = 'module.densenet121.features.transition' + str(i)
            h = F.relu(F.batch_norm(h,model.state_dict()[transi_pre+'.norm.running_mean'],
                model.state_dict()[transi_pre+'.norm.running_var'],
                model.state_dict()[transi_pre+'.norm.weight'],
                model.state_dict()[transi_pre+'.norm.bias'],training=True))
            h = F.conv2d(h,model.state_dict()[transi_pre+'.conv.weight'] + 
                w_d[size_list[k]:size_list[k+1]].reshape(model.state_dict()[transi_pre+'.conv.weight'].shape))
            k+=1
            h = F.avg_pool2d(h,kernel_size=2, stride=2)
    #norm 5
    h = F.relu(F.batch_norm(h,model.state_dict()['module.densenet121.features.norm5.running_mean'],
        model.state_dict()['module.densenet121.features.norm5.running_var'],
        model.state_dict()['module.densenet121.features.norm5.weight'],
        model.state_dict()['module.densenet121.features.norm5.bias'],training=True))
    #h = F.avg_pool2d(h, kernel_size=7, stride=1).view(h.size(0), -1)
    h = F.adaptive_avg_pool2d(h, (1, 1)).view(h.size(0), -1)
    logit_perturb = F.linear(h,model.state_dict()['module.densenet121.classifier.0.weight'],model.state_dict()['module.densenet121.classifier.0.bias'])
    
    return logit_perturb

def wcp_loss_torch(model, x, epsilon=8., dr=0.5, num_simulations=1, xi=1e-6):
    with torch.no_grad():
        _,logit = model(x)
    n_batch = x.shape[0]
    size_list = [0]
    size_sum = 0
    for name in model.state_dict():
        if "bias" in name or "norm" in name or "classifier" in name:
            continue
        size = model.state_dict()[name].numel()
        size_sum += size 
        size_list += [size_sum]
    d = xp.random.normal(size=x.shape)
    d /= (1e-12 + xp.max(xp.abs(d),range(1, len(d.shape)), keepdims=True))
    d /= xp.sqrt(1e-6 + xp.sum(d ** 2, range(1, len(d.shape)), keepdims=True))
    d = from_dlpack(d.toDlpack()).type(torch.FloatTensor).cuda()
    #d = from_dlpack(toDlpack(d)).type(torch.FloatTensor)
    
    d_weight = xp.random.normal(size=size_list[-1])
    d_weight /= (1e-12 + xp.max(xp.abs(d_weight)))
    d_weight /= xp.sqrt(1e-6 + xp.sum(d_weight ** 2))
    d_weight = from_dlpack(d_weight.toDlpack()).type(torch.FloatTensor).cuda()
    #d_weight = from_dlpack(toDlpack(d_weight)).type(torch.FloatTensor)
    
    drop_weight_list = []
    for name in model.state_dict():
        if "bias" in name or "norm" in name or "classifier" in name:
            continue
        drop_weight = xp.random.normal(size=model.state_dict()[name].numel() + 1)
        drop_weight /= (1e-12 + xp.max(xp.abs(drop_weight)))
        drop_weight /= xp.sqrt(1e-6 + xp.sum(drop_weight ** 2))
        drop_weight = from_dlpack(drop_weight.toDlpack()).type(torch.FloatTensor).cuda()
        drop_weight_list += [drop_weight]
    for _ in range(num_simulations):
        x_d = xi * d
        w_d = xi * d_weight
        x_d = Variable(x_d,requires_grad=True)
        w_d = Variable(w_d,requires_grad=True)
        w_d.retain_grad()
        drop_d1_list = []
        drop_d2_list = []
        for ii in range(2):
            drop_d1 = xi * (drop_weight_list[ii][:-1] + drop_weight_list[ii][-1])
            drop_d2 = xi * drop_weight_list[ii][:-1]
            drop_d1_list += [drop_d1]
            drop_d2_list += [drop_d2] 
#         for name, module in model._modules.items():
#             print(module.weight.data)
#             module.weight.data = module.weight + w_d.reshape(module.weight.data.shape)
#             print(module.weight.data)
#             print(name)
#             break
        logit_d = delta_forward(model, logit, x, x_d, w_d, size_list)
        logp_hat = F.log_softmax(logit_d, dim=1)
        kl_loss = F.kl_div(logp_hat,F.softmax(logit),reduction='batchmean')
        kl_loss.backward()
        d_weight = w_d.grad
        model.zero_grad()
        d_weight = d_weight / torch.sqrt(torch.sum(d_weight ** 2))
        d_weight = epsilon * d_weight
        logit_perturb = delta_forward(model, logit, x, x_d, d_weight, size_list)
        model.zero_grad()
        #compute loss
        logp_hat = F.log_softmax(logit_perturb, dim=1)
        kl_loss = F.kl_div(logp_hat,F.softmax(logit),reduction='batchmean')
    return kl_loss

# model = densenet.densenet121(pretrained=True, drop_rate=0)
# image = torch.rand(3,3,256,256)
# out = model(image)
# loss = wcp_loss_torch(model,image,out)
# print("loss:",loss)