import re
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import chainer
import chainer.functions as F_chainer
from chainer import Variable as V_chain
import cupy as xp
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from source.chainer_functions.loss import distance
from networks import densenet
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo


def delta_forward(model,logit, x, x_d, w_d, size_list):
    #h = x + x_d
    h = x
    # densenet
    k=1
    h = F.conv2d(h,model.state_dict()['features.conv0.weight'] + 
                 w_d[size_list[0]:size_list[k]].reshape(model.state_dict()['features.conv0.weight'].shape))
    h = F.max_pool2d(F.relu(F.batch_norm(h,model.state_dict()['features.norm0.running_mean'],
        model.state_dict()['features.norm0.running_var'],
        model.state_dict()['features.norm0.weight'],
        model.state_dict()['features.norm0.bias'])),kernel_size=3, stride=2, padding=1)
    #denseblock 1
    denselayer_num = [6,12,24,16]
    for i in range(1,5):
        for j in range(1,denselayer_num[i-1]+1):
        #denselayer 1
            h_temp = h
            dense_pre = 'features.denseblock'+str(i)+'.denselayer'+str(j)
            h = F.relu(F.batch_norm(h,model.state_dict()[dense_pre+'.norm1.running_mean'],
                model.state_dict()[dense_pre+'.norm1.running_var'],
                model.state_dict()[dense_pre+'.norm1.weight'],
                model.state_dict()[dense_pre+'.norm1.bias']))
            h = F.conv2d(h,model.state_dict()[dense_pre+'.conv1.weight'] + 
                w_d[size_list[k]:size_list[k+1]].reshape(model.state_dict()[dense_pre+'.conv1.weight'].shape))
            k+=1
            h = F.relu(F.batch_norm(h,model.state_dict()[dense_pre+'.norm2.running_mean'],
                model.state_dict()[dense_pre+'.norm2.running_var'],
                model.state_dict()[dense_pre+'.norm2.weight'],
                model.state_dict()[dense_pre+'.norm2.bias']))
            h = F.conv2d(h,model.state_dict()[dense_pre+'.conv2.weight'] + 
                w_d[size_list[k]:size_list[k+1]].reshape(model.state_dict()[dense_pre+'.conv2.weight'].shape),padding=1)
            h = torch.cat([h_temp,h], 1)
            k+=1
        # transition1
        if i < 4:
            transi_pre = 'features.transition' + str(i)
            h = F.relu(F.batch_norm(h,model.state_dict()[transi_pre+'.norm.running_mean'],
                model.state_dict()[transi_pre+'.norm.running_var'],
                model.state_dict()[transi_pre+'.norm.weight'],
                model.state_dict()[transi_pre+'.norm.bias']))
            h = F.conv2d(h,model.state_dict()[transi_pre+'.conv.weight'] + 
                w_d[size_list[k]:size_list[k+1]].reshape(model.state_dict()[transi_pre+'.conv.weight'].shape))
            k+=1
            h = F.avg_pool2d(h,kernel_size=2, stride=2)
    #norm 5
    h = F.relu(F.batch_norm(h,model.state_dict()['features.norm5.running_mean'],
        model.state_dict()['features.norm5.running_var'],
        model.state_dict()['features.norm5.weight'],
        model.state_dict()['features.norm5.bias']))
    #h = F.avg_pool2d(h, kernel_size=7, stride=1).view(h.size(0), -1)
    h = F.adaptive_avg_pool2d(h, (1, 1)).view(h.size(0), -1)
    logit_perturb = F.linear(h,model.state_dict()['classifier.weight'],model.state_dict()['classifier.bias'])
    


    h = F.relu(F.max_pool2d(F.conv2d(h,model.state_dict()['conv1.weight']+
                                     w_d.reshape(model.state_dict()['conv1.weight'].shape),
                                     model.state_dict()['conv1.bias']),2))
    h = F.relu(F.max_pool2d(F.conv2d(h,model.state_dict()['conv2.weight'],
                                     model.state_dict()['conv2.bias']),2))
    h = h.view(-1,320)
    h = F.relu(F.linear(h,model.state_dict()['fc1.weight'],model.state_dict()['fc1.bias']))
    #x = F.dropout(x, training=self.training)
    logit_perturb = F.linear(h,model.state_dict()['fc2.weight'],model.state_dict()['fc2.bias'])
    p1 = model(x)
    print("p1:",p1)
    print("p2:",F.softmax(logit_perturb))
    print("softmax:",F.softmax(logit))
    #kl_loss = distance(logit.detach().numpy(), logit_perturb.detach().numpy())
    logp_hat = F.log_softmax(logit_perturb, dim=1)
    kl_loss = F.kl_div(logp_hat,F.softmax(logit),reduction='batchmean')
    print("kl loss torch:",kl_loss)
    kl_loss.backward()
    d,d_weight = x_d.grad,w_d.grad
    model.zero_grad()
    print("d:",d)
    print("d weight:",d_weight)
    return d,d_weight

def wcp_loss_torch(model, x, logit, epsilon=8., dr=0.5, num_simulations=1, xi=1e-6):
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
    d = from_dlpack(toDlpack(d)).type(torch.FloatTensor)
    
    d_weight = xp.random.normal(size=size_list[-1])
    d_weight /= (1e-12 + xp.max(xp.abs(d_weight)))
    d_weight /= xp.sqrt(1e-6 + xp.sum(d_weight ** 2))
    d_weight = from_dlpack(toDlpack(d_weight)).type(torch.FloatTensor)
    
    drop_weight_list = []
    for name in model.state_dict():
        if "bias" in name or "norm" in name or "classifier" in name:
            continue
        drop_weight = xp.random.normal(size=model.state_dict()[name].numel() + 1)
        drop_weight /= (1e-12 + xp.max(xp.abs(drop_weight)))
        drop_weight /= xp.sqrt(1e-6 + xp.sum(drop_weight ** 2))
        drop_weight = from_dlpack(toDlpack(drop_weight)).type(torch.FloatTensor)
        drop_weight_list += [drop_weight]
    print("d weight shape:",d_weight.shape)
    print("d shape:",d.shape)
    for _ in range(num_simulations):
        x_d = xi * d
        w_d = xi * d_weight
        print(w_d.type())
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
        logit_d,d = delta_forward(model, logit, x, x_d, w_d, size_list)
        logit_d,d = delta_forward_chainer(model, logit, x, x_d, w_d, size_list)
        break
        logit_drop1 = drop_forward(model, x, drop_d1_list)
        logit_drop2 = drop_forward(model, x, drop_d2_list)       
        #kl_loss = distance(logit.data, logit_d)
        kl_loss = F.kl_div(F.softmax(logit),F.softmax(logit_d))
#         kl_loss_drop1 = distance(logit.data, logit_drop1)
#         kl_loss_drop2 = distance(logit.data, logit_drop2)
        kl_loss_drop1 = F.kl_div(F.softmax(logit),F.softmax(logit_drop1))
        kl_loss_drop2 = F.kl_div(F.softmax(logit),F.softmax(logit_drop2))
        kl_loss.backward()
        d,d_weight = x_d.grad,w_d.grad
        model.zero_grad()
        print("d:",d)
        print("d weight:",d_weight)
        break
        d, d_weight = chainer.grad([kl_loss], [x_d, w_d], enable_double_backprop=False)
        d = d / F.sqrt(F.sum(d ** 2, tuple(range(1, len(d.shape))), keepdims=True))
        d_weight = d_weight / F.sqrt(F.sum(d_weight ** 2))
        
        layer1_drop1, layer4_drop1, layer7_drop1 = chainer.grad([kl_loss_drop1], drop_d1_list, enable_double_backprop=False)
        layer1_drop2, layer4_drop2, layer7_drop2 = chainer.grad([kl_loss_drop2], drop_d2_list, enable_double_backprop=False)
        
        layer1_drop2 = F.reshape(F.sum(layer1_drop2), (1, 1))
        layer4_drop2 = F.reshape(F.sum(layer4_drop2), (1, 1))
        layer7_drop2 = F.reshape(F.sum(layer7_drop2), (1, 1))
        drop_weight_list[0] = F.flatten(F.concat([F.reshape(layer1_drop1, (-1, 1)), layer1_drop2], axis=0))
        drop_weight_list[1] = F.flatten(F.concat([F.reshape(layer4_drop1, (-1, 1)), layer4_drop2], axis=0))
        drop_weight_list[2] = F.flatten(F.concat([F.reshape(layer7_drop1, (-1, 1)), layer7_drop2], axis=0))
        drop_weight_list = [drop_weight_list[ii] / F.sqrt(F.sum(drop_weight_list[ii] ** 2)) for ii in range(3)]

    return d,d_weight,drop_weight_list

model = densenet.densenet121(pretrained=True, drop_rate=0)
image = torch.rand(3,3,256,256)
out = model(image)
_,_,output = wcp_loss_torch(model,image,out)