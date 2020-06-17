import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import contextlib

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
#CLASS_NUM = [1113, 6705, 514, 327, 1099, 115, 142]
#CLASS_NUM = [11559, 2776, 13317, 19894, 5782, 6331, 1431, 5302, 4667, 2303, 2516, 1686, 3385, 227] # chest
#CLASS_WEIGHT = torch.Tensor([10000/i for i in CLASS_NUM]).cuda()
#CLASS_WEIGHT = torch.Tensor([81176/i for i in CLASS_NUM]).cuda() #chest
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[1], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                _,pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            _,pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self, args):
        class_num = args.class_num
        class_weight = torch.Tensor([sum(class_num)/i for i in class_num]).cuda()
        self.base_loss = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        # print("output_softmax:",output_softmax)
        # print("target: ",target)
        return self.base_loss(output_softmax, target.long())


def bnm_loss(out_logits):
    """
    compute batch nuclear-norm maximization loss
    """
    A = F.softmax(out_logits, dim=1)
    L_bnm = -torch.norm(A,'nuc')
    return L_bnm

def bnm_loss_improve(out_logits):
    """
    compute batch nuclear-norm maximization loss refinement
    """
    A = F.softmax(out_logits, dim=1) + 0.000001
    B = -1.0 * A *torch.log(A)
    C = B.sum(dim=1)
    index = C.argsort(descending=True)[:12]
    D = A[index]
    L_bnm = -torch.norm(D,'nuc')
    return L_bnm


def get_UncertaintyLoss(method):
    assert method in METHODS
    
    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()
    
    if method == 'U-MultiClass':
        return Loss_MultiClass()

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits, args):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    class_num = args.class_num
    class_weight = torch.Tensor([sum(class_num)/i for i in class_num]).cuda()
    mse_loss = (input_softmax-target_softmax)**2 * class_weight
    return mse_loss

def cam_attention_map(activations, channel_weight):
    # activations 48*49*1024
    # channel_weight 48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    attention = attention.permute(1,0,2)
    attention = torch.sum(attention, -1)
    attention = torch.reshape(attention, (48, 7, 7))

    return attention

def cam_activation(batch_feature, channel_weight):
    # batch_feature = batch_feature.permute(0,2,3,1)#48 7 7 1024
    # activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))#48*49*1024
    
    # attention = activations.permute(1,0,2)#.mul(channel_weight)#49*48*1024
    # attention = attention.permute(1,2,0)#48*1024*49
    # attention = F.softmax(attention, -1)#48*1024*49

    # activations2 = activations.permute(0, 2, 1) #48 1024 49
    # activations2 = activations2 * attention 
    # activations2 = torch.sum(activations2, -1)#48*1024
    batch_feature = batch_feature.permute(0,2,3,1)
    #48*49*1024
    activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))
    
    #49*48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    #48*49*1024
    attention = attention.permute(1,0,2)
    #48*49
    attention = torch.sum(attention, -1)
    attention = F.softmax(attention, -1)

    activations2 = activations.permute(2, 0, 1) #1024*48*49
    activations2 = activations2 * attention 
    activations2 = torch.sum(activations2, -1) #1024*48
    #48 1024 
    activations2 = activations2.permute(1,0)

    return activations2

# def relation_mse_loss_cam(activations, ema_activations, model, label):
#     """Takes softmax on both sides and returns MSE loss

#     Note:
#     - Returns the sum over all examples. Divide by the batch size afterwards
#       if you want the mean.
#     - Sends gradients to inputs but not the targets.
#     """
#     weight = model.module.densenet121.classifier[0].weight
#     #48*1024
#     channel_weight = label.mm(weight)

#     activations = cam_activation(activations.clone(), channel_weight)
#     ema_activations = cam_activation(ema_activations.clone(), channel_weight)
    
#     assert activations.size() == ema_activations.size()

#     activations = torch.reshape(activations, (activations.shape[0], -1))
#     ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

#     similarity = activations.mm(activations.t())
#     norm_similarity = similarity / torch.norm(similarity, p=2)

#     ema_similarity = ema_activations.mm(ema_activations.t())
#     norm_ema_similarity = ema_similarity / torch.norm(ema_similarity, p=2)

#     similarity_mse_loss = (norm_similarity-norm_ema_similarity)**2
#     return similarity_mse_loss


def relation_mse_loss_cam(activations, ema_activations, model, label):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = model.module.densenet121.classifier[0].weight
    #48*1024
    channel_weight = label.mm(weight)

    activations = cam_activation(activations.clone(), channel_weight)
    ema_activations = cam_activation(ema_activations.clone(), channel_weight)
    
    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss

def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss


def feature_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    # similarity = activations.mm(activations.t())
    # norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    # norm_similarity = similarity / norm

    # ema_similarity = ema_activations.mm(ema_activations.t())
    # ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    # ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (activations-ema_activations)**2
    return similarity_mse_loss


def sigmoid_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    mse_loss = loss_fn(input_softmax, target_softmax)
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
