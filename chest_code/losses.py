import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import contextlib
from mixup import mixup_data_sup


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


def sharpen(x, T=0.5):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1, dis='kl',filter_batch=False,filter_prob=False,
                filter_num=8, is_sharpen=False, mixup = False):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.dis = dis
        self.filter_batch = filter_batch
        self.filter_num = filter_num
        self.is_sharpen = is_sharpen
        self.filter_batch_prob = filter_prob
        self.mixup = mixup

    def forward(self, model, x):
        if self.mixup:
            x,lam = mixup_data_sup(x)
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
            if self.is_sharpen:
                pred = sharpen(pred)
        if self.filter_batch_prob:
            A = pred + 1e-6
            B = torch.max(A, 1)[0]
            pred = A[B>0.3]
            if len(pred) == 0:
                return 0.0
        # if self.filter_batch:
        #     A = pred + 0.000001
        #     B = -1.0 * A *torch.log(A)
        #     C = B.sum(dim=1)
        #     index = C.argsort(descending=True)[-1*self.filter_num:]
        #     pred = pred[index]
        #     x = x[index]
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                if self.dis=='mse':
                    pred_hat_softmax = F.softmax(pred_hat, dim=1)
                    if self.is_sharpen:
                        pred_hat_softmax = sharpen(pred_hat_softmax)
                    adv_distance = F.mse_loss(pred_hat_softmax,pred)
                elif self.dis=='mae':
                    pred_hat_softmax = F.softmax(pred_hat, dim=1)
                    if self.is_sharpen:
                        pred_hat_softmax = sharpen(pred_hat_softmax)
                    adv_distance = F.l1_loss(pred_hat_softmax,pred)
                else:
                    logp_hat = F.log_softmax(pred_hat, dim=1)
                    adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            if self.dis=='mse':
                pred_hat_softmax = F.softmax(pred_hat, dim=1)
                lds = F.mse_loss(pred_hat_softmax,pred)
            elif self.dis=='mae':
                pred_hat_softmax = F.softmax(pred_hat, dim=1)
                lds = F.l1_loss(pred_hat_softmax,pred)
            else:
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
        class_num = args.class_num_dict
        class_weight = torch.Tensor([sum(class_num)/i for i in class_num]).cuda()
        self.base_loss = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        #output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        # print("output_softmax:",output_softmax)
        # print("target: ",target)
        return self.base_loss(output, target.long())


def bnm_loss(out_logits,filter_prob=False,filter_batch=False,filter_batch_num=-1):
    """
    compute batch nuclear-norm maximization loss
    """
    A = F.softmax(out_logits, dim=1)
    if filter_prob:
        B = torch.max(A, 1)[0]
        A = A[B<0.6]
        B = torch.max(A, 1)[0]
        A = A[B>0.25]
    if len(A) == 0:
        return 0.0
    if filter_batch:
        A = A + 1e-6
        B = -1.0 * A *torch.log(A)
        C = B.sum(dim=1)
        index = C.argsort(descending=True)[-1*filter_batch_num:]
        A = A[index]
    L_bnm = -torch.norm(A,'nuc')/A.shape[0]
    return L_bnm

def bnm_loss_improve(out_logits,filter_num=16):
    """
    compute batch nuclear-norm maximization loss refinement
    """
    A = F.softmax(out_logits, dim=1) + 0.000001
    B = -1.0 * A *torch.log(A)
    C = B.sum(dim=1)
    index = C.argsort(descending=True)[:filter_num]
    D = A[index]
    L_bnm = -torch.norm(D,'nuc')/D.shape[0]
    return L_bnm



def entropy_y_x(logit):
    soft_logit = F.softmax(logit, dim=1)
    return -torch.mean(torch.sum(soft_logit* F.log_softmax(logit,dim=1), dim=1))

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