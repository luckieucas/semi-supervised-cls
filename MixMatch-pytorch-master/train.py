from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wideresnet as models
from models.models import DenseNet121,DenseNet169,DenseNet201
import dataset.cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
import wandb
from metrics import compute_metrics_test
from config import CLASS_NAMES_DICTS,CLASS_NUM_DICTS
from losses import cross_entropy_loss

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=1400,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
# data
parser.add_argument('--task', type=str, default='skin', help='which task')
parser.add_argument('--exp', type=str, default='model-test', help='model tag')
parser.add_argument('--root_path', type=str, default='../../dataset/skin/training_data/',
                    help='dataset root dir')
parser.add_argument('--train_file', type=str, default='../../dataset/skin/training.txt')
parser.add_argument('--test_file', type=str, default='../../dataset/skin/testing.txt')
parser.add_argument('--val_file', type=str, default='../../dataset/skin/validation.txt')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
class_names = CLASS_NAMES_DICTS[args.task]
args.class_num_dict = CLASS_NUM_DICTS[args.task]
args.class_names = class_names


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    #--------------------training and validation data SETTINGS: DATA TRANSFORMS
    
    wandb.init(project=args.task + '-mixmatch',name=args.task+args.exp)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # # train and validation data transform
    # trainval_transforms_list = []
    # trainval_transforms_list.append(transforms.RandomResizedCrop(args.crop_size))
    # trainval_transforms_list.append(transforms.RandomHorizontalFlip())
    # trainval_transforms_list.append(transforms.ToTensor())
    # trainval_transforms_list.append(normalize)
    # trainval_transform_sequence = transforms.Compose(trainval_transforms_list)

    # #test data transfrom
    # test_transforms_list = []
    # test_transforms_list.append(transforms.Resize(args.resize))
    # test_transforms_list.append(transforms.ToTensor())
    # test_transform_sequence = transforms.Compose(test_transforms_list)
    print(f'==> Preparing cifar10')
    transform_train = transforms.Compose([
        #dataset.RandomPadandCrop(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    # transform_train = transforms.Compose([
    #     transforms.Resize(224),
    #     dataset.RandomPadandCrop(224),
    #     dataset.RandomFlip(),
    #     dataset.ToTensor(),
    # ])

    # transform_val = transforms.Compose([
    #     transforms.Resize(224),
    #     dataset.ToTensor(),
    # ])
    #train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('./data', args.n_labeled, transform_train=transform_train, transform_val=transform_val)
    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skin(args, args.n_labeled, 
            transform_train=transform_train, transform_val=transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = DenseNet121(7,True)
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = cross_entropy_loss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'noisy-cifar-10'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        wandb.log({'epoch':epoch+1,'lr':state['lr']})

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)
        wandb.log({'train_loss':train_loss,'val_loss':val_loss,'test_loss':test_loss,
            'train_acc':train_acc,'val_acc':val_acc,'test_acc':test_acc})

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        #targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            



            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outGT = torch.cat((outGT, targets), 0)
            outputs = model(inputs)
            outPRED = torch.cat((outPRED, outputs.data), 0)
            targets = targets.long()
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            #top1.update(prec1.item(), inputs.size(0))
            #top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        # top1=top1.avg,
                        # top5=top5.avg,
                        top1=0,
                        top5=0
                        )
            bar.next()
        bar.finish()
        # measure auc F1
    AUROCs, Accus, Senss, Specs, Pre, F1 = compute_metrics_test(outGT, outPRED, args, competition=True)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    Pre_avg = np.array(Pre).mean()
    F1_avg = np.array(F1).mean()
    wandb.log({'epoch':epoch,'TEST AUROC': AUROC_avg,'TEST Accus':Accus_avg,'TEST Senss':Senss_avg,
        'TEST Specs':Specs_avg,'TEST Pre':Pre_avg,'TEST F1':F1_avg})
    return (losses.avg, Accus_avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            param = param.float()
            ema_param = ema_param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()
