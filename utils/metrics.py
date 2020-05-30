# encoding: utf-8
import sys
import numpy as np
import torch
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics.ranking import roc_auc_score
sys.path.append("..")
#N_CLASSES = 7
#CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']


# N_CLASSES = 14
# CLASS_NAMES = [
#         'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
#         'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
#                   ]

def compute_AUCs(gt, pred, args, competition=True):
    """
    Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
        'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
    if args.task == 'chest':
        CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
         'Pneumothorax','Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'] 
    indexes = range(len(CLASS_NAMES))
    
    for i in indexes:
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            AUROCs.append(0)
    return AUROCs


def compute_metrics(gt, pred, args, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """
    CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
        'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
    if args.task == 'chest':
        CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
         'Pneumothorax','Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    if args.task == 'hip':
        CLASS_NAMES = ['Normal', 'ONFH_I', 'ONFH_II']
    # compute true accu
    correct = list(0. for i in range(len(CLASS_NAMES)))
    total = list(0. for i in range(len(CLASS_NAMES)))
    pred_true_acc = torch.max(pred, 1)[1]
    gt_true_acc = torch.max(gt, 1)[1]
    res = pred_true_acc == gt_true_acc
    train_correct = (pred_true_acc == gt_true_acc).sum()
    for label_idx in range(len(pred_true_acc)):
        label_single = gt_true_acc[label_idx]
        correct[label_single] += res[label_idx].item()
        total[label_single] += 1
    # compute true accu
    print("total:",total)
    print("correct:",correct)

    pred = torch.max(pred, 1)[1].unsqueeze(1).cpu() #add
    pred = torch.zeros(len(pred), len(CLASS_NAMES)).scatter_(1, pred, 1)
    AUROCs, Accus, Senss, Recas, Specs = [], [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    # if cfg.uncertainty == 'U-Zeros':
    #     gt_np[np.where(gt_np==-1)] = 0
    # if cfg.uncertainty == 'U-Ones':
    #     gt_np[np.where(gt_np==-1)] = 1
    pred_np = pred.numpy()
    
    
    THRESH = 0.18
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    #indexes = range(n_classes)
    
#     pdb.set_trace()

    indexes = range(len(CLASS_NAMES))
    
    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)
        
        try:
            #Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i])))
            Accus.append(correct[i]/total[i])
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Senss.append(sensitivity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing precision for {}.'.format(i))
            Senss.append(0)
        

        try:
            Specs.append(specificity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Specs.append(0)
    
    return AUROCs, Accus, Senss, Specs

def compute_metrics_test(gt, pred, args, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    AUROCs, Accus, Senss, Specs, Pre, F1 = [], [], [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    # if cfg.uncertainty == 'U-Zeros':
    #     gt_np[np.where(gt_np==-1)] = 0
    # if cfg.uncertainty == 'U-Ones':
    #     gt_np[np.where(gt_np==-1)] = 1
    pred_np = pred.cpu().detach().numpy()
    THRESH = 0.0
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    #indexes = range(n_classes)
    
#     pdb.set_trace()
    CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
        'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
    if args.task == 'chest':
        CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
         'Pneumothorax','Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'] 
    if args.task == 'hip':
        CLASS_NAMES = ['Normal','ONFH I','ONFH II']
    indexes = range(len(CLASS_NAMES))
    
    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)
        
        try:
            Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Senss.append(sensitivity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing precision for {}.'.format(i))
            Senss.append(0)
        

        try:
            Specs.append(specificity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Specs.append(0)

        try:
            Pre.append(precision_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Pre.append(0)
    
        try:
            F1.append(f1_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            F1.append(0)
    
    return AUROCs, Accus, Senss, Specs, Pre, F1
