import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import os

def dice_loss(input, target):
    eps = 1e-5
    # logits = torch.argmax(input,dim=1,keepdim = True)
    iflat = input.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + eps) /
    (iflat.sum() + tflat.sum() + eps)).mean()

def dice_loss_class(input, target, cutoff = 0.5):
    logits = torch.argmax(input,dim=1,keepdim = True)
    logits_myo = logits > cutoff
    target_myo = target > cutoff
    loss = dice_loss(logits_myo,target_myo)
    return loss

def dice_loss_TPM(input, target):
    logits = F.softmax(input,dim=1)
    target_all = F.one_hot(target,4)
    logits_LVmyo = logits[:,3,...] > 0.5
    target_LVmyo  = target_all[...,3]
    logits_RVmyo = logits[:,2,...] > 0.5
    target_RVmyo  = target_all[...,2]
    logits_BP = logits[:,1,...] > 0.5
    target_BP  = target_all[...,1]
    dice_sum = dice_loss(logits_LVmyo,target_LVmyo) + dice_loss(logits_RVmyo,target_RVmyo) + dice_loss(logits_BP,target_BP)
    return dice_sum    

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def HausdorffLoss(set1, set2):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Tensor where each row is an N-dimensional point.
    :param set2: Tensor where each row is an N-dimensional point.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """
    assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
    assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

    assert set1.size()[1] == set2.size()[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.size()[1], set2.size()[1])

    d2_matrix = cdist(set1, set2)
    # Modified Chamfer Loss
    term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
    term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

    res = term_1 + term_2

    return res

def AveragedHausdorffLoss(set1, set2, cutoff = 1.5):  
    ## Nslice x Height x Wideth
    mask = torch.argmax(set1,dim=1,keepdim =True)
    mask = mask > cutoff
    gt_mask = set2 > cutoff
    pred = torch.squeeze(mask).type(torch.float32) 
    gt = torch.squeeze(gt_mask).type(torch.float32) 

    batch_size = pred.shape[0]
    HDlosses = []
    if len(pred.shape) > 2:
        for b in range(batch_size):
            pred_b = pred[b, :, :]
            gt_b = gt[b, :, :]
            HDloss = HausdorffLoss(pred_b,gt_b)
            HDlosses.append(HDloss)
    else:
        HDloss = HausdorffLoss(pred,gt)
        HDlosses.append(HDloss)
    HDlosses = torch.stack(HDlosses)  
    res = HDlosses.mean()
    return res