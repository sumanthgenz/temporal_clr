import torch
import torchvision
import torchaudio
import numpy
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

#Vector-Wise
def hsphere_norm(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def cosine_similarity(x,y):
    # return torch.zeros(x.size(0))
    nx = hsphere_norm(x)
    ny = hsphere_norm(y)
    cos = torch.mm(x, y.t())/(torch.mm(nx, ny.t()))
    return min(max(cos, -1), 1)

def angular_similarity(x,y):
    cos = cosine_similarity(x, y)
    return 1 - np.arccos(cos)/np.pi

def kl_divergence(x, y):
    # return torch.zeros(x.size(0))
    denom_bound = 0.1
    return sum(x[i] * torch.log(x[i]/(x[i]+denom_bound)) for i in range(x.size(0)))

def l2_distance(x,y):
    # return np.linalg.norm(x-y)
    return hypersphere_norm(x-y)

def vector_couloumb(x, y, pos_pair, k=0.05, q1=1, q2=1):
    force = (k * q1 * q2) / (l2_distance(x, y)**2)
    if pos_pair:
        return -force
    return force

#Batch-Wise
#x, y have dims B * N, where B=bsz and N= latent_feature_dimensions
def infoNCE_loss(x, y, temp=0.5):
    #sim matrix dims = B * B, and hold pairwise (per sample) dot-product similarity for x, y views
    #pos_pairs dims = N, and specify which indices correspond to positive-pair dot products per sample in x
    pos_pairs = torch.arange(x.size(0)).cuda()
    sim_matrix = torch.mm(x, y.t())/temp
    # print(x.shape)
    # print(y.shape)
    # print(sim_matrix.shape)
    # print(pos_pairs.shape)
    loss = torch.nn.CrossEntropyLoss()(sim_matrix, pos_pairs)
    return loss

#x and y normalized to hypersphere 
def particle_contrastive_loss(x, y):
    k = 1/(4*np.pi*1e+1)
    K=1
    q1, q2, = 1, 1

    #dist matrix = 2(1-sim_matrix), negate diagnol (pos pairs), and divide all values by 1, then col_sum and mean
    dist_matrix = 2*(1 - torch.mm(x, y.t()))
    pos_pairs = torch.diag(-2*torch.diag(dist_matrix))
    dist_matrix += pos_pairs
    dist_matrix = torch.div(torch.ones(dist_matrix.shape).cuda(), dist_matrix)

    #when x=y, pos pairs 'inf' and 'nan' should be removed from dist_matrix
    # dist_matrix[dist_matrix == float('inf')] = 0
    # dist_matrix[torch.isnan(dist_matrix)] = 0

    force_loss = k * torch.einsum('ij->j', dist_matrix).mean()
    return force_loss


# lalign and lunif from https://arxiv.org/pdf/2005.10242.pdf
def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

def lunif(x, t=3):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

#alignnment and uniformity from https://github.com/CannyLab/aai/blob/davidmchan/features/hypersphere/aai/utils/torch/metrics.py
def alignment(x, y, alpha=2):
    return 1 - (x - y).norm(dim=1).pow(alpha).mean()

def uniformity(x, t=3):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return 1 - sq_pdist.mul(-t).exp().mean()

#implementation from aai/utils/torch/metrics.py
def _compute_mAP(logits, targets, threshold):  
    return torch.masked_select((logits > threshold) == (targets > threshold), (targets > threshold)).float().mean()

#implementation from aai/utils/torch/metrics.py
def compute_mAP(logits, targets, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)): 
    return torch.mean(torch.stack([_compute_mAP(logits, targets, t) for t in thresholds]))

#implementation from aai/utils/torch/metrics.py
def compute_accuracy(logits, ground_truth, top_k=1):
    """Computes the precision@k for the specified values of k.
    https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
    """
    batch_size = ground_truth.size(0)
    _, pred = logits.topk(top_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(ground_truth.reshape(1, -1).expand_as(pred))
    correct_k = correct[:top_k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)
