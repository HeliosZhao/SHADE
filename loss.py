"""
Loss.py
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from config import cfg
import ipdb

def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        ce_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        ce_weight = None


    print("standard cross entropy")
    criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean',
                                       ignore_index=datasets.ignore_label).cuda()

    criterion_val = nn.CrossEntropyLoss(reduction='mean',
                                       ignore_index=datasets.ignore_label).cuda()
    return criterion, criterion_val

def get_loss_aux(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        ce_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        ce_weight = None

    print("standard cross entropy")
    criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean',
                                    ignore_index=datasets.ignore_label).cuda()

    return criterion

def masked_feat_dist(f1, f2, mask=None):
    '''
    f1: seg model features 2B,C,H,W
    f2: imgnet model features 2B,C,H,W
    '''

    feat_diff = F.mse_loss(f1, f2, reduction='none') # B,C,H,W
    pw_feat_dist = feat_diff.mean(1)

    if mask is not None:
        pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
    return torch.mean(pw_feat_dist)

imnet_feature_dist_lambda=0.005
imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
imnet_feature_dist_scale_min_ratio=0.75

def calc_feat_dist(gt, feat_imnet, feat, num_classes):
    # lay = -1
    '''
    gt B,H,W
    feat_imnet  B,C,H,W
    feat  B,C,H,W
    '''
    if imnet_feature_dist_classes is not None:
        fdclasses = torch.tensor(imnet_feature_dist_classes, device=gt.device)
        scale_factor = gt.shape[-1] // feat.shape[-1]
        gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                            imnet_feature_dist_scale_min_ratio,
                                            num_classes,
                                            255).long().detach()
        # ipdb.set_trace()
        fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1) # ...None == unsqueeze(-1)
        if not fdist_mask.sum():
            return torch.tensor(0., device=gt.device)
        feat_dist = masked_feat_dist(feat, feat_imnet, fdist_mask)

    else:
        feat_dist = masked_feat_dist(feat, feat_imnet)
    
    if torch.isnan(feat_dist):
        ipdb.set_trace()

    return feat_dist



def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor > 1
    bs, orig_h, orig_w = gt.shape
    # assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out, num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out