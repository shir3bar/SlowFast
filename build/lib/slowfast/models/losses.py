#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.nn as nn

class WeightedCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, reduction='mean', weight=None):
        super().__init__(reduction=reduction, weight=torch.tensor([4.0,1.0]).cuda())

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "weighted_cross_entropy": WeightedCrossEntropy,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
}

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
