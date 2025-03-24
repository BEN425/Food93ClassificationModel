'''
loss.py

Defines loss function for training and inference
'''

import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

#? Use BCE (Binary Cross Entropy) loss for multilabel classfication
#? Use focal loss for unbalanced dataset

# loss = nn.CrossEntropyLoss()
loss = nn.BCELoss()

def cal_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    '''
    Calculate BCE loss between `logits` and `label`
    '''
    return loss(logits, label)

def cal_focal_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    '''
    Calculate sigmoid focal loss between `logits` and `label`. 
    `sigmoid_focal_loss` implicitly applies sigmoid function to logits.
    '''
    return sigmoid_focal_loss(logits, label, reduction="mean")

def cal_l2_regularization(model, beta=1e-4) -> torch.Tensor:
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if "conv" in name and param.requires_grad:
            l2_reg += 0.5 * torch.sum(param ** 2)
    return beta * l2_reg
