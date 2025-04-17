'''
loss.py

Defines loss function for training and inference
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def cal_class_focal_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    class_freq: torch.Tensor,
    gamma: float = 2,
) -> torch.Tensor:
    """
    Calculate sigmoid focal loss between `logits` and `label`.
    
    `sigmoid_focal_loss` implicitly applies sigmoid function to logits.

    Args:
        class_freq `Tensor`: Class frequency of the dataset. Each value is in range [0, 1]
        gamma `float`: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
    """
    
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
    p_t = p * label + (1 - p) * (1 - label)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = (1 - class_freq) * label + class_freq * (1 - label)
    loss = alpha_t * loss

    loss = loss.mean()

    return loss

def cal_l2_regularization(model, beta=1e-4) -> torch.Tensor:
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if "conv" in name and param.requires_grad:
            l2_reg += 0.5 * torch.sum(param ** 2)
    return beta * l2_reg
