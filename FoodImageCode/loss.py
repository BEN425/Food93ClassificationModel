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
bce_loss = nn.BCELoss()

def cal_bce_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    '''
    Calculate BCE loss between `logits` and `label`
    '''
    return bce_loss(logits, label)

def cal_class_focal_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    class_alpha: torch.Tensor = None,
    gamma: float = 2,
) -> torch.Tensor:
    """
    Calculate sigmoid focal loss between `logits` and `label`.
    
    `sigmoid_focal_loss` implicitly applies sigmoid function to logits.

    Arguments :
        `logits` and `label` are both of dimenison `[batch_size, num_classes]`
        `logits` is model output (without sigmoid)
        `label` is multi-hot encoded ground-truth labels
        class_alpha   `Tensor`: Alpha α of focal loss for each class. Each value is in range [0, 1]. Default is 0.25
        gamma `float`: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default is 2.
    """
    
    # BCE Loss
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
    
    # Gamma
    p_t = p * label + (1 - p) * (1 - label)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Alpha
    class_alpha = .25 if class_alpha is None else class_alpha
    alpha_t = (1 - class_alpha) * label + class_alpha * (1 - label)
    loss = alpha_t * loss

    return loss.mean()

def cal_l2_regularization(model, beta=1e-4) -> torch.Tensor:
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if "conv" in name and param.requires_grad:
            l2_reg += 0.5 * torch.sum(param ** 2)
    return beta * l2_reg
