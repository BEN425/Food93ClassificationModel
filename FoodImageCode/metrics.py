'''
metrics.py

Defines matrics to evaluate the model
Defines a function to evaluate the model on validation set
'''


import torch
import torch.nn as nn

from loss import cal_loss, cal_focal_loss, cal_l2_regularization, cal_class_focal_loss
from rich import get_console

console = get_console()

#? Calculate accuracy for multi-label classification

# Calculate TP, FP, FN and TN for a batch
def cal_tp_fp_fn_tn(logits: torch.Tensor, label: torch.Tensor):
    '''
    Calculate tp, fp, fn and tn for a batch of predictions and labels
    
    Arguments:
        logits and labels of a batch
        `logits` and `label` are both of dimenison `[batch_size, num_classes]`
    Return:
        `tp`, `fp`, `fn`, `tn` of the batch, of dimension `[num_classes]`
    '''
    
    logits = torch.round(logits) # Thresold = .5

    tp = torch.sum((logits * label) != 0, dim=0)
    fp = torch.sum((logits * (label - 1)) != 0, dim=0)
    fn = torch.sum(((logits - 1) * label) != 0, dim=0)
    tn = torch.sum(((logits - 1) * (label - 1)) != 0, dim=0)
    
    return tp, fp, fn, tn

# Calculate F1 score for a batch
def cal_f1_score_acc(logits: torch.Tensor, label: torch.Tensor, cls_freq=False):
    '''
    Calculate F1 score for a batch of predictions and labels
    F1 = 2 * prec * recall / (prec + recall)
    
    Arguments:
        `logits` and `label` are both of dimenison `[batch_size, num_classes]`
        `logits` is sigmoid of model outputs
    
    Returns:
        `dict` contains `microf1`, `macrof1` and `micro_acc`. Note that `micro_acc`
        is not suitable for multi-label classification
    '''

    # Calculate TP, FP, FN, TN
    tp, fp, fn, tn = cal_tp_fp_fn_tn(logits, label) # Each class
    total_tp = torch.sum(tp)
    total_fp = torch.sum(fp)
    total_fn = torch.sum(fn)
    total_tn = torch.sum(tn)
    
    # Accuracy, not suitable for multi-label
    micro_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)

    # Micro F1
    
    precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall_micro    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    microF1 = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) \
        if (precision_micro + recall_micro) > 0 else 0

    # Macro F1

    precision_cls = torch.where(
        condition = (tp + fp) != 0,
        input = tp / (tp + fp),
        other = torch.zeros_like(tp)
    )
    recall_cls = torch.where(
        condition = (tp + fn) != 0,
        input = tp / (tp + fn),
        other = torch.zeros_like(tp)
    )

    precision_ma = torch.mean(precision_cls)
    recall_ma = torch.mean(recall_cls)

    macroF1 = 2 * (precision_ma * recall_ma) / (precision_ma + recall_ma) \
        if (precision_ma + recall_ma) > 0 else 0
    clsF1 = 2 * (precision_cls * recall_cls) / (precision_cls + recall_cls + 1e-9)

    if cls_freq :
        return {
            "microf1": microF1, 
            "macrof1": macroF1, 
            "micro_acc": micro_acc,
            "classf1": clsF1
        }
    else :
        return {
            "microf1": microF1, 
            "macrof1": macroF1, 
            "micro_acc": micro_acc
        }

    
# Calculate hamming accuracy and zero accuracy for a batch
def cal_ham_zero_acc(logits: torch.Tensor, label: torch.Tensor):
    '''
    Calculate hamming accuracy and zero accuracy for a batch of predictions and labels
    
    Hamming accuracy is the ratio of correct predictions to all predicted labels.
    Zero accuracy is the ratio of correct images to all images.
    
    Arguments:
        `logits` and `label` are both of dimenison `[batch_size, num_classes]`
        `logits` should be either 0 or 1
    '''
    
    err = (logits != label).sum(dim=1)
    ham_loss = err.sum() / logits.numel()
    zero_acc = 1 - err.count_nonzero() / len(logits)
    
    return {
        "ham_loss": ham_loss,
        "zero_acc": zero_acc
    }

# Calclate acccuracy for single label
#? Not used in multi-label classification
def cal_acc(logits: torch.Tensor, label: torch.Tensor):
    top_pred = logits.argmax(1, keepdim=True)
    correct = top_pred.eq(label.view_as(top_pred)).sum()
    return correct.float() / label.shape[0]

# Evaluate model on the validation set
def evaluate_valid_dataset(model, dataloader, device):
    record_dict = {
        "valid_total_loss": 0,
        "valid_bce_loss"  : 0,
        "valid_l2_reg"    : 0,
        "valid_microf1"   : 0,
        "valid_macrof1"   : 0,
        "valid_micro_acc" : 0
    }
    
    model.eval() # Evaluation mode
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device,  dtype=torch.float32)
            
            logits = torch.sigmoid(model(img))
            # Loss
            bce_loss = cal_loss(logits, label)
            # l2_reg = cal_l2_regularization(model)
            total_loss = bce_loss
            
            valid_metrics_results = cal_f1_score_acc(logits, label)

            record_dict["valid_total_loss"] += total_loss.item()
            record_dict["valid_bce_loss"]   += bce_loss.item()
            # record_dict["valid_l2_reg"]     += l2_reg.item()
            record_dict["valid_microf1"]    += valid_metrics_results["microf1"]
            record_dict["valid_macrof1"]    += valid_metrics_results["macrof1"]
            record_dict["valid_micro_acc"]  += valid_metrics_results["micro_acc"]
    
        # record each epoch loss
        record_dict["valid_total_loss"] /= len(dataloader)
        record_dict["valid_bce_loss"]   /= len(dataloader)
        # record_dict["valid_l2_reg"]     /= len(dataloader)
        record_dict["valid_microf1"]    /= len(dataloader)
        record_dict["valid_macrof1"]    /= len(dataloader)
        record_dict["valid_micro_acc"]  /= len(dataloader)
    
    return record_dict

# Evaluate model on the validation set
# Use focal loss instead of BCS loss, remove L2 regularization
# Add hamming accuracy and zero accuracy
def evaluate_valid_dataset_new(model, dataloader, device):
    record_dict = {
        "valid_ham_loss"  : 0,
        "valid_zero_acc"  : 0,
        "valid_total_loss": 0,
        "valid_focal_loss": 0,
        "valid_microf1"   : 0,
        "valid_macrof1"   : 0,
        "valid_micro_acc" : 0
    }
    
    model.eval() # Evaluation mode
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device,  dtype=torch.float32)
            
            logits = model(img)
            logits_sigmoid = torch.sigmoid(logits)
            pred = torch.round(logits_sigmoid) # Threshold = 0.5
            
            # Loss
            focal_loss = cal_focal_loss(logits, label)
            total_loss = focal_loss
            
            valid_metrics_results = cal_f1_score_acc(logits_sigmoid, label)
            valid_acc_results = cal_ham_zero_acc(pred, label)

            record_dict["valid_total_loss"] += total_loss.item()
            record_dict["valid_focal_loss"] += focal_loss.item()
            record_dict["valid_microf1"]    += valid_metrics_results["microf1"]
            record_dict["valid_macrof1"]    += valid_metrics_results["macrof1"]
            record_dict["valid_micro_acc"]  += valid_metrics_results["micro_acc"]
            record_dict["valid_ham_loss"]   += valid_acc_results["ham_loss"]
            record_dict["valid_zero_acc"]   += valid_acc_results["zero_acc"]
    
        # record each epoch loss
        record_dict["valid_total_loss"] /= len(dataloader)
        record_dict["valid_focal_loss"] /= len(dataloader)
        record_dict["valid_microf1"]    /= len(dataloader)
        record_dict["valid_macrof1"]    /= len(dataloader)
        record_dict["valid_micro_acc"]  /= len(dataloader)
        record_dict["valid_ham_loss"]   /= len(dataloader)
        record_dict["valid_zero_acc"]   /= len(dataloader)
    
    return record_dict

# Evaluate model on the validation set
# Use class frequency as weight factor of focal loss, ɑ
def evaluate_valid_dataset_new2(model, dataloader, class_freq, device):
    record_dict = {
        "valid_ham_loss"  : 0,
        "valid_zero_acc"  : 0,
        "valid_total_loss": 0,
        "valid_focal_loss": 0,
        "valid_microf1"   : 0,
        "valid_macrof1"   : 0,
        "valid_micro_acc" : 0
    }
    
    model.eval() # Evaluation mode
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):
            print(idx)
            img = img.to(device)
            label = label.to(device,  dtype=torch.float32)
            
            logits = model(img)
            logits_sigmoid = torch.sigmoid(logits)
            pred = torch.round(logits_sigmoid) # Threshold = 0.5
            
            # Loss
            # focal_loss = cal_focal_loss(logits, label)
            focal_loss = cal_class_focal_loss(logits, label, class_freq)
            total_loss = focal_loss
            
            valid_metrics_results = cal_f1_score_acc(logits_sigmoid, label)
            valid_acc_results = cal_ham_zero_acc(pred, label)

            record_dict["valid_total_loss"] += total_loss.item()
            record_dict["valid_focal_loss"] += focal_loss.item()
            record_dict["valid_microf1"]    += valid_metrics_results["microf1"]
            record_dict["valid_macrof1"]    += valid_metrics_results["macrof1"]
            record_dict["valid_micro_acc"]  += valid_metrics_results["micro_acc"]
            record_dict["valid_ham_loss"]   += valid_acc_results["ham_loss"]
            record_dict["valid_zero_acc"]   += valid_acc_results["zero_acc"]
    
        # record each epoch loss
        record_dict["valid_total_loss"] /= len(dataloader)
        record_dict["valid_focal_loss"] /= len(dataloader)
        record_dict["valid_microf1"]    /= len(dataloader)
        record_dict["valid_macrof1"]    /= len(dataloader)
        record_dict["valid_micro_acc"]  /= len(dataloader)
        record_dict["valid_ham_loss"]   /= len(dataloader)
        record_dict["valid_zero_acc"]   /= len(dataloader)
    
    return record_dict
