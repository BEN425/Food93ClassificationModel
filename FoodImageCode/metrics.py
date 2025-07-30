'''
metrics.py

Defines matrics to evaluate the model
Defines a function to evaluate the model on dataset
'''

# TODO: Add mIoU metrics to S2C

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as Dataloader

import numpy as np

from cam import class_activation_map
from loss import cal_bce_loss, cal_l2_regularization, cal_class_focal_loss
from rich import get_console

console = get_console()

# Calculate TP, FP, FN and TN for a batch
def cal_tp_fp_fn_tn(pred: torch.Tensor, label: torch.Tensor):
    '''
    Calculate tp, fp, fn and tn for a batch of predictions and labels
    
    Arguments :
        pred `Tensor`: Prediction result, containing only 0 and 1
        label `Tensor`: Ground-truth label, containing only 0 and 1
        `pred` and `label` are both of dimenison `[batch_size, num_classes]`

    Return :
        `tp`, `fp`, `fn`, `tn` of, of dimension `[num_classes]`
    '''

    tp = torch.sum((pred * label) != 0, dim=0)
    fp = torch.sum((pred * (label - 1)) != 0, dim=0)
    fn = torch.sum(((pred - 1) * label) != 0, dim=0)
    tn = torch.sum(((pred - 1) * (label - 1)) != 0, dim=0)
    
    return tp, fp, fn, tn

# Calculate F1 score for a batch
def cal_f1_score_acc(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor,
    alt_macrof1 = False,
    class_acc = False
):
    '''
    Calculate F1 score for a batch of predictions and labels
    F1 = 2 * prec * recall / (prec + recall)

    Calculateing accuracy on a batch of data instead of the entire dataset will cause
    inaccurate result, consider computing after all data are evaluated.

    Micro accuracy will be dominated by True Negative (TN), which is not suitable for multi-label classification

    The original macro F1 method is calculating F1 scores of each class and then taking the mean.
    The alternate method is calculating macro precision and macro recall and then calculating the F1 score
    
    Arguments :
        `tp`, `fp`, `fn`, `tn` are of dimension `[num_classes]`
        alt_macrof1: Use alternate macro F1 method
        class_acc: Return accuracy of each class, including precision, recall and F1 score
    Returns :
        `dict` contains `microf1`, `macrof1` and `micro_acc`, contains `class_precision`, `class_recall`, `class_f1` if `class_f1` is True
    
    '''


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

    f1_cls = torch.where(
        condition = (precision_cls + recall_cls) > 0,
        input = 2 * (precision_cls * recall_cls) / (precision_cls + recall_cls),
        other = torch.zeros_like(recall_cls)
    )
    
    macroF1_alt = 2 * (precision_ma * recall_ma) / (precision_ma + recall_ma) \
        if (precision_ma + recall_ma) > 0 else 0
    macroF1 = f1_cls.mean().item()

    metrics = {
        "microf1": microF1,
        "macrof1": macroF1_alt if alt_macrof1 else macroF1,
        "micro_acc": micro_acc,
    }

    if class_acc:
        metrics["class_f1"] = f1_cls
        metrics["class_precision"] = precision_cls
        metrics["class_recall"] = recall_cls
    return metrics

# Calculate hamming accuracy and zero accuracy for a batch
#? Not used in training
@torch.no_grad()
def cal_ham_zero_acc(pred: torch.Tensor, label: torch.Tensor):
    '''
    Calculate hamming loss and zero accuracy for a batch of predictions and labels
    
    Hamming loss is the ratio of incorrect predictions to all predicted labels.
    Zero accuracy is the ratio of correct-predicted data to all data.
    
    Arguments :
        pred `Tensor`: Prediction result, containing only 0 and 1
        label `Tensor`: Ground-truth label, containing only 0 and 1
        `pred` and `label` are both of dimenison `[batch_size, num_classes]`
    
    Return :
        `dict` contains `ham_loss` and `zero_acc`
    '''
    
    err = (pred != label).sum(dim=1)
    ham_loss = err.sum() / pred.numel()
    zero_acc = 1 - err.count_nonzero() / len(pred)
    
    return {
        "ham_loss": ham_loss,
        "zero_acc": zero_acc
    }

# Calculate numbers of error predictions
@torch.no_grad()
def cal_error_nums(pred: torch.Tensor, label: torch.Tensor) :
    '''
    A helper function to calculate numbers of error prediction in a batch.
    Return values are used for hamming loss and zero accuracy after the entire dataset is evaluated.

    Arguments :
        pred `Tensor`: Prediction result, containing only 0 and 1
        label `Tensor`: Ground-truth label, containing only 0 and 1
        `pred` and `label` are both of dimenison `[batch_size, num_classes]`
    
    Return :
        err_label: Number of wrong predicted labels
        err_data: Number of wrong predicted data
    '''

    err_batch = (pred != label).sum(dim=1)
    err_label = err_batch.sum().item()
    err_data  = err_batch.count_nonzero().item()
    return err_label, err_data

# Calclate acccuracy for single label
#? Not used in multi-label classification
def cal_acc(logits: torch.Tensor, label: torch.Tensor):
    top_pred = logits.argmax(1, keepdim=True)
    correct = top_pred.eq(label.view_as(top_pred)).sum()
    return correct.float() / label.shape[0]

# Evaluate the model on a dataset
# Use specific weight factor of focal loss α for each class
@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataloader: Dataloader,
    cate_num: int,
    device,
    class_alpha: torch.Tensor = None,
    gamma: float = 2
) -> "dict[str, float]" :
    '''
    Evaluate model on a dataset
    Use specific weight factor of focal loss ɑ for each class. Use another Macro F1 method
    
    Arguments :
        cate_num `int`: Number of categories
        class_f1: Return F1 score of each class
        class_alpha `Tensor`: Alpha α of focal loss for each class. Each value is in range [0, 1]
        gamma `float`: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns :
        `dict` containing evaluation results
    '''

    record_dict = {
        "valid_ham_loss"  : 0,
        "valid_zero_acc"  : 0,
        "valid_total_loss": 0,
        "valid_microf1"   : 0,
        "valid_macrof1"   : 0,
        "valid_micro_acc" : 0
    }

    # TP, TN, FN, FP of each class, for calculating Macro F1
    tp = torch.zeros(cate_num).to(device)
    fp = torch.zeros(cate_num).to(device)
    fn = torch.zeros(cate_num).to(device)
    tn = torch.zeros(cate_num).to(device)
    # For calculating Hamming acc and Zero acc
    err_label = 0
    err_data = 0
    label_count = 0
    data_count = 0

    model.eval() # Evaluation mode
    for idx, (img, label, _) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device, dtype=torch.float32)
        
        out = model(img)["pred"]
        logits = torch.sigmoid(out)
        pred = torch.round(logits) # Threshold = 0.5
        
        # Loss
        loss = cal_class_focal_loss(out, label, class_alpha, gamma)

        # Metrics
        tp_fp_fn_tn = cal_tp_fp_fn_tn(pred, label)
        valid_err_cor = cal_error_nums(pred, label)
        label_count += label.numel()
        data_count += len(label)

        record_dict["valid_total_loss"] += loss.item()
        tp += tp_fp_fn_tn[0]
        fp += tp_fp_fn_tn[1]
        fn += tp_fp_fn_tn[2]
        tn += tp_fp_fn_tn[3]
        err_label += valid_err_cor[0]
        err_data  += valid_err_cor[1]

    # Record loss and metrics
    valid_metrics_results = cal_f1_score_acc(tp, fp, fn, tn)
    #? Use `float` to ensure all values are not Tensor
    record_dict["valid_microf1"]     = float(valid_metrics_results["microf1"])
    record_dict["valid_macrof1"]     = float(valid_metrics_results["macrof1"])
    record_dict["valid_micro_acc"]   = float(valid_metrics_results["micro_acc"])
    record_dict["valid_ham_loss"]    = float(err_label / label_count)
    record_dict["valid_zero_acc"]    = float(1 - err_data / data_count)
    record_dict["valid_total_loss"] /= len(dataloader)

    return record_dict

# Same with `evaluate_dataset` but also include accuracy of each class, total tp, fp, fn and tn
@torch.no_grad()
def evaluate_dataset_class_acc( 
    model: nn.Module,
    dataloader: Dataloader,
    cate_num: int,
    device,
    class_alpha: torch.Tensor = None,
    gamma: float = 2,
    to_list = False
) -> "dict[str, float]" :
    '''
    Evaluate model on a dataset
    Use specific weight factor of focal loss ɑ for each class. Use another Macro F1 method
    
    Arguments :
        cate_num `int`: Number of categories
        class_f1: Return F1 score of each class
        class_alpha `Tensor`: Alpha α of focal loss for each class. Each value is in range [0, 1]
        gamma `float`: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        to_list `bool`: Convert the return values of Tensor to list

    Returns :
        `dict` containing evaluation results
    '''

    record_dict = {
        "valid_ham_loss"        : 0,
        "valid_zero_acc"        : 0,
        "valid_total_loss"      : 0,
        "valid_microf1"         : 0,
        "valid_macrof1"         : 0,
        "valid_micro_acc"       : 0,
        "valid_class_f1"        : None,
        "valid_class_precision" : None,
        "valid_class_recall"    : None,
        "tp"                    : None,
        "fp"                    : None,
        "fn"                    : None,
        "tn"                    : None,
        "conf_matrix"           : None,
    }

    # TP, TN, FN, FP of each class, for calculating Macro F1
    tp = torch.zeros(cate_num, dtype=torch.int).to(device)
    fp = torch.zeros(cate_num, dtype=torch.int).to(device)
    fn = torch.zeros(cate_num, dtype=torch.int).to(device)
    tn = torch.zeros(cate_num, dtype=torch.int).to(device)
    conf_matrix = torch.zeros((cate_num, cate_num), dtype=torch.int).to(device)
    # For calculating Hamming acc and Zero acc
    err_label = 0
    err_data = 0
    label_count = 0
    data_count = 0

    model.eval() # Evaluation mode
    for idx, (img, label, _) in enumerate(dataloader):
        img: torch.Tensor = img.to(device)
        label: torch.Tensor = label.to(device, dtype=torch.float32)
        
        out = model(img)["pred"]
        logits = torch.sigmoid(out)
        pred = torch.round(logits) # Threshold = 0.5
        
        # Loss
        loss = cal_class_focal_loss(out, label, class_alpha, gamma)

        # Metrics
        tp_fp_fn_tn = cal_tp_fp_fn_tn(pred, label)
        valid_err_cor = cal_error_nums(pred, label)
        label_count += label.numel()
        data_count += len(label)

        record_dict["valid_total_loss"] += loss.item()
        tp += tp_fp_fn_tn[0]
        fp += tp_fp_fn_tn[1]
        fn += tp_fp_fn_tn[2]
        tn += tp_fp_fn_tn[3]
        err_label += valid_err_cor[0]
        err_data  += valid_err_cor[1]

        # Confusion Matrix
        for l, p in zip(label, pred) :
            cls_ids = l.nonzero().long()
            for id in cls_ids :
                p_ = p.clone().int()
                p_[ cls_ids[id != cls_ids] ] = 0
                conf_matrix[id] += p_

    # Record loss and metrics
    valid_metrics_results = cal_f1_score_acc(tp, fp, fn, tn, class_acc=True)
    #? Use `float` to ensure all values are not Tensor
    record_dict["valid_microf1"]          = float(valid_metrics_results["microf1"])
    record_dict["valid_macrof1"]          = float(valid_metrics_results["macrof1"])
    record_dict["valid_micro_acc"]        = float(valid_metrics_results["micro_acc"])
    record_dict["valid_ham_loss"]         = float(err_label / label_count)
    record_dict["valid_zero_acc"]         = float(1 - err_data / data_count)
    record_dict["valid_total_loss"]      /= len(dataloader)
    record_dict["valid_class_f1"]         = valid_metrics_results["class_f1"]
    record_dict["valid_class_precision"]  = valid_metrics_results["class_precision"]
    record_dict["valid_class_recall"]     = valid_metrics_results["class_recall"]
    record_dict["tp"] = tp.cpu()
    record_dict["fp"] = fp.cpu()
    record_dict["fn"] = fn.cpu()
    record_dict["tn"] = tn.cpu()
    record_dict["conf_matrix"] = conf_matrix.cpu()

    if to_list and isinstance(record_dict["valid_class_f1"], torch.Tensor) and \
        isinstance(record_dict["valid_class_precision"], torch.Tensor) and \
        isinstance(record_dict["valid_class_recall"], torch.Tensor) and \
        isinstance(record_dict["conf_matrix"], torch.Tensor) :
        
        record_dict["valid_class_f1"] = record_dict["valid_class_f1"].tolist()
        record_dict["valid_class_precision"] = record_dict["valid_class_precision"].tolist()
        record_dict["valid_class_recall"] = record_dict["valid_class_recall"].tolist()
        record_dict["conf_matrix"] = record_dict["conf_matrix"].tolist()

    return record_dict

if __name__ == "__main__":
    pred = torch.Tensor([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    label = torch.Tensor([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
    ])
    
    err_label, err_data = cal_error_nums(pred, label)
    print(err_label, err_data)