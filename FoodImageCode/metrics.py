'''
metrics.py

Defines matrics to evaluate the model
Defines a function to evaluate the model on validation set
'''


import torch
import torch.nn as nn
from torch.utils.data import DataLoader as Dataloader

from loss import cal_bce_loss, cal_l2_regularization, cal_class_focal_loss
from rich import get_console

console = get_console()

#? Calculate accuracy for multi-label classification

# Calculate TP, FP, FN and TN for a batch
def cal_tp_fp_fn_tn(pred: torch.Tensor, label: torch.Tensor):
    '''
    Calculate tp, fp, fn and tn for a batch of predictions and labels
    
    Arguments :
        pred `Tensor`: Prediction result, containing only 0 and 1
        label `Tensor`: Ground-truth label, containing only 0 and 1
        `pred` and `label` are both of dimenison `[batch_size, num_classes]`

    Return :
        `tp`, `fp`, `fn`, `tn` of the batch, of dimension `[num_classes]`
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
    class_f1 = False
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
        class_f1: Return F1 score of each class
    Returns :
        `dict` contains `microf1`, `macrof1` and `micro_acc`, contains `class_f1` if `class_f1` is True
    
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
    if class_f1:
        metrics["class_f1"] = f1_cls
    return metrics

# Calculate hamming accuracy and zero accuracy for a batch
def cal_ham_zero_acc(logits: torch.Tensor, label: torch.Tensor):
    '''
    Calculate hamming loss and zero accuracy for a batch of predictions and labels
    
    Hamming loss is the ratio of incorrect predictions to all predicted labels.
    Zero accuracy is the ratio of correct-predicted data to all data.
    
    Arguments :
        `logits` and `label` are both of dimenison `[batch_size, num_classes]`
        `logits` should be either 0 or 1
    
    Return :
        `dict` contains `ham_loss` and `zero_acc`
    '''
    
    err = (logits != label).sum(dim=1)
    ham_loss = err.sum() / logits.numel()
    zero_acc = 1 - err.count_nonzero() / len(logits)
    
    return {
        "ham_loss": ham_loss,
        "zero_acc": zero_acc
    }

# Calculate numbers of error predictions
def cal_error_nums(pred: torch.Tensor, label: torch.Tensor) :
    '''
    A helper function to calculate numbers of error prediction in a batch.
    Used for Hamming accuracy and Zero accuracy after the entire dataset is evaluated.

    Arguments :
        pred `Tensor`: Prediction result, containing only 0 and 1
        label `Tensor`: Ground-truth label, containing only 0 and 1
        `pred` and `label` are both of dimenison `[batch_size, num_classes]`
    
    Return :
        err_label: Number of wrong predicted labels
        correct_data: Number of correct predicted data
    '''

    err_batch = (pred != label).sum(dim=1)
    err_label = err_batch.sum().item()
    correct_data = err_batch.count_nonzero().item()
    return err_label, correct_data

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
    class_f1: bool = False,
    class_alpha: torch.Tensor = None,
    gamma: float = 2
) -> "dict[str, float]" :
    '''
    Evaluate model on a dataset
    Use specific weight factor of focal loss ɑ for each class. Use another Macro F1 method
    
    Arguments :
        cate_num `int`: Number of categories
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
    errors = 0
    corrects = 0
    label_count = 0
    data_count = 0

    model.eval() # Evaluation mode
    for idx, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device, dtype=torch.float32)
        
        out = model(img)
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
        errors   += valid_err_cor[0]
        corrects += valid_err_cor[1]

    # Record loss and metrics
    valid_metrics_results = cal_f1_score_acc(tp, fp, fn, tn, class_f1=class_f1)
    record_dict["valid_microf1"]     = valid_metrics_results["microf1"]
    record_dict["valid_macrof1"]     = valid_metrics_results["macrof1"]
    record_dict["valid_micro_acc"]   = valid_metrics_results["micro_acc"]
    record_dict["valid_ham_loss"]    = errors / label_count
    record_dict["valid_zero_acc"]    = 1 - corrects / data_count
    record_dict["valid_total_loss"] /= len(dataloader)
    if class_f1:
        record_dict["class_f1"] = valid_metrics_results["class_f1"]

    return record_dict
