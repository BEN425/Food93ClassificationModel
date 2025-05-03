'''
training_loop.py

Defines `Trainer` class and training loop for the training task
'''

import os
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from rich.progress import track
from ema_pytorch import EMA

from loss import cal_bce_loss, cal_l2_regularization, cal_class_focal_loss
from metrics import cal_f1_score_acc, cal_tp_fp_fn_tn, cal_error_nums, evaluate_dataset

from rich import get_console
console = get_console()

class Trainer():
    '''
    Trainere for the food database
    
    `training loop` defines the training process. It trains the model for the given epochs.
    The training results and model checkpoints are saved in `Results` folder.
    The log files are created by tensorboard `SummaryWriter`
    
    Arguments :
        dataset `dict[str, FoodDataset]`: Contains "train" and "valid" keys for train set and val set
        model `ModifiedResNet50`: CNN classification model to train
        opt: `Optimizer`: Pytorch Optimizer
        device: `int|str`: CPU or CPU
        cfg: Configs
    '''
    
    def __init__(
            self,
            dataset: "dict[str, DataLoader]",
            model: nn.Module,
            opt: torch.optim.Optimizer,
            device,
            cfg: "dict[str]"
        ):
        
        self.train_dataloader = dataset["train"]
        self.valid_dataloader = dataset["valid"]
        self.class_num = cfg["MODEL"]["CATEGORY_NUM"]
        self.model = model
        self.cfg = cfg
        self.opt = opt
        self.device = device
        self.ema = EMA(self.model)
        
        # Resume from the lsat checkpoint
        if cfg["RESUME"]:
            ema_checkpoint = torch.load(cfg["CHECKPOINT_PATH"])
            self.ema.ema_model.load_state_dict(ema_checkpoint["model_ema"])
        self.ema.eval()
        
        # Specify saved model name and path
        date = f"{datetime.datetime.now().month}_{datetime.datetime.now().day}"
        save_model_path = os.path.join(cfg["SAVE_DIR"], "checkpoints", date)
        log_path = os.path.join(cfg["SAVE_DIR"], "logs", date)
        os.makedirs(save_model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        self.save_model_name = os.path.join(
            save_model_path,
            f"{date}_{cfg['MODEL']['NAME']}")
        
        # Create SummaryWriter and specify log path
        self.writer = SummaryWriter(log_dir=log_path)
    
    def train(
        self,
        start_epoch: int,
        end_epoch: int,
        cfg: dict,
        class_alpha: torch.Tensor = None,
        gamma: float = 2,
    ):
        '''
        Define the training loop. Train the model on given epochs and config.

        Save the model checkpoint and evaluation results of each epoch.
        
        Arguments :
            start_epoch `int`
            end_epoch `int`
            cfg: Configs,
            class_alpha `Tensor`: Alpha α of focal loss for each class. Each value is in range [0, 1]
            gamma `float`: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default is `2`.
        `class_alpha` and `gamma` is not used in `bce` loss.

        Returns :
            `list` containing evaluation results of each epoch
        '''
        
        
        # Calculate progress bar
        batch_size = cfg["BATCH_SIZE"]
        total = len(self.train_dataloader.dataset) // batch_size
        class_alpha = class_alpha.to(self.device) if class_alpha is not None else None
        
        # Store results of each epoch
        record_epoch = []
        
        for epoch in range(start_epoch, end_epoch):
            console.print( "====================")
            console.print(f"{f'Epoch {epoch}/{start_epoch}-{end_epoch-1}':^20}")
            console.print( "====================")
            
            # Record metrics of the epoch
            record_dict = {
                "train_total_loss": 0,
                "train_microf1"   : 0,
                "train_macrof1"   : 0,
                "train_micro_acc" : 0,
                "train_ham_loss"  : 0,
                "train_zero_acc"  : 0,
            }

            # TP, TN, FN, FP of each class, for calculating Macro F1
            tp = torch.zeros(self.class_num).to(self.device)
            fp = torch.zeros(self.class_num).to(self.device)
            fn = torch.zeros(self.class_num).to(self.device)
            tn = torch.zeros(self.class_num).to(self.device)
            # For calculating Hamming acc and Zero acc
            errors = 0
            corrects = 0
            label_count = 0
            data_count = 0
            
            self.model.train() # Training mode
            for idx, (img, label) in tqdm(
                enumerate(self.train_dataloader),
                total=total
            ):
                
                img, label = img.to(self.device), label.to(self.device).to(torch.float32)
                
                out = self.model(img)       # Model output
                logits = torch.sigmoid(out) # Probablility of each class
                pred = torch.round(logits)  # Prediction result, threshold = 0.5
                
                # Loss
                #? focal loss implicitly applies sigmoid
                loss = cal_class_focal_loss(out, label, class_alpha, gamma)
                
                # Metrics
                tp_fp_fn_tn = cal_tp_fp_fn_tn(pred, label)
                train_err_cor = cal_error_nums(pred, label)
                label_count += label.numel()
                data_count += len(label)

                # Back propagation
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()
                self.ema.update()

                # Record each iter loss
                record_dict["train_total_loss"] += loss.item()
                tp += tp_fp_fn_tn[0]
                fp += tp_fp_fn_tn[1]
                fn += tp_fp_fn_tn[2]
                tn += tp_fp_fn_tn[3]
                errors   += train_err_cor[0]
                corrects += train_err_cor[1]
    
            # Record each epoch loss
            train_metrics_results = cal_f1_score_acc(tp, fp, fn, tn)
            record_dict["train_microf1"]     = train_metrics_results["microf1"]
            record_dict["train_macrof1"]     = train_metrics_results["macrof1"]
            record_dict["train_micro_acc"]   = train_metrics_results["micro_acc"]
            record_dict["train_total_loss"]  = errors / label_count
            record_dict["train_ham_loss"]    = 1 - corrects / data_count
            record_dict["train_zero_acc"]   /= len(self.train_dataloader)

            # Evaluate on validation set
            if cfg["TEST_METRICS"]:
                valid_results = evaluate_dataset(
                    self.model,
                    self.valid_dataloader,
                    self.class_num,
                    self.device,
                    class_alpha,
                    gamma
                )
                record_dict.update(valid_results)
            
            # Save metrics and checkpoint of the epoch
            self._monitor(record_dict, epoch)
            self._save_checkpoint(record_dict, epoch)
            record_dict["epoch"] = epoch
            record_epoch.append(record_dict)

        self.writer.flush()

        return record_epoch

    def _monitor(self, record_dict: "dict[str, float]", epoch: int):
        for key, value in record_dict.items():
            self.writer.add_scalar(key, value, epoch)
        console.print(record_dict)
    
    def _save_checkpoint(self, record_dict: "dict[str, float]", epoch: int):
        checkpoint_name = f"{self.save_model_name}_epoch_{epoch}.pth.tar"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "model_ema": self.ema.ema_model.state_dict(),
                "opt": self.opt.state_dict(),
                "record_dict": record_dict,    
            },
            checkpoint_name
        )
        console.print(f"Saved checkpoint: \"{checkpoint_name}\"")
