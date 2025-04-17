'''
training_loop.py

Defines `Trainer` class and training loop for the training task
'''

import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
from tqdm import tqdm, trange
from rich.progress import track

from ema_pytorch import EMA

from loss import cal_loss, cal_l2_regularization, cal_focal_loss, cal_class_focal_loss
from metrics import cal_f1_score_acc, evaluate_valid_dataset, evaluate_valid_dataset_new, cal_ham_zero_acc, evaluate_valid_dataset_new2

from rich import get_console
console = get_console()

class Trainer():
    '''
    Trainere for the food database
    
    `training loop` defines the training process. It trains the model for the given epochs.
    The training results and model checkpoints are saved in `Results` folder.
    The log files are created by tensorboard `SummaryWriter`
    
    Argument:
        dataset `dict[str, FoodDataset]`: Contains "train" and "valid" keys for train set and val set
        model `ModifiedResNet50`: CNN classification model to train
        opt: `Optimizer`: Pytorch Optimizer
        device: `int|str`: CPU or CPU
        cfg: Configs
    '''
    
    def __init__(
            self,
            dataset: "dict[str, torch.utils.data.DataLoader]",
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
        
        # Define saved model name
        date = f"{datetime.datetime.now().month}_{datetime.datetime.now().day}"
        save_model_path = os.path.join(cfg["SAVE_DIR"], "checkpoints")
        os.makedirs(save_model_path, exist_ok=True)
        self.save_model_name = os.path.join(
            save_model_path,
            f"{date}_{cfg['MODEL']['NAME']}")
        self.writer = SummaryWriter(log_dir = os.path.join(
            cfg["SAVE_DIR"],
            "logs",
            date
        ))
        
    def training_loop(self, start_epoch: int, end_epoch: int, cfg):
        '''
        Define the training loop
        
        Argument:
            start_epoch `int`
            end_epoch `int`
            cfg: Configs
        '''
        
        
        # Calculate progress bar
        batch_size = cfg["BATCH_SIZE"]
        total = len(self.train_dataloader.dataset) / batch_size
        
        for epoch in range(start_epoch, end_epoch):
            console.print( "====================")
            console.print(f"{f'Epoch {epoch}/{start_epoch}-{end_epoch-1}':^20}")
            console.print( "====================")
            
            # Record metrics of the epoch
            record_dict = {
                "train_total_loss": 0,
                "train_bce_loss"  : 0,
                "train_l2_reg"    : 0,
                "train_microf1"   : 0,
                "train_macrof1"   : 0,
                "train_micro_acc" : 0
            }
            
            self.model.train() # Training mode
            
            for idx, (img, label) in tqdm(
                enumerate(self.train_dataloader), total=total
            ):
                
                img, label = img.to(self.device), label.to(self.device).to(torch.float32)
                
                #? Use sigmoid for multi-label classification
                logits = torch.sigmoid(self.model(img))
                # Loss
                bce_loss = cal_loss(logits, label)
                # l2_reg = cal_l2_regularization(self.model)
                total_loss = bce_loss
                # Metrics
                train_metrics_results = cal_f1_score_acc(logits, label)

                # Back propagation
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                self.ema.update()

                # Record each iter loss
                record_dict["train_total_loss"] += total_loss.item()
                record_dict["train_bce_loss"]   += bce_loss.item()
                # record_dict["train_l2_reg"]     += l2_reg.item()
                record_dict["train_microf1"]    += train_metrics_results["microf1"]
                record_dict["train_macrof1"]    += train_metrics_results["macrof1"]
                record_dict["train_micro_acc"]  += train_metrics_results["micro_acc"]
                
            # Record each epoch loss
            record_dict["train_total_loss"] /= len(self.train_dataloader)
            record_dict["train_bce_loss"]   /= len(self.train_dataloader)
            # record_dict["train_l2_reg"]     /= len(self.train_dataloader)
            record_dict["train_microf1"]    /= len(self.train_dataloader)
            record_dict["train_macrof1"]    /= len(self.train_dataloader)
            record_dict["train_micro_acc"]  /= len(self.train_dataloader)

            # Evaluate validation set
            if cfg["TEST_METRICS"]:
                valid_results = evaluate_valid_dataset(
                    self.model, 
                    self.valid_dataloader, 
                    self.device
                )
                for key, value in valid_results.items():
                    record_dict[key] = value
            
            # Save metrics and checkpoint of the epoch  
            self._monitor(record_dict, epoch)
            self._save_checkpoint(record_dict, epoch)

        self.writer.flush()

    def training_loop_new(self, start_epoch: int, end_epoch: int, cfg):
        '''
        Define the training loop
        
        U.se focal loss instead of BCE loss. Remove L2 regularization
        
        Argument:
            start_epoch `int`
            end_epoch `int`
            cfg: Configs
        '''
        
        
        # Calculate progress bar
        batch_size = cfg["BATCH_SIZE"]
        total = len(self.train_dataloader.dataset) / batch_size
                
        # Calculate class frequencies for focal loss parameter ɑ
        console.print("Calculating class frequencies...")
        cls_freq = self._load_class_frequency().to(self.device)
        console.print(cls_freq)
        
        for epoch in range(start_epoch, end_epoch):
            console.print( "====================")
            console.print(f"{f'Epoch {epoch}/{start_epoch}-{end_epoch-1}':^20}")
            console.print( "====================")
            
            # Record metrics of the epoch
            record_dict = {
                "train_total_loss": 0,
                "train_focal_loss": 0,
                "train_microf1"   : 0,
                "train_macrof1"   : 0,
                "train_micro_acc" : 0,
                "train_ham_loss"  : 0,
                "train_zero_acc"  : 0,
            }
            
            self.model.train() # Training mode
            
            for idx, (img, label) in tqdm(
                enumerate(self.train_dataloader),
                total=total
            ):
                
                img, label = img.to(self.device), label.to(self.device).to(torch.float32)
                
                logits = self.model(img)
                logits_sigmoid = torch.sigmoid(logits)
                pred = torch.round(logits_sigmoid) # Threshold = 0.5
                
                # Loss
                #? focal loss implicitly applies sigmoid
                # focal_loss = cal_focal_loss(logits, label)
                focal_loss = cal_class_focal_loss(logits, label, cls_freq)
                total_loss = focal_loss
                # Metrics
                train_metrics_results = cal_f1_score_acc(logits_sigmoid, label)
                train_acc_results = cal_ham_zero_acc(pred, label)

                # Back propagation
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                self.ema.update()

                # Record each iter loss
                record_dict["train_total_loss"] += total_loss.item()
                record_dict["train_focal_loss"] += focal_loss.item()
                record_dict["train_microf1"]    += train_metrics_results["microf1"]
                record_dict["train_macrof1"]    += train_metrics_results["macrof1"]
                record_dict["train_micro_acc"]  += train_metrics_results["micro_acc"]
                record_dict["train_ham_loss"]   += train_acc_results["ham_loss"]
                record_dict["train_zero_acc"]   += train_acc_results["zero_acc"]
    
            # Record each epoch loss
            record_dict["train_total_loss"] /= len(self.train_dataloader)
            record_dict["train_focal_loss"] /= len(self.train_dataloader)
            record_dict["train_microf1"]    /= len(self.train_dataloader)
            record_dict["train_macrof1"]    /= len(self.train_dataloader)
            record_dict["train_micro_acc"]  /= len(self.train_dataloader)
            record_dict["train_ham_loss"]   /= len(self.train_dataloader)
            record_dict["train_zero_acc"]   /= len(self.train_dataloader)

            # Evaluate validation set
            if cfg["TEST_METRICS"]:
                # valid_results = evaluate_valid_dataset_new(
                #     self.model, 
                #     self.valid_dataloader, 
                #     self.device
                # )
                valid_results = evaluate_valid_dataset_new2(
                    self.model, 
                    self.valid_dataloader,
                    cls_freq,
                    self.device
                )
                for key, value in valid_results.items():
                    record_dict[key] = value
            
            # Save metrics and checkpoint of the epoch  
            self._monitor(record_dict, epoch)
            self._save_checkpoint(record_dict, epoch)

        self.writer.flush()

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

    def _load_class_frequency(self) :
        cls_freq = torch.zeros(self.class_num)
        
        with open(os.path.join(self.cfg["DATA_BASE_DIR"], "..", "class_freq.txt"), "r") as f:
            for i, line in enumerate(f.readlines()) :
                cls_freq[i] = float(line)
        
        return cls_freq
