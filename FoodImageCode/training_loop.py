'''
training_loop.py

Defines `Trainer` class and training loop for the training task
After each epoch, the model checkpoint and evaluation results are saved in `Results` folder.
'''

import os
import datetime

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from rich.progress import track
from ema_pytorch import EMA

from loss import cal_class_focal_loss, cal_ssc_loss, select_best_sam_mask_by_cam_overlap, denorm, calc_cpm_loss
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
        dataset `dict[str, FoodDataset]`: Contains `train` and `valid` keys for train set and val set
        model `ModifiedResNet50`: CNN classification model to train
        sam `SAM`: SAM model used in CPM. Will not be used if "USE_CPM" is `False` in `cfg`
        opt: `Optimizer`: Pytorch Optimizer
        device: `int|str`: CPU or CPU
        cfg: Configs
    '''
    
    def __init__(
            self,
            dataset: "dict[str, DataLoader]",
            model: nn.Module,
            opt: torch.optim.Optimizer,
            device: torch.types.Device,
            cfg: "dict[str]",
            sam: nn.Module = None,
            using_ddp: bool = False
        ):
        
        self.train_dataloader = dataset["train"]
        self.valid_dataloader = dataset["valid"]
        self.class_num: int = cfg["MODEL"]["CATEGORY_NUM"]
        self.model = model
        self.sam = sam
        self.cfg = cfg
        self.opt = opt
        self.device = device
        self.gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        self.using_ddp = using_ddp
        self.ema = EMA(self.model.module if using_ddp else self.model)
        
        # Resume from the lsat checkpoint
        if cfg["RESUME"]:
            ema_checkpoint = torch.load(cfg["CHECKPOINT_PATH"])
            self.ema.ema_model.load_state_dict(ema_checkpoint["model_ema"])
        self.ema.eval()
        
        # Specify saved model name and path
        date = f"{datetime.datetime.now().month}_{datetime.datetime.now().day}"
        sub_dir = self.cfg.get("SAVE_SUB_NAME", date)
        save_model_path = os.path.join(cfg["SAVE_DIR"], "checkpoints", sub_dir)
        log_path = os.path.join(cfg["SAVE_DIR"], "logs", sub_dir)

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
            gamma `float`: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples
        `class_alpha` and `gamma` is not used in `bce` loss.

        Returns :
            `list` containing evaluation results of each epoch
        '''
        
        
        # Calculate progress bar
        batch_size = cfg["BATCH_SIZE"]
        total = len(self.train_dataloader.dataset) // batch_size
        
        # Store results of each epoch
        record_epoch = []
        
        for epoch in range(start_epoch, end_epoch):
            # console.print( "====================")
            # console.print(f"{f'Epoch {epoch}/{start_epoch}-{end_epoch-1}':^20}")
            # console.print( "====================")

            if self.using_ddp :
                self.train_dataloader.sampler.set_epoch(epoch)
                self.valid_dataloader.sampler.set_epoch(epoch)
            
            # Record metrics of the epoch
            record_dict = {
                "train_total_loss": 0,
                "train_cls_loss"  : 0,
                "train_cpm_loss"  : 0,
                "train_ssc_loss"  : 0,
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
            err_label = 0
            err_data  = 0
            label_count = 0
            data_count  = 0
            
            self.model.to(self.device)
            self.ema.to(self.device)
            if self.sam is not None :
                self.sam.to(self.device)
                self.sam.eval()

            self.model.train() # Training mode
            for idx, (img, label, sam_img) in tqdm(
                enumerate(self.train_dataloader),
                desc=f"GPU[{self.gpu_id}]: Epoch {epoch}/{start_epoch}-{end_epoch-1}",
                # position=self.gpu_id,
                leave=True,
                total=total
            ):
                
                self.model.train()
                img, label = img.to(self.device), label.to(self.device).to(torch.float32)
                
                result = self.model(img)       # Model output
                out = result["pred"]
                logits = torch.sigmoid(out) # Probablility of each class
                pred = torch.round(logits)  # Prediction result, threshold = 0.5
                
                # Classification Loss
                #? focal loss implicitly applies sigmoid
                loss_cls = cal_class_focal_loss(out, label, class_alpha, gamma)
                
                # SSC loss
                if self.cfg["USE_SSC"] :
                    feature_vec = result["feat"]
                    se = sam_img.to(self.device)
                    loss_ssc = cal_ssc_loss(se, feature_vec, target_size=(224, 224)) # Use smaller size to save memory
                    # se = se.unsqueeze(0).long() #? 2x512x512, Unsqueeze?
                else :
                    loss_ssc = torch.Tensor([0]).to(self.device)
                
                # CPM loss
                if self.cfg["USE_CPM"] :
                    self.model.eval()
                    cam_main, cam_ms = self._multi_scale_cam(img, label)
                    max_points = self._sample_local_max(
                        img.shape,
                        label,
                        cam_ms,
                        size_sam = 512,
                        threshold = 0.2,
                    )
                    pgt_sam = self._aggregate_sam_cam(img, cam_ms, max_points, size_sam=512, mask_select=2)
                    self.model.train()
                    loss_cpm = calc_cpm_loss(cam_main, pgt_sam)
                else :
                    loss_cpm = torch.Tensor([0]).to(self.device)
                
                self.model.train()
                
                # Total loss
                loss = loss_cls + loss_ssc + loss_cpm
                
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
                record_dict["train_cls_loss"]   += loss_cls.item()
                record_dict["train_cpm_loss"]   += loss_cpm.item()
                record_dict["train_ssc_loss"]   += loss_ssc.item()
                record_dict["train_total_loss"] += loss.item()
                tp += tp_fp_fn_tn[0]
                fp += tp_fp_fn_tn[1]
                fn += tp_fp_fn_tn[2]
                tn += tp_fp_fn_tn[3]
                err_label += train_err_cor[0]
                err_data  += train_err_cor[1]
    
            # Record each epoch loss
            train_metrics_results = cal_f1_score_acc(tp, fp, fn, tn)
            #? Use `float` to ensure all values are scalar
            record_dict["train_microf1"]     = float(train_metrics_results["microf1"])
            record_dict["train_macrof1"]     = float(train_metrics_results["macrof1"])
            record_dict["train_micro_acc"]   = float(train_metrics_results["micro_acc"])
            record_dict["train_ham_loss"]    = float(err_label / label_count)
            record_dict["train_zero_acc"]    = float(1 - err_data / data_count)
            record_dict["train_total_loss"] /= len(self.train_dataloader)
            record_dict["train_cls_loss"]   /= len(self.train_dataloader)
            record_dict["train_ssc_loss"]   /= len(self.train_dataloader)
            record_dict["train_cpm_loss"]   /= len(self.train_dataloader)

            # Evaluate on validation set

            if cfg["TEST_METRICS"] :
                valid_results = evaluate_dataset(
                    self.model,
                    self.valid_dataloader,
                    self.class_num,
                    self.device,
                    class_alpha=class_alpha,
                    gamma=gamma
                )
                record_dict.update(valid_results)
            
            # Save metrics and checkpoint of the epoch
            # Only record on the main process
            # if self.is_main :
            self._monitor(record_dict, epoch)
            self._save_checkpoint(record_dict, epoch)
            record_dict["epoch"] = epoch
            record_epoch.append(record_dict)

        # Close writer
        self.writer.flush()
        self.writer.close()

        return record_epoch

    ### CPM ###

    @torch.no_grad()
    def _multi_scale_cam(self, image_tensor: torch.Tensor, label_tensor: torch.Tensor) -> "tuple[torch.Tensor]" :
        '''
        Generate CAM with image scaled by 0.5, 1, 1.5 and 2
        
        Arguments :
            image_tensor `Tensor` "[B, 3, H, W]": Input image Tensor
            label_tensor `Tensor` "[B, CLS]": Ground-truth label
        
        Return :
            cam_main `Tensor` "[B, CLS+1, H, W]": CAM of original scale image with additional background channel
            cam_ms `Tensor` "[B, CLS, H, W]": Result CAM
        '''
        
        B, C, H, W = image_tensor.shape
        scales = [0.5, 1.0, 1.5, 2.0]
        # Sum of CAM values from all scales
        cam_ms = torch.zeros((B, self.class_num, H, W), device=self.device) # [B, CLS, H, W]
        
        # print(f"cam_ms: {cam_ms.shape}")
        # print("Generating CAM of scaled images...")
        
        # Do CAM with different image scales
        for scale in scales:
            img_scaled = F.interpolate(image_tensor, scale_factor=scale, mode="bilinear", align_corners=True)
            out = self.model(img_scaled)
            cam = out["cam"] # [B, CLS, 14, 14]
                
            # if isinstance(cam, np.ndarray):
            #     cam = torch.tensor(cam, device=DEVICE)
            # if cam.ndim == 3:
            #     cam = cam.unsqueeze(0)

            # Resize CAM to image size
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
            cam = F.relu(cam)
            
            # Only keep channels of ground-truth labels
            cam *= label_tensor.view(B, self.class_num, 1, 1)
            
            # Save CAM of original scale for later use
            if scale == 1 :
                cam_main = cam
                # Normalization
                cam_max = F.adaptive_max_pool2d(cam_main, (1, 1))
                cam_main = cam_main / (cam_max + 1e-5)
                # Append background channel
                cam_bg = 1 - cam_main.max(dim=1, keepdims=True)[0]
                cam_main = torch.cat((cam_main, cam_bg), dim=1)
            
            # print(f"\t{scale}: {cam.shape}") 
            
            cam_ms += cam

        # Normalization, each channel divided by max value
        cam_max = F.adaptive_max_pool2d(cam_ms, (1, 1))
        cam_ms = cam_ms / (cam_max + 1e-5)
        
        return cam_main, cam_ms

    @torch.no_grad()
    def _sample_local_max(
        self,
        img_shape: torch.Size,
        label_tensor: torch.Tensor,
        cam_ms: torch.Tensor,
        size_sam: int,
        threshold: float = 0.5,
    ) -> "dict[int, dict[int, torch.Tensor]]" :
        '''
        Sample local maximum points from CAM
        
        Arguments :
            img_shape `tuple` "(B, C, H, W)": Shape of input image
            label_tensor `Tensor` "[B, CLS]": Ground-truth label
            cam_ms `Tensor` "[B, CLS, H, W]": CAM from `multi_scale_cam`
            threshold `float`: Threshold value to determine whether to include a local maximum
            size_sam `int`: Size of SAM map
        
        Returns :
            `dict[int, dict[int, torch.Tensor]]`
            ```
            {
                batch_num `int`: {
                    ground_truth_cls_id `int`: Tensor[point_num, 2]
                }
            }
            ```
            The outer dict contains maximum points of each batch.
            The inner dict contains maximum points of CAM in each ground-truth class.
        '''
        
        B, C, H, W = img_shape
        
        # All local max points
        '''
        points_all = {
            batch_num `int`: {
                ground_truth_cls_id `int`: Tensor[point_num, 2]
            }
        }
        '''
        points_all: "dict[int, dict[int, torch.Tensor]]" = {}
        
        for b in range(B):
            points_img: "dict[int, torch.Tensor]" = {}
            
            # Only process ground-truth class
            for cls in label_tensor[b].nonzero(as_tuple=False)[:, 0]:
                cls = cls.item()
                # CAM of batch `i` and class `ct`
                cam_target = cam_ms[b, cls] # [H, W]
                cam_target_np = cam_target.cpu().detach().numpy()
                
                # Find global maximum
                peak_max = torch.unsqueeze( (cam_target == torch.max(cam_target)).nonzero()[0], dim=0 )
                peak_max = peak_max.cpu().detach().numpy()

                # Find local maximums
                cam_filtered = ndi.maximum_filter(cam_target_np, size=3, mode='constant')
                peaks_temp = peak_local_max(cam_filtered, min_distance=20)
                # Only select value larger than threshold
                peaks_valid = peaks_temp[ cam_target_np[peaks_temp[:, 0], peaks_temp[:, 1]] > threshold ]

                if len(peaks_valid) == 0 :
                    peaks = peak_max
                else:
                    peaks = np.concatenate((peak_max, peaks_valid[1:]), axis=0)
                
                # Scale the point coordinate to SAM map size
                points = np.flip(peaks, axis=-1) * size_sam / H
                points = torch.from_numpy(points).float().to(self.device)
                points_img[cls] = points
            
            points_all[b] = points_img
        
        return points_all

    @torch.no_grad()
    def _aggregate_sam_cam(
        self,
        image_tensor: torch.Tensor,
        cam_ms: torch.Tensor,
        max_points: "dict[int, dict[int, torch.Tensor]]",
        threshold: float = 0,
        do_denorm: bool = False,
        size_sam: int = None,
        mask_select: int = -1,
    ) -> torch.Tensor :
        '''
        Aggregate SAM map and CAM to generate pseudo masks.
        
        Multiply SAM score with average CAM value within the SAM mask region.
        Then decide the class of each pixel by which class has the maximum score
        
        Arguments :
            image_tensor `Tensor` "[B, 3, H, W]": Input image Tensor
            cam_ms `Tensor` "[B, num_class, H, W]": CAM from `multi_scale_cam`
            max_points `dict[int, dict[int, Tensor]]`: Maximum points from `sample_local_max`
            threshold `float`: Threshold to determine whether a pixel is set to background class. Default is 0
            do_denorm `bool`: Whether to denormalize the input image
            size_sam `int`: Resize input image to the size before inputting to SAM. Default is original size (no resize)
            mask_select `int`: Which mask to select from SAM output. Set -1 to auto-select. Default is -1. SSC used 2
        
        Returns :
            `Tensor` "[B, H, W]": Result mask. Each pixel is in range `[0, CLS_NUM]` indicating the class of the pixel.
            `[0, CLS_NUM-1]` are original class ids. Id `CLS_NUM` is background.
        '''

        B, C, H, W = image_tensor.shape
        
        if size_sam is None : size_sam = H
        
        # De-normalization
        if do_denorm :
            image_denorm = denorm(image_tensor) * 255
            image_sam = F.interpolate(image_denorm, size=(size_sam, size_sam), mode='bilinear', align_corners=True)
        else :
            image_sam = F.interpolate(image_tensor, size=(size_sam, size_sam), mode='bilinear', align_corners=True)
        image_sam = image_sam.to(torch.uint8)
        
        # SAM encoder, embed images
        features_sam = self.sam(
            run_encoder_only=True,
            transformed_image=image_tensor,
            original_image_size=(H, W)
        )
        
        # SAM confidence score
        sam_conf = -1e5 * torch.ones_like(cam_ms) # [B, CLS, H, W]
        
        for b in range(B):
            for cls, point_all in max_points[b].items() :
                points = point_all.unsqueeze(0) # [1, point_num, 2]
                # Point labels to SAM. Use 1 for foreground
                points_label = torch.ones_like(points[:, :, 0]) # [1, point_num]

                # SAM decoder, predict 3 masks and their confidence scores
                # masks: [1, 3, H, W]
                # confs: [1, 1, 3]
                masks, _, confs = self.sam(
                    run_decoder_only = True,
                    features_sam = features_sam[b].unsqueeze(0),
                    original_image_size = (H, W),
                    point_coords = points,
                    point_labels = points_label,
                    multimask_output = True
                )

                # Select the best mask by mIoU between SAM mask and CAM
                #? SSC selects mask 2
                if mask_select < 0 :
                    target_idx, target_mask = select_best_sam_mask_by_cam_overlap(
                        masks,
                        cam_region = cam_ms[b, cls],
                        target_size = (H, W),
                        threshold = 0.1
                    )
                else :
                    target_idx = mask_select
                    target_mask = masks[0, mask_select]
                
                # Check whether mask selection failed
                if (mask_select < 0 and target_idx < 0) or target_idx >= confs.shape[1]:
                    target_idx = 2
                    target_mask = masks[0, 2]
                    # print(f"[Warning] Invalid best_idx={target_idx}, confs.shape={confs.shape}")
                    # continue  # skip invalid
                
                target_conf = confs[0, target_idx].unsqueeze(0).unsqueeze(0)
                target_conf = F.interpolate(target_conf, (H,W), mode='bilinear', align_corners=False)[0,0]
                target_score = target_conf[target_mask]
                # CAM score: Region-wise average with SAM mask region
                cam_mean = cam_ms[b, cls][target_mask].mean() if target_mask.any() else 0.0
                # Confidence score: Multiplfy CAM score and SAM score
                sam_conf[b, cls][target_mask] = target_score * cam_mean # [B, CLS, H, W]

        # For each pixel, find the class with the max confidence score
        # `torch.max` returns the maximum value itself and index of the value in the Tensor
        # `pgt_score`: The maximum confidence score of each pixel among all classes (not used)
        # `pgt_sam`: The class of each pixel
        pgt_sam: torch.Tensor; pgt_score: torch.Tensor;
        pgt_score, pgt_sam = sam_conf.max(dim=1) # [B, H, W]
        # Pixels with confidence score < 0 are set to background class
        pgt_sam[pgt_score < threshold] = 93
        pgt_score[pgt_score < threshold] = 0
        
        return pgt_sam

    ### Utility ###

    def _monitor(self, record_dict: "dict[str, float]", epoch: int):
        for key, value in record_dict.items():
            self.writer.add_scalar(key, value, epoch)
        if self.gpu_id == 0 :
            console.print(record_dict)
    
    def _save_checkpoint(self, record_dict: "dict[str, float]", epoch: int):
        checkpoint_name = f"{self.save_model_name}_epoch_{epoch}.pth.tar"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.module.cpu().state_dict() if self.using_ddp else self.model.cpu().state_dict(),
                "model_ema": self.ema.cpu().ema_model.state_dict(),
                "opt": self.opt.state_dict(),
                "record_dict": record_dict,    
            },
            checkpoint_name
        )
        console.print(f"Saved checkpoint: \"{checkpoint_name}\"")

if __name__ == "__main__" :
    import time
    
    start = time.time()
    
    for _ in range(100000) :
        a = torch.rand(224, 224)
        
        # a_f = a.view(-1)
        # b = torch.argmax(a_f)
        # coord_w = torch.div(b, 224, rounding_mode="trunc")
        # coord_h = b % 224
        # peak_max = torch.cat((coord_w.view(1, 1), coord_h.view(1, 1)), dim=-1)
        
        # a_f = a.view(-1)
        # row, col = divmod(a_f.argmax().item(), 224)
        # peak_max = torch.tensor([[row, col]])
        
        peak_max = torch.unsqueeze((a==torch.max(a)).nonzero()[0], dim=0)
    
    end = time.time()
    print(end - start)
    
    # 1: 32.867; f: 33.097
    # 2: 32.027; f: 31.912
    # 3: 29.521; c: 29.007