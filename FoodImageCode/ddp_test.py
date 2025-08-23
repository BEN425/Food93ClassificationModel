import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
# For DDP with multiple GPUs
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.multiprocessing as mp

from dataset import TestDataset
from model.ResNet_modified import ModifiedResNet
from loss import cal_class_focal_loss, cal_ssc_loss, select_best_sam_mask_by_cam_overlap, denorm, calc_cpm_loss
from metrics import cal_f1_score_acc, cal_tp_fp_fn_tn, cal_error_nums, evaluate_dataset

import pillow_avif # AVIF format support for PIL

from tqdm import tqdm

from rich import get_console
console = get_console()

RESUME = False
TEST_METRICS = True


# Initialization for DDP
def ddp_setup() :
    #! Explicity specify addr and port might get error
    #! RuntimeError: nonce == returnedNonce INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/TCPStore.cpp":418, please report a bug to PyTorch. Ping failed, invalid nonce returned
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"

    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

# Load pretraind ResNet50 model
def load_model() -> nn.Module :
    
    console.print("Model: ResNet50")
    model = ModifiedResNet(
        in_channels  = 3,
        out_channels = 64,
        num_classes  = 93, 
    )

    # Load the pretrained weights into ModifiedResNet
    pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()

    # Filter out unnecessary keys and find out which layers match
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items() if k in model_dict \
        and v.shape == model_dict[k].shape and k not in
        [
            "fc.weight", 
            "fc.bias", 
            "layer3.0.conv1.weight", 
            "layer3.0.conv2.weight", 
            "layer3.0.conv3.weight", 
            "layer4.0.conv1.weight", 
            "layer4.0.conv2.weight", 
            "layer4.0.conv3.weight"
        ]
    }

    # Update the model dictionary with the pretrained weights
    model_dict.update(pretrained_dict_filtered)
    # Load the state_dict into the model
    model.load_state_dict(model_dict)

    return model

# Load datasets from csv file and apply preprocessing
def load_dataset(using_ddp: bool = False, rank: int = 0) -> "dict[str, DataLoader]":
    # Transforms
    train_trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop(224),
        # Apply random transform to augment images
        
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # transforms.RandomAffine(
        #     degrees=5,
        #     translate=(0.02, 0.02),
        #     scale=(0.95, 1.05),
        #     shear=2
        # ),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])
    valid_trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])

    # Load dataset and dataloader

    # Split batch to multiple GPUs
    train_ba_size = 16 // dist.get_world_size() if using_ddp else 8
    valid_ba_size = 16 // dist.get_world_size() if using_ddp else 8

    train_csv = "/work/u6140562/FoodS2C/FoodImageCode/csv/ai_single_food_200/AllFoodImage_train_ratio811.csv"
    valid_csv = "/work/u6140562/FoodS2C/FoodImageCode/csv/ai_single_food_200/AllFoodImage_valid_ratio811.csv"
    train_dataset = TestDataset(train_csv, transform=train_trfs, hsv=False, root="/work/u6140562/FoodS2C/")
    valid_dataset = TestDataset(valid_csv, transform=valid_trfs, hsv=False, root="/work/u6140562/FoodS2C/")

    #? Use `DistributedSampler` for DDP 
    train_sampler = DistributedSampler(train_dataset, seed=200, rank=rank, shuffle=True)  if using_ddp else None
    valid_sampler = DistributedSampler(valid_dataset, seed=200, rank=rank, shuffle=False) if using_ddp else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_ba_size,
        drop_last=True,
        shuffle=not using_ddp,
        num_workers=4,
        sampler=train_sampler
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_ba_size,
        drop_last=True, shuffle=False,
        num_workers=4,
        sampler=valid_sampler
    )

    dataset = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    return dataset

if __name__ == "__main__" :
    try :
        
        ##### Initialization #####
        
        world_size = int(os.environ.get("WORLD_SIZE", 1)) 
        using_ddp = world_size > 1
        if using_ddp :
            ddp_setup()
            rank = int(os.environ.get("LOCAL_RANK", 0))
            device = rank % torch.cuda.device_count()
            console.print(f"DDP with {world_size} GPUs")
        else :
            rank = 0
            device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
        console.print(f"world_size = {world_size}; ddp = {using_ddp}; rank = {rank}; device = {device}")
    
        ##### Load dataset #####
        
        console.print("Loading dataset...")
        dataset = load_dataset(using_ddp, rank)
        train_dataloader = dataset["train"]
        valid_dataloader = dataset["valid"]
        
        ##### Load model #####
        
        console.print("Loading model...")
        
        # Load classification model
        model = load_model()
        model = model.to(device)
        start_epoch, end_epoch = 0, 5
        if using_ddp :
            ddp_model = DDP(model, device_ids=[device], find_unused_parameters=True)

        ###### Load optimizer #####

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=0.0001,
            weight_decay=0.9
        )
        
        ##### Resume from checkpoint #####
        
        if RESUME : 
            console.print("Resuming from checkpoint...")

            checkpoint = torch.load("/work/u6140562/FoodS2C/FoodImageCode/Results/checkpoints/8_2_aisingle_200/8_3_ModifiedResNet50_93_epoch_0.pth.tar")
            if using_ddp :
                ddp_model.module.load_state_dict(checkpoint["model"])
            else :
                model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["opt"])
            start_epoch += checkpoint["epoch"] + 1
            end_epoch += checkpoint["epoch"]
            
            console.print(f"Resumed from epoch {start_epoch} to {end_epoch}")
        
        ##### Training #####
        
        record_epoch = []
            
        save_dir = "temp"
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(start_epoch, end_epoch) :
            
            if using_ddp :
                train_dataloader.sampler.set_epoch(epoch)
                valid_dataloader.sampler.set_epoch(epoch)
            
            # Progress bar
            if rank == 0 :
                train_dataloader_bar = tqdm(
                    enumerate(train_dataloader),
                    desc=f"GPU[{rank}]: Epoch {epoch}/{start_epoch}-{end_epoch-1}",
                    # position=self.gpu_id,
                    leave=True,
                    total=len(train_dataloader)
                )
            else :
                train_dataloader_bar = enumerate(train_dataloader)

            # Record metrics of the epoch
            record_dict = {
                "train_loss"      : 0,
                "train_microf1"   : 0,
                "train_macrof1"   : 0,
                "train_micro_acc" : 0,
                "train_ham_loss"  : 0,
                "train_zero_acc"  : 0,
                "valid_loss"      : 0,
                "valid_microf1"   : 0,
                "valid_macrof1"   : 0,
                "valid_micro_acc" : 0,
                "valid_ham_loss"  : 0,
                "valid_zero_acc"  : 0,
            }

            # TP, TN, FN, FP of each class, for calculating Macro F1
            tp = torch.zeros(93).to(device)
            fp = torch.zeros(93).to(device)
            fn = torch.zeros(93).to(device)
            tn = torch.zeros(93).to(device)
            # For calculating Hamming acc and Zero acc
            err_label = 0
            err_data  = 0
            label_count = 0
            data_count  = 0
            record_epoch.append(record_dict)

            model.to(device)
            model.train()
            
            for idx, (img, label, sam_img) in train_dataloader_bar :
                print(f"[{rank}]: {idx}")
                opt.zero_grad()
                
                img, label = img.to(device), label.to(device).to(torch.float32)
                result = model(img)         # Model output
                out = result["pred"]
                logits = torch.sigmoid(out) # Probablility of each class
                pred = torch.round(logits)  # Prediction result, threshold = 0.5
                loss = cal_class_focal_loss(out, label, mean=True)
                
                loss.backward()
                opt.step()
                
                # Metrics
                tp_fp_fn_tn = cal_tp_fp_fn_tn(pred, label)
                train_err_cor = cal_error_nums(pred, label)
                label_count += label.numel()
                data_count += len(label)

                # Record each iter loss
                record_dict["train_loss"] += loss.item()
                tp += tp_fp_fn_tn[0]
                fp += tp_fp_fn_tn[1]
                fn += tp_fp_fn_tn[2]
                tn += tp_fp_fn_tn[3]
                err_label += train_err_cor[0]
                err_data  += train_err_cor[1]
            
            train_metrics_results = cal_f1_score_acc(tp, fp, fn, tn)
            record_dict["train_microf1"]     = float(train_metrics_results["microf1"])
            record_dict["train_macrof1"]     = float(train_metrics_results["macrof1"])
            record_dict["train_micro_acc"]   = float(train_metrics_results["micro_acc"])
            record_dict["train_ham_loss"]    = float(err_label / label_count)
            record_dict["train_zero_acc"]    = float(1 - err_data / data_count)
            record_dict["train_loss"]       /= len(train_dataloader)

            # Test validation set
            if TEST_METRICS :
                model.eval()
                
                # TP, TN, FN, FP of each class, for calculating Macro F1
                tp = torch.zeros(93).to(device)
                fp = torch.zeros(93).to(device)
                fn = torch.zeros(93).to(device)
                tn = torch.zeros(93).to(device)
                # For calculating Hamming acc and Zero acc
                err_label = 0
                err_data  = 0
                label_count = 0
                data_count  = 0
                
                for idx, (img, label, _) in enumerate(valid_dataloader):
                    img = img.to(device)
                    label = label.to(device, dtype=torch.float32)
                    
                    out = model(img)["pred"]
                    logits = torch.sigmoid(out)
                    pred = torch.round(logits) # Threshold = 0.5
                    loss = cal_class_focal_loss(out, label)
                    
                    # Metrics
                    tp_fp_fn_tn = cal_tp_fp_fn_tn(pred, label)
                    train_err_cor = cal_error_nums(pred, label)
                    label_count += label.numel()
                    data_count += len(label)

                    # Record each iter loss
                    record_dict["valid_loss"] += loss.item()
                    tp += tp_fp_fn_tn[0]
                    fp += tp_fp_fn_tn[1]
                    fn += tp_fp_fn_tn[2]
                    tn += tp_fp_fn_tn[3]
                    err_label += train_err_cor[0]
                    err_data  += train_err_cor[1]

                valid_metrics_results = cal_f1_score_acc(tp, fp, fn, tn)
                record_dict["valid_microf1"]     = float(valid_metrics_results["microf1"])
                record_dict["valid_macrof1"]     = float(valid_metrics_results["macrof1"])
                record_dict["valid_micro_acc"]   = float(valid_metrics_results["micro_acc"])
                record_dict["valid_ham_loss"]    = float(err_label / label_count)
                record_dict["valid_zero_acc"]    = float(1 - err_data / data_count)
                record_dict["valid_loss"]       /= len(valid_dataloader)

            # Show result and save checkpoint
            if rank == 0 :
                console.print(record_dict)
                checkpoint_name = f"cp_epoch_{epoch}.pth.tar"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict() if not using_ddp else ddp_model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "record_dict": record_dict,
                    },
                    os.path.join(save_dir, checkpoint_name)
                )
    
    
    except Exception :
        console.print_exception()
    
    finally :
        # Ensure the process group is destroyed
        if using_ddp :
            dist.destroy_process_group()
            console.print(f"Destroyed process group {rank}")

        import gc
        torch.cuda.empty_cache()
        gc.collect()
