import os
import random
import json

import numpy as np
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

from dataset import FoodDataset
from training_loop import Trainer
from model.ResNet_modified import ModifiedResNet
from cfgparser import CfgParser

import pillow_avif # AVIF format support for PIL

from rich import get_console
console = get_console()


### Main Function ###

def main(cfg: dict) :
    try :

        # Initialization
        console.print("Initializing...")

        results = None
        world_size = int(os.environ.get("WORLD_SIZE", 1)) 
        using_ddp = world_size > 1
        if using_ddp :
            ddp_setup()
            rank = int(os.environ.get("LOCAL_RANK", 0))
            device = rank % torch.cuda.device_count()
            console.print(f"DDP with {world_size} GPUs")
        else :
            rank = 0
            device = torch.device(f"cuda:{cfg['GPU_ID']}")
        
        init_seed(cfg["SEED"])
        
        # Load dataloader
        console.print("Loading dataset...")
        dataset = load_dataset(cfg, using_ddp, rank)
        
        # Load model
        console.print("Loading model...")
        model = load_model(cfg)
        model = model.to(device)
        start_epoch, end_epoch = 0, cfg["EPOCHS"]
        if using_ddp :
            ddp_model = DDP(model, device_ids=[device])

        # Optimizer, change to AdamW
        # opt = torch.optim.SGD(
        #     model.parameters(),
        #     lr=cfg["MODEL"]["LR"],
        #     momentum=cfg["MODEL"]["MOMENTUM"],
        #     weight_decay=cfg["MODEL"]["WEIGHT_DECAY"]
        # )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["MODEL"]["LR"],
            weight_decay=cfg["MODEL"]["WEIGHT_DECAY"]
        )

        # Resume from checkpoint
        if cfg["RESUME"] : 
            console.print("Resuming from checkpoint...")

            checkpoint = torch.load(cfg["CHECKPOINT_PATH"])
            if using_ddp :
                ddp_model.module.load_state_dict(checkpoint["model"])
            else :
                model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["opt"])
            start_epoch += checkpoint["epoch"] + 1
            end_epoch += checkpoint["epoch"]
            console.print(f"Resumed from epoch {start_epoch} to {end_epoch}")

        # Load class frequencies and entropies for focal loss parameter ɑ
        # console.print("Loading class frequencies and entropies...")
        # cls_freq = load_class_frequency(cfg)
        # cls_entropy = load_class_entropy(cfg)
        # cls_entropy /= cls_entropy.max()
        # cls_entropy *= .99
        # console.print(cls_freq)
        # console.print(cls_entropy)

        # if device == 0 :
        #     dist.barrier()

        # Training
        console.print("Training...")
        trainer = Trainer(
            dataset,
            ddp_model if using_ddp else model,
            opt,
            device,
            cfg,
            using_ddp
        )
        results = trainer.train(
            start_epoch, end_epoch, cfg,
            class_alpha=cfg["LOSS"]["ALPHA"],
            gamma=cfg["LOSS"]["GAMMA"],
        )

    finally :

        # Write training result and config to json file
        if rank == 0 and results is not None :

            # Print training result
            console.print(f"Training results (Epoch {results[-1]['epoch']}):")
            console.print(results[-1])

            # Store result locally
            result_path = os.path.join(cfg["SAVE_DIR"], "logs", cfg["SAVE_SUB_NAME"], "results.json")
            json_write = {}
            json_write["config"] = cfg
            json_write["result"] = results
            with open(result_path, "w") as file :
                json.dump(json_write, file, indent=4, ensure_ascii=False)
            console.print(f"Training results saved at \"{result_path}\"")

        # Ensure the process group is destroyed
        if using_ddp :
            dist.destroy_process_group()
            console.print("Destroyed process group.")

# Initialize random seeds
def init_seed(seed: int, cuda_deterministic=True) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = not cuda_deterministic

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
def load_model(cfg) -> nn.Module :
    model = ModifiedResNet(
        in_channels     = cfg["MODEL"]["INCHANNELS"],
        out_channels    = 64,
        num_classes     = cfg["MODEL"]["CATEGORY_NUM"],
        use_cbam_layers = cfg["MODEL"]["CBAM"], 
        use_se_layers   = cfg["MODEL"]["SENET"]  
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

# Load class frequency and entropy for focal loss
# Class frequency is loaded from "Database/class_freq.txt"
# Class entropy is loaded from "Database/class_entropy.txt"
# Execute "utility/calc_class_freq_entropy.py" to generate the data
def load_class_frequency(cfg) -> torch.Tensor :
        cls_freq = torch.zeros(cfg["MODEL"]["CATEGORY_NUM"])
        
        with open(os.path.join(cfg["DATA_BASE_DIR"], "..", "class_freq.txt"), "r") as file :
            for i, line in enumerate(file.readlines()) :
                cls_freq[i] = float(line)
        
        return cls_freq
def load_class_entropy(cfg) -> torch.Tensor :
    cls_entropy = torch.zeros(cfg["MODEL"]["CATEGORY_NUM"])
    
    with open(os.path.join(cfg["DATA_BASE_DIR"], "..", "class_entropy.txt"), "r") as file :
        for i, line in enumerate(file.readlines()) :
            cls_entropy[i] = float(line)
    
    return cls_entropy * .99

# Load datasets from csv file and apply preprocessing
def load_dataset(cfg: dict, using_ddp: bool = False, rank: int = 0) -> "dict[str, DataLoader]":
    # Transforms
    # TODO: Calculate mean and std of dataset
    train_trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        # Apply random transform to augment images
        
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # transforms.RandomAffine(
        #     degrees=5,
        #     translate=(0.02, 0.02),
        #     scale=(0.95, 1.05),
        #     shear=2
        # ),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        # transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])
    valid_trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        # transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])

    # Load dataset and dataloader

    # Split batch to multiple GPUs
    train_ba_size = cfg["BATCH_SIZE"]       // dist.get_world_size() if using_ddp else cfg["BATCH_SIZE"]
    valid_ba_size = cfg["EVAL_BATCH_SIZE"]  // dist.get_world_size() if using_ddp else cfg["EVAL_BATCH_SIZE"]

    train_dataset = FoodDataset(cfg["TRAIN_CSV_DIR"], transform=train_trfs, hsv=False, root=cfg["ROOT"])
    valid_dataset = FoodDataset(cfg["VALID_CSV_DIR"], transform=valid_trfs, hsv=False, root=cfg["ROOT"])

    #? Use `DistributedSampler` for DDP 
    train_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True)  if using_ddp else None
    valid_sampler = DistributedSampler(valid_dataset, rank=rank, shuffle=False) if using_ddp else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_ba_size,
        drop_last=True,
        shuffle=not using_ddp,
        num_workers=cfg["WORKERS"],
        sampler=train_sampler
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_ba_size,
        drop_last=True, shuffle=False,
        num_workers=cfg["WORKERS"],
        sampler=valid_sampler
    )

    dataset = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    return dataset

if __name__ == "__main__":
    import time
    start = time.time()
    
    console.print("Parsing config...")

    try :
        cfgparser = CfgParser(config_path="./cfg/Setting.yml")
        cfg = cfgparser.cfg_dict
        console.print(cfg)
        console.print("This is the config of training.")
        console.print(
            "Make sure to run \"make_image_csv.py\""
            ", set up configs and complete preliminaries.", style="yellow")
        input("Press ENTER to continue > ")
        main(cfg)
    except Exception :
        console.print_exception(show_locals=False)
    finally :
        end = time.time()
        dt = end - start
        console.print(f"Time used = {dt}")
