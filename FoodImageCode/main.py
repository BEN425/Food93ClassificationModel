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
import torch.multiprocessing as mp

from dataset import FoodDataset
from training_loop import Trainer
from model.ResNet_modified import ModifiedResNet
from cfgparser import CfgParser

import pillow_avif # AVIF format support for PIL

from rich import get_console
console = get_console()


### Main Function ###

def main(device: int, cfg: dict, world_size: int = 1) :
    console.print("Initializing...")
    init_seed(cfg["SEED"])
    ddp_setup(device, world_size)
    
    # device = rank
    
    console.print("Loading dataset...")
    dataset = load_dataset(cfg)
    
    # Load model
    console.print("Loading model...")
    model = load_model(cfg)
    model = model.to(device)
    start_epoch, end_epoch = 0, cfg["EPOCHS"]
    model = DDP(model, device_ids=[cfg["GPU_ID"]])

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
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch += checkpoint["epoch"]
        end_epoch += checkpoint["epoch"]

    # Load class frequencies and entropies for focal loss parameter ɑ
    console.print("Loading class frequencies and entropies...")
    cls_freq = load_class_frequency(cfg)
    cls_entropy = load_class_entropy(cfg)
    cls_entropy /= cls_entropy.max()
    # console.print(cls_freq)
    # console.print(cls_entropy)

    # Training
    console.print("Training...")
    trainer = Trainer(dataset, model, opt, device, cfg)
    results = trainer.train(
        start_epoch, end_epoch, cfg,
        class_alpha=cls_freq.to(device),
        gamma=2
    )

    # Print training result
    console.print(f"Training results (Epoch {results[-1]['epoch']}):")
    console.print(results[-1])

    # Store result locally
    with open("results.json", "w") as file :
        json.dump(results, file, indent=4, ensure_ascii=False)
    console.print("Training results saved at \"results.json\"")

# Initialize random seeds
def init_seed(seed: int, cuda_deterministic=True) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = not cuda_deterministic

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Load pretraind ResNet50 model
def load_model(cfg) -> nn.Module :
    model = ModifiedResNet(3, 64, cfg["MODEL"]["CATEGORY_NUM"])

    # Load the pretrained weights into ModifiedResNet
    pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()

    # Filter out unnecessary keys and find out which layers match
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items() if k in model_dict and k not in 
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
def load_dataset(cfg) -> "dict[str, DataLoader]":
    # Transforms
    train_trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        # Apply random transform to augment images
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])
    valid_trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])

    # Load dataset and dataloader
    train_dataset = FoodDataset(cfg["TRAIN_CSV_DIR"], transform=train_trfs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        drop_last=True,
        num_workers=cfg["WORKERS"],
        sampler=DistributedSampler(train_dataset)
    )
    valid_dataset = FoodDataset(cfg["VALID_CSV_DIR"], transform=valid_trfs)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["EVAL_BATCH_SIZE"],
        drop_last=True, shuffle=False,
        num_workers=cfg["WORKERS"],
        sampler=DistributedSampler(valid_dataset, shuffle=False)
    )

    dataset = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    return dataset

if __name__ == "__main__":
    
    console.print("Parsing config...")

    try :
        cfgparser = CfgParser(config_path="./cfg/Setting.yml")
        cfg = cfgparser.cfg_dict

        epochs = cfg["EPOCHS"]
        batch_size = cfg["BATCH_SIZE"]
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(cfg, world_size,), nprocs=world_size, join=True)
        # main(cfg)
    except Exception :
        console.print_exception(show_locals=False)
