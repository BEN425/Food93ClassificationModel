import os
import random
import shutil
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
import albumentations as A
import cv2


from dataset import FoodDataset, FoodDatasetWithMasks, TestDataset
from training_loop import Trainer
from model.ResNet_modified import ModifiedResNet
from cfgparser import CfgParser

import pillow_avif # AVIF format support for PIL

from rich import get_console
import logging
console = get_console()
logger  = logging.getLogger("FoodImageCode")


### Main Function ###

def main(cfg: dict) :
    try :

        ##### Initialization #####
        
        logger.info("Initializing...")
        console.print("Initializing...")
        results = None
        
        init_seed(cfg["SEED"])

        # Setup DDP
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
        
        logger.info(f"world_size = {world_size}; ddp = {using_ddp}; device = {device}; seed = {cfg['SEED']}")
        
        ##### Load dataset #####
        
        logger.info("Loading dataset...")
        console.print("Loading dataset...")
        dataloaders = load_dataset(cfg, using_ddp, rank)
        
        ##### Load model #####
        
        logger.info("Loading model...")
        console.print("Loading model...")
        
        # Load classification model
        model = load_model(cfg)
        start_epoch, end_epoch = 0, cfg["EPOCHS"]
        
        # Load SAM model
        if cfg["USE_CPM"] :
            console.print("Loading SAM model...")
            sam = load_sam_model(cfg)
        else :
            sam = None
        
        logger.info(f"Model: \"{cfg['MODEL']['TYPE']}, {cfg['MODEL']['NAME']}\"\n{model}")
        logger.info(f"SAM: \"{cfg['SAM_DIR']}\"\n{sam}")
        
        ###### Load optimizer #####

        opt_name = cfg["MODEL"]["OPTIMIZER"].strip().lower()

        # SGD
        if "sgd" in opt_name :
            opt = torch.optim.SGD(
                model.parameters(),
                lr=cfg["MODEL"]["LR"],
                momentum=cfg["MODEL"]["MOMENTUM"],
                weight_decay=cfg["MODEL"]["WEIGHT_DECAY"]
            )
        # Default: AdamW
        else :
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["MODEL"]["LR"],
                weight_decay=cfg["MODEL"]["WEIGHT_DECAY"]
            )

        logger.info(f"Optimizer:\n{opt}")

        ##### Resume from checkpoint #####
        
        if using_ddp : dist.barrier()
        if cfg["RESUME"] : 
            console.print("Resuming from checkpoint...")

            checkpoint = torch.load(
                cfg["CHECKPOINT_PATH"],
                map_location={"cuda:0": f"cuda:{device}"} if using_ddp else "cpu"
            )
            model.load_state_dict(checkpoint["model"])
            model = model.to(device)
            opt.load_state_dict(checkpoint["opt"])
            start_epoch += checkpoint["epoch"] + 1
            end_epoch += checkpoint["epoch"]
            
            console.print(f"Resumed from epoch {start_epoch} to {end_epoch}")
            logger.info(f"Resumed from \"{cfg['CHECKPOINT_PATH']}\": Epoch {start_epoch} to {end_epoch}")
        else :
            logger.info(f"Start training from epoch {start_epoch} to {end_epoch}")
        model = model.to(device)
        
        ##### Wrap model with DDP #####
        
        if using_ddp :
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, find_unused_parameters=True)
        
        ##### Training #####

        # Load class frequencies and entropies for focal loss parameter ɑ
        console.print("Loading class weights...")
        cls_weight = load_class_weight(cfg, device)
        
        logger.info(f"Class Weights:\n{cls_weight}")

        if using_ddp : dist.barrier()
        # Training
        logger.info("Setup trainer...")
        console.print("Training...")
        trainer = Trainer(
            dataloaders = dataloaders,
            model = model,
            opt = opt,
            device = device,
            cfg = cfg,
            sam = sam,
            using_ddp = using_ddp,
            logger=logger,
        )
        
        logger.info("Start training...")
        results = trainer.train(
            start_epoch, end_epoch, cfg,
            class_alpha=cls_weight,
            gamma=cfg["LOSS"]["GAMMA"],
        )
        logger.info("Training finished.")

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

### Initialization Functions ###

# Initialize random seeds and set deterministic algorithms
def init_seed(seed: int, cuda_deterministic=True) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = not cuda_deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

# Initialization for DDP, not used in single GPU
def ddp_setup() :
    #! Explicity specify addr and port might get error
    #! RuntimeError: nonce == returnedNonce INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/TCPStore.cpp":418, please report a bug to PyTorch. Ping failed, invalid nonce returned
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"

    import datetime

    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=30))
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

### Loader Functions ###

# Load pretraind classification model
def load_model(cfg: dict) -> nn.Module :
    
    # Load ResNet38
    if cfg["MODEL"]["TYPE"].strip().lower() == "resnet38" :
        from model.ResNet_38d import Net_CAM, convert_mxnet_to_torch
        
        console.print("Model: ResNet38")
        model = Net_CAM(D=256, C=cfg["MODEL"]["CATEGORY_NUM"])
        model.load_state_dict(
            convert_mxnet_to_torch(os.path.join(cfg["ROOT"], "pretrained/resnet_38d.params")),
            strict=False
        )
    # Load ResNet50 (default model)
    else :
        console.print("Model: ResNet50")
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

# Load SAM model
def load_sam_model(cfg: dict) -> nn.Module :
    from segment_anything import sam_model_registry

    sam_name = None
    model_names = ["vit_h", "vit_l", "vit_b"]
    for name in model_names :
        if name in cfg["SAM_DIR"].lower():
            sam_name = name
            break
    
    assert sam_name is not None, f"SAM model name should be one of {model_names}"
    
    return sam_model_registry[sam_name](checkpoint=cfg["SAM_DIR"])

# Load class weights for focal loss
def load_class_weight(cfg: dict, device: torch.device) -> "float | torch.Tensor" :
    alpha = cfg["LOSS"]["ALPHA"]
    
    # Setting alpha as hyperparameter
    if isinstance(alpha, (int, float)) : return alpha
    
    # Setting alpha for each class
    if isinstance(alpha, (list, tuple)) :
        assert len(alpha) == cfg["MODEL"]["CATEGORY_NUM"], "ALPHA list length should equal to category numbers"
        return torch.tensor(alpha).to(device)
    
    # Read alpha from a file
    if isinstance(alpha, str) :
        console.print(f"Loading class weights from \"{alpha}\"")
        with open(alpha, "r") as file :
            alpha_list = [float(line) for line in file.readlines()]
        assert len(alpha_list) == cfg["MODEL"]["CATEGORY_NUM"], "ALPHA list length should equal to category numbers"
        
        log_path = os.path.join(cfg["SAVE_DIR"], "logs", cfg["SAVE_SUB_NAME"])
        shutil.copyfile(alpha, os.path.join(log_path, os.path.basename(alpha)))
        
        return torch.tensor(alpha_list).to(device)
    
    console.print("Class alpha is not set, using default value 0.25")
    return 0.25

# Load datasets from csv file and apply preprocessing
def load_dataset(cfg: dict, using_ddp: bool = False, rank: int = 0) -> "dict[str, DataLoader]":
    
    # Initialize random seed of dataloader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Transforms
    if cfg["USE_SSC"] :
        train_trfs = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            A.CenterCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5, interpolation=cv2.INTER_LINEAR),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Affine(
                rotate=(-5, 5), 
                translate_percent=(0.02, 0.02), 
                scale=(0.95, 1.05), 
                shear=2, 
                interpolation=cv2.INTER_LINEAR, 
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.5
            ),
        ])
        valid_trfs = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ToTensorV2()
        ])
    else :
        train_trfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.CenterCrop(224),
            # Apply random transform to augment images
            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                shear=2
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        valid_trfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.CenterCrop(224),
            # transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    logger.info(f"Train transforms:\n{train_trfs}")
    logger.info(f"Valid transforms:\n{valid_trfs}")

    # Load dataset and dataloader

    #? Split batch to multiple GPUs
    train_ba_size = cfg["BATCH_SIZE"]      // dist.get_world_size() if using_ddp else cfg["BATCH_SIZE"]
    valid_ba_size = cfg["EVAL_BATCH_SIZE"] // dist.get_world_size() if using_ddp else cfg["EVAL_BATCH_SIZE"]

    if cfg["USE_SSC"] :
        train_dataset = FoodDatasetWithMasks(cfg["TRAIN_CSV_DIR"], transform=train_trfs, hsv=False, root=cfg["ROOT"], sam_dir=cfg["SAM_MASK_DIR"])
        valid_dataset = FoodDatasetWithMasks(cfg["VALID_CSV_DIR"], transform=valid_trfs, hsv=False, root=cfg["ROOT"], sam_dir=cfg["SAM_MASK_DIR"])
    else :
        train_dataset = FoodDataset(cfg["TRAIN_CSV_DIR"], transform=train_trfs, hsv=False, root=cfg["ROOT"])
        valid_dataset = FoodDataset(cfg["VALID_CSV_DIR"], transform=valid_trfs, hsv=False, root=cfg["ROOT"])

    #? Use `DistributedSampler` for DDP 
    train_sampler = DistributedSampler(train_dataset, seed=cfg["SEED"], rank=rank, shuffle=True)  if using_ddp else None
    valid_sampler = DistributedSampler(valid_dataset, seed=cfg["SEED"], rank=rank, shuffle=False) if using_ddp else None

    # Create a generator for deterministic dataloading
    g = torch.Generator()
    g.manual_seed(cfg["SEED"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_ba_size,
        drop_last=True,
        shuffle=not using_ddp, #? Sampler handles shuffling for DDP
        num_workers=cfg["WORKERS"],
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=g
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_ba_size,
        drop_last=True, shuffle=False,
        num_workers=cfg["WORKERS"],
        sampler=valid_sampler,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # for i, data in enumerate(train_dataloader) :
    #     console.print(f"{i}: {data[0]}")
    #     if i >= 5 : break
    
    # exit()

    dataset = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    return dataset


if __name__ == "__main__":
    import time
    import pprint
    start = time.time()
    
    console.print("Parsing config...")

    try :
        # Read config
        config_path = os.environ.get("CONFIG_PATH", "./cfg/Setting.yml")
        cfgparser = CfgParser(config_path=config_path)
        cfg = cfgparser.cfg_dict
        log_path = os.path.join(cfg["SAVE_DIR"], "logs", cfg["SAVE_SUB_NAME"])
        
        console.print(f"Cconfig path: \"{config_path}\"")
        console.print(cfg)
        console.print(
            "Make sure to set up environment"
            ", set up configs and complete preliminaries.", style="yellow")
        input("Press ENTER to continue > ")
        
        # Check directory empty to prevent accidentally overwriting results
        # if os.listdir(log_path) :
        #     raise Exception(f"Result folder \"{log_path}\" is not empty. Please change \"SAVE_SUB_NAME\" in config file or remove the results.")
        
        # Setup logger
        os.makedirs(log_path, exist_ok=True)
        log_file = open(
            os.path.join(log_path, "record.log"),
            "a" if cfg["RESUME"] else "w"
        )
        handler = logging.StreamHandler(log_file)
        formatter = logging.Formatter(
            "[%(levelname)-5s][%(asctime)s] (%(filename)s:%(lineno)d) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info(f"Config path: \"{config_path}\"")
        logger.info(f"Config:\n{pprint.pformat(cfg, indent=4)}")

        # Record csv files
        shutil.copyfile(cfg["ALL_CSV_DIR"],   os.path.join(log_path, os.path.basename(cfg["ALL_CSV_DIR"])))
        shutil.copyfile(cfg["TRAIN_CSV_DIR"], os.path.join(log_path, os.path.basename(cfg["TRAIN_CSV_DIR"])))
        shutil.copyfile(cfg["VALID_CSV_DIR"], os.path.join(log_path, os.path.basename(cfg["VALID_CSV_DIR"])))
        shutil.copyfile(cfg["TEST_CSV_DIR"],  os.path.join(log_path, os.path.basename(cfg["TEST_CSV_DIR"])))
        # Record config file
        shutil.copyfile(config_path, os.path.join(log_path, os.path.basename(config_path)))
        
        main(cfg)
    except Exception :
        logger.error("Error", exc_info=True)
        console.print_exception(show_locals=False)
    finally :
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        end = time.time()
        dt = end - start
        console.print(f"Time used = {dt:.3f}")
        logger.info(f"Time used = {dt:.3f}")
        if logger.hasHandlers() :
            log_file.close()
