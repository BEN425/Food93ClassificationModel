import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import random
import numpy as np
from cfgparser import CfgParser

from dataset import FoodDataset
from training_loop import Trainer
from model.ResNet_modified import ModifiedResNet

import pillow_avif # AVIF format support for PIL

from rich import get_console
console = get_console()


### Main Function ###

def main(cfg: "dict[str, int|bool|float|str]"):
    console.print("Loading model and datasets...")
    
    device = torch.device(f"cuda:{cfg['GPU_ID']}") 
    
    init_seed(cfg["SEED"])
    dataset, model, opt = load_objs(cfg, device)
    
    start_epoch, end_epoch = 0, cfg["EPOCHS"]
    if cfg["RESUME"]:
        model, opt, start_epoch, end_epoch = resume_setting(
            cfg["CHECKPOINT_PATH"], model, opt, start_epoch, end_epoch)
    
    trainer = Trainer(dataset, model, opt, device, cfg)
    trainer.training_loop_new(start_epoch, end_epoch, cfg)

# Initialize random seeds
def init_seed(seed: int, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = not cuda_deterministic

# Load ResNet50 model
def load_model(cfg):
    
    model = ModifiedResNet(3, 64, cfg["MODEL"]["CATEGORY_NUM"])
    pretrained_resnet = models.resnet50(pretrained=True)

    # Load the pretrained weights into ModifiedResNet
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()

    # Filter out unnecessary keys and find out which layers match
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items() if k in model_dict and k not in 
        ["fc.weight", 
         "fc.bias", 
         "layer3.0.conv1.weight", 
         "layer3.0.conv2.weight", 
         "layer3.0.conv3.weight", 
         "layer4.0.conv1.weight", 
         "layer4.0.conv2.weight", 
         "layer4.0.conv3.weight"]}

    # Update the model dictionary with the pretrained weights
    model_dict.update(pretrained_dict_filtered)
    # Load the state_dict into the model
    model.load_state_dict(model_dict)
    return model

# Load datasets, model and optimizer
def load_objs(cfg, device):
    trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        # Apply random transform to augment images
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])
    train_dataset = FoodDataset(cfg["TRAIN_CSV_DIR"], transform=trfs)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg["BATCH_SIZE"], drop_last=True, shuffle=True, num_workers=cfg["WORKERS"])
    valid_dataset = FoodDataset(cfg["VALID_CSV_DIR"], transform=trfs)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=cfg["EVAL_BATCH_SIZE"], drop_last=True, shuffle=False, num_workers=cfg["WORKERS"])

    dataset = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }
    
    model = load_model(cfg)
    # opt = torch.optim.SGD(
    #     model.parameters(),
    #     lr=cfg["MODEL"]["LR"],
    #     momentum=cfg["MODEL"]["MOMENTUM"],
    #     weight_decay=cfg["MODEL"]["WEIGHT_DECAY"]
    # )
    # Change to AdamW optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["MODEL"]["LR"],
        weight_decay=cfg["MODEL"]["WEIGHT_DECAY"]
    )
    return dataset, model.to(device), opt

# Resume from checkpoint
def resume_setting(checkpoint_dir, model, opt, start_epoch, end_epoch):
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["opt"])
    start_epoch += checkpoint["epoch"]
    end_epoch += checkpoint["epoch"]
    return model, opt, start_epoch, end_epoch

if __name__ == "__main__":
    
    console.print("Parsing config...")

    try :
        cfgparser = CfgParser(config_path="./cfg/Setting.yml")
        cfg = cfgparser.cfg_dict
        main(cfg)
    except Exception :
        console.print_exception(show_locals=True)