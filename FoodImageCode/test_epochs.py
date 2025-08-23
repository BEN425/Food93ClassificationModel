import os
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import evaluate_dataset_class_acc
from dataset import FoodDataset
from model.ResNet_modified import ModifiedResNet

from rich import get_console
console = get_console()

cp_dir = "/work/u6140562/FoodS2C/FoodImageCode/Results/checkpoints/8_2_aisingle_200"
test_csv = "/work/u6140562/FoodS2C/FoodImageCode/csv/ai_single_food_200/AllFoodImage_valid_ratio811.csv"
out_csv = "/work/u6140562/FoodS2C/test/focal25_loss.csv"
max_epoch = 15

def get_epoch(name: str) :
    return int(os.path.basename(name).split(".", maxsplit=1)[0].split("_")[-1])

cp_list = [entry for entry in os.scandir(cp_dir) if entry.is_file() and entry.name.endswith(".pth.tar")]
cp_sorted = sorted(cp_list, key=lambda entry: get_epoch(entry.path))

device = torch.device("cuda")
trfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
])
dataset = FoodDataset(test_csv, root="/work/u6140562/FoodS2C", transform=trfs)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

cls_list = []
with open("/work/u6140562/FoodS2C/Database/class.txt", "r") as file :
    cls_list = [line.split()[-1] for line in file.readlines()]
console.print(cls_list)

record = []
for epoch, entry in enumerate(cp_sorted) :
    if epoch >= max_epoch : break
    console.print(entry.name)
    
    cp_path = entry.path
    
    cp = torch.load(cp_path, map_location="cpu")
    model = ModifiedResNet(3, 64, 93)
    model.load_state_dict(cp["model"])
    model.to(device)
    
    result = evaluate_dataset_class_acc(model, dataloader, 93, device)
    result["epoch"] = epoch
    record.append(result)
    console.print(result)

    with open(out_csv, "w", newline="") as file :
        writer = csv.writer(file)
        # Write header
        writer.writerow(cls_list)
        # Write loss
        for r in record :
            writer.writerow(r["valid_class_loss"].tolist())
