'''
Find images in the csv file from database
'''

import os
import shutil
from csv import reader
from itertools import compress

ROOT_DIR = "/home/msoc/ben_aifood/"
CSV_PATH = "FoodImageCode/csv/ai_single_food_300/AllFoodImage_valid_ratio811.csv"
OUTDIR   = "aisingle_300"


csv_path = os.path.join(ROOT_DIR, CSV_PATH)
out_dir = os.path.join(ROOT_DIR, f"test/{OUTDIR}")
print(f"csv_path: {csv_path}")
print(f"out_dir: {out_dir}")
input("Press ENTER to continue > ")

# Read class name
with open(os.path.join(ROOT_DIR, "Database/class.txt"), 'r') as file :
    lines = file.read().split("\n")
cls_names = [line.split()[1] for line in lines]

# Create folders to store found images
os.makedirs(out_dir, exist_ok=True)
for cls_name in cls_names :
    new_cls_folder = os.path.join(out_dir, cls_name)
    os.makedirs(new_cls_folder, exist_ok=True)

with open(csv_path, 'r') as file :
    csv_reader = reader(file)

    for line in csv_reader :
        # Get image path and multi-hot label
        img_path, *labels = line
        img_path = os.path.join(ROOT_DIR, img_path)
        labels = [int(i) for i in labels]
        # Convert label to corresponding class id
        label_ids = compress(range(len(labels)), labels)
        
        # Copy image in database to new folder
        for id in label_ids :
            shutil.copyfile(
                img_path,
                os.path.join(out_dir, cls_names[id], os.path.basename(img_path))
            )
