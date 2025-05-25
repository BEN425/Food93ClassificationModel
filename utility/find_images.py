'''
Find images in the csv file from database
'''

import os
import shutil
from csv import reader
from itertools import compress

CSV_NAME = "AllFoodImage_test_ratio811.csv"
DATABASE = "single_food_with_preprocess_0503"
OUTDIR   = "test"


# Read class name
with open("../Database/class.txt", 'r') as file :
    lines = file.read().split("\n")
cls_names = [line.split()[1] for line in lines]

# Create folders to store found images
os.makedirs(f"../test/{OUTDIR}", exist_ok=True)
for cls_name in cls_names :
    new_cls_folder = f"../test/{OUTDIR}/{cls_name}"
    os.makedirs(new_cls_folder, exist_ok=True)

with open(f"../test/{CSV_NAME}", 'r') as file :
    csv_reader = reader(file)

    for line in csv_reader :
        # Get image path and multi-hot label
        img_path, *labels = line
        labels = [int(i) for i in labels]
        # Convert label to corresponding class id
        label_ids = compress(range(len(labels)), labels)
        
        # Copy image in database to new folder
        for id in label_ids :
            shutil.copyfile(img_path, f"../test/{OUTDIR}/{cls_names[id]}/{os.path.basename(img_path)}")
