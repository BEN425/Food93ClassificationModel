'''
make_image_csv.py

This script goes through all files in FoodSeg103 database, converts FoodSeg103 labels to 93 labels.
The  paths and labels are stored in a csv file
'''


import os
import csv
from glob import glob

from PIL import Image
import numpy as np
from rich import get_console
console = get_console()

from cfgparser import CfgParser

# Read path from yaml file
cfgparser = CfgParser(config_path="./cfg/Setting.yml")
cfg = cfgparser.cfg_dict

# Get path
foodseg_path = cfg["FOODSEG_DIR"]
csv_path = cfg["FOODSEG_CSV_DIR"]

# Get mapping table of FoodSeg103 to 93 categories
id_mapping = {}
with open(os.path.join(foodseg_path, "..", "id_mapping.csv"), "r") as file :
    reader = csv.reader(file)
    for foodseg_id, food93_id in reader :
        id_mapping[int(foodseg_id)] = int(food93_id)

# Get path of all images and labels
# image extension is .jpg, label extension is .png
# Sort the list based on filename so that the image order of 2 lists is the same
image_path_list = sorted(
    glob(os.path.join(foodseg_path, "Images", "img_dir", "**", "*.jpg"), recursive=True),
    key=lambda x : int( x.split("/")[-1].rsplit(".", maxsplit=1)[0] )
)
label_path_list = sorted(
    glob(os.path.join(foodseg_path, "Images", "ann_dir", "**", "*.png"), recursive=True),
    key=lambda x : int( x.split("/")[-1].rsplit(".", maxsplit=1)[0] )
)

# Check the path of images and labels
# image_set = set(path.split("/")[-1].rsplit(".", maxsplit=1)[0] for path in image_path_list)
# label_set = set(path.split("/")[-1].rsplit(".", maxsplit=1)[0] for path in label_path_list)
# print(image_path.__len__())
# print(label_path.__len__())
# print(image_set - label_set, label_set - image_set)

# Read label images to get image-level labels
image_labels = {}
for image_path, label_path in zip(image_path_list, label_path_list) :
    # Read image and convert to array
    label_img = Image.open(label_path)
    label_arr = np.asarray(label_img)
    
    # Get unique pixel values as image-level label
    foodseg_label = np.unique(label_arr)
    food93_label = [id_mapping[label] for label in foodseg_label if label in id_mapping]
    if not food93_label : continue
    image_labels[image_path] = food93_label
    
# Store image path and labels in a csv file
with open(csv_path, "w") as file :
    writer = csv.writer(file)
    for path, labels in image_labels.items() :
        # Convert labels to multi-hot format
        mh_label = [0] * cfg["MODEL"]["CATEGORY_NUM"]
        for label in labels :
            mh_label[label] = 1
        writer.writerow((path, *mh_label))
