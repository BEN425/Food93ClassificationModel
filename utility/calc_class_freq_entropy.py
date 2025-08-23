'''
Calculate class weights for focal loss
'''

import os
from glob import glob
import torch

DATABASE = "aisingle_food_preprocess_0820"

# Get the class lists
with open("../Database/class.txt", "r") as file :
    cls = list(line.split()[1] for line in file.readlines())
    
cls_counts = torch.zeros(len(cls))

# Get all folders of categories
cls_folders = glob(f"../Database/{DATABASE}/*/*/*/*")

# Count image numbers and calculate class frequency
for folder in cls_folders :
    name = os.path.basename(folder)
    cls_counts[ cls.index(name) ] += len(os.listdir(folder))

total = cls_counts.sum()

# Calculate class entropy
cls_entropy = -torch.log2(cls_counts / total)

# cls_target = total / (cls_counts * 93) # freq1
# cls_target = total / (cls_counts * 2)
cls_target = 4 * ( torch.sigmoid(total / (cls_counts) / 100) - 0.5 ) # freq2
# cls_target = 1 - cls_counts / total

# Write output

with open(f"../Database/class_freq.txt", "w") as file :
    file.writelines(f"{i}\n" for i in cls_target)

# with open(f"../Database/class_entropy.txt", "w") as file :
#     file.writelines(f"{i}\n" for i in cls_entropy)
