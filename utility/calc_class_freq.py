'''
Calculate class frequencies (ratio of class images to total images)
'''

from glob import glob
from os import listdir
import torch

DATABASE = "AI_SingleFood_database_0310"

# Get the class lists
with open("../Database/class.txt", "r") as file :
    cls = list(line.split()[1] for line in file.readlines())
    
cls_freq = torch.zeros(len(cls))

# Get all folders of categories
cls_folders = glob(f"../Database/{DATABASE}/*/*/*/*")

# Count image numbers and calculate class frequency
for folder in cls_folders :
    name = folder.split("/")[-1]
    cls_freq[ cls.index(name) ] += len(listdir(folder))

cls_freq /= cls_freq.sum()

# Write output

with open(f"../Database/class_freq.txt", "w") as file :
    file.writelines(f"{i}\n" for i in cls_freq)
