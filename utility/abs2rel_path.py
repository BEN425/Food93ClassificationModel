'''
Convert absolute path to relative path in csv file
'''


import os
import csv
import re

ROOT = '/home/msoc/ben/'
CSV_PATH = "../FoodImageCode/csv/AllFoodImage_valid_ratio811.csv"

new_labels = []
with open(CSV_PATH, "r") as file :
    reader = csv.reader(file.readlines())
    for abs_path, *labels in reader :
        rel_path = os.path.relpath(abs_path, ROOT).replace("\\", "/")
        sim_path = re.sub(r"/?\S*/", "", rel_path)

        new_labels.append((rel_path, *labels))
        # new_labels.append((sim_path, *labels))

with open(CSV_PATH.rsplit("/", maxsplit=1)[-1], "w", newline="") as file :
    writer = csv.writer(file)
    writer.writerows(new_labels)
