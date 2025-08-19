import os
import csv
import json

from torch import Tensor

RESULT_PATH = "/home/msoc/ben_aifood/FoodImageCode/Results/logs/7_24_aisingle/results.json"
OUT_PATH = "/home/msoc/ben_aifood/FoodImageCode/Results/logs/7_24_aisingle/"

with open(RESULT_PATH, "r") as file :
    results = json.load(file)["result"]

# Write metrics
with open(os.path.join(OUT_PATH, "metrics.csv"), "w", newline="") as file :
    writer = csv.writer(file)
    
    # Write header
    header = ["epoch"] + [key for key, value in results[0].items() \
        if ("train" in key or "valid" in key) and not isinstance(value, (dict, list, Tensor)) and value is not None]
    writer.writerow(header)
    
    # Write metrics
    for res in results :
        writer.writerow(res[key] for key in header)

# Get class names
with open("/home/msoc/ben_aifood/Database/class.txt", "r") as file :
    cls_names = [line.split()[1] for line in file.readlines()]

list_data_name = [key for key, value in results[0].items() \
    if "valid" in key and isinstance(value, list)]

# Write metrics of each class
for name in list_data_name :
    with open(os.path.join(OUT_PATH, f"{name}.csv"), "w", newline="") as file :
        writer = csv.writer(file)
        
        writer.writerow(["epoch"] + cls_names)
        for res in results :
            writer.writerow([res["epoch"]] + res[name])
