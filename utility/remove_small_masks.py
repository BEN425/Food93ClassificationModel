'''
remove_small_masks.py

Remove SAM masks which are smaller than a theshold
'''


import os
from glob import glob
import numpy as np

from matplotlib import pyplot as plt
from rich.progress import track

from rich import print

# Threshold for a single mask region
REGION_T = 100
# Threshold for the entire mask
MASK_T = 1000

mask_path = "../sam_map/single_food_with_preprocess_0503"
entries = glob(os.path.join(mask_path, "**/*.np?"), recursive=True)
# test_path = "../sam_map/single_food_with_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/Peach/Peach_2.jpg.npz"
# entries = [test_path]

# Calculate size of a region with BFS, and remove small region
def remove_small_region(mask: np.ndarray) -> np.ndarray :
    traveled = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    
    # BFS
    def traverse(r, c) :
        val = mask[r][c]
        pixels = []
        queue = []
        queue.append((r, c))
        
        count = 0
        while queue :
            y, x = queue.pop(0)
            if traveled[y][x] == 1 or mask[y][x] != val : continue
            
            traveled[y][x] = 1
            count += 1
            pixels.append((y, x))
            
            if y - 1 >= 0 :
                queue.append((y - 1, x))
            if y + 1 < h :
                queue.append((y + 1, x))
            if x - 1 >= 0 :
                queue.append((y, x - 1))
            if x + 1 < w :
                queue.append((y, x + 1))
        
        return count, pixels
    
    # Skip background
    traveled[mask == -1] = 1
    for i in range(h) :
        for j in range(w) :
            
            if traveled[i][j] : continue

            counts, pixels = traverse(i, j)
            # traveled[pixels] = 1
            # print(f"{(i, j)}, {mask[i][j]}, {len(pixels)}")
            # if counts > 100 : print(f"{(i, j)}, {mask[i][j]}, {counts}")
            if counts < REGION_T :
                mask[pixels] = -1

for entry in track(entries, total=len(entries)) :
    if not os.path.isfile(entry) : continue
    
    if entry.endswith(".npy") :
        mask = np.load(entry)
    elif entry.endswith(".npz") :
        mask = np.load(entry)["arr_0"]
    else :
        continue
    
    mask_data = np.unique(mask, return_counts=True)
    # remove_small_region(mask)
    for val, count in zip(*mask_data) :
        # Skip background
        if val == -1 : continue
        
        if count < MASK_T :
            mask[mask == val] = -1
    
    new_path = entry.replace("sam_map", "sam_map_1")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    np.savez_compressed(new_path, mask)
