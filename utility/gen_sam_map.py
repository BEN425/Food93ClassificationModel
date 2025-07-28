'''
gen_sam_map.py

Generate SAM map of images in dataset
'''

import os
from glob import glob
import logging

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from rich.progress import track

logging.basicConfig(
    level=logging.INFO,
    filename="get_sam_map.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATABASE = "single_food_with_preprocess_0503"
SAM = "vit_h"

root_path = '/home/msoc/ben_s2c/S2C'
data_path = f'/home/msoc/ben_aifood/Database/{DATABASE}'
result_path = f'/home/msoc/ben_s2c/S2C/sam_map/{DATABASE}'

logging.info(f"device: {device}")
logging.info(f"database: {data_path}")
logging.info(f"sam: {SAM}")

sam_path = root_path + f'/pretrained/sam_{SAM}.pth'
sam = sam_model_registry[SAM](checkpoint=sam_path)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)

failed = []
entries = glob(os.path.join(data_path, "**", "*.*"), recursive=True)

logging.info(f"Image count: {len(entries)}")

for j, entry in track(
    enumerate(entries), total=len(entries)
) :
    
    # Check image
    if not entry.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")) :
        logging.debug(f"Non-image: {entry}")
        continue
    
    try :
        
        # Get output path
        out_path = os.path.join(result_path, os.path.relpath(entry, data_path))
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        
        # Open image and generate masks
        image = Image.open(entry).convert("RGB")
        img_array = np.asarray(image, dtype=np.uint8)
        masks = mask_generator.generate(img_array)
        
        if len(masks) == 0 : continue
        
        # Create mask image
        temp = np.full((img_array.shape[0], img_array.shape[1]), -1, dtype=int)
        for i, mask in enumerate(reversed(masks)):
            temp[mask['segmentation']] = i
        
        np.savez_compressed(out_path, temp)
        
        if j % 100 == 0 and i != 0 :
            logging.info(f"Progress: {i}/{len(entries)}")
    
    except Exception as e:
        logging.error(f"Failed image '{entry}': {e}")
        failed.append(entry)

with open("failed.txt", "w") as f :
    for item in failed:
        f.write(f"{item}\n")

logging.info(f"Failed: {len(failed)}")
logging.info("Completed")
