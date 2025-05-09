from glob import glob
import os
from PIL import Image
import pillow_avif

from rich import print

DATABASE = "single_food_preprocess_0503"

image_list = glob(f"../Database/{DATABASE}/**/*.*", recursive=True)

failed = []
for entry in image_list :
    try :
        img = Image.open(entry)
    except Exception :
        failed.append(entry)

print(f"Failed count: {len(failed)}")
print(failed)

input("Press enter to delete failed images > ")

for entry in failed :
    os.remove(entry)

print("Done")
