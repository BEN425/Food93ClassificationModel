'''
Check all files in database.
Remove non-image files and images failed to open.
'''


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

if not failed :
    print("No failed images. Done.")
    exit(0)

input("Press enter to delete failed images > ")

for entry in failed :
    os.remove(entry)

print("Done")
