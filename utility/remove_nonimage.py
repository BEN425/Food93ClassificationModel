'''
Remove all non-image files in database
'''

from glob import glob
from os import remove
from rich import print

DATABASE = "AI_SingleFood_database_0310"

files = glob(f"../Database/{DATABASE}/**/*.*")
formats = ("jpg", "jpeg", "png", "avif", "webp")

removed_lst = []

for entry in files :
    if entry.split(".")[-1] not in formats :
        remove(entry)
        removed_lst.append(entry)

print(f"Removed {len(removed_lst)} non-image files:", removed_lst)
