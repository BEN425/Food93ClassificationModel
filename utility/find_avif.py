'''
Find avif files in database
'''

from glob import glob
from rich import print

DATABASE = "AI_SingleFood_database_0310"

lsts = glob(f"../Database/{DATABASE}/**/*.avif", recursive=True)
print(lsts)