'''
Get all tensorbaord log from a root directory
'''

import os

from openpyxl import Workbook

from get_tensorboard_log import get_log_data, write_csv, write_worksheet


ROOT_PATH = "../FoodImageCode/Results/logs/"
EXCLUDE = []

wb = Workbook()
for entry in os.scandir(ROOT_PATH) :
    if not entry.is_dir() or entry.name in EXCLUDE :
        continue

    try :
        data_dict = get_log_data(entry.path)
    except Exception :
        print(f"Failed to get log data: {entry.name}")
        continue
    
    try :
        write_csv(data_dict, os.path.join(ROOT_PATH, f"{entry.name}.csv"))

        ws = wb.create_sheet(entry.name)
        write_worksheet(ws, data_dict)
    except Exception :
        print(f"Failed to write result to csv/xlsx: {entry.name}")
        continue


wb.save(os.path.join(ROOT_PATH, "all.xlsx"))
wb.close()
