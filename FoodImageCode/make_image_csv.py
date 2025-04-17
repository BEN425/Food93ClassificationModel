'''
make_image_csv.py

This script goes through all files in the database, and stores the paths and labels in a csv file
Then, the csv file is splited into 3 csv files for train, test and val sets.
'''


import os
import random
import csv

from rich import get_console
console = get_console()

from cfgparser import CfgParser

# Read path from yaml file
cfgparser = CfgParser(config_path="./cfg/Setting.yml")
cfg = cfgparser.cfg_dict
database_path  = cfg["DATA_BASE_DIR"]
all_csv_path   = cfg["ALL_CSV_DIR"]
train_csv_path = cfg["TRAIN_CSV_DIR"]
test_csv_path  = cfg["TEST_CSV_DIR"]
val_csv_path   = cfg["VALID_CSV_DIR"]

# Initialize seed
random.seed(cfg["SEED"])

# Check whether a file is an image file
def check_image(file: str) :
    filename = file.lower()
    return filename.split(".")[-1].lower() in [
        "jpg", "jpeg", "png", "avif", "webp"
    ]


### Walk through all files in database ###

# Key: category name. Value: category index
categories_dict: "dict[str, int]" = {}
# Key: file name.
# Value: An dict contains `path` for image path and `labels` for image labels
'''
Example: {
    "image1.jpg": {
        "path": "./ ... / image.jpg",
        "labels": set(1, 2, 3),
    },
    ...
}
'''
images: "dict[ str, dict[str, str|set[int]] ]" = {}

# Read ids of all classes
with open(os.path.join(database_path, "..", "class.txt"), "r") as file :
    lines = (line.split() for line in file.readlines())
    for class_id, class_name in lines :
        categories_dict[class_name] = int(class_id)

count = 0
mismatch_cate_count = 0
mismatch_categories = []

#* Top level: Six categorie, starts with number (ex: 1_CerealsGrainsTubersAndRoots)
for six_category in os.scandir(database_path):
    six_cate_name = six_category.name
    
    # Skip specific categories and other files
    if six_cate_name.startswith("4") or six_cate_name.startswith("7") or not six_category.is_dir() :
        continue
    
    #* 1st level: starts with letter (ex: A_CeralsGrainsTubersAndRoots)
    for first_category in os.scandir(six_category.path):
        if not first_category.is_dir() : continue
        
        #* 2nd level: starts with letter+number (ex: A1_RiceAndProducts)
        for second_category in os.scandir(first_category.path):
            if not second_category.is_dir() : continue
            
            #* 3rd level: Food name (ex: Congee)
            for third_category in os.scandir(second_category.path):
                dirname = third_category.name
                if not third_category.is_dir() or dirname == "Sesame":
                    continue
                
                # Check whether folder name matches the class name
                if dirname not in categories_dict:
                    mismatch_cate_count += 1
                    mismatch_categories.append(dirname)
                    continue

                #* 4th level: image files (.jpg, .jpeg, .png, etc)
                for image_file in os.scandir(third_category.path) :
                    filename = image_file.name
                    if not image_file.is_file() or not check_image(filename) :
                        continue

                    # Avoid repeated file name
                    if filename not in images:
                        images[filename] = {
                            "path": image_file.path,
                            "labels": set()
                        }
                    
                    images[filename]["labels"].add(categories_dict[dirname])
                    count += 1

print(f"Total images: {len(images)}")
# Check whether there are any mismatch classes
if mismatch_cate_count != 0 :
    print(f"Mismatched classes ({mismatch_cate_count}):", mismatch_categories)
    exit(1)


### Save all images in a csv file

with open(all_csv_path, "w", newline="") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerows((image["path"], *image["labels"]) \
        for image in images.values())

### Put images in the corresponding category ###

# Key: category id
# Value: list of images. Each image is a dict from `images`
category_to_images: "dict[int, list[dict]]" = {}
try :
    for image in images.values() :
        # Only uses one class to seperate train, test and val sets
        category = next(iter(image["labels"]))
        if category not in category_to_images:
            category_to_images[category] = []
        category_to_images[category].append(image)
except Exception :
    console.print_exception(show_locals=True)
    exit(1)


### Seperate images for train, valid and test sets (8:1:1) ###

train_set = []
valid_set = []
test_set  = []

for category, image_lst in category_to_images.items():
    random.shuffle(image_lst)
    train_size = int(len(image_lst) * 0.8)
    valid_size = int(len(image_lst) * 0.1)
    
    train_set.extend(image_lst[:train_size])
    valid_set.extend(image_lst[train_size : train_size+valid_size])
    test_set.extend(image_lst[train_size+valid_size:])


### Write seperated image data into csv files ###

def write_to_csv(file_path: str, dataset: "list[dict]") -> None:
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # writer.writerow(["Image_Path", "Category"])  # CSV header
        for image in dataset :
            # Convert labels to multi-hot format
            mh_label = [0] * cfg["MODEL"]["CATEGORY_NUM"]
            for label in image["labels"] :
                mh_label[label] = 1
            writer.writerow((image["path"], *mh_label))

# Write split datasets to csv files
write_to_csv(train_csv_path, train_set)
write_to_csv(val_csv_path,   valid_set)
write_to_csv(test_csv_path,  test_set)
