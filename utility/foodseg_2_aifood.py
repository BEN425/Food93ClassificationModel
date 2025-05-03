'''
Generate a conversion file from FoodSeg103 categories to 93 Food categories
Required FoodSeg103 database placed at `Database/FoodSeg103`

Database/FoodSeg_cate_mapping.csv : FoodSeg103 category names to 93 Food category names
Database/class.txt : 93 Food category ids and names
Database/FoodSeg103/category_id.txt : Foodseg103 category ids and names
'''

from rich import print

### Convert FoodSeg103 category name to 93 Food category name

name_mapping = {}
skip_cates = []

with open("../Database/FoodSeg_cate_mapping.csv", "r") as file :
    lines = file.readlines()

for line in lines :
    foodseg_name, *food93_name = line.split(",")
    foodseg_name = foodseg_name.strip()
    
    # Not 1 to 1 mapping, skip category
    if len(food93_name) != 1 :
        skip_cates.append(foodseg_name)
        continue
    
    # Take the first element and remove space
    food93_name = food93_name[0].split("_")[-1].strip().replace(" ", "")
    
    name_mapping[foodseg_name] = food93_name

### Convert FoodSeg103 category id to Food93 category id

id_mapping = {}
foodseg_ids = {}
food93_ids = {}

#  Get FoodSeg103 category ids

with open("../Database/FoodSeg103/category_id.txt", "r") as file :
    lines = file.readlines()
    
for line in lines :
    id_, name = line.split(maxsplit=1)
    foodseg_ids[name.strip()] = int(id_)

#  Get Food93 category ids

with open("../Database/class.txt", "r") as file :
    lines = file.readlines()
    
for line in lines :
    id_, name = line.split(maxsplit=1)
    food93_ids[name.strip()] = int(id_)

# Create category mapping

for foodseg_name, food93_name in name_mapping.items() :
    # Skip FoodSeg103 categories
    if foodseg_name in skip_cates : continue
    
    id_mapping[foodseg_ids[foodseg_name]] = food93_ids[food93_name]

# Store mapping table
with open("../Database/id_mapping.csv", "w") as file :
    file.writelines(f"{id1},{id2}\n" for id1, id2 in id_mapping.items())

print(f"Mapping: \n{id_mapping}")
print(f"Skipped: \n{skip_cates}")
