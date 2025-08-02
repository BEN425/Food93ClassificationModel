import os
import csv

ROOT_DIR = "/home/msoc/ben_aifood/"
img_lists = VAL_IMGS_300 = [
    # Bunashimeji
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/Bunashimeji/Bunashimeji_36.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/Bunashimeji/Bunashimeji_46.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/Bunashimeji/Bunashimeji_227.jpg",
    # Radish
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G2_LightGreenVegetables/Radish/Radish_202.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G2_LightGreenVegetables/Radish/Radish_132.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G2_LightGreenVegetables/Radish/Radish_8.jpg",
    # Zucchini
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G4_GourdVegetables/Zucchini/Zucchini_30.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G4_GourdVegetables/Zucchini/Zucchini_440.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G4_GourdVegetables/Zucchini/Zucchini_202.jpg",
    # SweetPepper
    "Database/aisingle_food_preprocess_0503/3_FishMeatAndEgg/D_Meat/D2_BeefAndProducts/Beef/Beef_ 304.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G1_DarkGreenAndYellowVegetables/SweetPepper/SweetPepper_80.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G1_DarkGreenAndYellowVegetables/SweetPepper/SweetPepper_381.jpg",
    # Pineapple
    "Database/aisingle_food_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/Pineapple/Pineapple_90.jpg",
    "Database/aisingle_food_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/Pineapple/Pineapple_339.jpg",
    "Database/aisingle_food_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/Pineapple/Pineapple_111.jpg",
    # Shiitake
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/ShiitakeMushrooms/ShiitakeMushrooms_104.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/ShiitakeMushrooms/ShiitakeMushrooms_169.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/ShiitakeMushrooms/ShiitakeMushrooms_26.jpg",
    # Onion
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G2_LightGreenVegetables/Onions/Onions_65.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G2_LightGreenVegetables/Onions/Onions_57.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G2_LightGreenVegetables/Onions/Onions_145.jpg",
    # Agaric
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/Agaric/Agaric_86.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/Agaric/Agaric_186.jpg",
    "Database/aisingle_food_preprocess_0503/5_Vegetable/G_Vegetables/G6_Mushrooms/Agaric/Agaric_99.jpg",
    # CherryTomato
    "Database/aisingle_food_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/CherryTomato/CherryTomato_127.jpg",
    "Database/aisingle_food_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/CherryTomato/CherryTomato_33.jpg",
    "Database/aisingle_food_preprocess_0503/6_Fruit/H_Fruits/H1_FreshFruits/CherryTomato/CherryTomato_594.jpg"
]

csv_file = "/home/msoc/ben_aifood/FoodImageCode/csv/ai_single_food_200/AllFoodImage.csv"

file_names = {}
labels = []

with open(csv_file, "r", newline="") as file :
    reader = csv.reader(file)
    for name, *label in reader :
        file_names[name] = [int(i) for i in label]
    
for img in img_lists :
    labels.append(file_names[os.path.join(ROOT_DIR, img)])
    
print(labels)