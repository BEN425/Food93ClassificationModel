'''
Check the folder name and the class lists
Find mismatch label names
'''


from os import scandir

DATABASE = "AI_SingleFood_database_0310"

with open("../Database/class.txt", "r") as file :
    cls = set(line.split()[1] for line in file.readlines())

scan_cls = set()
for layer1 in scandir(f"../Database/{DATABASE}") :
    for layer2 in scandir(layer1.path) :
        for layer3 in scandir(layer2.path) :
            for layer4 in scandir(layer3.path) :
                
                scan_cls.add(layer4.name)

print("Labels not found in class list:", cls - scan_cls)
print("Labels not found in database:  ", scan_cls - cls)
