'''
Count number of images of each class
Plot bar chart of the numbers
'''


from os import scandir
from rich import print
import matplotlib
matplotlib.use("webagg")
from matplotlib import pyplot as plt
from collections import OrderedDict

DATABASE = "AI_SingleFood_database_0310"
counts = OrderedDict()

for layer1 in scandir(f"../Database/{DATABASE}") :
    for layer2 in scandir(layer1.path) :
        for layer3 in scandir(layer2.path) :
            for layer4 in scandir(layer3.path) :
                counts[layer4.name] = len(list(scandir(layer4.path)))

sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

plt.bar(*zip(*sorted_counts))
plt.xticks(rotation=45, ha="right", fontsize=5)
plt.show()

print(sorted_counts)
