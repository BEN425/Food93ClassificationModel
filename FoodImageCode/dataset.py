'''
dataset.py

This file defines `FoodDataset` class for the food database
'''

import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import csv
from PIL import Image


class FoodDataset(data.Dataset):
    '''
    Dataset containing SingleFood and AIFood
    
    Arguments :
        csv_path `str`: Path to csv file containing paths and labels of all image files. \
        CSV file is seperated by comma, the first item is path to the image, the rests are labels in multi-hot format.
        Ex (an image with label 2 and 4): `./path/to/image.jpg,0,0,1,0,1`
        root `str`: Base path of image paths in the csv file. If None, the original path is used
        transform `Transform`: Transformation to be apply on images. If not specified, \
        a default transform is applied to convert PIL Image to Tensor.
        hsv `bool`: Add 3 extra channels for HSV to the image (Total 6 channels: RGB + HSV)
    '''
    
    def __init__(self, 
            csv_path: str,
            root: str = None,
            transform = None,
            hsv = False,
        ):
        
        # Use default transformation if not specified
        # Crop the image to 244x244 for ResNet50 input
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
        ]) if transform is None else transform
        
        # List containing paths and labels of all images
        # Each label is in multi-hot form
        self.datalist = []
        with open(csv_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for img_path, *label in csv_reader:
                self.datalist.append((
                    img_path,
                    list(int(i) for i in label)
                ))
        
        self.root = root
        self.add_hsv = hsv
        
        #print(self.datalist)

    def _get_image(self, img_path: str) -> torch.Tensor:
        '''
        Open an image and apply the transform
        '''
        
        if self.root is not None :
            img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # Add 3 extra channels for HSV
        if self.add_hsv :
            img_hsv = img.convert("HSV")
            img_hsv = self.transform(img_hsv)
            img = torch.vstack((img, img_hsv))

        return img

    def __getitem__(self, index: int):
        '''
        Get an image from dataset
        
        Return :
            img `Tensor`: Image in Tensor format
            label `Tensor`: multi-hot label
        '''
        
        img_path, label = self.datalist[index]
        img = self._get_image(img_path)
        return img, torch.tensor(label)
    
    def __len__(self):
        return len(self.datalist)

if __name__ == "__main__":

    csv_dir = "/home/msoc/SingleImageFoodCode/SingleFoodImage_test_ratio811.csv"


    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = FoodDataset(
        csv_path=csv_dir,
        transform=transform
    )
    
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    for idx, (img, label) in enumerate(dataloader):
        pass
        #print(img.shape, label)
