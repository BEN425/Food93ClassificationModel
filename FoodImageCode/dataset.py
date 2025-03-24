'''
dataset.py

This file defines `FoodDataset` class for the food database
'''

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import csv
from PIL import Image


class FoodDataset(data.Dataset):
    '''
    Dataset containing SingleFood and AIFood
    
    Arguments:
        root `str`: Path to csv file containing paths and labels of all image files. \
        CSV file is seperated by comma, the first item is path to the image, the rests are labels in multi-hot format.
        Ex (an image with label 2 and 4): `./path/to/image.jpg,0,0,1,0,1`
        transform `Transform`: Transformation to be apply on images. If not specified, \
        a default transform is applied to convert PIL Image to Tensor.
    '''
    
    def __init__(self, 
            root: str, 
            transform
        ):
        
        self.transform = transform
        
        # List containing paths and labels of all images
        # Each label is in multi-hot form
        self.datalist = []
        with open(root, "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for img_path, *label in csv_reader:
                self.datalist.append((
                    img_path,
                    list(int(i) for i in label)
                ))
        
        #print(self.datalist)

    def get_image(self, img_path: str, transform=None) -> torch.Tensor:
        '''
        Open an image and apply the transform
        '''
        
        img = Image.open(img_path).convert("RGB")
        # Default transformation
        # Crop the image to 244x244 for ResNet50 input
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
            ])
        img = transform(img)
        return img

    def __getitem__(self, index: int):
        '''
        Get an image from dataset
        
        Return:
            img `Tensor`: Image in Tensor format
            label `Tensor`: multi-hot label
        '''
        
        img_path, label = self.datalist[index]
        img = self.get_image(img_path, self.transform)
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
        root=csv_dir,
        transform=transform
    )
    
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    for idx, (img, label) in enumerate(dataloader):
        pass
        #print(img.shape, label)
