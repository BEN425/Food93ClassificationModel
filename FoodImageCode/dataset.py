'''
dataset.py

This file defines `FoodDataset` class for the food database
'''

import os

import numpy as np
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
    
    def __getitem__(self, index: int) -> torch.Tensor :
        '''
        Get an image from dataset
        
        Return :
            img `Tensor` "[C, H, W]": Image in Tensor format
            label `Tensor` "[1, CLS]": multi-hot label
        '''
        
        img_path, label = self.datalist[index]
        img = self._get_image(img_path)
        
        return img, torch.tensor(label), 0 # Return 3 values for compability
    
    def __len__(self):
        return len(self.datalist)

class FoodDatasetWithMasks(data.Dataset):
    '''
    Dataset containing SingleFood and AIFood
    Returns image Tensor and SAM mask
    
    Arguments :
        csv_path `str`: Path to csv file containing paths and labels of all image files. \
            CSV file is seperated by comma, the first item is path to the image, the rests are labels in multi-hot format.
        Ex (an image with label 2 and 4): `./path/to/image.jpg,0,0,1,0,1`
        root `str`: Base path of image paths in the csv file. If None, the original path is used
        is_sam_dir_root `bool`: Whether `sam_dir` is the root directory of sam maps. \
            If `False`, `root` will be appended at the front of `sam_dir`
        sam_dir `str`: Path to the directory of sam maps, \
            which contains a folder structure like Database and each map is a `.npz` file.
        transform `Transform`: Transformation to be apply on images. If not specified, \
            a default transform is applied to convert PIL Image to Tensor.
        hsv `bool`: Add 3 extra channels for HSV to the image (Total 6 channels: RGB + HSV)
    '''
    
    def __init__(self, 
            csv_path: str,
            root: str = None,
            sam_dir: str = None,
            is_sam_dir_root = False,
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
        
        if not is_sam_dir_root :
            self.sam_dir = os.path.join(root, sam_dir) \
                if sam_dir is not None and root is not None else None
        else :
            self.sam_dir = sam_dir
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
    
    def _get_segment(self, seg_path: str) -> np.ndarray :
            '''
            Open sam segments from .npz file
            '''
            
            if self.sam_dir is not None :
                seg_path = os.path.relpath(seg_path, "Database")
                seg_path = os.path.join(self.sam_dir, seg_path)
            seg_path = seg_path + ".npz"
            seg = np.load(seg_path)["arr_0"]
            return seg
        
    def __getitem__(self, index: int) -> "tuple[torch.Tensor, torch.Tensor, np.ndarray]" :
        '''
        Get an image from dataset
        
        Return :
            img `Tensor` "[C, H, W]: Image in Tensor format
            label `Tensor` "[1, CLS]": multi-hot label
            seg `Tensor` "[H, W]": Corresponding SAM-processed image
        '''
        
        img_path, label = self.datalist[index]
        img = self._get_image(img_path)
        seg = self._get_segment(img_path)
        
        return img, torch.tensor(label), seg
    
    def __len__(self):
        return len(self.datalist)


if __name__ == "__main__":

    # csv_dir = "/home/msoc/SingleImageFoodCode/SingleFoodImage_test_ratio811.csv"


    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize((256, 256)),
    #         transforms.CenterCrop(224),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    # dataset = FoodDataset(
    #     csv_path=csv_dir,
    #     transform=transform
    # )
    
    # dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    # for idx, (img, label) in enumerate(dataloader):
        # pass
        #print(img.shape, label)

    from rich import print

    csv_dir = "/home/msoc/ben_s2c/food_ian/csv/single_food/AllFoodImage_train_ratio811.csv"
    root_dir = "/home/msoc/ben_aifood"
    sam_dir = "sam_map/"

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = FoodDataset(
        csv_path=csv_dir,
        root=root_dir,
        sam_dir=sam_dir,
        transform=transform
    )
    
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    for idx, (img, label, sam) in enumerate(dataloader):
        # pass
        print(img.shape, label, sam.shape)
        input()