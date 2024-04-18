#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For managing COCO dataset
# from pycocotools.coco import COCO

# For creating and managing folder/ files
import glob
import os
import shutil

# For managing images
from PIL import Image
import skimage.io as io

# Basic libraries
import numpy as np
import pandas as pd
import random
import cv2

# For plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# For importing models and working with them
## Torch
import torch
import torch.utils.data # for Dataset
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

## Torchvision
import torchvision
from torchvision.transforms import transforms

# For creating train - test splits
from sklearn.model_selection import train_test_split

import pathlib
import pylab
import requests
from io import BytesIO
from pprint import pprint
from tqdm import tqdm
import time
from imutils import paths

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline



# In[ ]:





# In[4]:


class SegmentationDataset(Dataset):

    def __init__(self, original_images, altered_images, masks, transforms=None):
        self.original_images = original_images
        self.altered_images = altered_images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        original_image = self.original_images[idx]
        altered_image = self.altered_images[idx]
        mask = self.masks[idx]

        if self.transforms:
            original_image = self.transforms(original_image)
            altered_image = self.transforms(altered_image)
            mask = self.transforms(mask)

        return original_image, altered_image, mask

def load_dataset(dataset_path, num_folders):
    print("Starting loading dataset")
    original_images = []
    altered_images = []
    masks = []

    folders = sorted(os.listdir(dataset_path))[:num_folders]
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Find the common ID from the filenames
            common_id = None
            for file in os.listdir(folder_path):
                if file.endswith('_0.jpg') or file.endswith('_0.png'):
                    common_id = file.split('_')[0]  # Extract common ID from filename
                    break  # Stop after finding the first _0.jpg or _0.png file

            if common_id is None:
                print(f"No altered image found for folder {folder}")
                continue

            # Construct file paths for original image, altered image, and mask
            orig_path = os.path.join(folder_path, folder+'_orig.jpg')
            altered_ext = 'jpg' if os.path.exists(os.path.join(folder_path, common_id+'_0.jpg')) else 'png'
            altered_path = os.path.join(folder_path, common_id+'_0.'+altered_ext)
            mask_path = os.path.join(folder_path, common_id+'_0_mask.png')

            try:
                orig_img = Image.open(orig_path).convert("RGB")
                altered_img = Image.open(altered_path).convert("RGB")
                mask_img = Image.open(mask_path).convert("L")  # Convert to grayscale

                if orig_img is None or altered_img is None or mask_img is None:
                    print(f"Failed to load images for folder {folder}")
                    continue

                original_images.append(orig_img)
                altered_images.append(altered_img)
                masks.append(mask_img)
            except Exception as e:
                print(f"Error loading images for folder {folder}: {e}")

    return original_images, altered_images, masks


# In[5]:




