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

class SegmentationDataset(Dataset):
    def __init__(self, csv_file, images_dir, binary_masks_dir, bounding_box_masks_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.binary_masks_dir = binary_masks_dir
        self.bounding_box_masks_dir = bounding_box_masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.images_dir, self.df.iloc[idx]['image'])
        binary_mask_filename = os.path.join(self.binary_masks_dir, str(self.df.iloc[idx]['filename']))
        bounding_box_mask_filename = os.path.join(self.bounding_box_masks_dir, str(self.df.iloc[idx]['filename']))

        # Load images
        original_img = Image.open(img_filename).convert("RGB")
        if self.df.iloc[idx]['forged'] == 1:
            # Load binary mask for forged images
            binary_mask = Image.open(binary_mask_filename).convert("L")
            bounding_box_mask = Image.open(bounding_box_mask_filename).convert("RGB")
            label = torch.tensor(1)  # Forged
        else:
            # Create black mask for non-forged images
            binary_mask = Image.new("L", original_img.size, color=0)
            # Copy original image for bounding box mask
            bounding_box_mask = original_img.copy()
            label = torch.tensor(0)  # Not forged

        if self.transform:
            original_img = self.transform(original_img)
            binary_mask = self.transform(binary_mask)
            bounding_box_mask = self.transform(bounding_box_mask)

        return original_img, bounding_box_mask, binary_mask, label







