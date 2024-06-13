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
    def __init__(self, images_dir, binary_masks_dir, bounding_box_masks_dir, transform=None):
        self.image_paths = list(paths.list_images(images_dir))
        self.binary_mask_paths = list(paths.list_images(binary_masks_dir))
        self.bounding_box_mask_paths = list(paths.list_images(bounding_box_masks_dir))
        self.transform = transform

        # Ensure the lists are sorted for consistent ordering
        self.image_paths.sort()
        self.binary_mask_paths.sort()
        self.bounding_box_mask_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_filename = self.image_paths[idx]
        binary_mask_filename = self.binary_mask_paths[idx]
        bounding_box_mask_filename = self.bounding_box_mask_paths[idx]

        # Load images
        original_img = Image.open(img_filename).convert("RGB")
        binary_mask = Image.open(binary_mask_filename).convert("L")
        bounding_box_mask = Image.open(bounding_box_mask_filename).convert("RGB")

        if self.transform:
            original_img = self.transform(original_img)
            binary_mask = self.transform(binary_mask)
            bounding_box_mask = self.transform(bounding_box_mask)

        return original_img, bounding_box_mask, binary_mask







