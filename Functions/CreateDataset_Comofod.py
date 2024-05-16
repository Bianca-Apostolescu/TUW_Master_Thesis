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


def load_dataset(dataset_path):
    print("Starting loading dataset")
    original_images = []
    altered_images = []
    masks = []

    for file in os.listdir(dataset_path):
        if file.endswith('_B.png'):
            common_id = file.split('_B')[0]  # Extract common ID from filename

            # Construct file paths for original image, altered image, and mask
            orig_path = os.path.join(dataset_path, f"{common_id}_O.png")
            altered_path = os.path.join(dataset_path, f"{common_id}_F.png")
            mask_path = os.path.join(dataset_path, f"{common_id}_B.png")

            try:
                orig_img = Image.open(orig_path).convert("RGB")
                altered_img = Image.open(altered_path).convert("RGB")
                mask_img = Image.open(mask_path).convert("L")  # Convert to grayscale

                original_images.append(orig_img)
                altered_images.append(altered_img)
                masks.append(mask_img)
            except Exception as e:
                print(f"Error loading images for file {file}: {e}")

    return original_images, altered_images, masks






