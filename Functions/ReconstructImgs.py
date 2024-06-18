#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


def reconstruct_image_from_patches(patches, image_shape, patch_size, channels):
    num_patches = patches.shape[0]
    patch_height, patch_width = patch_size

    # Calculate the number of patches that fit in each dimension
    num_patches_h = (image_shape[0] + patch_height - 1) // patch_height
    num_patches_w = (image_shape[1] + patch_width - 1) // patch_width

    # Initialize the reconstructed image
    if channels == 1:
        reconstructed_image = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
    elif channels == 3:
        reconstructed_image = np.zeros((image_shape[0], image_shape[1], channels), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    # Loop through each patch and place it in the reconstructed image
    for idx in range(num_patches):
        row_idx = idx // num_patches_w
        col_idx = idx % num_patches_w

        start_h = row_idx * patch_height
        start_w = col_idx * patch_width

        end_h = min(start_h + patch_height, image_shape[0])
        end_w = min(start_w + patch_width, image_shape[1])

        if channels == 1:
            reconstructed_image[start_h:end_h, start_w:end_w] = patches[idx].squeeze()[:end_h-start_h, :end_w-start_w]
        elif channels == 3:
            reconstructed_image[start_h:end_h, start_w:end_w, :] = patches[idx].transpose((1, 2, 0))[:end_h-start_h, :end_w-start_w, :]
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

    return reconstructed_image



