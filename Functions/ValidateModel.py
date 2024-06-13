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

import EarlyStopping as stopping

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline




def validate_model(model, dataloader, loss_function, device, channels, dataset_type):
  print("Validating...")

  totalValLoss = 0

  model.eval()

  with torch.no_grad():
      # total_val_loss = 0

      if dataset_type == 'comofod' or dataset_type == 'imd':
        for orig_images, altered_images, masks in dataloader:
            
            orig_images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)
            
            if channels == 3:
              # For 3 channels - only the altered image as input
              pred_masks = model(altered_images) # validate on altered_images

            elif channels == 6:
              # For 6 channels - altered + original image as input (concat on channel dim)
              input_tensor = torch.cat([orig_images, altered_images], dim=1)
              pred_masks = model(input_tensor)
            

            # # For 6 channels - altered + original image as input (concat on width dim)
            # # Split the predicted masks back into two halves
            # batch_size = orig_images.size(0)
            # pred_masks_orig, pred_masks_altered = torch.split(pred_masks, batch_size, dim=0)

            # # Compute loss separately for original and altered images
            # loss_orig = loss_function(pred_masks_orig, masks)
            # loss_altered = loss_function(pred_masks_altered, masks)

            # # Total loss is the sum of losses for original and altered images
            # val_loss = loss_orig + loss_altered

            val_loss = loss_function(pred_masks, masks)
            totalValLoss += val_loss.item()

      elif dataset_type == 'sroie':
        for orig_images, altered_images, masks in dataloader:
            
            orig_images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)
            
            if channels == 3:
              # For 3 channels - only the altered image as input
              pred_masks = model(orig_images) # validate on altered_images

            elif channels == 6:
              # For 6 channels - altered + original image as input (concat on channel dim)
              input_tensor = torch.cat([orig_images, altered_images], dim=1)
              pred_masks = model(input_tensor)
            

            val_loss = loss_function(pred_masks, masks)
            totalValLoss += val_loss.item()

  avg_val_loss = totalValLoss / len(dataloader)

  return avg_val_loss





