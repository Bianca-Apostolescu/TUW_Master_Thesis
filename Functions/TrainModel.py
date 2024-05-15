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



def train_model(model, dataloader, loss_function, optim, device):
  print("Training...")

  totalTrainLoss = 0

  model.train() # unet.train.train() 

  for orig_images, altered_images, masks in dataloader:
      images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)

      optim.zero_grad()
      input_tensor = torch.cat([images, altered_images], dim=1) # channel
      pred_masks = model(input_tensor) # they are not binary => the binary masks are displayed using the vizualize function with a threshold
      # pred_masks = torch.sigmoid(pred_masks)

      # print(f"training images = {images.shape}")
      # print(f"training altered_images = {altered_images.shape}")
      # print(f"training input_tensor = {input_tensor.shape}")

      # print(f"training pred_masks = {pred_masks.shape}")
      # print(f"training masks = {masks.shape}")

      # # Split the predicted masks back into two halves
      # batch_size = images.size(0)
      # pred_masks_orig, pred_masks_altered = torch.split(pred_masks, batch_size, dim=0)

      # # Compute loss separately for original and altered images
      # loss_orig = loss_function(pred_masks_orig, masks)
      # loss_altered = loss_function(pred_masks_altered, masks)

      # # Total loss is the sum of losses for original and altered images
      # loss = loss_orig + loss_altered

      loss = loss_function(pred_masks, masks)
      loss.backward()
      optim.step()

      pred_masks = (pred_masks > 0.5).float()

      totalTrainLoss += loss.item()

  avg_train_loss = totalTrainLoss / len(dataloader)

  return avg_train_loss



