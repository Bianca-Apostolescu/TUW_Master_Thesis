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
import wandb

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



def plot_results(lent, orig_images, altered_images, masks, pred_masks, dataset_type):
    
    if dataset_type == 'comofod' or dataset_type == 'imd':
      for i in range (0, lent): # range (0, lent) - for every image combination
            
            # Log image(s)
            wandb.log(
                    {"original_images": [wandb.Image(orig_images.cpu().numpy()[i].transpose(1, 2, 0), caption = "Original_Image")],
                    "altered_images": [wandb.Image(altered_images.cpu().numpy()[i].transpose(1, 2, 0), caption = "Altered_Image")],
                    "gt_masks": [wandb.Image(masks.cpu().numpy()[i][0], caption = "GT_Mask")],
                    "pred_masks": [wandb.Image(pred_masks.cpu().numpy()[i][0], caption = "Pred_Mask")]
                    })

    elif dataset_type == 'sroie':
      for i in range (0, lent): # range (0, lent) - for every image combination
            
            # Log image(s)
            wandb.log(
                    {"original_images": [wandb.Image(orig_images.cpu().numpy()[i].transpose(1, 2, 0), caption = "Original_Image")],
                    "bb_masks": [wandb.Image(altered_images.cpu().numpy()[i].transpose(1, 2, 0), caption = "BB_Mask")],
                    "gt_masks": [wandb.Image(masks.cpu().numpy()[i][0], caption = "GT_Mask")],
                    "pred_masks": [wandb.Image(pred_masks.cpu().numpy()[i][0], caption = "Pred_Mask")]
                    })
          
  




