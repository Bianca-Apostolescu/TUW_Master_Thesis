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



def plot_results(lent, orig_images, altered_images, masks, pred_masks):
    
    for i in range (0, lent): # range (0, lent) - for every image combination
          plt.figure(figsize = (12, 12))

          # plot original image
          plt.subplot(141)
          plt.imshow(orig_images.cpu().numpy()[i].transpose(1, 2, 0))
          plt.title('Original Image')
          plt.axis('off')

          # plot altered image
          plt.subplot(142)
          plt.imshow(altered_images.cpu().numpy()[i].transpose(1, 2, 0))
          plt.title('Altered Image')
          plt.axis('off')

          # plot ground truth mask
          plt.subplot(143)
          plt.imshow(masks.cpu().numpy()[i][0], cmap='gray')
          plt.title('Ground Truth Mask')
          plt.axis('off')

          # plot predicted mask
          plt.subplot(144)
          plt.imshow(pred_masks.cpu().numpy()[i][0], cmap='gray')
          plt.title('Predicted Mask')
          plt.axis('off')

          # display the plot
          plt.show()


# In[5]:




