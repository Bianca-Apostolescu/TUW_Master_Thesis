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


def validate_model(model, dataloader, steps, loss_function, optim, device):
  print("Validating...")

  model.eval()

  totalTrainLoss = 0

  # loop over the training set
  with torch.no_grad():
    for i, (x, y) in enumerate(dataloader):

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = loss_function(pred, y)

        # add the loss to the total training loss so far
        totalTrainLoss += loss


    avgTrainLoss = totalTrainLoss / steps

    return avgTrainLoss


# In[5]:




