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

import PlotResults as pr


# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics.classification import Dice, BinaryJaccardIndex
# from torchmetrics.detection import IntersectionOverUnion
# from torchmetrics import JaccardIndex


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline



def test_model(model, dataloader, loss_function, device):
    print("Testing...")


    conf_matrix = 0
    totalTestLoss = 0
    accuracy, recall, precision, f1_score, dice_score, iou = 0, 0, 0, 0, 0, 0 


    model.eval()
    with torch.no_grad():
        
        for orig_images, altered_images, masks in dataloader:
            orig_images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)
            pred_masks = model(altered_images) # Testing for altered images 
            
            # tTransform both masks into binary - just to be sure 
            masks = (masks > 0.5).float()
            pred_masks = (pred_masks > 0.5).float()

            # Check if they are binary
            # print(f"binary masks = {((masks == 0) | (masks == 1)).all()}")
            # print(f"binary pred masks = {((pred_masks == 0) | (pred_masks == 1)).all()}")
            
            test_loss = loss_function(pred_masks, masks)
            totalTestLoss += test_loss.item()

            # Check if tensors
            # print("masks is a PyTorch tensor." if torch.is_tensor(masks) else "masks is not a PyTorch tensor.")
            # print("pred_masks is a PyTorch tensor." if torch.is_tensor(pred_masks) else "pred_masks is not a PyTorch tensor.")


            # Plot results - images 
            print('\n')
            lent = orig_images.cpu().numpy().shape[0]
            pr.plot_results(lent, orig_images, altered_images, masks, pred_masks)

            # Flatten the masks tensors
            masks = masks.view(-1)
            pred_masks = pred_masks.view(-1)

            # Torch Metrics
            metric = BinaryAccuracy()
            metric.update(pred_masks, masks)
            accuracy += metric.compute()

            metric = BinaryPrecision()
            metric.update(pred_masks, masks)
            precision += metric.compute()

            metric = BinaryRecall()
            metric.update(pred_masks.to(torch.uint8), masks.to(torch.uint8))
            recall += metric.compute()

            metric = BinaryF1Score()
            metric.update(pred_masks, masks)
            f1_score += metric.compute()

            metric = BinaryJaccardIndex().to(device)
            metric.update(pred_masks, masks)
            iou += metric.compute()

            metric = Dice().to(device)
            metric.update(pred_masks.to(device), masks.long().to(device))
            dice_score += metric.compute()

            
            
    avg_test_loss   = totalTestLoss / len(dataloader)
    avg_accuracy    = accuracy / len(dataloader)
    avg_precision   = precision / len(dataloader)
    avg_recall      = recall / len(dataloader)
    avg_f1_score    = f1_score / len(dataloader)
    avg_dice_score  = dice_score / len(dataloader)
    avg_iou         = iou / len(dataloader)

    return avg_test_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_dice_score, avg_iou

