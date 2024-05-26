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
from segmentation_models_pytorch.decoders.fpn.model import FPNDecoder
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
import segmentation_models_pytorch as smp

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
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix



# Functions - to have them separated in files
import CalcMetrics as cm
import BinaryMasks as bm
import TrainModel as trModel
import ValidateModel as valModel
import TestModel as testModel
import CreateDataset_Comofod as com
import CreateDataset_IMD2020 as imd
import DisplayMetrics as dm
import PlotResults as pr
import EarlyStopping as stopping
import DiceLoss as dcloss
import GCANet as gca
# import MainLoop as main


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline



def main_loop(original_images, altered_images, masks, transforms_train, transforms_test, model_type, channels, dataset_type, wb_name, lr, batch_size, epochs, test_split, valid_split):
    
    wandb.login()

    


    for tts in test_split:
        print("[INFO] TEST_SPLIT = {} ...".format(tts))

        print("Splits, Datasets, and Dataloaders")
        startTime = time.time()

        train_orig_images, test_orig_images, train_altered_images, test_altered_images, train_masks, test_masks = train_test_split(original_images, altered_images, masks, test_size = tts, random_state = 42)
        train_orig_images, val_orig_images, train_altered_images, val_altered_images, train_masks, val_masks = train_test_split(train_orig_images, train_altered_images, train_masks, test_size = valid_split, random_state = 42)

        # Create datasets and data loaders for training, validation, and testing sets
        if dataset_type == 'comofod':
          train_dataset = com.SegmentationDataset(train_orig_images, train_altered_images, train_masks, transforms = transforms_train)
          val_dataset   = com.SegmentationDataset(val_orig_images,   val_altered_images,   val_masks,   transforms = transforms_test)
          test_dataset  = com.SegmentationDataset(test_orig_images,  test_altered_images,  test_masks,  transforms = transforms_test)
        elif dataset_type == 'imd':
          train_dataset = imd.SegmentationDataset(train_orig_images, train_altered_images, train_masks, transforms = transforms_train)
          val_dataset   = imd.SegmentationDataset(val_orig_images,   val_altered_images,   val_masks,   transforms = transforms_test)
          test_dataset  = imd.SegmentationDataset(test_orig_images,  test_altered_images,  test_masks,  transforms = transforms_test)

        train_loader = DataLoader(train_dataset, shuffle = True,  batch_size = batch_size)
        val_loader   = DataLoader(val_dataset,   shuffle = False, batch_size = batch_size)
        test_loader  = DataLoader(test_dataset,  shuffle = False, batch_size = batch_size)


        endTime = time.time()
        print("[INFO] Total time taken to create the dataset and dataloader: {:.2f}s".format(endTime - startTime))

        # calculate steps per epoch for training set
        trainSteps = len(train_dataset) // batch_size
        testSteps  = len(test_dataset) // batch_size
        valSteps   = len(val_dataset) // batch_size

        print(f"trainSteps = {trainSteps}, testSteps = {testSteps}, valSteps = {valSteps}")

        for epoch in epochs:

          if model_type == 'GCA':
            gcanet = gca.GCANet(in_c = channels, out_c = 1, only_residual = True).to(device)
            model = gcanet

          elif model_type == 'unet':
            unet = smp.Unet(
                  encoder_name = "resnet101",
                  encoder_weights = "imagenet",
                  in_channels = channels,  # 3 channels for the image
                  classes = 1,  # 1 class => binary mask
                  activation = 'sigmoid'
                ).to(device)
            model = unet

          # Initialize loss function and optimizer
          # lossFunc = nn.BCEWithLogitsLoss()
          lossFunc = dcloss.DiceBCELoss()
          opt = torch.optim.Adam(model.parameters(), lr = lr)

          wandb.init(
            project = wb_name,
            name = "init_metrics_run_" + "tts" + str(tts) + "_ep" + str(epoch), 
            # Track hyperparameters and run metadata
            config = {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch": batch_size
                    },
            )


          print("[INFO] Training the network for {} epochs...".format(epoch))

          startTime = time.time()

          for e in tqdm(range(epoch)):

              #### TRAINING LOOP ####
              avg_train_loss = trModel.train_model(model, train_loader, lossFunc, opt, device, channels)
              

              #### VALIDATION LOOP ####
              avg_val_loss = valModel.validate_model(model, val_loader, lossFunc, device, channels)

              early_stopping = stopping.EarlyStopping(patience = 5, verbose = True)

              # Check if validation loss has improved
              early_stopping(avg_val_loss)

              # If validation loss hasn't improved, break the loop
              if early_stopping.early_stop:
                  print("Early stopping")


              # Log the losses to WandB
              wandb.log(
                    {
                    "Epoch": e,
                    "Train Loss": avg_train_loss,
                    "Valid Loss": avg_val_loss,
                    }
                    )
                
          # Display total time taken to perform the training
          endTime = time.time()
          print("[INFO] Total time taken to train and validate the model: {:.2f}s".format(endTime - startTime))


          #### TESTING LOOP ####
          avg_test_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_dice_score, avg_iou, TP, TN, FP, FN = testModel.test_model(model, test_loader, lossFunc, device, channels)

          print(f"avg_accuracy = {avg_accuracy}, avg_precision = {avg_precision}, avg_recall = {avg_recall}, avg_f1_score = {avg_f1_score}, avg_dice_score = {avg_dice_score}, avg_iou = {avg_iou}")
          print(f"TP = {TP}")
          print(f"TN = {TN}")
          print(f"FP = {FP}")
          print(f"FN = {FN}")


          wandb.log(
                {
                "Accuracy": avg_accuracy,
                "Precision": avg_precision,
                "Recall": avg_recall,
                "F1-Score": avg_f1_score,
                "DICE": avg_dice_score,
                "IOU": avg_iou
                }
                )

