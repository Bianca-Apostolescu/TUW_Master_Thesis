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
import TrainModel as train
import ValidateModel as val
import CreateDataset as crd
import DisplayMetrics as dm
import PlotResults as pr
# import MainLoop as main


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline



# In[ ]:





# In[4]:


def main_loop(model, original_images, altered_images, masks, transforms_train, transforms_test, wb_name, lr, batch_size, epochs, test_split, valid_split):
    
    wandb.login()

    # wandb.init(
    #         project = wb_name,
    #         # name = "init_metrics_run_" + epoch, 
    #         # Track hyperparameters and run metadata
    #         config = {
    #                 "learning_rate": lr,
    #                 "epochs": epochs,
    #                 "batch": batch_size
    #                 },
    #         )

    # Initialize loss function and optimizer
    lossFunc = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr = lr)



    for tts in test_split:
        print("[INFO] TEST_SPLIT = {} ...".format(tts))

        print("Splits, Datasets, and Dataloaders")
        startTime = time.time()

        train_orig_images, test_orig_images, train_altered_images, test_altered_images, train_masks, test_masks = train_test_split(original_images, altered_images, masks, test_size = tts, random_state = 42)
        train_orig_images, val_orig_images, train_altered_images, val_altered_images, train_masks, val_masks = train_test_split(train_orig_images, train_altered_images, train_masks, test_size = valid_split, random_state = 42)

        # Create datasets and data loaders for training, validation, and testing sets
        train_dataset = crd.SegmentationDataset(train_orig_images, train_altered_images, train_masks, transforms = transforms_train)
        val_dataset   = crd.SegmentationDataset(val_orig_images,   val_altered_images,   val_masks,   transforms = transforms_test)
        test_dataset  = crd.SegmentationDataset(test_orig_images,  test_altered_images,  test_masks,  transforms = transforms_test)

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

            wandb.init(
              project = wb_name,
              name = "init_metrics_run_epochs_" + str(epoch), 
              # Track hyperparameters and run metadata
              config = {
                      "learning_rate": lr,
                      "epochs": epochs,
                      "batch": batch_size
                      },
              )

            # wandb.name("init_metrics_run_epochs_" + epoch)


            # Initialize dictionary to store training history
            # H = {"train_loss": [], "val_loss": [], "test_loss": []}
            # Perf_Metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
            # Dice_IOU = {"dice": [], "iou": []}

            print("[INFO] Training the network for {} epochs...".format(epoch))

            startTime = time.time()

            for e in tqdm(range(epoch)):

                # initialize variables for output
                totalTrainLoss, totalTestLoss, totalValLoss = 0, 0, 0
                accuracy_scores, precision_score, recall_scores, f1_scores = 0, 0, 0, 0
                dice_scores, iou_scores = 0, 0



                #### TRINING LOOP ####
                model.train() # unet.train.train() 
                for orig_images, altered_images, masks in train_loader:
                    images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)

                    opt.zero_grad()
                    pred_masks = model(altered_images) # they are not binary => the binary masks are displayed using the vizualize function with a threshold
                    # pred_masks = torch.sigmoid(pred_masks)

                    loss = lossFunc(pred_masks, masks)
                    loss.backward()
                    opt.step()

                    pred_masks = (pred_masks > 0.5).float()

                    totalTrainLoss += loss.item()

                avg_train_loss = totalTrainLoss / len(train_loader)




                #### VALIDATION LOOP ####
                model.eval()
                with torch.no_grad():
                    total_val_loss = 0
                    for orig_images, altered_images, masks in val_loader:
                        orig_images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)
                        pred_masks = model(orig_images)
                        val_loss = lossFunc(pred_masks, masks)

                        totalValLoss += val_loss.item()

                avg_val_loss = totalValLoss / len(val_loader)

                # Log the losses to WandB
                wandb.log(
                      {
                      "Epoch": e,
                      "Train Loss": avg_train_loss,
                      # "Train Acc": acc_train,
                      "Valid Loss": avg_val_loss,
                      # "Valid Acc": acc_valid,
                      }
                      )
                  



            #### TESTING LOOP ####
            conf_matrix = 0
            accuracy, recall, precision, f1_score, dice_coefficient, iou = 0, 0, 0, 0, 0, 0 

            masks_list = []
            pred_masks_list = []

            model.eval()
            with torch.no_grad():
                for orig_images, altered_images, masks in test_loader:
                    orig_images, altered_images, masks = orig_images.to(device), altered_images.to(device), masks.to(device)
                    pred_masks = model(orig_images)

                    masks_np = masks.cpu().detach().numpy().argmax(axis=1).reshape(-1)
                    pred_masks_np = pred_masks.cpu().detach().numpy().argmax(axis=1).reshape(-1)
                    conf_matrix += confusion_matrix(masks_np, pred_masks_np)

                    # Append masks and predicted masks to lists
                    masks_list.append(masks_np)
                    pred_masks_list.append(pred_masks_np)

                # tn = conf_matrix[0][0]
                # tp = conf_matrix[1][1]
                # fp = conf_matrix[0][1]
                # fn = conf_matrix[1][0]

                masks_all = np.concatenate(masks_list)
                pred_masks_all = np.concatenate(pred_masks_list)

                tn = conf_matrix[0][0] if conf_matrix.shape[0] > 0 and conf_matrix.shape[1] > 0 else 0
                tp = conf_matrix[0][0] if conf_matrix.shape[0] > 0 and conf_matrix.shape[1] > 0 else 0
                fp = conf_matrix[0][0] if conf_matrix.shape[0] > 0 and conf_matrix.shape[1] > 0 else 0
                fn = conf_matrix[0][0] if conf_matrix.shape[0] > 0 and conf_matrix.shape[1] > 0 else 0


                accuracy = np.sum(np.diag(conf_matrix)/np.sum(conf_matrix))
                recall = tp/(tp + fn)
                precision = tp/(tp + fp)
                f1_score = (2 * precision * recall) / (precision + recall)
                dice_coefficient = (2 * tp) / (2 * tp + fp + fn)

                # masks_all = masks_all.cpu().detach().numpy()
                # pred_masks_all = pred_masks_all.cpu().detach().numpy()

                # Compute intersection and union
                intersection = np.logical_and(masks_all, pred_masks_all)
                union = np.logical_or(masks_all, pred_masks_all)

                iou = intersection.sum() / union.sum()

                wandb.log(
                      {
                      "Accuracy": accuracy,
                      "Precision": precision,
                      "Recall": recall,
                      "F1-Score": f1_score,
                      "DICE": dice_coefficient,
                      "IOU": iou,
                      }
                      )


                


    #         # Averaging performance metrics and updating training history
    #         avgTrainLoss = totalTrainLoss / trainSteps
    #         avgValLoss = totalValLoss / valSteps
    #         avgTestLoss = totalTestLoss / testSteps

    #         accuracy_scores = accuracy_scores / testSteps #testSteps
    #         precision_score = precision_score / testSteps
    #         recall_scores = recall_scores / testSteps
    #         f1_scores = f1_scores / testSteps

    #         dice_scores = dice_scores / testSteps
    #         iou_scores = iou_scores / testSteps

    #         # update our training history
    #         H["train_loss"].append(avgTrainLoss)
    #         H["val_loss"].append(avgValLoss)
    #         H["test_loss"].append(avgTestLoss)

    #         # update our performance metrics history
    #         Perf_Metrics["accuracy"].append(accuracy_scores)
    #         Perf_Metrics["precision"].append(precision_score)
    #         Perf_Metrics["recall"].append(recall_scores)
    #         Perf_Metrics["f1_score"].append(f1_scores)


    #         # update our dice and IOU score history
    #         Dice_IOU["dice"].append(dice_scores)
    #         Dice_IOU["iou"].append(iou_scores)

    #         # Print training loss
    #         print("[INFO] EPOCH: {}/{}".format(e + 1, epoch))
    #         print("Train loss: {:.6f}".format(avgTrainLoss))

    #         # # Log the metrics to WandB
    #         #     wandb.log(
    #         #           {
    #         #           "Accuracy": avg_train_loss,
    #         #           "Precision": acc_train,
    #         #           "Recall": avg_val_loss,
    #         #           "F1-Score": acc_valid,
    #         #           "DICE":,
    #         #           "IOU":,
    #         #           }
    #         #           )

                

    # #             wandb.log({"accuracy": accuracy_scores, "precision": precision_score, "loss": avgTrainLoss})

            # Display total time taken to perform the training
            endTime = time.time()
            print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime - startTime))


            # Display performance metrics
            # print('\n')
            # dm.display_metrics(Perf_Metrics, Dice_IOU)


            # Plot results - images 
            print('\n')
            lent = orig_images.cpu().numpy().shape[0]
            pr.plot_results(lent, orig_images, altered_images, masks, pred_masks)



            # # plot the training loss
            # plt.style.use("ggplot")
            # plt.figure()
            # plt.plot(H["train_loss"], label = "train_loss")
            # # plt.plot(H["val_loss"], label="val_loss")
            # # plt.plot(H["test_loss"], label="test_loss")
            # plt.title("Training, Validation and Test Loss on Dataset")
            # plt.xlabel("Epoch #")
            # plt.ylabel("Loss")
            # plt.legend()
            # plt.show()


            print('\n')

