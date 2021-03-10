#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:24:34 2021

@author: arevell
"""

import sys
import copy
import json
import os
import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
from os.path import join
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import bct

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

#import custom
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.seizureSpread import echomodel
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.dataClass import DataClassSfc, DataClassJson

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#%% Paths
fnameiEEGusernamePassword = join("/media","arevell","sharedSSD","linux", "ieegorg.json")
fnameJSON = join("/media","arevell","sharedSSD","linux", "data", "metadata", "iEEGdataRevell.json")
BIDS = join("/media","arevell","sharedSSD","linux", "data", "BIDS")
deepLearningModelsPath = "/media/arevell/sharedSSD/linux/data/deepLearningModels/seizureSpread"
dataset = "PIER"
session = "implant01"

#opening files
with open(fnameiEEGusernamePassword) as f: usernameAndpassword = json.load(f)
with open(fnameJSON) as f: jsonFile = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]



#%% Model parameters
fsds = 128 #"sampling frequency, down sampled"
annotationLayerName = "seizureChannelBipolar"
#annotationLayerName = "seizure_spread"
secondsBefore = 20
secondsAfter = 5
window = 10 #window of eeg for training/testing. In seconds
skipWindow = 0.25 #Next window is skipped over in seconds 
time_step, skip = int(window*fsds), int(skipWindow*fsds)
montage = "bipolar"
prewhiten = True


verbose = 1
training_epochs = 10
batch_size = 2**10
optimizer_n = 'adam'
learn_rate = 0.01
beta_1 = 0.9
beta_2=0.999
amsgrad=False
dropout=0.3
n_features = 1
input_shape = (time_step,  n_features)

#%% Get files and relevant patient information to train model

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = DataClassJson.DataClassJson(jsonFile)
#Get patients who have seizure annotations by channels (individual channels have seizure markings on ieeg.org)
patientsWithAnnotations = DataJson.get_patientsWithSeizureChannelAnnotations()
#Split Training and Testing sets BY PATIENT
train, test = echomodel.splitDataframeTrainTest(patientsWithAnnotations, "subject", trainSize = 0.66)


#%% Data exploration: finding patients with seizures to annotate on iEEG.org

i=0
sub = list(jsonFile["SUBJECTS"].keys() )[i]
fname_iEEG =  jsonFile["SUBJECTS"][sub]["Events"]["Ictal"]["1"]["FILE"]

fname_iEEG = "HUP197_phaseII_D01"
annotations, annotationsSeizure, annotationsUEOEEC = downloadiEEGorg.get_natus(username, password, fname_iEEG = fname_iEEG, annotationLayerName = "Imported Natus ENT annotations")

#%% Get data




#get training data
for i in range(len(train)):
    sub = np.array(train["subject"])[i]
    idKey = np.array(train["idKey"])[i]
    AssociatedInterictal = np.array(train["AssociatedInterictal"])[i]
    if i ==0: #intialize
        X_train, y_train, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey, 
                                                                              AssociatedInterictal, username, password, 
                                                                              annotationLayerName, BIDS = BIDS, dataset= dataset, session = session, 
                                                                              secondsBefore = secondsBefore, 
                                                                              secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey, 
                                                                  AssociatedInterictal, username, password, 
                                                                  annotationLayerName, BIDS = BIDS, dataset= dataset, session = session, 
                                                                  secondsBefore = secondsBefore, 
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
        X_train = np.concatenate([X_train, X], axis = 0)
        y_train = np.concatenate([y_train, y], axis = 0)

#get testing data
for i in range(len(test)):
    sub = np.array(test["subject"])[i]
    idKey = np.array(test["idKey"])[i]
    AssociatedInterictal = np.array(test["AssociatedInterictal"])[i]
    if i ==0: #intialize
        X_test, y_test, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey, 
                                                                            AssociatedInterictal, username, password, 
                                                                            annotationLayerName, BIDS = BIDS, dataset= dataset, session = session, 
                                                                            secondsBefore = secondsBefore, 
                                                                            secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey, 
                                                                  AssociatedInterictal, username, password, 
                                                                  annotationLayerName, BIDS = BIDS, dataset= dataset, session = session, secondsBefore = secondsBefore, 
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
        X_train = np.concatenate([X_test, X], axis = 0)
        y_train = np.concatenate([y_test, y], axis = 0)





#%% Model training

version = 0
# Wavenet
filepath = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "wavenet", training_epochs = 3, batch_size=2**10, learn_rate = 0.001)


# 1dCNN
filepath = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "1dCNN", training_epochs = 3, batch_size=2**10, learn_rate = 0.001)


# lstm
filepath = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "lstm", training_epochs = 3, batch_size=2**10, learn_rate = 0.001)







#%% Evaluate model
version = 0
fpath_model = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
yPredictProbability = echomodel.modelPredict(fpath_model, X_test) 
echomodel.modelEvaluate(yPredictProbability, X_test, y_test, title = "Wavenet Performance") 



fpath_model = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
yPredictProbability = echomodel.modelPredict(fpath_model, X_test) 
echomodel.modelEvaluate(yPredictProbability, X_test, y_test, title = "1dCNN Performance") 



fpath_model = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
yPredictProbability = echomodel.modelPredict(fpath_model, X_test) 
echomodel.modelEvaluate(yPredictProbability, X_test, y_test, title = "LSTM Performance") 


#%%


    