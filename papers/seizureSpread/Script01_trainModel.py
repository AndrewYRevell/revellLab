#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:24:34 2021

@author: arevell
"""

#%% 01 Import

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
import pkg_resources

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

#import custom
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.seizureSpread import echomodel
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.dataclass import dataclass_SFC, dataclass_iEEG_metadata
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#% 02 Paths and files
fnameiEEGusernamePassword = join("/media","arevell","sharedSSD","linux", "ieegorg.json")
metadataDir =  join("/media","arevell","sharedSSD","linux", "data", "metadata")
fnameJSON = join(metadataDir, "iEEGdataRevell.json")
BIDS = join("/media","arevell","sharedSSD","linux", "data", "BIDS")
BIDSpenn = join(BIDS, "PIER")
BIDSmusc = join(BIDS, "MIER")
deepLearningModelsPath = "/media/arevell/sharedSSD/linux/data/deepLearningModels/seizureSpread"
datasetiEEG = "derivatives/seizure_spread/iEEG_data"
session = "implant01"


revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
atlasPath = join(tools, "atlases", "atlases" )
atlasLabelsPath = join(tools, "atlases", "atlasLabels" )
atlasfilesPath = join(tools, "atlases", "atlasMetadata.json")
MNItemplatePath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz")
MNItemplateBrainPath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz")


atlasLocaliztionDir = join(BIDS, "derivatives", "atlasLocalization")
atlasLocalizationFunctionDirectory = join(revellLabPath, "packages", "atlasLocalization")


#opening files
with open(fnameiEEGusernamePassword) as f: usernameAndpassword = json.load(f)
with open(fnameJSON) as f: jsonFile = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]


# Get files and relevant patient information to train model

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = dataclass_iEEG_metadata.dataclass_iEEG_metadata(jsonFile)
#Get patients who have seizure annotations by channels (individual channels have seizure markings on ieeg.org)
patientsWithAnnotations = DataJson.get_patientsWithSeizureChannelAnnotations()
#Split Training and Testing sets BY PATIENT
train, test = echomodel.splitDataframeTrainTest(patientsWithAnnotations, "subject", trainSize = 0.66)



#% 03 Project parameters
fsds = 128 #"sampling frequency, down sampled"
annotationLayerName = "seizureChannelBipolar"
#annotationLayerName = "seizure_spread"
secondsBefore = 180
secondsAfter = 180
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



#%% 04 Electrode and atlas localization
#May not need to run - mainly used to help annotate EEG and know what electrode contacts are where

#atl.atlasLocalizationBIDSwrapper(["RID0380"], atlasLocalizationFunctionDirectory, atlasLocaliztionDir, atlasPath, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, atlasLocaliztionDir, rerun = True)
atl.atlasLocalizationFromBIDS(BIDS, "PIER", "RID0380", "preop3T", "3D", join(atlasLocaliztionDir, "sub-RID0380", "electrodenames_coordinates_native_and_T1.csv") , atlasLocalizationFunctionDirectory,
                              MNItemplatePath , MNItemplateBrainPath, atlasPath, atlasLabelsPath, atlasLocaliztionDir, rerun = True )

atl.atlasLocalizationBIDSwrapper(["RID0267", "RID0259", "RID0250", "RID0241", "RID0240", "RID0238", "RID0230"], atlasLocalizationFunctionDirectory, atlasLocaliztionDir, atlasPath, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, atlasLocaliztionDir, multiprocess = True)
atl.atlasLocalizationBIDSwrapper(["RID0279"], atlasLocalizationFunctionDirectory, atlasLocaliztionDir, atlasPath, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, atlasLocaliztionDir)
atl.atlasLocalizationBIDSwrapper(["RID0252"], atlasLocalizationFunctionDirectory, atlasLocaliztionDir, atlasPath, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, atlasLocaliztionDir)



#RID0279 really poor preimplant clinical MRI
#RID0476 has R/L AVM burst?
#RID0472- has structural malformation
#RID0338 - re run for bias correction using research 3T
#RID 337 missing
#RID 307 parietal something
#RID 280 run with high resolution

#267
#%% Data exploration: finding patients with seizures to annotate on iEEG.org

i=0
sub = list(jsonFile["SUBJECTS"].keys() )[i]
fname_iEEG =  jsonFile["SUBJECTS"][sub]["Events"]["Ictal"]["1"]["FILE"]

fname_iEEG = "HUP214_phaseII_D01"
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
                                                                              annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                              secondsBefore = secondsBefore,
                                                                              secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                  AssociatedInterictal, username, password,
                                                                  annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
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
                                                                            annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                            secondsBefore = secondsBefore,
                                                                            secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                  AssociatedInterictal, username, password,
                                                                  annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore,
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten)
        X_train = np.concatenate([X_test, X], axis = 0)
        y_train = np.concatenate([y_test, y], axis = 0)





#%% Model training

version = 1
# Wavenet
filepath = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "wavenet", training_epochs = 3, batch_size=2**10, learn_rate = 0.001)


# 1dCNN
filepath = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "1dCNN", training_epochs = 30, batch_size=2**10, learn_rate = 0.001)


# lstm
filepath = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "lstm", training_epochs = 3, batch_size=2**10, learn_rate = 0.001)







#%% Evaluate model
version = 1
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


