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
from revellLab.paths import constants_paths as paths
from revellLab.packages.utilities import utils


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#%%
#% 02 Paths and files
fnameiEEGusernamePassword = paths.IEEG_USERNAME_PASSWORD
metadataDir =  paths.METADATA
fnameJSON = join(metadataDir, "iEEGdataRevell.json")
BIDS = paths.BIDS
deepLearningModelsPath = paths.DEEP_LEARNING_MODELS
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
train,  test = echomodel.splitDataframeTrainTest(patientsWithAnnotations, "subject", trainSize = 0.66)



#% 03 Project parameters
fsds = 128 #"sampling frequency, down sampled"
annotationLayerName = "seizureChannelBipolar"
#annotationLayerName = "seizure_spread"
secondsBefore = 180
secondsAfter = 180
window = 1 #window of eeg for training/testing. In seconds
skipWindow = 1#Next window is skipped over in seconds
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
for i in range(2):
    sub = np.array(train["subject"])[i]
    idKey = np.array(train["idKey"])[i]
    AssociatedInterictal = np.array(train["AssociatedInterictal"])[i]
    if i ==0: #intialize
        X_train, y_train, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                              AssociatedInterictal, username, password,
                                                                              annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                              secondsBefore = secondsBefore,
                                                                              secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                  AssociatedInterictal, username, password,
                                                                  annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                  secondsBefore = secondsBefore,
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow)
        X_train = np.concatenate([X_train, X], axis = 0)
        y_train = np.concatenate([y_train, y], axis = 0)

#get testing data
for i in range(2):
    sub = np.array(test["subject"])[i]
    idKey = np.array(test["idKey"])[i]
    AssociatedInterictal = np.array(test["AssociatedInterictal"])[i]
    if i ==0: #intialize
        X_test, y_test, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                            AssociatedInterictal, username, password,
                                                                            annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                            secondsBefore = secondsBefore,
                                                                            secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                  AssociatedInterictal, username, password,
                                                                  annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore,
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow)
        X_test = np.concatenate([X_test, X], axis = 0)
        y_test = np.concatenate([y_test, y], axis = 0)





#%% Model training

version = 11
# Wavenet
filepath = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "wavenet", training_epochs = 10, batch_size=2**10, learn_rate = 0.001)


# 1dCNN
filepath = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "1dCNN", training_epochs = 10, batch_size=2**10, learn_rate = 0.001)


# lstm
filepath = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "lstm", training_epochs = 10, batch_size=2**10, learn_rate = 0.001)







#%% Evaluate model
version = 11
fpath_model = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
yPredictProbability = echomodel.modelPredict(fpath_model, X_test)
echomodel.modelEvaluate(yPredictProbability, X_test, y_test, title = "Wavenet Performance")
#plt.savefig("papers/seizureSpread/plotting/performance_wavenet.pdf")
#plt.savefig("papers/seizureSpread/plotting/performance_wavenet.png")


fpath_model = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
yPredictProbability = echomodel.modelPredict(fpath_model, X_test)
echomodel.modelEvaluate(yPredictProbability, X_test, y_test, title = "1dCNN Performance")
#plt.savefig("papers/seizureSpread/plotting/performance_1dcnn.pdf")
#plt.savefig("papers/seizureSpread/plotting/performance_1dcnn.png")

fpath_model = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
yPredictProbability = echomodel.modelPredict(fpath_model, X_test)
echomodel.modelEvaluate(yPredictProbability, X_test, y_test, title = "LSTM Performance")
#plt.savefig("papers/seizureSpread/plotting/performance_lstm.pdf")
#plt.savefig("papers/seizureSpread/plotting/performance_lstm.png")

#%%

#Calculate line length


lineLength_arr_train, lineLengthNorm_train = echobase.line_length_x_train(X_train)
lineLength_arr_test, lineLengthNorm_test = echobase.line_length_x_train(X_test)
#%%


#sns.scatterplot(x = np.argmax(y_train, axis = -1), y = lineLength_arr_train)
#sns.scatterplot(x = np.argmax(y_test, axis = -1), y = lineLength_arr_test)

lineLength_arr_test_norm  = lineLength_arr_test/np.max(lineLength_arr_train)
tmp = np.array([1-lineLength_arr_test_norm, lineLength_arr_test_norm]   ).T

lineLength_arr_train_norm  = lineLength_arr_train/np.max(lineLength_arr_train)
sns.scatterplot(x = np.argmax(y_test, axis = -1), y = lineLength_arr_test_norm)
ll_pos = lineLength_arr_train_norm[np.where(np.argmax(y_train, axis = -1) == 1)]
ll_neg = lineLength_arr_train_norm[np.where(np.argmax(y_train, axis = -1) == 0)]
#fig, axes = utils.plot_make(r = 1)
#sns.histplot(ll_pos, binrange= [0,0.5], binwidth=0.01, ax = axes , color = "red") 
#sns.histplot(ll_neg, binrange= [0,0.5], binwidth=0.01 , ax = axes, color = "blue")

thresh = echomodel.get_optimal_threshold(ll_pos, ll_neg)
echomodel.modelEvaluate(tmp, X_test, y_test, title = "Line Length Performance", threshold = thresh)
#plt.savefig("papers/seizureSpread/plotting/performance_line_length.pdf")
#plt.savefig("papers/seizureSpread/plotting/performance_line_length.png")

#%%
#power

power_train = echobase.get_power_x_train(X_train, fsds)
power_test = echobase.get_power_x_train(X_test, fsds)

"""
ch = 2
fig, axes = utils.plot_make(r = 2)
sns.lineplot(x = range(power_train.shape[1]), y = power_train[ch,:], ax = axes[0])
sns.lineplot(x = range(X_train.shape[1]), y = X_train[ch,:,0], ax = axes[1])
axes[0].text(x = 0, y =0, s = int(y_test[ch,1]), size = 30, va = "top" )
#axes[0].set_ylim([0,1])
"""
print("calculating power sum")
power_train_sum = np.zeros(len(X_train))
power_test_sum = np.zeros(len(X_test))
for ch in range(len(power_train)):
    print(f"\r{np.round((ch+1)/len(power_train)*100,2)}%   ", end = "\r")
    power_train_sum[ch] = echomodel.integrate(range(power_train.shape[1]), power_train[ch,:])
    #power_train_sum[ch] = np.mean(power_train[ch,:])
for ch in range(len(power_test)):
    print(f"\r{np.round((ch+1)/len(power_test)*100,2)}%   ", end = "\r")
    power_test_sum[ch] = echomodel.integrate(range(power_test.shape[1]), power_test[ch,:])
    #power_test_sum[ch] = np.mean(power_test[ch,:])



power_test_sum_norm  = power_test_sum/np.max(power_train_sum)
tmp = np.array([1-power_test_sum_norm, power_test_sum_norm]   ).T


sns.scatterplot(x = np.argmax(y_test, axis = -1), y = power_test_sum_norm)
ll_pos = power_test_sum_norm[np.where(np.argmax(y_test, axis = -1) == 1)]
ll_neg = power_test_sum_norm[np.where(np.argmax(y_test, axis = -1) == 0)]


thresh = echomodel.get_optimal_threshold(ll_pos, ll_neg)

echomodel.modelEvaluate(tmp, X_test, y_test, title = "Power", threshold = thresh)

plt.savefig("papers/seizureSpread/plotting/performance_power.pdf")
plt.savefig("papers/seizureSpread/plotting/performance_power.png")
#%% Variance 


var_train = echobase.get_varaince_x_train(X_train)
var_test = echobase.get_varaince_x_train(X_test)

var_test_norm  = var_test/np.max(var_train)
tmp = np.array([1-var_test_norm, var_test_norm]   ).T


sns.scatterplot(x = np.argmax(y_test, axis = -1), y = var_test_norm)
ll_pos = var_test_norm[np.where(np.argmax(y_test, axis = -1) == 1)]
ll_neg = var_test_norm[np.where(np.argmax(y_test, axis = -1) == 0)]

thresh = echomodel.get_optimal_threshold(ll_pos, ll_neg)
echomodel.modelEvaluate(tmp, X_test, y_test, title = "Variance", threshold = thresh)
plt.savefig("papers/seizureSpread/plotting/performance_var.pdf")
plt.savefig("papers/seizureSpread/plotting/performance_var.png")

# %% Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# LL, Pwr, Var
features_train = np.vstack([lineLength_arr_train, power_train_sum, var_train]).T
features_test = np.vstack([lineLength_arr_test, power_test_sum, var_test]).T
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
clf.fit(features_train, y_train)
y_pred = clf.predict(features_test)
metrics.accuracy_score(y_test, y_pred)
feature_names = ["ll", "pwr", "var"]
feature_importance = pd.DataFrame(columns = ["feature", "importance"])
for name, score in zip(feature_names, clf.feature_importances_):
    feature_importance = feature_importance.append(dict(feature = name, importance = score),ignore_index=True)
    print(name, score)
tmp = clf.predict_proba(features_test)[1]
echomodel.modelEvaluate(tmp, X_test, y_test, title = f"Random Forest;\nLL ({np.round(feature_importance.loc[0][1],2)}), Pwr ({np.round(feature_importance.loc[1][1],2)}), Var ({np.round(feature_importance.loc[2][1],2)})", threshold =0.5)
plt.savefig("papers/seizureSpread/plotting/performance_RF_ll_pwr_var.pdf")
plt.savefig("papers/seizureSpread/plotting/performance_RF_ll_pwr_var.png")

# Pwr, Var
features_train = np.vstack([power_train_sum, var_train]).T
features_test = np.vstack([power_test_sum, var_test]).T
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
clf.fit(features_train, y_train)
y_pred = clf.predict(features_test)
metrics.accuracy_score(y_test, y_pred)
feature_names = ["pwr", "var"]
feature_importance = pd.DataFrame(columns = ["feature", "importance"])
for name, score in zip(feature_names, clf.feature_importances_):
    feature_importance = feature_importance.append(dict(feature = name, importance = score),ignore_index=True)
    print(name, score)
tmp = clf.predict_proba(features_test)[1]
echomodel.modelEvaluate(tmp, X_test, y_test, title = f"Random Forest;\nPwr ({np.round(feature_importance.loc[0][1],2)}), Var ({np.round(feature_importance.loc[1][1],2)})", threshold =0.5)
plt.savefig("papers/seizureSpread/plotting/performance_RF_pwr_var.pdf")
plt.savefig("papers/seizureSpread/plotting/performance_RF_pwr_var.png")

# LL, Var
features_train = np.vstack([lineLength_arr_train, var_train]).T
features_test = np.vstack([lineLength_arr_test, var_test]).T
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
clf.fit(features_train, y_train)
y_pred = clf.predict(features_test)
metrics.accuracy_score(y_test, y_pred)
feature_names = ["ll", "var"]
feature_importance = pd.DataFrame(columns = ["feature", "importance"])
for name, score in zip(feature_names, clf.feature_importances_):
    feature_importance = feature_importance.append(dict(feature = name, importance = score),ignore_index=True)
    print(name, score)
tmp = clf.predict_proba(features_test)[1]
echomodel.modelEvaluate(tmp, X_test, y_test, title = f"Random Forest;\nLL ({np.round(feature_importance.loc[0][1],2)}), Var ({np.round(feature_importance.loc[1][1],2)})", threshold =0.5)
plt.savefig("papers/seizureSpread/plotting/performance_RF_ll_var.pdf")
plt.savefig("papers/seizureSpread/plotting/performance_RF_ll_var.png")

# LL, Pwr
features_train = np.vstack([lineLength_arr_train, power_train_sum]).T
features_test = np.vstack([lineLength_arr_test, power_test_sum]).T
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
clf.fit(features_train, y_train)
y_pred = clf.predict(features_test)
metrics.accuracy_score(y_test, y_pred)
feature_names = ["ll", "pwr"]
feature_importance = pd.DataFrame(columns = ["feature", "importance"])
for name, score in zip(feature_names, clf.feature_importances_):
    feature_importance = feature_importance.append(dict(feature = name, importance = score),ignore_index=True)
    print(name, score)
tmp = clf.predict_proba(features_test)[1]
echomodel.modelEvaluate(tmp, X_test, y_test, title = f"Random Forest;\nLL ({np.round(feature_importance.loc[0][1],2)}), Pwr ({np.round(feature_importance.loc[1][1],2)})", threshold =0.5)
plt.savefig("papers/seizureSpread/plotting/performance_RF_ll_pwr.pdf")
plt.savefig("papers/seizureSpread/plotting/performance_RF_ll_pwr.png")