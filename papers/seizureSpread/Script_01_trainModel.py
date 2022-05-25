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

project_folder = "derivatives/seizure_spread"

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
train,  test = echomodel.splitDataframeTrainTest(patientsWithAnnotations, "subject", trainSize = 0.9)

#test_ind = 0
#test = patientsWithAnnotations[patientsWithAnnotations["subject"].isin([patientsWithAnnotations["subject"][test_ind]])]
#train = copy.deepcopy(patientsWithAnnotations)
#train.drop(train[train['subject'] == train["subject"][test_ind]].index, inplace = True)



#% 03 Project parameters
fsds = 128 #"sampling frequency, down sampled"
annotationLayerName = "seizureChannelBipolar"
#annotationLayerName = "seizure_spread"
secondsBefore = 180
secondsAfter = 180
window = 1#window of eeg for training/testing. In seconds
skipWindow = 0.5#Next window is skipped over in seconds
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

fname_iEEG = "HUP202_phaseII"
fname_iEEG = "HUP215_phaseII_D03"
fname_iEEG = "HUP155_phaseII"
fname_iEEG = "HUP224_phaseII"
fname_iEEG = "HUP196_phaseII"
fname_iEEG = "HUP189_phaseII"
fname_iEEG = "HUP215_phaseII_D04"
annotations, annotationsSeizure, annotationsUEOEEC = downloadiEEGorg.get_natus(username, password, fname_iEEG = fname_iEEG, annotationLayerName = "Imported Natus ENT annotations")

#%% Get data




#get training data
for i in range(len(train)):
    print(f"\n\n\n\n{i}/{len(train)}\n\n\n\n")
    sub = np.array(train["subject"])[i]
    idKey = np.array(train["idKey"])[i]
    AssociatedInterictal = np.array(train["AssociatedInterictal"])[i]
    if i ==0: #intialize
        X_train, y_train, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                              AssociatedInterictal, username, password,
                                                                              annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                              secondsBefore = secondsBefore,
                                                                              secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow, fsds = fsds)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                  AssociatedInterictal, username, password,
                                                                  annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                  secondsBefore = secondsBefore,
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow, fsds = fsds)
        X_train = np.concatenate([X_train, X], axis = 0)
        y_train = np.concatenate([y_train, y], axis = 0)

#get testing data
for i in range(len(test)):
    print(f"\n\n\n\n{i}/{len(test)}\n\n\n\n")
    sub = np.array(test["subject"])[i]
    idKey = np.array(test["idKey"])[i]
    AssociatedInterictal = np.array(test["AssociatedInterictal"])[i]
    if i ==0: #intialize
        X_test, y_test, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                            AssociatedInterictal, username, password,
                                                                            annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session,
                                                                            secondsBefore = secondsBefore,
                                                                            secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow, fsds = fsds)
    else:
        X, y, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey,
                                                                  AssociatedInterictal, username, password,
                                                                  annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore,
                                                                  secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow, fsds = fsds)
        X_test = np.concatenate([X_test, X], axis = 0)
        y_test = np.concatenate([y_test, y], axis = 0)



#=



#%%



names_train = "_".join(list(train["subject"]))
names_test = "_".join(list(test["subject"]))

full_analysis = [X_train, X_test, y_train, y_test, fsds, secondsBefore, secondsAfter, window, skipWindow, montage]

full_analysis_location_file_basename = f"train_{names_train}_test_{names_test}.pickle"

#######
analysis1 = "train_RID0194_RID0278_RID0320_RID0420_RID0536_RID0536_RID0572_RID0583_RID0596_RID0650_test_RID0502_RID0595_RID0648.pickle"






full_analysis_location = join(BIDS, project_folder, f"training_and_testing")
full_analysis_location_file_basename = "train_RID0194_RID0278_RID0320_RID0420_RID0536_RID0536_RID0572_RID0583_RID0596_RID0650_test_RID0502_RID0595_RID0648.pickle"
full_analysis_location_file = join(full_analysis_location, full_analysis_location_file_basename)
#with open(full_analysis_location_file, 'wb') as f: pickle.dump(full_analysis, f)

with open(full_analysis_location_file, 'rb') as f:[X_train, X_test, y_train, y_test, fsds, secondsBefore, secondsAfter, window, skipWindow, montage] = pickle.load(f)


#%% Model training

version = 17
# Wavenet
filepath = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "wavenet", training_epochs = 5, batch_size=2**9, learn_rate = 0.001)

#batch_size=2**11
# 1dCNN
filepath = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "1dCNN", training_epochs = 5, batch_size=2**9, learn_rate = 0.001)


# lstm
filepath = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "lstm", training_epochs = 3, batch_size=2**9, learn_rate = 0.001)







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
#%% absolute slope 


abs_slope_train = np.abs(np.divide(np.diff(X_train, axis=0), 1/fsds))
abs_slope_test = np.abs(np.divide(np.diff(X_test, axis=0), 1/fsds))


#abs_slope_ii = np.abs(np.divide(np.diff(dataII_scalerDS, axis=0), 1/fsds))
#sigma_ii = np.nanstd( abs_slope_ii ,  axis=0)
#sigma_ii = np.mean(sigma_ii)

#abs_slope_all_windows_normalized = abs_slope_all_windows/sigma_ii
#abs_slope_normalized = np.nanmean(abs_slope_all_windows_normalized, axis = 1)

absolute_slope_train_sum = np.zeros(len(X_train))
absolute_slope_test_sum = np.zeros(len(X_test))
for ch in range(len(abs_slope_train)):
    print(f"\r{np.round((ch+1)/len(abs_slope_train)*100,2)}%   ", end = "\r")
    absolute_slope_train_sum[ch] = np.nanmean(abs_slope_train[ch,:,0])
    #absolute_slope_train_sum[ch] = np.mean(absolute_slope_train[ch,:])
for ch in range(len(abs_slope_test)):
    print(f"\r{np.round((ch+1)/len(abs_slope_test)*100,2)}%   ", end = "\r")
    absolute_slope_test_sum[ch] = np.nanmean(abs_slope_test[ch,:,0])
    #absolute_slope_test_sum[ch] = np.mean(absolute_slope_test[ch,:])





absolute_slope_test_sum_norm  = absolute_slope_test_sum/np.max(power_train_sum)
tmp = np.array([1-absolute_slope_test_sum_norm, absolute_slope_test_sum_norm]   ).T


sns.scatterplot(x = np.argmax(y_test, axis = -1), y = absolute_slope_test_sum_norm)
ll_pos = power_test_sum_norm[np.where(np.argmax(y_test, axis = -1) == 1)]
ll_neg = power_test_sum_norm[np.where(np.argmax(y_test, axis = -1) == 0)]


thresh = echomodel.get_optimal_threshold(ll_pos, ll_neg)

echomodel.modelEvaluate(tmp, X_test, y_test, title = "Power", threshold = thresh)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

version = 11
yPredictProbability_WN = echomodel.modelPredict(join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5"), X_test)
yPredictProbability_CNN = echomodel.modelPredict(join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5"), X_test)
yPredictProbability_LSTM = echomodel.modelPredict( join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5"), X_test)

yPredictProbability_absolute_slope = np.array([1-absolute_slope_test_sum_norm, absolute_slope_test_sum_norm]   ).T
yPredictProbability_line_length = np.array([1-lineLength_arr_test_norm, lineLength_arr_test_norm]   ).T
yPredictProbability_pwr = np.array([1-power_test_sum_norm, power_test_sum_norm]   ).T

Y_predict = np.zeros(yPredictProbability_WN.shape[0])
Y_predict[yPredictProbability_WN[:,1]  >=0.99]=1


Y = np.argmax(y_test, axis=-1).reshape(y_test.shape[0], 1)

fpr, tpr, threshold = metrics.roc_curve(Y, yPredictProbability_WN[:,1], pos_label = 1 )
metrics.roc_auc_score(Y , yPredictProbability_WN[:,1])


metrics.roc_auc_score(Y, yPredictProbability_WN[:,1])
metrics.roc_auc_score(Y, yPredictProbability_CNN[:,1])
metrics.roc_auc_score(Y, yPredictProbability_LSTM[:,1])
metrics.roc_auc_score(Y, yPredictProbability_absolute_slope[:,1])
metrics.roc_auc_score(Y, yPredictProbability_line_length[:,1])
metrics.roc_auc_score(Y, yPredictProbability_pwr[:,1])




target_names = ['Interictal', 'Ictal']
print(metrics.classification_report(Y.reshape([-1]), Y_predict, target_names=target_names))

metrics.precision_recall_fscore_support(Y.reshape([-1]), Y_predict)



















#%%%%%%%%%%%

#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

#cross validation


#%%% Get data separately and save
#get testing data
for i in range(len(patientsWithAnnotations)):
    print(f"\n\n\n\n{i}/{len(patientsWithAnnotations)}\n\n\n\n")
    sub = np.array(patientsWithAnnotations["subject"])[i]
    idKey = np.array(patientsWithAnnotations["idKey"])[i]
    AssociatedInterictal = np.array(patientsWithAnnotations["AssociatedInterictal"])[i]

    X_test, y_test, data, dataII, dataAnnotations = DataJson.get_dataXY(sub, idKey, AssociatedInterictal, username, password, annotationLayerName, BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore, secondsAfter = secondsAfter, montage = montage, prewhiten = prewhiten, window = window , skipWindow = skipWindow, fsds = fsds)
    
        
    names = "_".join(list([sub,idKey ]))
  

    analysis = [X_test, y_test, fsds, secondsBefore, secondsAfter, window, skipWindow, montage]

    analysis_location_file_basename = f"sub_{names}.pickle"


    analysis_location = join(BIDS, project_folder, f"training_and_testing")
    analysis_location_file = join(analysis_location, analysis_location_file_basename)
    #with open(full_analysis_location_file, 'wb') as f: pickle.dump(full_analysis, f)

    with open(analysis_location_file, 'wb') as f: pickle.dump(analysis, f)

#%%
kfold_auc = pd.DataFrame( columns = ["k" , "WN" , "CNN" , "LSTM", "absolute_slope", "line_length", "power"] )

for k in range(len(patientsWithAnnotations)):
    
    print(f"\n\n{k}/{len(patientsWithAnnotations)}")
    sub = np.array(patientsWithAnnotations["subject"])[k]
    idKey = np.array(patientsWithAnnotations["idKey"])[k]
    names = "_".join(list([sub,idKey ]))

    analysis_location_file_basename = f"sub_{names}.pickle"
    analysis_location = join(BIDS, project_folder, f"training_and_testing")
    analysis_location_file = join(analysis_location, analysis_location_file_basename)
    
    with open(analysis_location_file, 'rb') as f:[X_test, y_test, fsds, secondsBefore, secondsAfter, window, skipWindow, montage] = pickle.load(f)
    
    
    #get training
    train = copy.deepcopy(patientsWithAnnotations)
    train.drop(train[train['subject'] == train["subject"][k]].index, inplace = True)
    for m in range(len(train)):
        sub_t = np.array(train["subject"])[m]
        idKey_t = np.array(train["idKey"])[m]
        names_t = "_".join(list([sub_t,idKey_t ]))

        analysis_location_file_basename_t = f"sub_{names_t}.pickle"
        analysis_location_t = join(BIDS, project_folder, f"training_and_testing")
        analysis_location_file_t = join(analysis_location_t, analysis_location_file_basename_t)
        
        if m == 0:
            with open(analysis_location_file, 'rb') as f:[X_train, y_train, fsds, secondsBefore, secondsAfter, window, skipWindow, montage] = pickle.load(f)
        else:
            with open(analysis_location_file, 'rb') as f:[X, y, fsds, secondsBefore, secondsAfter, window, skipWindow, montage] = pickle.load(f)
            
            X_train = np.concatenate([X_train, X], axis = 0)
            y_train = np.concatenate([y_train, y], axis = 0)
     
            
    
    # LL
    version = 100 + k
    
    feature_name = "line_length"
    filepath = join(deepLearningModelsPath, f"single_features/{feature_name}_v{version:03d}.hdf5")
    if utils.checkIfFileDoesNotExist(filepath, printBOOL=False):
        lineLength_arr_train, lineLengthNorm_train = echobase.line_length_x_train(X_train)
        lineLength_arr_test, lineLengthNorm_test = echobase.line_length_x_train(X_test)
        
        lineLength_arr_test_norm  = lineLength_arr_test/np.max(lineLength_arr_train)
        
        analysis = [lineLength_arr_train , lineLengthNorm_train, lineLength_arr_test, lineLengthNorm_test, lineLength_arr_test_norm]
        with open(filepath, 'wb') as f: pickle.dump(analysis, f)
    else:
        with open(filepath, 'rb') as f:[lineLength_arr_train , lineLengthNorm_train, lineLength_arr_test, lineLengthNorm_test, lineLength_arr_test_norm] = pickle.load(f)
        


    #power
    
    feature_name = "power"
    filepath = join(deepLearningModelsPath, f"single_features/{feature_name}_v{version:03d}.hdf5")
    if utils.checkIfFileDoesNotExist(filepath, printBOOL=False):
        power_train = echobase.get_power_x_train(X_train, fsds)
        power_test = echobase.get_power_x_train(X_test, fsds)
        print("calculating power sum")
        power_train_sum = np.zeros(len(X_train))
        power_test_sum = np.zeros(len(X_test))
        for ch in range(len(power_train)):
            #print(f"\r{np.round((ch+1)/len(power_train)*100,2)}%   ", end = "\r")
            power_train_sum[ch] = echomodel.integrate(range(power_train.shape[1]), power_train[ch,:])
        for ch in range(len(power_test)):
            #print(f"\r{np.round((ch+1)/len(power_test)*100,2)}%   ", end = "\r")
            power_test_sum[ch] = echomodel.integrate(range(power_test.shape[1]), power_test[ch,:])
        
        power_test_sum_norm  = power_test_sum/np.max(power_train_sum)
        
        analysis = [power_train , power_test, power_train_sum, power_test_sum, power_test_sum_norm]
        with open(filepath, 'wb') as f: pickle.dump(analysis, f)
    else:
        with open(filepath, 'rb') as f:[power_train , power_test, power_train_sum, power_test_sum, power_test_sum_norm] = pickle.load(f)
        
        
        
    #% absolute slope 

    feature_name = "absolute_slope"
    filepath = join(deepLearningModelsPath, f"single_features/{feature_name}_v{version:03d}.hdf5")
    if utils.checkIfFileDoesNotExist(filepath, printBOOL=False):

        abs_slope_train = np.abs(np.divide(np.diff(X_train, axis=0), 1/fsds))
        abs_slope_test = np.abs(np.divide(np.diff(X_test, axis=0), 1/fsds))
    
    
        absolute_slope_train_sum = np.zeros(len(X_train))
        absolute_slope_test_sum = np.zeros(len(X_test))
        for ch in range(len(abs_slope_train)):
            #print(f"\r{np.round((ch+1)/len(abs_slope_train)*100,2)}%   ", end = "\r")
            absolute_slope_train_sum[ch] = np.nanmean(abs_slope_train[ch,:,0])
        for ch in range(len(abs_slope_test)):
            #print(f"\r{np.round((ch+1)/len(abs_slope_test)*100,2)}%   ", end = "\r")
            absolute_slope_test_sum[ch] = np.nanmean(abs_slope_test[ch,:,0])
    
        absolute_slope_test_sum_norm  = absolute_slope_test_sum/np.max(power_train_sum)
        analysis = [abs_slope_train , abs_slope_test, absolute_slope_train_sum, absolute_slope_test_sum, absolute_slope_test_sum_norm]
        with open(filepath, 'wb') as f: pickle.dump(analysis, f)
    else:
        with open(filepath, 'rb') as f: [abs_slope_train , abs_slope_test, absolute_slope_train_sum, absolute_slope_test_sum, absolute_slope_test_sum_norm] = pickle.load(f)
        

    

    #deep learning
    version = 100 + k
    training_epochs= 1
    batch_size =2**9
    returnOpposite = True
    
    
    """
    learn_rate = 0.05# 0.001
    # Wavenet
    filepath = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
    if utils.checkIfFileDoesNotExist(filepath, printBOOL=False, returnOpposite=returnOpposite):
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "wavenet", training_epochs = training_epochs, batch_size=batch_size, learn_rate = learn_rate)
        """
    learn_rate = 0.02# 0.001
    #batch_size=2**11
    # 1dCNN
    filepath = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5")
    if utils.checkIfFileDoesNotExist(filepath, printBOOL=False, returnOpposite=returnOpposite):
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "1dCNN", training_epochs = training_epochs, batch_size=batch_size, learn_rate = learn_rate)
            
    # lstm
    """
    learn_rate = 0.01# 0.001
    filepath = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
    if utils.checkIfFileDoesNotExist(filepath, printBOOL=False, returnOpposite=returnOpposite):
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "lstm", training_epochs = training_epochs, batch_size=batch_size, learn_rate = learn_rate)
              """     

    
    version =  100 + k
    #yPredictProbability_WN = echomodel.modelPredict(join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5"), X_test)
    yPredictProbability_CNN = echomodel.modelPredict(join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5"), X_test)
    #yPredictProbability_LSTM = echomodel.modelPredict( join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5"), X_test)
    """
    if all((np.isnan(yPredictProbability_LSTM)).flatten()):
        filepath = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5")
        if utils.checkIfFileDoesNotExist(filepath, printBOOL=False, returnOpposite=returnOpposite):
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            score, hx = echomodel.modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "lstm", training_epochs = training_epochs, batch_size=batch_size, learn_rate = learn_rate)
            yPredictProbability_LSTM = echomodel.modelPredict( join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5"), X_test)
            """

    
    yPredictProbability_absolute_slope = np.array([1-absolute_slope_test_sum_norm, absolute_slope_test_sum_norm]   ).T
    yPredictProbability_line_length = np.array([1-lineLength_arr_test_norm, lineLength_arr_test_norm]   ).T
    yPredictProbability_pwr = np.array([1-power_test_sum_norm, power_test_sum_norm]   ).T
    

    Y = np.argmax(y_test, axis=-1).reshape(y_test.shape[0], 1)
    
    
    
    #fpr, tpr, threshold = metrics.roc_curve(Y, yPredictProbability_WN[:,1], pos_label = 1 )
    #metrics.roc_auc_score(Y , yPredictProbability_WN[:,1])
    
    
    #auc_WN = metrics.roc_auc_score(Y, yPredictProbability_WN[:,1])
    auc_CNN = metrics.roc_auc_score(Y, yPredictProbability_CNN[:,1])
    #auc_LSTM= metrics.roc_auc_score(Y, yPredictProbability_LSTM[:,1])
    auc_absolute_slope =  metrics.roc_auc_score(Y, yPredictProbability_absolute_slope[:,1])
    auc_line_length = metrics.roc_auc_score(Y, yPredictProbability_line_length[:,1])
    auc_power = metrics.roc_auc_score(Y, yPredictProbability_pwr[:,1])
    
    
    auc_WN= 0.75
    auc_LSTM= 0.75
    kfold_auc = kfold_auc.append(  dict( k = k, WN = auc_WN, CNN = auc_CNN, LSTM = auc_LSTM, absolute_slope = auc_absolute_slope, line_length = auc_line_length, power = auc_power), ignore_index = True)
    print(f"auc_WN: {auc_WN}")
    print(f"auc_CNN: {auc_CNN}")
    print(f"auc_LSTM: {auc_LSTM}")
    print(f"auc_absolute_slope {auc_absolute_slope}")
    print(f"auc_line_length: {auc_line_length}")
    print(f"auc_power: {auc_power}")
 
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_IDs = ["WN", "CNN", "LSTM", "absolute_slope", "line_length", "power"]
palette = {"WN": "#1d5e9e", "CNN": "#73ace5", "LSTM": "#7373e5", "absolute_slope": "#961c1d", "line_length": "#d16a6a" , "power": "#d19e6a" }
palette_dark = {"WN": "#0a2036", "CNN": "#1e60a1", "LSTM": "#3737b3", "absolute_slope": "#250b0b", "line_length": "#5b1c1c" , "power": "#5b3c1c" }



df = pd.melt(kfold_auc, id_vars = ["k"], var_name = ["model"], value_name = "AUC")

fig, axes = utils.plot_make(size_length=5)
sns.boxplot(ax = axes, data = df, x = "model", y = "AUC" , order=model_IDs, fliersize=0, palette=palette, medianprops=dict(color="black", lw = 4))

for i,artist in enumerate(axes.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    #artist.set_facecolor('None')

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    for j in range(i*6,i*6+6):
        line = axes.lines[j]
        #line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
        
for i, tick in enumerate(axes.xaxis.get_major_ticks()):
    tick.label.set_fontsize(6)              
        
axes.set_xlabel("model",fontsize=20)

sns.swarmplot(ax = axes,data = df, x = "model", y = "AUC" , order=model_IDs, palette=palette_dark)



##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################



#%%


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