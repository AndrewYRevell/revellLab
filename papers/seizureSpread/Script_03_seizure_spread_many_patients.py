#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:52:41 2022

@author: arevell
"""


import sys
import os
import json
import copy
import time
import bct
import glob
import math
import random
import pickle
import pingouin
import scipy
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
import statsmodels.api as sm
from scipy import signal, stats
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename

from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

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
from revellLab.papers.seizureSpread import seizurePattern
from revellLab.packages.diffusionModels import diffusionModels as DM

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Plotting parameters
custom_params = {"axes.spines.right": False, "axes.spines.top": False, 'figure.dpi': 300,
                 "legend.frameon": False, "savefig.transparent": True}
sns.set_theme(style="ticks", rc=custom_params,  palette="pastel")
sns.set_context("talk")
aspect = 50
kde_kws = {"bw_adjust": 2}

#%
#% 02 Paths and files
fnameiEEGusernamePassword = paths.IEEG_USERNAME_PASSWORD
metadataDir =  paths.METADATA
fnameJSON = join(metadataDir, "iEEGdataRevell_seizure_severity.json")
BIDS = paths.BIDS
deepLearningModelsPath = paths.DEEP_LEARNING_MODELS
datasetiEEG = "derivatives/seizure_spread/iEEG_data"
datasetiEEG_preprocessed = "derivatives/seizure_spread/preprocessed" ##################################################
datasetiEEG_spread = "derivatives/seizure_spread/seizure_spread_measurements"
session = "implant01"
SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"

#%
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
atlasPath = join(tools, "atlases", "atlases" )
atlasLabelsPath = join(tools, "atlases", "atlasLabels" )
atlasfilesPath = join(tools, "atlases", "atlasMetadata.json")
MNItemplatePath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz")
MNItemplateBrainPath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz")


atlasLocaliztionDir = join(BIDS, "derivatives", "atlasLocalization")
atlasLocalizationFunctionDirectory = join(revellLabPath, "packages", "atlasLocalization")


#% 03 Project parameters
fsds = 128 #"sampling frequency, down sampled"
annotationLayerName = "seizureChannelBipolar"
#annotationLayerName = "seizure_spread"
secondsBefore = 180
secondsAfter = 180
window = 1 #window of eeg for training/testing. In seconds
skipWindow = 0.1#Next window is skipped over in seconds
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

#opening files
with open(fnameiEEGusernamePassword) as f: usernameAndpassword = json.load(f)
with open(fnameJSON) as f: jsonFile = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]



#% Get files and relevant patient information to train model

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = dataclass_iEEG_metadata.dataclass_iEEG_metadata(jsonFile)
patientsWithseizures = DataJson.get_patientsWithSeizuresAndInterictal()

unique_patients = np.array(patientsWithseizures.subject)
v = {}
for i, x in enumerate(unique_patients):
    if x not in v:
        v[x] = i
        
indexes = np.array([v.get(x) for x in list(v.keys()) ])

def prob_threshold_moving_avg(prob_array, fsds, skip, threshold = 0.9, smoothing = 20):
    windows, nchan = prob_array.shape
   
    w = int(smoothing*fsds/skip)
    probability_arr_movingAvg = np.zeros(shape = (windows - w + 1, nchan))
    
    for c in range(nchan):
        probability_arr_movingAvg[:,c] =  echobase.movingaverage(prob_array[:,c], w)
        
    probability_arr_threshold = copy.deepcopy(probability_arr_movingAvg)
    probability_arr_threshold[probability_arr_threshold > threshold] = 1
    probability_arr_threshold[probability_arr_threshold <= threshold] = 0
        
    return probability_arr_movingAvg, probability_arr_threshold


def get_start_times(secondsBefore, skipWindow, fsds, channels, start, stop, probability_arr_threshold):
    
    nchan = probability_arr_threshold.shape[1]
    seizure_start =int((secondsBefore-start)/skipWindow)
    seizure_stop = int((secondsBefore + stop)/skipWindow)
    
    probability_arr_movingAvg_threshold_seizure = probability_arr_threshold[seizure_start:,:]
    spread_start = np.argmax(probability_arr_movingAvg_threshold_seizure == 1, axis = 0)
    
    for c in range(nchan): #if the channel never starts seizing, then np.argmax returns the index as 0. This is obviously wrong, so fixing this
        if np.all( probability_arr_movingAvg_threshold_seizure[:,c] == 0  ) == True:
            spread_start[c] = len(probability_arr_movingAvg_threshold_seizure)
    
    
    spread_start_loc = ( (spread_start + seizure_start)  *skipWindow*fsds).astype(int)
    markers = spread_start_loc
    channel_order = np.argsort(spread_start)
    channel_order_labels = np.array(channels)[channel_order]
    print(np.array(channels)[channel_order])
    return spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels

#%%
i=0 
#for i in indexes[39]:#range(23,25):
for i in range(0,100):

    RID = np.array(patientsWithseizures["subject"])[i]
    idKey = np.array(patientsWithseizures["idKey"])[i]
    AssociatedInterictal = np.array(patientsWithseizures["AssociatedInterictal"])[i]
    seizure_length = patientsWithseizures.length[i]
    
    #Check if preprocessed data is available
    
    #%
    #check if preprocessed is already saved
    fname = DataJson.get_fname_ictal(RID, "Ictal", idKey, dataset= datasetiEEG, session = session, startUsec = None, stopUsec= None, startKey = "EEC", secondsBefore = secondsBefore, secondsAfter = secondsAfter )
    
    preprocessed_location = join(BIDS, datasetiEEG_preprocessed, f"sub-{RID}" )
    utils.checkPathAndMake( preprocessed_location, preprocessed_location, printBOOL=False)
    
    preprocessed_file_basename = f"{splitext(fname)[0]}_preprocessed.pickle"
    preprocessed_file = join(preprocessed_location, preprocessed_file_basename)
    
    
    #check if preproceed file exists, and if it does, load that instead of dowloading ieeg data and performing preprocessing on that
    if utils.checkIfFileExists( preprocessed_file, printBOOL=False ):
        print(f"\n{RID} {i} PREPROCESSED FILE EXISTS")
        with open(preprocessed_file, 'rb') as f: [dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels] = pickle.load(f)
        df, fs, ictalStartIndex, ictalStopIndex = DataJson.get_precitalIctalPostictal(RID, "Ictal", idKey, username, password,BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore, secondsAfter = secondsAfter)
        
        
    else:
        df, fs, ictalStartIndex, ictalStopIndex = DataJson.get_precitalIctalPostictal(RID, "Ictal", idKey, username, password,BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore, secondsAfter = secondsAfter)
        df_interictal, _ = DataJson.get_iEEGData(RID, "Interictal", AssociatedInterictal, username, password, BIDS = BIDS, dataset= datasetiEEG, session = session, startKey = "Start")
        
        df = df.fillna(0)
        df_interictal = df_interictal.fillna(0)
     
        
        print(f"\n{RID} {i} PREPROCESSING")
        
        
        dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels = DataJson.preprocessNormalizeDownsample(df, df_interictal, fs, fsds, montage = montage)
    
        pickle_save = [dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels]
        with open(preprocessed_file, 'wb') as f: pickle.dump(pickle_save, f)
        
        
    ########################################
    #########################################
    #######################################
    
    #deploying model
    ########################################
    #########################################
    #######################################
    
    _, nchan = data_scalerDS.shape
    
    np.unique(patientsWithseizures.subject) 
    tmp = df.columns

    version = 11
    fpath_wavenet = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
    fpath_1dCNN = join(deepLearningModelsPath, f"1dCNN/v{version:03d}.hdf5")
    fpath_lstm = join(deepLearningModelsPath, f"lstm/v{version:03d}.hdf5")
    
    
    #%window to fit model
    data_scalerDS_X = echomodel.overlapping_windows(data_scalerDS, time_step, skip)
    windows, _, _ = data_scalerDS_X.shape
    

    
    #CHECKING IF MODEL FILES EXIST
    
    spread_location = join(BIDS, datasetiEEG_spread, f"v{version:03d}", f"sub-{RID}" )
    utils.checkPathAndMake( spread_location, spread_location, printBOOL=False)
    
    spread_location_file_basename = f"{splitext(fname)[0]}_spread.pickle"
    spread_location_file = join(spread_location, spread_location_file_basename)
    
    if utils.checkIfFileExists( spread_location_file , printBOOL=False):
        print(f"\n{RID} {i} SPREAD FILE EXISTS")
        #with open(spread_location_file, 'rb') as f:[probWN, probCNN, probLSTM, data_scalerDS, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
    else:
        modelWN = load_model(fpath_wavenet)
        modelCNN= load_model(fpath_1dCNN)
        modelLSTM = load_model(fpath_lstm)
        
        probWN = np.zeros(shape = (windows, nchan))  
        probCNN = np.zeros(shape = (windows, nchan))  
        probLSTM = np.zeros(shape = (windows, nchan))  
            
        #%
        for c in range(nchan):
            print(f"\r{np.round((c+1)/nchan*100,2)}%     ", end = "\r")
            ch_pred =     data_scalerDS_X[:,:,c].reshape(windows, time_step, 1    )
            probWN[:,c] =  modelWN.predict(ch_pred, verbose=0)[:,1]
            probCNN[:,c] =  modelCNN.predict(ch_pred, verbose=0)[:,1]
            probLSTM[:,c] =  modelLSTM.predict(ch_pred, verbose=0)[:,1]
        
        pickle_save = [probWN, probCNN, probLSTM, data_scalerDS, channels, window, skipWindow, secondsBefore, secondsAfter]
        with open(spread_location_file, 'wb') as f: pickle.dump(pickle_save, f)
            
        
    ########################################### 
    #calculate absolute slope
    ########################################### 
    feature_name = "absolute_slope"
    
    #CHECKING IF MODEL FILES EXIST
    spread_location_sf = join(BIDS, datasetiEEG_spread, "single_features", f"sub-{RID}" )
    utils.checkPathAndMake( spread_location_sf, spread_location_sf, printBOOL=False)
    
    spread_location_sf_file_basename = f"{splitext(fname)[0]}_{feature_name}.pickle"
    spread_location_sf_file = join(spread_location_sf, spread_location_sf_file_basename)
    
    if utils.checkIfFileExists( spread_location_sf_file , printBOOL=False):
        print(f"\n{RID} {i} {feature_name} EXISTS")
        #with open(spread_location_sf_file, 'rb') as f:[abs_slope_normalized_tanh, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
    else:
        abs_slope_all_windows = np.abs(np.divide(np.diff(data_scalerDS_X, axis=1), 1/fsds))
        abs_slope_ii = np.abs(np.divide(np.diff(dataII_scalerDS, axis=0), 1/fsds))
        sigma_ii = np.nanstd( abs_slope_ii ,  axis=0)
        sigma_ii = np.mean(sigma_ii)
        
        abs_slope_all_windows_normalized = abs_slope_all_windows/sigma_ii
        abs_slope_normalized = np.nanmean(abs_slope_all_windows_normalized, axis = 1)
        
        #Tanh
        multiplier = 1e-1 
        abs_slope_normalized_tanh = utils.apply_tanh(abs_slope_normalized, multiplier = multiplier)
        
        pickle_save = [abs_slope_normalized_tanh, channels, window, skipWindow, secondsBefore, secondsAfter]
        with open(spread_location_sf_file, 'wb') as f: pickle.dump(pickle_save, f)
        
    
    #########
    pic_name = splitext(spread_location_sf_file_basename)[0] + "_PICTURE_05_absolute_slope.png"
    if not utils.checkIfFileExists( join(spread_location_sf, pic_name), printBOOL=False):
    
        THRESHOLD = 0.4
        SMOOTHING = 20 #in seconds
        prob_array= abs_slope_normalized_tanh
        
        
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
        
        
        seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.4, vmax = 1, center=None, smoothing = 20 , title = "Absolute Slope", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)    
        
        plt.savefig(join(spread_location_sf, pic_name) )
        plt.show()
        
    ########################################### 
    #calculate Line Length
    ########################################### 
    feature_name = "line_length"
    
    #CHECKING IF MODEL FILES EXIST
    spread_location_sf = join(BIDS, datasetiEEG_spread, "single_features", f"sub-{RID}" )
    utils.checkPathAndMake( spread_location_sf, spread_location_sf, printBOOL=False)
    
    spread_location_sf_file_basename = f"{splitext(fname)[0]}_{feature_name}.pickle"
    spread_location_sf_file = join(spread_location_sf, spread_location_sf_file_basename)
    
    if utils.checkIfFileExists( spread_location_sf_file , printBOOL=False):
        print(f"\n{RID} {i} {feature_name} EXISTS")
        #with open(spread_location_sf_file, 'rb') as f:[probLL_tanh, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
    else:
        probLL, probLL_norm = echobase.lineLengthOfArray(data_scalerDS_X)
        #apply tanh function to map LL to seizure probability
        multiplier = 2e-3 #multiplier multiplies the abolute LL value to put into the tanh function. LL values are very large, so thats why multiplier is very small
        probLL_tanh = utils.apply_tanh(probLL, multiplier = multiplier)
        
        pickle_save = [probLL_tanh, channels, window, skipWindow, secondsBefore, secondsAfter]
        with open(spread_location_sf_file, 'wb') as f: pickle.dump(pickle_save, f)
    
    #########
    pic_name = splitext(spread_location_sf_file_basename)[0] + "_PICTURE_04_line_length.png"
    if not utils.checkIfFileExists( join(spread_location_sf, pic_name), printBOOL=False):
        THRESHOLD = 0.5
        SMOOTHING = 20 #in seconds
        prob_array= probLL_tanh
        
        
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
        
        
        seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.4, vmax = 1, center=None, smoothing = 20 , title = "Line Length", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)
        
        plt.savefig(join(spread_location_sf, pic_name) )
        plt.show()
        
    
        
        
        #%%
        
 
    ########################################### 
    ########################################### 
    ########################################### 
    ########################################### 
    ########################################### 
    ########################################### 
    #PLOTTING 
    #% Optional
    
    
    
    seizure_start = int((secondsBefore-0)/skipWindow)
    seizure_stop = int((secondsAfter + seizure_length+10)/skipWindow)
    #%

    # %Wavenet
    pic_name = splitext(spread_location_file_basename)[0] + "_PICTURE_01_wavenet.png"
    if not utils.checkIfFileExists( join(spread_location, pic_name)):
    
        THRESHOLD = 0.7
        SMOOTHING = 20 #in seconds
        prob_array= probWN
        
        
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        #sns.heatmap( probability_arr_movingAvg.T , cbar=False)      
        #sns.heatmap( probability_arr_threshold.T , cbar=False)    
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
        
        seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.4, vmax = 1, center=None, smoothing = 20 , title = "Wavenet", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)
        
        plt.savefig(join(spread_location, pic_name) )
        plt.show()
    
    #%CNN
    
    pic_name = splitext(spread_location_file_basename)[0] + "_PICTURE_02_1dCNN.png"
    if not utils.checkIfFileExists( join(spread_location, pic_name)):
        THRESHOLD = 0.7
        SMOOTHING = 20 #in seconds
        prob_array= probCNN
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
        
        
        seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.5, vmax = 1, center=None, smoothing = 20 , title = "1D CNN", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)
        
        plt.savefig(join(spread_location, pic_name) )
        plt.show()
        
    #%LSTM
        
    pic_name = splitext(spread_location_file_basename)[0] + "_PICTURE_03_LSTM.png"
    if not utils.checkIfFileExists( join(spread_location, pic_name)):
    
        THRESHOLD = 0.6
        SMOOTHING = 20 #in seconds
        prob_array= probLSTM
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
        
        
        
        
        seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.5, vmax = 1, center=None, smoothing = 20 , title = "LSTM", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)
        
        plt.savefig(join(spread_location, pic_name) )
        plt.show()
    
    
    pic_to_check = join(spread_location,  splitext(spread_location_file_basename)[0] + "_PICTURE_00_EEG_ZOOM_ORDERED.png")
    if not utils.checkIfFileExists( pic_to_check):
    
    
    
    
        #####################################
        #% Plotting
        #for plotting eeg 
        fsfs = 64
        data_DSDS = DataJson.downsample(data_scaler, fs, fsfs)
        
        plot_before = 10
        
        
        
        THRESHOLD = 0.7
        SMOOTHING = 20 #in seconds
        prob_array= probWN
        
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
        
        #getting markers
        markers = (spread_start_loc/fsds - (secondsBefore-plot_before))*fsfs
        total_stop = data_DSDS[(secondsBefore-plot_before)*fsfs:int((secondsBefore+ seizure_length + plot_before))*fsfs,:].shape[0]
        markers[np.where(markers > total_stop)] = total_stop
        
    
        
        aspect=275
        #not ordered
        DataJson.plot_eeg(data_DSDS, fsfs,  startSec = (secondsBefore-plot_before), stopSec = (secondsBefore+ seizure_length + plot_before)  , markers =markers, dpi = 250, aspect=aspect, height=nchan/aspect/11, nchan=nchan, channel_names_show = True, channel_names = channels, channel_size = 2, labelpad = -20, markers2 = [(plot_before)*fsfs])  
        
        pic_name = splitext(spread_location_file_basename)[0] + "_PICTURE_00_EEG_UNORDERED.png"
        plt.savefig(join(spread_location, pic_name) )
        plt.show()
        
        
        # ordered
        DataJson.plot_eeg(data_DSDS[:,channel_order], fsfs,  startSec = (secondsBefore-plot_before), stopSec = (secondsBefore+ seizure_length + plot_before)  , markers = markers[channel_order], dpi = 250, aspect=aspect, height=nchan/aspect/11, nchan=nchan, channel_names_show = True, channel_names = channel_order_labels, channel_size = 20, labelpad = -20, markers2 = [(plot_before)*fsfs])  
        
        pic_name = splitext(spread_location_file_basename)[0] + "_PICTURE_00_EEG_ORDERED.png"
        plt.savefig(join(spread_location, pic_name) )
        plt.show()
        
        # ordered Larger
        aspect=50
        DataJson.plot_eeg(data_DSDS[:,channel_order], fsfs,  startSec = (secondsBefore-plot_before), stopSec = (secondsBefore+ seizure_length + plot_before)  , markers = markers[channel_order], dpi = 250, aspect=aspect, height=nchan/aspect/10, nchan=nchan,  channel_names_show = True, channel_names = channel_order_labels, channel_size = 20, labelpad = -20, markers2 = [(plot_before)*fsfs])  
        
        pic_name = splitext(spread_location_file_basename)[0] + "_PICTURE_00_EEG_ZOOM_ORDERED.png"
        plt.savefig(join(spread_location, pic_name) )
        plt.show()
        
        




    ########################################### 
    ########################################### 
    ########################################### 
    #Single features
    ########################################### 
    ########################################### 
    ########################################### 
        
        
        
    probLL = np.zeros(shape = (windows, nchan))  
    abs_slope = np.zeros(shape = (windows, nchan))  
    
    ########################################### 
    #calculate Line Length
    ########################################### 

    probLL, probLL_norm = echobase.lineLengthOfArray(data_scalerDS_X)
    #apply tanh function to map LL to seizure probability
    multiplier = 2e-3 #multiplier multiplies the abolute LL value to put into the tanh function. LL values are very large, so thats why multiplier is very small
    probLL_tanh = utils.apply_tanh(probLL, multiplier = multiplier)

    sns.heatmap( probLL_tanh.T )  
    THRESHOLD = 0.5
    SMOOTHING = 20 #in seconds
    prob_array= probLL_tanh
    
    
    probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
    
    spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
    
    
    seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.4, vmax = 1, center=None, smoothing = 20 , title = "Line Length", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)
    
    
    
    
    
    
    
    ########################################### 
    #calculate BB power
    ########################################### 
    power, power_total = echobase.get_power_over_windows(data_scalerDS_X, fsds)
    
    #Tanh
    multiplier = 7e-2 
    power_total_tanh = utils.apply_tanh(power_total, multiplier = multiplier)
    
    #########
    THRESHOLD = 0.3
    SMOOTHING = 20 #in seconds
    prob_array= power_total_tanh
    
    
    probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
    
    spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
    
    
    seizurePattern.plot_probheatmaps(prob_array[:,channel_order], fsds, skip, threshold=THRESHOLD, vmin = 0.4, vmax = 1, center=None, smoothing = 20 , title = "Power Broadband", title_channel1 = channel_order_labels[0], title_channel2 = channel_order_labels[1], channel_names = channel_order_labels)
    
    
    ########################################### 
    #calculate strength, pearson FC
    ########################################### 
    data_scalerDS_X.shape
    
    fc_degree = np.zeros(shape = (windows, nchan))
    for w in range(windows):
        print(f"{w}/{windows}")
        fc_pearson = echobase.pearson_connectivity(data_scalerDS_X[w,:,:], fs)
        fc_pearson_values = abs(fc_pearson[0])
        fc_degree[w,:] = np.sum(fc_pearson_values, axis = 0)

        
        #calculate absolute degree
    ########################################### 
    ########################################### 
    ########################################### 
    ########################################### 
    ########################################### 
    ########################################### 
        