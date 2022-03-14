#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:05:45 2021

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


#% 02 Paths and files
fnameiEEGusernamePassword = paths.IEEG_USERNAME_PASSWORD
metadataDir =  paths.METADATA
fnameJSON = join(metadataDir, "iEEGdataRevell.json")
BIDS = paths.BIDS
deepLearningModelsPath = paths.DEEP_LEARNING_MODELS
datasetiEEG = "derivatives/seizure_spread/iEEG_data"
session = "implant01"
SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"

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

#opening files
with open(fnameiEEGusernamePassword) as f: usernameAndpassword = json.load(f)
with open(fnameJSON) as f: jsonFile = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]


#plotting parameters
aspect = 50

#% Get files and relevant patient information to train model

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = dataclass_iEEG_metadata.dataclass_iEEG_metadata(jsonFile)
patientsWithseizures = DataJson.get_patientsWithSeizuresAndInterictal()

#%%Get data


tmp = np.unique(patientsWithseizures["subject"])
i = 86#RID0309: 33-43 ; RID0278:19; RID0596: 86
RID = np.array(patientsWithseizures["subject"])[i]
idKey = np.array(patientsWithseizures["idKey"])[i]
AssociatedInterictal = np.array(patientsWithseizures["AssociatedInterictal"])[i]


#%%
df, fs, ictalStartIndex, ictalStopIndex = DataJson.get_precitalIctalPostictal(RID, "Ictal", idKey, username, password,BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore, secondsAfter = secondsAfter)
df_interictal, _ = DataJson.get_iEEGData(RID, "Interictal", AssociatedInterictal, username, password, BIDS = BIDS, dataset= datasetiEEG, session = session, startKey = "Start")

                   
print("preprocessing")
dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels = DataJson.preprocessNormalizeDownsample(df, df_interictal, fs, fsds, montage = montage)


data, data_avgref, data_ar, data_filt, channels = echobase.preprocess(df, fs, fsds, montage = montage)
data_scalerDSDS = DataJson.downsample(data_scaler, fs, 16)
data_arDSDS = DataJson.downsample(data_ar, fs, 16)

nsamp, nchan = data_filt.shape



# %% Plot eeg  Optional
#DataJson.plot_eeg(data_scalerDSDS, 16, markers = [secondsBefore*16], dpi = 25, aspect=aspect*2, height=nchan/aspect/2)  

#for plotting eeg of 595
#data_DSDS = DataJson.downsample(data_filt, fs, 16)
#DataJson.plot_eeg(data_DSDS, 16, markers = [secondsBefore*16], dpi = 25, aspect=aspect*2, height=nchan/aspect/2)  

#%%Meaure seizure spread
version = 11
fpath_wavenet = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
fpath_1dCNN = join(deepLearningModelsPath, f"1dCNN/v{version:03d}.hdf5")
fpath_lstm = join(deepLearningModelsPath, f"lstm/v{version:03d}.hdf5")


modelWN = load_model(fpath_wavenet)
modelCNN= load_model(fpath_1dCNN)
modelLSTM = load_model(fpath_lstm)

#%window to fit model
data_scalerDS_X = echomodel.overlapping_windows(data_scalerDS, time_step, skip)
windows, _, _ = data_scalerDS_X.shape

probWN = np.zeros(shape = (windows, nchan))  
probCNN = np.zeros(shape = (windows, nchan))  
probLSTM = np.zeros(shape = (windows, nchan))  
    


  
#%%


for c in range(nchan):
    print(f"\r{np.round((c+1)/nchan*100,2)}%     ", end = "\r")
    ch_pred =     data_scalerDS_X[:,:,c].reshape(windows, time_step, 1    )
    probWN[:,c] =  modelWN.predict(ch_pred, verbose=0)[:,1]
    probCNN[:,c] =  modelCNN.predict(ch_pred, verbose=0)[:,1]
    probLSTM[:,c] =  modelLSTM.predict(ch_pred, verbose=0)[:,1]
        


def prob_threshold_moving_avg(prob_array, fsds, skip, threshold = 0.9, smoothing = 20):
    nchan = prob_array.shape[1]
    w = int(smoothing*fsds/skip)
    probability_arr_movingAvg = np.zeros(shape = (windows - w + 1, nchan))
    
    for c in range(nchan):
        probability_arr_movingAvg[:,c] =  echobase.movingaverage(prob_array[:,c], w)
        
    probability_arr_threshold = copy.deepcopy(probability_arr_movingAvg)
    probability_arr_threshold[probability_arr_threshold > threshold] = 1
    probability_arr_threshold[probability_arr_threshold <= threshold] = 0
        
    return probability_arr_movingAvg, probability_arr_threshold

#%% Optional

seizurePattern.plot_probheatmaps(probWN, fsds, skip, threshold=0.7, vmin = 0., vmax = 1, center=None, smoothing = 20)
seizurePattern.plot_probheatmaps(probCNN, fsds, skip, threshold=0.9, vmin = 0.4, vmax = 1, center=None, smoothing = 20)
seizurePattern.plot_probheatmaps(probLSTM, fsds, skip, threshold=0.6, vmin = 0.5, vmax = 1, center=None, smoothing = 20)



# %%
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
THRESHOLD = 0.6
SMOOTHING = 20 #in seconds
prob_array= probLSTM


probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
sns.heatmap( probability_arr_movingAvg.T , cbar=False)      
sns.heatmap( probability_arr_threshold.T , cbar=False)    
    
    
spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 20, 20, probability_arr_threshold)
    
#%%  Optional
# Ridgeline
###
DataJson.plot_eeg(probability_arr_movingAvg, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    
DataJson.plot_eeg(probability_arr_threshold, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    
###

#%%getting start times


#%%Calculate Line length and getting top channels 
print(RID)
SMOOTHING = 20 #in seconds

_, _, spread_start_loc_WM, channel_order_WN, channel_order_labels_WN = get_start_times(secondsBefore, skipWindow, fsds, channels, 20, 20, prob_threshold_moving_avg(probWN, fsds, skip, threshold = 0.9, smoothing = SMOOTHING)[1]  )
_, _, spread_start_loc_CNN, channel_order_CNN, channel_order_labels_CNN = get_start_times(secondsBefore, skipWindow, fsds, channels, 20, 20, prob_threshold_moving_avg(probCNN, fsds, skip, threshold = 0.9, smoothing = SMOOTHING)[1]  )
_, _, spread_start_loc_LSTM, channel_order_LSTM, channel_order_labels_LSTM = get_start_times(secondsBefore, skipWindow, fsds, channels, 20, 20, prob_threshold_moving_avg(probLSTM, fsds, skip, threshold = 0.9, smoothing = SMOOTHING)[1]  )


spearmanr( channel_order_WN, channel_order_CNN)[0]
spearmanr( channel_order_WN, channel_order_LSTM)[0]
spearmanr( channel_order_CNN, channel_order_LSTM)[0]




lineLength_arr_train, lineLengthNorm_train = echobase.lineLengthOfArray(data_scalerDS_X)
lineLength_arr_train_II = echobase.line_length_array_2d(dataII_scalerDS)
tmp =  lineLength_arr_train/lineLength_arr_train_II
tmp1 = tmp/np.max(tmp)


print(RID)
_, _, spread_start_loc_LL, channel_order_LL, channel_order_labels_LL = get_start_times(secondsBefore, skipWindow, fsds, channels, 20, 20, prob_threshold_moving_avg(tmp1, fsds, skip, threshold = 0.2, smoothing = SMOOTHING)[1]  )



spearmanr( channel_order_WN, channel_order_LL)[0]
spearmanr( channel_order_CNN, channel_order_LL)[0]
spearmanr( channel_order_LSTM, channel_order_LL)[0]



sns.heatmap( tmp1[:,channel_order_LL].T )  
sns.heatmap( probWN[:,channel_order_WN].T )  



#calculating spread 

print(RID)

timeee = 40
print(len(np.where(spread_start_loc_WM/fsds - secondsBefore < timeee)[0])/nchan)
print(len(np.where(spread_start_loc_CNN/fsds - secondsBefore < timeee)[0])/nchan)
print(len(np.where(spread_start_loc_LSTM/fsds - secondsBefore < timeee)[0])/nchan)
print(len(np.where(spread_start_loc_LL/fsds - secondsBefore < timeee)[0])/nchan)


timeee = 20
print(len(np.where(spread_start_loc_WM/fsds - secondsBefore < timeee)[0])/nchan)
print(len(np.where(spread_start_loc_CNN/fsds - secondsBefore < timeee)[0])/nchan)
print(len(np.where(spread_start_loc_LSTM/fsds - secondsBefore < timeee)[0])/nchan)
print(len(np.where(spread_start_loc_LL/fsds - secondsBefore < timeee)[0])/nchan)

#%%Plotting


WN_per = np.array([1, 0.8,0.75,0.625,0.2857142857,1,0.1,0.6,0.875,0.7777777778])
CNN_per = np.array([0.75,0.6,0.25,0.25,0.2857142857,0.875,0.1,0.4,0.875,0.8888888889])
LL = np.array([0.75,0.8,0.125,0.125,0.4285714286,0.5,0,0.6,0.125,0.5555555556])

df_per = pd.concat([pd.DataFrame(WN_per),pd.DataFrame(CNN_per), pd.DataFrame(LL)], axis= 1)
df_per.columns = ["WN", "CNN", "LL"]

fig, axes = utils.plot_make()
sns.boxplot(data = df_per, palette = ["#90b2e5","#9990e5", "#e5c390"], fliersize=0)
sns.swarmplot(data = df_per, palette = ["#163460","#1e1660", "#604216"])
sns.despine()


palette = ["#90b2e5","#9990e5", "#e5c390"]
plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")

for a in range(len( axes.artists)):
    mybox = axes.artists[a]
    # Change the appearance of that box
    mybox.set_facecolor(palette[a])
    mybox.set_edgecolor(palette[a])
    #mybox.set_linewidth(3)
count = 0
a = 0
for line in axes.get_lines():
    line.set_color("#222222")
    count = count + 1
    if count % 5 ==0:
        a = a +1
    if count % 5 ==0: #set mean line
        line.set_color("#222222")
        #line.set_ls("-")
        #line.set_lw(2.5)



axes.text(x = 1, y = 1.1, s = np.round(stats.ttest_rel(WN_per, LL )[1],5) )
axes.text(x = 1, y = 1.02, s = np.round(stats.ttest_rel(WN_per, CNN_per )[1],5) )
axes.text(x = 1, y = 0.95, s = np.round(stats.ttest_rel(CNN_per, LL )[1],5) )
stats.ttest_rel(WN_per, LL )[1]


plt.savefig( f"SOZ.png", transparent=True, dpi = 300)
plt.savefig( f"SOZ.pdf", transparent=True)
#################
WN_spread_good = [0.09923664122,0.03448275862,0.2857142857,0.1904761905]
WN_spread_poor = [0.3950617284,0.4505494505,0.656,0.3495145631]
CNN_spread_good = [0.2442748092,0.1034482759,0.3736263736,0.2222222222]
CNN_spread_poor = [0.7037037037,0.5054945055,0.936,0.5922330097]
LL_spread_good = [0.06106870229,0.01724137931,0.2197802198,0.1904761905]
LL_spread_poor = [0.1234567901,0.3736263736,0.368,0.4077669903]

df_spread = pd.DataFrame(np.vstack(  [WN_spread_good,WN_spread_poor,CNN_spread_good,CNN_spread_poor,LL_spread_good,LL_spread_poor] ))
df_spread.columns = ["patient1","patient2","patient3","patient4"]


df_spread["algorithm"] = ["WN", "WN", "CNN", "CNN", "LL", "LL"]
df_spread["outcome"] = ["good", "poor", "good", "poor","good", "poor"]
df_spread_log = pd.melt(df_spread, id_vars = ["algorithm", "outcome"])

palette = ["#417bd2", "#90b2e5","#6d60da", "#9990e5", "#daa960", "#e5c390"]
palette = ["#90b2e5", "#417bd2","#9990e5", "#6d60da", "#e5c390", "#daa960"]



fig, axes = utils.plot_make()
sns.boxplot(data = df_spread_log, x = "algorithm", y ="value", hue = "outcome",fliersize=0)
sns.swarmplot(data = df_spread_log, color = "#222222",x = "algorithm", y ="value", hue = "outcome",dodge = True)
sns.despine()
axes.legend([],[], frameon=False)

plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")

for a in range(len( axes.artists)):
    mybox = axes.artists[a]
    # Change the appearance of that box
    mybox.set_facecolor(palette[a])
    mybox.set_edgecolor(palette[a])
    #mybox.set_linewidth(3)
count = 0
a = 0
for line in axes.get_lines():
    line.set_color("#222222")
    count = count + 1
    if count % 5 ==0:
        a = a +1
    if count % 5 ==0: #set mean line
        line.set_color("#222222")
        #line.set_ls("-")
        #line.set_lw(2.5)
        
        
stats.ttest_ind(WN_spread_good, WN_spread_poor )[1]
stats.ttest_ind(CNN_spread_good, CNN_spread_poor )[1]
stats.ttest_ind(LL_spread_good, LL_spread_poor )[1]

utils.cohend(WN_spread_good, WN_spread_poor)
utils.cohend(CNN_spread_good, CNN_spread_poor)
utils.cohend(LL_spread_good, LL_spread_poor)

axes.text(x = 0, y = 1.1, s = np.round(stats.ttest_ind(WN_spread_good, WN_spread_poor )[1],5) )
axes.text(x = 1, y = 1.1, s = np.round(stats.ttest_ind(CNN_spread_good, CNN_spread_poor )[1],5) )
axes.text(x = 2, y = 1.1, s = np.round(stats.ttest_ind(LL_spread_good, LL_spread_poor )[1],5) )

axes.text(x = 0, y = 1, s = np.round(utils.cohend(WN_spread_good, WN_spread_poor),5) )
axes.text(x = 1, y = 1, s = np.round(utils.cohend(CNN_spread_good, CNN_spread_poor),5) )
axes.text(x = 2, y = 1, s = np.round(utils.cohend(LL_spread_good, LL_spread_poor),5) )

plt.savefig( f"spread_cohensd.png", transparent=True, dpi = 300)
plt.savefig( f"spread_cohensd.pdf", transparent=True)
#%% Optional
fig, axes = utils.plot_make()
sns.heatmap(  prob_threshold_moving_avg(tmp1, fsds, skip, threshold = 0.1, smoothing = SMOOTHING)[0][320:460,channel_order_LL].T, center = 0.05, vmin=0., vmax = 0.5, xticklabels=False, yticklabels=False, cbar = False )  
#plt.savefig( f"seizure_probmap_LL1.pdf", transparent=True)
#plt.savefig( f"seizure_probmap_LL1.png", transparent=True, dpi = 300)

fig, axes = utils.plot_make()
sns.heatmap(  prob_threshold_moving_avg(tmp1, fsds, skip, threshold = 0.1, smoothing = SMOOTHING)[1][320:460,channel_order_LL].T, center = 0.05, vmin=0., vmax = 1, xticklabels=False, yticklabels=False, cbar = False )  
#plt.savefig( f"seizure_probmap_LL2.pdf", transparent=True)
#plt.savefig( f"seizure_probmap_LL2.png", transparent=True, dpi = 300)

fig, axes = utils.plot_make()
sns.heatmap(  tmp1[350:460,channel_order_LL].T, center = 0.05, vmin=0., vmax = 0.5, xticklabels=False, yticklabels=False, cbar = False )  
#plt.savefig( f"seizure_probmap_LL3.pdf", transparent=True)
#plt.savefig( f"seizure_probmap_LL3.png", transparent=True, dpi = 300)


#sns.heatmap( lineLengthNorm_train.T , cbar=False)  

#%% Optional

DataJson.plot_eeg(probability_arr_threshold[:,channel_order], fsds, nchan = 50, dpi = 300, fill=True, aspect=200, height=0.05)    
#%% Optional
DataJson.plot_eeg(data_scalerDS, fsds, markers = spread_start_loc, nchan = 5, dpi = 300, aspect=200, height=0.1)    


#DataJson.plot_eeg(data_scalerDS[:,channel_order], fsds, markers = spread_start_loc[channel_order], nchan = nchan, dpi = 300, aspect=200, height=0.05)    


 
DataJson.plot_eeg(data_avgref, fs, markers =  ( (spread_start + seizure_start)  *skipWindow*fs).astype(int)[0:5], nchan = 5, dpi = 300, aspect=20, height=0.5)    


#%% Optional; Visualize


DataJson.plot_eeg(probability_arr_movingAvg, fsds, nchan = 5, dpi = 300, fill=True, aspect=35, height=0.3)    
DataJson.plot_eeg(data, fs, nchan = 5, dpi = 300, aspect=35)    
DataJson.plot_eeg(data_scalerDS, fsds, nchan = 5, dpi = 300, aspect=35)    




    
#%%diffusion model
#SC= SC_hat
sub = RID

ses = basename(glob.glob( join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{SESSION_RESEARCH3T}"))[0])[4:]
sc_paths = glob.glob(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{ses}", "matrices", "*connectogram.txt"))
len(sc_paths)

save_loc = join(paths.SEIZURE_SPREAD_ATLASES_PROBABILITIES, f"sub-{sub}")
with open(paths.ATLAS_FILES_PATH) as f: atlas_metadata_json = json.load(f)
#%%
atlas_index = 5
probability_arr_input = probability_arr_movingAvg

diffusion_model_type=0
time_steps = 3
threshold = 0.05
gradient = 0.1

SC_threshold =  0.4
log_normalize = False

spread_before = 10
spread_after = 20

corrs_atlas, atlas_name = DM.compute_diffusion_model_correlations_for_atlas(sub, paths, atlas_index, secondsBefore, skipWindow, probability_arr_input, channels, SESSION_RESEARCH3T, session, diffusion_model_type, threshold=threshold, time_steps = time_steps, gradient = gradient, spread_before = spread_before, spread_after = spread_after, SC_threshold = SC_threshold, log_normalize = log_normalize, visualize = True, r_to_visualize = 217)

#%%

DM.create_nifti_corrs_brain(corrs_atlas, sub, paths, SESSION_RESEARCH3T, atlas_name)



#%%
atlas_index = 5
probability_arr_input = probability_arr_movingAvg

diffusion_model_type=0
time_steps = 5
threshold = 0.025
gradient = 0.1

SC_threshold =  0.4
log_normalize = True

spread_before = 20
spread_after = 30




#For RID595
atlas_index = 5
probability_arr_input = probability_arr_movingAvg

diffusion_model_type=0
time_steps = 3
threshold = 0.05
gradient = 0.1

SC_threshold =  0.4
log_normalize = False

spread_before = 10
spread_after = 20


#%% finding correlations and saving nifti files


total = len(sc_paths)
for a in range(total):
    #if a>=21:
    #    time_steps = 8
        
    atlas_index = a
    corrs_atlas, atlas_name = DM.compute_diffusion_model_correlations_for_atlas(sub, paths, atlas_index, secondsBefore, skipWindow, probability_arr_input, channels, SESSION_RESEARCH3T, session, diffusion_model_type, threshold=threshold, time_steps = time_steps, gradient = gradient, spread_before = spread_before, spread_after = spread_after, SC_threshold = SC_threshold, log_normalize = log_normalize, visualize = False)
    if a ==0:
        copy_T1 = True
    else:
        copy_T1 = False
    DM.create_nifti_corrs_brain(corrs_atlas, sub, paths, SESSION_RESEARCH3T, atlas_name)
    #time_steps = 4




#%% averaging files



count = 0

for a in range(0, total):
    sc_atlas = sc_paths[a]
    atlas_name = basename(sc_atlas).split(".")[1]
    
    utils.printProgressBar(a, len(sc_paths), suffix=atlas_name)
    
    save_loc_atlas = join(save_loc, f"{atlas_name}.nii.gz")
    if count ==0:
        img_data = utils.load_img_data(save_loc_atlas)
        img_data_all = np.zeros(shape = (img_data.shape[0], img_data.shape[1], img_data.shape[2], total ))
    else:
        img_data = utils.load_img_data(save_loc_atlas)
        
    img_data[np.where(img_data < -0.99)] = np.nan
    img_data_all[:,:,:,a] = img_data
    count = count +1
    
img_data_avg = np.nanmean(img_data_all, axis = 3)

#removing regions outside brain
brain_mask = glob.glob(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{ses}", "atlas_registration", "*desc-preproc_T1w_brain_std.nii.gz"))[0]
img_mask_data = utils.load_img_data(brain_mask)
#utils.show_slices(brain_mask)


img_data_avg[np.where(img_mask_data <= 0)] = np.nan

utils.show_slices(img_data_avg, data_type="data", cmap = "viridis")

img, img_data  = utils.load_img_and_data(save_loc_atlas)
utils.save_nib_img_data(img_data_avg, img,  join(save_loc, "average.nii.gz"))





#%% Making pictures of atlases


atlas_names_all_standard = []
for k in range(len(atlas_metadata_json["STANDARD"].keys())):
    key_item = list(atlas_metadata_json["STANDARD"].keys())[k]
    atlas_metadata_json["STANDARD"][key_item]["name"]
    
    atlas_names_all_standard.append(basename(atlas_metadata_json["STANDARD"][key_item]["name"]).split(".")[0])
    
    
total = len(sc_paths)
for a in range(total):
    sc_atlas = sc_paths[a]
    atlas_name = basename(sc_atlas).split(".")[1]
    utils.printProgressBar(a, len(sc_paths), suffix=atlas_name)
    
    save_loc_atlas = join(save_loc, f"{atlas_name}.nii.gz")
    save_loc_T1 = glob.glob(join(save_loc, f"*T1w_std.nii.gz"))[0]
    #save_loc_T1 = glob.glob(join(save_loc, f"*post_op.nii.gz"))[0]
    save_loc_atlas_figure_path = join(save_loc, "atlas_figures")
    
    
    utils.checkPathAndMake(save_loc_atlas_figure_path,save_loc_atlas_figure_path, printBOOL=False)
    save_loc_atlas_figure = join(save_loc_atlas_figure_path, f"{atlas_name}.nii.gz")
    
    img_data = utils.load_img_data(save_loc_atlas)
    img_data[np.where(img_data < -0.9)] = np.nan
    
    
    T1_data = utils.load_img_data(save_loc_T1)
    T1_data = T1_data/np.nanmax(T1_data)
    T1_data[np.where(T1_data <0.1)] = np.nan
    T1_data[np.where(T1_data < 0.0)] = np.nan
    
    #Calculating averages 
    if a ==0:
        averages_all = np.zeros(shape = (img_data.shape[0], img_data.shape[1], img_data.shape[2], total))
        averages_all[:] = np.nan
        averages_all[:,:,:,a] = img_data
        averages = img_data
    else:
        averages_all[:,:,:,a] = img_data
        averages = np.nanmean(averages_all , axis = 3)
    
    # Slices
    
    #RID0440
    slice_axial = 0.6
    slice_coronal = 0.68
    slice_sag = 0.39
    #RID 365
    slice_axial = 0.32
    slice_coronal = 0.63
    slice_sag = 0.39
    
    
    #RID 595
    slice_axial = 0.32
    slice_coronal = 0.59
    slice_sag = 0.39
    
    slice_atlas =  [img_data[:, :, int((img_data.shape[2]*slice_axial)) ], img_data[:,  int((img_data.shape[2]*slice_coronal)), : ], img_data[ int((img_data.shape[2]*slice_sag)),:,: ]]
    
    slice_average = [averages[:, :, int((averages.shape[2]*slice_axial)) ], averages[:,  int((averages.shape[2]*slice_coronal)), : ], averages[ int((averages.shape[2]*slice_sag)),:,: ]]
   
    slice_T1 = [T1_data[:, :, int((T1_data.shape[2]*slice_axial)) ], T1_data[:,  int((T1_data.shape[2]*slice_coronal)), : ], T1_data[ int((T1_data.shape[2]*slice_sag)),:,: ]]
   
    plt.style.use('dark_background')
    
    #im = plt.imread( join(paths.SEIZURE_SPREAD_ATLASES_PROBABILITIES, "atlas_pictures", f"{atlas_name}.png" )  )
     
    cmap="RdBu_r"
    
    
    fig = plt.figure(constrained_layout=False, dpi=300, figsize=(8, 5))
    gs1 = fig.add_gridspec(nrows=2, ncols=3, left=0, right=0.7, bottom=0, top=1, wspace=0.00, hspace =-0.3)
    gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.7, right=1, bottom=0, top=1, wspace=0.00, hspace = 0.00)
    axes = []
    for r in range(2): #standard
        for c in range(3):
            axes.append(fig.add_subplot(gs1[r, c]))
    axes.append(fig.add_subplot(gs2[0,0]))        
    
    
    vmin_atlas, vmax_atlas = np.min([np.nanmin(sublist) for sublist in slice_atlas]), np.min([np.nanmax(sublist) for sublist in slice_atlas])
    vmin_avg, vmax_avg = np.min([np.nanmin(sublist) for sublist in slice_average]), np.min([np.nanmax(sublist) for sublist in slice_average])
    for i in range(7):
        if i <6:
            if i < 3:
                axes[i].imshow(scipy.ndimage.rotate(slice_T1[i], -90, order = 0)   , cmap="gray", origin="lower")
                axes[i].imshow(scipy.ndimage.rotate(slice_atlas[i], -90, order = 0), cmap=cmap, origin="lower", vmin = vmin_atlas, vmax = vmax_atlas*0.9)
            else:
                axes[i].imshow(scipy.ndimage.rotate(slice_T1[i-3],-90, order = 0), cmap="gray", origin="lower")
                axes[i].imshow(scipy.ndimage.rotate(slice_average[i-3], -90, order = 0), cmap=cmap, origin="lower", vmin = vmin_avg, vmax = vmax_avg*0.9)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].axis("off")
 
    

    im = plt.imread( join(paths.SEIZURE_SPREAD_ATLASES_PROBABILITIES, "atlas_pictures", f"{atlas_name}.png" )  )
    if atlas_name in atlas_names_all_standard:
        ind = np.where(atlas_name == np.array(atlas_names_all_standard) )[0][0]
        atlas_key = list(atlas_metadata_json["STANDARD"].keys())[ind]
        
        atlas_title = atlas_metadata_json["STANDARD"][atlas_key]["shortname"]
    else:
        atlas_size = int(atlas_name[11:18])
        atlas_version = int(atlas_name[20:])
        atlas_title = f"Random Atlas {atlas_size} v{atlas_version}"
    
    atlas_subplot = fig.add_axes([0.58,0.3,0.4,0.4], anchor='NE', zorder=1)
    atlas_subplot.imshow(im)
    atlas_subplot.axis('off')
    atlas_subplot.text(im.shape[0]/2, 0 , f"{atlas_title}" , ha = "center")

    axes[1].text(slice_atlas[1].T.shape[0]/2, slice_atlas[1].T.shape[1]*0.9 , f"{atlas_title}" , ha = "center")
    axes[4].text(slice_atlas[1].T.shape[0]/2, -slice_atlas[1].T.shape[1]*0.1 , "Average" , ha = "center")

    axes[1].text(slice_atlas[1].T.shape[0], slice_atlas[1].T.shape[1]*0.5 , "L" , ha = "right")
    axes[1].text(0, slice_atlas[1].T.shape[1]*0.5 , "R" , ha = "left")
    
    plt.savefig(join(save_loc_atlas_figure_path, f"atlas_SOZ_{a:03}.png"), transparent=False, dpi = 300)
    plt.savefig(join(save_loc_atlas_figure_path, f"atlas_SOZ_{a:03}.pdf"), transparent=False)
    
    plt.show()
    """
    if save:
        if saveFilename == None:
            raise Exception("No file name was given to save figures")
    """
    #plt.show()
    #plt.style.use('default')

#%%
#for beginning plot
a = 0
sc_atlas = sc_paths[a]
atlas_name = basename(sc_atlas).split(".")[1]
utils.printProgressBar(a, len(sc_paths), suffix=atlas_name)

save_loc_atlas = join(save_loc, f"{atlas_name}.nii.gz")
save_loc_T1 = glob.glob(join(save_loc, f"*T1w_std.nii.gz"))[0]
#save_loc_T1 = glob.glob(join(save_loc, f"*post_op.nii.gz"))[0]
save_loc_atlas_figure_path = join(save_loc, "atlas_figures")


utils.checkPathAndMake(save_loc_atlas_figure_path,save_loc_atlas_figure_path, printBOOL=False)
save_loc_atlas_figure = join(save_loc_atlas_figure_path, f"{atlas_name}.nii.gz")

img_data = utils.load_img_data(save_loc_atlas)
img_data[np.where(img_data < -0.9)] = np.nan

T1_data = utils.load_img_data(save_loc_T1)
T1_data = T1_data/np.nanmax(T1_data)
T1_data[np.where(T1_data <0.1)] = np.nan
#Calculating averages 
if a ==0:
    averages_all = np.zeros(shape = (img_data.shape[0], img_data.shape[1], img_data.shape[2], total))
    averages_all[:] = np.nan
    averages_all[:,:,:,a] = img_data
    averages = img_data
else:
    averages_all[:,:,:,a] = img_data
    averages = np.nanmean(averages_all , axis = 3)

# Slices

#RID0440
slice_axial = 0.6
slice_coronal = 0.68
slice_sag = 0.39
#RID 365
slice_axial = 0.32
slice_coronal = 0.63
slice_sag = 0.39


#RID 595
slice_axial = 0.32
slice_coronal = 0.59
slice_sag = 0.39

slice_atlas =  [img_data[:, :, int((img_data.shape[2]*slice_axial)) ], img_data[:,  int((img_data.shape[2]*slice_coronal)), : ], img_data[ int((img_data.shape[2]*slice_sag)),:,: ]]

slice_average = [averages[:, :, int((averages.shape[2]*slice_axial)) ], averages[:,  int((averages.shape[2]*slice_coronal)), : ], averages[ int((averages.shape[2]*slice_sag)),:,: ]]

slice_T1 = [T1_data[:, :, int((T1_data.shape[2]*slice_axial)) ], T1_data[:,  int((T1_data.shape[2]*slice_coronal)), : ], T1_data[ int((T1_data.shape[2]*slice_sag)),:,: ]]

plt.style.use('dark_background')   

vmin_atlas, vmax_atlas = np.min([np.nanmin(sublist) for sublist in slice_atlas]), np.min([np.nanmax(sublist) for sublist in slice_atlas])
vmin_avg, vmax_avg = np.min([np.nanmin(sublist) for sublist in slice_average]), np.min([np.nanmax(sublist) for sublist in slice_average])
fig = plt.figure(constrained_layout=False, dpi=300, figsize=(8, 5))
gs1 = fig.add_gridspec(nrows=2, ncols=3, left=0, right=0.7, bottom=0, top=1, wspace=0.00, hspace =-0.3)
gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.7, right=1, bottom=0, top=1, wspace=0.00, hspace = 0.00)
axes = []
for r in range(2): #standard
    for c in range(3):
        axes.append(fig.add_subplot(gs1[r, c]))
axes.append(fig.add_subplot(gs2[0,0]))        

for i in range(7):
    if i <6:
        if i < 3:
            axes[i].imshow(scipy.ndimage.rotate(slice_T1[i], -90, order = 0)   , cmap="gray", origin="lower")
            #axes[i].imshow(scipy.ndimage.rotate(slice_atlas[i], -90, order = 0), cmap=cmap, origin="lower", vmin = vmin_atlas, vmax = vmax_atlas*0.9)
        else:
            axes[i].imshow(scipy.ndimage.rotate(slice_T1[i-3],-90, order = 0), cmap="gray", origin="lower")
            #axes[i].imshow(scipy.ndimage.rotate(slice_average[i-3], -90, order = 0), cmap=cmap, origin="lower", vmin = vmin_avg, vmax = vmax_avg*0.9)
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].axis("off")



im = plt.imread( join(paths.SEIZURE_SPREAD_ATLASES_PROBABILITIES, "atlas_pictures", f"{atlas_name}.png" )  )
if atlas_name in atlas_names_all_standard:
    ind = np.where(atlas_name == np.array(atlas_names_all_standard) )[0][0]
    atlas_key = list(atlas_metadata_json["STANDARD"].keys())[ind]
    
    atlas_title = atlas_metadata_json["STANDARD"][atlas_key]["shortname"]
else:
    atlas_size = int(atlas_name[11:18])
    atlas_version = int(atlas_name[20:])
    atlas_title = f"Random Atlas {atlas_size} v{atlas_version}"

atlas_subplot = fig.add_axes([0.58,0.3,0.4,0.4], anchor='NE', zorder=1)
atlas_subplot.imshow(im)
atlas_subplot.axis('off')
atlas_subplot.text(im.shape[0]/2, 0 , f"{atlas_title}" , ha = "center")

axes[1].text(slice_atlas[1].T.shape[0]/2, slice_atlas[1].T.shape[1]*0.9 , f"{atlas_title}" , ha = "center")
axes[4].text(slice_atlas[1].T.shape[0]/2, -slice_atlas[1].T.shape[1]*0.1 , "Average" , ha = "center")

axes[1].text(slice_atlas[1].T.shape[0], slice_atlas[1].T.shape[1]*0.5 , "L" , ha = "right")
axes[1].text(0, slice_atlas[1].T.shape[1]*0.5 , "R" , ha = "left")

plt.savefig(join(save_loc_atlas_figure_path, f"atlas_SOZ_blank4.png"), transparent=False, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"atlas_SOZ_blank4.pdf"), transparent=False)

plt.show()    

#%%
#Plot RID0595 Data visualization
DataJson.plot_eeg(data_scalerDSDS[(secondsBefore-5)*16:-(secondsAfter-10)*16,channel_order], 16, markers = [5*16], dpi = 300, aspect=aspect*3, height=0.1, color = "black", lw= 2, hspace = -0.8)  
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_eeg1.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_eeg1.pdf"), transparent=True)

seizurePattern.plot_probheatmaps(probLSTM[1800:2600,channel_order], fsds, skip, threshold=0.9, vmin = 0.5, vmax = 1, center=None, smoothing = 20)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm1.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm1.pdf"), transparent=True)

fig, axes = utils.plot_make()
sns.heatmap( probLSTM[1860:2460,:].T , cbar=False , center = 0.5, vmin=0.5, xticklabels=False, yticklabels=False, ax = axes)   
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm2.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm2.pdf"), transparent=True)

fig, axes = utils.plot_make()   
sns.heatmap( probLSTM[1860:2460,channel_order].T , cbar=False , center = 0.5, vmin=0.5, xticklabels=False, yticklabels=False) 
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm3.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm3.pdf"), transparent=True)

fig, axes = utils.plot_make()        
sns.heatmap( probability_arr_movingAvg[1800:2260,channel_order].T , cbar=False, center = 0.5, vmin=0.5 ,xticklabels=False, yticklabels=False)      
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm4.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm4.pdf"), transparent=True)

fig, axes = utils.plot_make()        
sns.heatmap( probability_arr_threshold[1800:2260,channel_order].T , cbar=False, center = 0.5, vmin=0.5 ,xticklabels=False, yticklabels=False)    
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm5.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"seizure_probmap_lstm5.pdf"), transparent=True)

#%% RID0595 Atlas vis
a=5
sc_atlas = sc_paths[a]
atlas_name = basename(sc_atlas).split(".")[1]
print(f"\n\n\n{atlas_name}\n\n\n")

SC = utils.read_DSI_studio_Txt_files_SC(sc_atlas)
SC_regions = utils.read_DSI_studio_Txt_files_SC_return_regions(sc_atlas, atlas_name)

dm_data = DM.diffusionModels(SC, SC_regions)  
SC_hat = dm_data.preprocess_SC(SC, SC_threshold, log_normalize = log_normalize)    


time_steps = 6
node_state = dm_data.LTM(SC_hat, seed = 215, time_steps = time_steps, threshold = 0.025)


diffusion_atlas = []
atlas_label_path = glob.glob( join(paths.ATLAS_LABELS, f"{atlas_name}.csv"))
if len(atlas_label_path) > 0:
    atlas_label = atlas_label_path[0]
    atlas_label = pd.read_csv(atlas_label, header=1)
else:
    atlas_label = pd.DataFrame(  dict( region= SC_regions.astype(int), label =  SC_regions))



    
atlas_path = glob.glob(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{ses}", "atlas_registration", f"*{atlas_name}.nii.gz"))[0]
img, img_data = utils.load_img_and_data(atlas_path)
img_data_timestep = copy.deepcopy(img_data)
img_data_timestep[np.where(img_data_timestep > 0 )] = 1
img_data_timestep[np.where(img_data_timestep == 0 )] = np.nan
img_data_timestep[np.where(img_data_timestep == 1 )] = 0

#loop thru regions and replace regions with activation
img_data_timestep_all = np.zeros(shape =(img_data.shape[0],img_data.shape[1],img_data.shape[2],time_steps))


for t in range(time_steps):
    utils.printProgressBar(t+1,time_steps)
    ind = np.where(node_state[t,:] == 1 )[0]
    for r in range(len(ind)):
        region = ind[r]
        
        label = atlas_label["region"][region]
        img_data_timestep[np.where(img_data ==label )] = 1
    img_data_timestep_all[:,:,:,t] = img_data_timestep


slice_start = 113


save_loc_T1 = glob.glob(join(save_loc, f"*T1w_std.nii.gz"))[0]
T1_data = utils.load_img_data(save_loc_T1)
T1_data = T1_data/np.nanmax(T1_data)
T1_data[np.where(T1_data <0.1)] = np.nan
T1_data[:,  slice_start, : ]
np.nanmax(T1_data)
#Plotting
save_loc_atlas_figure_path = join(save_loc, "atlas_figures")
plt.style.use('dark_background')   
fig, axes = utils.plot_make(r= 2, c = 3)
axes = axes.flatten()

for t in range(time_steps):
    axes[t].imshow(scipy.ndimage.rotate(T1_data[:,  slice_start, : ],-90, order = 0),  cmap = "gray", origin = "lower")
    if t > 0:
        axes[t].imshow(scipy.ndimage.rotate(img_data_timestep_all[:,slice_start,:, t-1].T, 0, order = 0),  cmap = "Reds", origin = "lower", vmin = 0, vmax = 1)
    axes[t].set_xticklabels([])
    axes[t].set_yticklabels([])
    axes[t].set_xticks([])
    axes[t].set_yticks([])
    axes[t].axis("off")
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}.png"), transparent=False, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}.pdf"), transparent=False)

#%

node_state_spread_start = np.zeros(len(node_state.T))
for n in range(len(node_state.T)):
    where_start = np.where(node_state[:,n]==1)[0]
    if len(where_start) > 0:
        node_state_spread_start[n] = where_start[0]
    else:
        node_state_spread_start[n] = len(node_state)

np.argsort(node_state_spread_start)

sns.heatmap(node_state[0:5,np.argsort(node_state_spread_start)].T)

plt.style.use('default') 
fig, axes = utils.plot_make(r= 1, c = 1, size_length = 6)
sns.heatmap(node_state[0:5,:].T, cbar = False, xticklabels=False, yticklabels=False, ax = axes)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}_heatmap.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}_heatmap.pdf"), transparent=True)
#%%
plt.style.use('default') 
roi = 215
corrs_atlas, atlas_name = DM.compute_diffusion_model_correlations_for_atlas(sub, paths, atlas_index, secondsBefore, skipWindow, probability_arr_input, channels, SESSION_RESEARCH3T, session, diffusion_model_type, threshold=threshold, time_steps = 5, gradient = gradient, spread_before = spread_before, spread_after = spread_after, SC_threshold = SC_threshold, log_normalize = log_normalize, visualize = True, r_to_visualize = roi)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}_correlation_roi_{roi}.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}_correlation_roi_{roi}.pdf"), transparent=True)

roi=45
corrs_atlas, atlas_name = DM.compute_diffusion_model_correlations_for_atlas(sub, paths, atlas_index, secondsBefore, skipWindow, probability_arr_input, channels, SESSION_RESEARCH3T, session, diffusion_model_type, threshold=threshold, time_steps = 5, gradient = gradient, spread_before = spread_before, spread_after = spread_after, SC_threshold = SC_threshold, log_normalize = log_normalize, visualize = True, r_to_visualize = roi)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}_correlation_roi_{roi}.png"), transparent=True, dpi = 300)
plt.savefig(join(save_loc_atlas_figure_path, f"diffusion_model_{atlas_name}_correlation_roi_{roi}.pdf"), transparent=True)
#%%

"""
Nnull = 2
corrs_null = np.zeros(shape = (Nnull, len(corrs)))
corrs_null_order = corrs_null[:,ind]
#corrs_order_null_df = pd.DataFrame(corrs_null_order).melt()
for nu in range(Nnull):
    print(f"\r{np.round((nu+1)/Nnull*100,2)}", end = "\r")
    SC_null = bct.randmio_und(SC, 30)[0]
    dm_data = DM.diffusionModels(SC, SC_regions, spread, spread_regions, electrodeLocalization) 
    SC_hat = dm_data.preprocess_SC(SC_null, 0.4)  
    corrs_null[nu,:] = get_LTM(SC_hat)

"""
#%%
"""
sfc_datapath = "/media/arevell/sharedSSD/linux/papers/brain_atlases/data/data_processed/aggregated_data"
iEEG_filename =   "HUP138_phaseII"

start_time_usec = 416023190000
stop_time_usec = 416112890000 
fname = os.path.join(sfc_datapath,   f"sub-{RID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )
fname = os.path.join(sfc_datapath,   f"sub-RID0320_HUP140_phaseII_D02_331979921071_332335094986_data.pickle" )


os.path.exists(fname)

if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = dataclass_SFC.dataclass_sfc( **all_data  ) 


atlas = sfc_data.get_atlas_names()[1]   
SC, SC_regions = sfc_data.get_structure(atlas)
electrodeLocalization = sfc_data.get_electrodeLocalization(atlas)
"""


#SC= SC_distInv
"""
tmp_sc = "/media/arevell/data/linux/data/BIDS/derivatives/structural_connectivity/structural_matrices/sub-RID0278/ses-research3Tv01/matrices/sub-RID0278.AAL2.count.pass.connectogram.txt"
tmp_sc = "/media/arevell/data/linux/data/BIDS/derivatives/structural_connectivity/structural_matrices/sub-RID0278/ses-research3Tv01/matrices/sub-RID0278.BN_Atlas_246_1mm.count.pass.connectogram.txt"
SC2 = utils.read_DSI_studio_Txt_files_SC(tmp_sc)

sns.heatmap(SC, square=True)
sns.heatmap(SC2, square=True)


pearsonr(utils.getUpperTriangle(SC), utils.getUpperTriangle(SC2))

#SC = utils.log_normalize_adj(SC)
#SC2= utils.log_normalize_adj(SC2)
sns.scatterplot(x =  utils.getUpperTriangle(SC),y =  utils.getUpperTriangle(SC2))

SC2= utils.read_DSI_studio_Txt_files_SC(tmp_sc)
#null
#SC_null = bct.randmio_und(SC, 30)[0]

#sns.heatmap(SC, square=True)   
#sns.heatmap(SC_null, square=True)   

#SC = SC_null
"""

#sns.heatmap(node_state_mod_resample.T, cbar=False) 

#sns.heatmap(spreadMod_state.T, cbar=False) 

#node_state2 = signal.resample(node_state, spreadMod_state.shape[0])
#sns.heatmap(node_state2.T, cbar=False)   
#node_state



#%%build line length


data_scalerDS_X.shape

vector = data_scalerDS_X[0,:,0]
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def lineLength(vector):
    l = len(vector)
    length = np.zeros(shape = l)
    x = np.array(range(l))
    for v in range(l-1):
        length[v] = distance( (x[v], vector[v]) , (x[v+1], vector[v+1])  )
    lineLen = np.sum(length)
    return lineLen

ll = lineLength(vector)

lineLength_arr = np.zeros(shape = (windows, nchan))  
for c in range(nchan):
    print(c)
    for win in range(windows):
        lineLength_arr[win, c] = lineLength(data_scalerDS_X[win, :, c])

#normalize LL
lineLength_arr
for c in range(nchan):
    lineLength_arr[:,c] = lineLength_arr[:,c]/np.max(lineLength_arr[:,c])

#%%
fig, ax = plt.subplots(3,2, figsize=(10,10), dpi =300)

sns.lineplot( x = range(windows),  y= lineLength_arr[:,0],    ci=None, ax = ax[0,0])    
sns.lineplot( x = range(windows),  y= lineLength_arr[:,2],    ci=None, ax = ax[0,1])    
    
sns.heatmap( lineLength_arr.T , ax = ax[1,0])    

THRESHOLD_LL = 0.7
SMOOTHING = 20 #in seconds
    
lineLength_arr_threshold = copy.deepcopy(lineLength_arr)
lineLength_arr_threshold[lineLength_arr_threshold>THRESHOLD_LL] = 1
lineLength_arr_threshold[lineLength_arr_threshold<=THRESHOLD_LL] = 0


sns.heatmap( lineLength_arr_threshold.T,  ax = ax[1,1] )    



w = int(SMOOTHING*fsds/skip)
lineLength_arr_movingAvg = np.zeros(shape = (windows - w + 1, nchan))

for c in range(nchan):
    lineLength_arr_movingAvg[:,c] =  echobase.movingaverage(lineLength_arr[:,c], w)
    
    
sns.heatmap( lineLength_arr_movingAvg.T , ax = ax[2,0] )      
    
    
lineLength_arr_movingAvg_threshold = copy.deepcopy(lineLength_arr_movingAvg)

lineLength_arr_movingAvg_threshold[lineLength_arr_movingAvg_threshold>THRESHOLD_LL] = 1
lineLength_arr_movingAvg_threshold[lineLength_arr_movingAvg_threshold<=THRESHOLD_LL] = 0


sns.heatmap( lineLength_arr_movingAvg_threshold.T,  ax = ax[2,1] )      
    
    
#sns.heatmap(SC, square=True)   
#sns.heatmap(SC_hat, square=True)   
    
    
#%% effect of distance


centroidsPath = os.path.join(path, "data", "raw","atlases", "atlasCentroids", "AAL2_centroid.csv")
    
    
centroids = pd.read_csv(centroidsPath)       
    
    
#get distance matrix
SC_dist = np.zeros((len(centroids),len(centroids)  ))
SC_distInv = np.zeros((len(centroids),len(centroids)  ))
for i in range(len(centroids)):
    print(i)
    for j in range(len(centroids)):
        p1 = np.array([centroids.iloc[i].x,centroids.iloc[i].y,centroids.iloc[i].z])
        p2 = np.array([centroids.iloc[j].x,centroids.iloc[j].y,centroids.iloc[j].z])

        SC_dist[i,j] = np.sqrt(np.sum((p1-p2)**2, axis=0))
        if i!=j:
            SC_distInv[i,j] = 1/(np.sqrt(np.sum((p1-p2)**2, axis=0)))
    
    
sns.heatmap(SC_dist, square=True)    
sns.heatmap(SC_distInv, square=True)    
    
    
spearmanr(SC_distInv.flatten(),SC_hat.flatten() )  [0] 
    
    
    