#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:05:45 2021

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
from dataclasses import dataclass
from os.path import join
from matplotlib import pyplot
from  matplotlib import colors
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from scipy import signal
import bct

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

#import custom
#import custom
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.seizureSpread import echomodel, seizurePattern
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.dataClass import DataClassSfc, DataClassJson
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#%% Paths
fnameiEEGusernamePassword = join("/media","arevell","sharedSSD","linux", "ieegorg.json")
fnameJSON = join("/media","arevell","sharedSSD","linux", "data", "metadata", "iEEGdataRevell.json")
BIDS = join("/media","arevell","sharedSSD","linux", "data", "BIDS")
deepLearningModelsPath = "/media/arevell/sharedSSD/linux/data/deepLearningModels/seizureSpread"
pathFigure = join("/media", "arevell", "sharedSSD", "linux", "figures")
dataset = "PIER"
session = "implant01"

#opening files
with open(fnameiEEGusernamePassword) as f: usernameAndpassword = json.load(f)
with open(fnameJSON) as f: jsonFile = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]



#%% Model parameters
fsds = 128 #"sampling frequency, down sampled"
window = 10 #window of eeg for training/testing. In seconds
skipWindow = 0.25 #Next window is skipped over in seconds 
time_step, skip = int(window*fsds), int(skipWindow*fsds)
verbose = 1
secondsBeforeSpread = 60
secondsAfterSpread = 60
montage = "bipolar"
prewhiten = True

#plotting parameters
aspect = 50

#%% Get files and relevant patient information to train model

pathFigureOverview = join(pathFigure, "overview")

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = DataClassJson.DataClassJson(jsonFile)

version = 0
fpath_wavenet = join(deepLearningModelsPath,f"wavenet/v{version:03d}.hdf5") #version 7 of wavenet is good
fpath_1dCNN = join(deepLearningModelsPath,f"1dCNN/v{version:03d}.hdf5") #version 7 of wavenet is good
fpath_lstm = join(deepLearningModelsPath,f"lstm/v{version:03d}.hdf5") #version 7 of wavenet is good
modelWN = load_model(fpath_wavenet)
modelCNN= load_model(fpath_1dCNN)
modelLSTM = load_model(fpath_lstm)

#%%Meaure seizure spread


#%Get data


patientsWithseizures = DataJson.get_patientsWithSeizuresAndInterictal()

#%% generate example EEG

i = 42
sub = np.array(patientsWithseizures["subject"])[i]
idKey = np.array(patientsWithseizures["idKey"])[i]
AssociatedInterictal = np.array(patientsWithseizures["AssociatedInterictal"])[i]

df, fs, ictalStartIndex, ictalStopIndex = DataJson.get_precitalIctalPostictal(sub, "Ictal", idKey, username, password, BIDS = BIDS, dataset= dataset, session = session, secondsBefore = secondsBeforeSpread, secondsAfter = secondsAfterSpread)
df_interictal, _ = DataJson.get_iEEGData(sub, "Interictal", AssociatedInterictal, username, password, BIDS = BIDS, dataset= dataset, session = session, startKey = "Start")
print("\n\n\nPreprocessing\n\n\n")
dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels = DataJson.preprocessNormalizeDownsample(df, df_interictal, fs, fsds, montage = montage, prewhiten = prewhiten)
data, data_ref, _, _, channels = echobase.preprocess(df, fs, fsds, montage = montage, prewhiten = prewhiten)



data_plotFilt = echobase.elliptic_bandFilter(data_ref, int(fs))[0]

nsamp, nchan = df.shape

#c = [62,61,60,59,58,53,52,51,43,42,36,32,31,27,21,17,7,6,5]#car
c = [47,45,64, 63,62,59,40,33,27,26, 18, 17,16,6,5] #bipolar
#c.reverse()
c_copy = copy.deepcopy(c)
fs_plot = copy.deepcopy(fs)

data_scalerDSPlot = data_scalerDS[:,c]

data_scalerDSPlot_X = echomodel.overlapping_windows(data_scalerDSPlot, time_step, skip)
windows, _, _ = data_scalerDSPlot_X.shape

probWN = np.zeros(shape = (windows, len(c)))  
probCNN = np.zeros(shape = (windows, len(c)))  
probLSTM = np.zeros(shape = (windows, len(c)))  
    

for ch in range(len(c)):
    print(ch)
    ch_pred =     data_scalerDSPlot_X[:,:,ch].reshape(windows, time_step, 1    )
    probWN[:,ch] =  modelWN.predict(ch_pred, verbose=1)[:,1]
    probCNN[:,ch] =  modelCNN.predict(ch_pred, verbose=1)[:,1]
    probLSTM[:,ch] =  modelLSTM.predict(ch_pred, verbose=1)[:,1]
        

    
smoothing = 20
w = int(smoothing*fsds/skip)
mvgAvg = np.zeros(shape = (windows - w + 1, len(c)))
for ch in range(len(c)):
    mvgAvg[:,ch] =  echobase.movingaverage(probWN[:,ch], w)  
    
    
#Calculate Raw Features

#linelength, linelengthNorm = echobase.lineLengthOfArray(data_scalerDSPlot_X)



#%% generate example EEG
fnameFig = join(pathFigureOverview, "exampleEEGspread.png")
DataJson.plot_eeg(data_scaler[int(ictalStartIndex-1*fs_plot):int(ictalStopIndex+2*fs_plot),c], fs_plot, nchan = None, dpi = 300, aspect=30, height=0.5, hspace=-0.87, lw = 0.8, savefig = False, pathFig = fnameFig)  


#%making pattern


fnameFigChannels = join(pathFigureOverview, "exampleProbabilityChannels.png")
seizurePattern.plot_eegGradient(mvgAvg, fsds,  startInd = 0, stopInd = 500, lw=4, fill = True, savefig = False, pathFig = fnameFigChannels,hspace= -0.5 , height=0.5,   )



#%%Heatmap of probabilities

seizurePattern.plot_heatmapSingle(probWN[0:600,:], fsds, skip, figsize = (5,2.5), threshold = 0.8, smoothing = 10, vmin = 0.3, vmax = 1.1)
fnameFig = join(pathFigureOverview, "seizurePatternHeatmap.png")
#plt.savefig( fnameFig, transparent=True)

seizurePattern.plot_heatmapSingleThreshold(probWN[0:600,:], fsds, skip, figsize = (5,2.5), threshold = 0.8, smoothing = 10, vmax = 1.15)
fnameFig = join(pathFigureOverview, "seizurePatternHeatmapThreshold.png")
#plt.savefig( fnameFig, transparent=True)


#%% Heatmap of Line length



#seizurePattern.plot_heatmapSingle(linelength[200:500,:], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = 0, vmax = 10000)




#%% heatmap of power














#%% heatmap  node degree



















#%% gnerate example single channel EEG
i = 82
sub = np.array(patientsWithseizures["subject"])[i]
idKey = np.array(patientsWithseizures["idKey"])[i]
AssociatedInterictal = np.array(patientsWithseizures["AssociatedInterictal"])[i]

df, fs, ictalStartIndex, ictalStopIndex = DataJson.get_precitalIctalPostictal(sub, "Ictal", idKey, username, password,BIDS = BIDS, dataset= dataset, session = session, secondsBefore = secondsBeforeSpread, secondsAfter = secondsAfterSpread)
df_interictal, _ = DataJson.get_iEEGData(sub, "Interictal", AssociatedInterictal, username, password, BIDS = BIDS, dataset= dataset, session = session, startKey = "Start")
print("\n\n\nPreprocessing\n\n\n")
dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels = DataJson.preprocessNormalizeDownsample(df, df_interictal, fs, fsds, montage = montage, prewhiten = prewhiten)
data, data_avgref, _, _, channels = echobase.preprocess(df, fs, fsds, montage = montage, prewhiten = prewhiten)
dataII, data_avgrefII, _, _, channels = echobase.preprocess(df_interictal, fs, fsds, montage = montage, prewhiten = prewhiten)

data_plotFilt = echobase.elliptic_bandFilter(data_avgref, int(fs))[0]
data_plotFiltII = echobase.elliptic_bandFilter(data_avgrefII, int(fs))[0]

fs_plot = 32
data_plotFiltDS = DataJson.downsample(data_plotFilt, fs, fs_plot)
data_plotFiltDSII = DataJson.downsample(data_plotFiltII, fs, fs_plot)
data_plotFiltDS2_X = echomodel.overlapping_windows(DataJson.downsample(data_plotFilt, fs, fsds), time_step, skip)
data_plotFiltDSII_X = echomodel.overlapping_windows(DataJson.downsample(data_plotFiltII, fs, fsds), time_step, skip)
data_scalerDS2 = DataJson.downsample(data_scalerDS, fsds, fs_plot)
data_scalerIIDS2 = DataJson.downsample(dataII_scalerDS, fsds, fs_plot)
c = [3]
c2 = [0,1,2,3,4,5]
c3 = np.array(range(50)).tolist()
data_scalerDSPlot = data_scalerDS[:,c2]

data_scalerDSPlot_X = echomodel.overlapping_windows(data_scalerDSPlot, time_step, skip)
windows, _, _ = data_scalerDSPlot_X.shape

probWN = np.zeros(shape = (windows, len(c2)))  
probCNN = np.zeros(shape = (windows, len(c2)))  
probLSTM = np.zeros(shape = (windows, len(c2)))  
     

#for onset
data_scalerDS_X = echomodel.overlapping_windows(data_scalerDS, time_step, skip)
windows, _, nchan = data_scalerDS_X.shape

probWN = np.zeros(shape = (windows, nchan)  )
probCNN = np.zeros(shape = (windows, nchan))  
probLSTM = np.zeros(shape = (windows, nchan))  
    

for ch in range(nchan):
    print(ch)
    ch_pred =     data_scalerDS_X[:,:,ch].reshape(windows, time_step, 1    )
    probWN[:,ch] =  modelWN.predict(ch_pred, verbose=1)[:,1]
    probCNN[:,ch] =  modelCNN.predict(ch_pred, verbose=1)[:,1]
    probLSTM[:,ch] =  modelLSTM.predict(ch_pred, verbose=1)[:,1]
        
data_plotFiltPIII= np.concatenate([data_plotFiltDS, data_plotFiltDSII], axis = 0)
    
    
# for raw features

#line length
#linelength, linelengthNorm = echobase.lineLengthOfArray(data_scalerDS_X)
linelengthFilt, linelengthNormFilt = echobase.lineLengthOfArray(data_plotFiltDS2_X)



#power


power = signal.welch(data_scalerDS_X[:,:,:], fsds, axis=1)[1]
powerSum = np.sum(power, axis = 1)
powerNorm = powerSum / powerSum.max(axis=0)


#pearson correlations

pearson = np.zeros(shape = (nchan, nchan, windows))
for win in range(windows):
    print(win)
    pearson[:,:,win] = echobase.pearson_connectivity(data_scalerDS_X[win,:,:], fsds)[0]

sns.heatmap(pearson[:,:,100], square=True  , vmin = -0.4, vmax = 1)


threshPos = 0.4
threshNeg = -0.2
pearsonBinary = copy.deepcopy(pearson)
pearsonBinary[ np.abs(pearsonBinary) < threshPos] = 0
pearsonBinaryPos = copy.deepcopy(pearson)
pearsonBinaryPos[ pearsonBinaryPos < threshPos] = 0
pearsonBinaryNeg = copy.deepcopy(pearson)
pearsonBinaryNeg[ pearsonBinaryNeg > threshNeg] = 0



degree = bct.degrees_und(pearsonBinary[:,:,:]).T
degreePos = bct.degrees_und(pearsonBinaryPos[:,:,:]).T
degreeNeg = bct.degrees_und(pearsonBinaryNeg[:,:,:]).T

strength =  bct.strengths_und(np.abs( pearson[:,:,:])).T

strengthPos = np.zeros(shape = (windows, nchan))
strengthNeg = np.zeros(shape = (windows, nchan))
for win in range(windows):
    strengthPos[win, :], strengthNeg[win, :] , _, _= bct.strengths_und_sign( pearson[:,:,win])


strengthNegAbs = np.abs(strengthNeg)

#normalize
degreeNorm = degree / degree.max(axis=0)
degreePosNorm = degreePos / degreePos.max(axis=0)
degreeNegNorm = degreeNeg / degreeNeg.max(axis=0)
strengthNorm = strength / strength.max(axis=0)
strengthPosNorm = strengthPos / strengthPos.max(axis=0)
strengthNegNorm = strengthNegAbs / strengthNegAbs.max(axis=0)

#%% geerate example single channel EEG
fnameFig = join(pathFigureOverview, "exampleEEGspread_single_channel.png")
DataJson.plot_eeg(data_plotFiltDS[(int(secondsBeforeSpread*fs_plot- 1*fs_plot)):(int((secondsBeforeSpread+89)*fs_plot- 85*fs_plot)),c], fs_plot, nchan = None, dpi = 300, aspect=3, height=5, hspace=-0, lw = 13, savefig = False, pathFig = fnameFig)  
fnameFig = join(pathFigureOverview, "exampleEEGspread_single_channel2.png")
DataJson.plot_eeg(data_plotFiltDS[(int(secondsBeforeSpread*fs_plot- 10*fs_plot)):(int((secondsBeforeSpread+89)*fs_plot- 71*fs_plot)),c], fs_plot, nchan = None, dpi = 300, aspect=3, height=5, hspace=-0, lw = 7, savefig = False, pathFig = fnameFig)  


fnameFig = join(pathFigureOverview, "exampleEEG_periIctal.png")
DataJson.plot_eeg(data_plotFiltDS[(int(secondsBeforeSpread*fs_plot- 15*fs_plot)):(int((secondsBeforeSpread+89)*fs_plot- 55*fs_plot)),c], fs_plot, nchan = None, dpi = 300, aspect=3, height=5, hspace=-0, lw = 5, savefig = False, pathFig = fnameFig)  
fnameFig = join(pathFigureOverview, "exampleEEG_interictal.png")
DataJson.plot_eeg(data_plotFiltDSII[(int(secondsBeforeSpread*fs_plot- 30*fs_plot)):(int((secondsBeforeSpread+89)*fs_plot- 71*fs_plot)),c], fs_plot, nchan = None, dpi = 300, aspect=3, height=5, hspace=-0, lw = 5, savefig = False, pathFig = fnameFig)  


fnameFig = join(pathFigureOverview,"Class0Class1", "class0_01.png")
interictal01 = data_scalerIIDS2[fs_plot*40:fs_plot*55,10].reshape(-1,1)
ictal01 = data_scalerDS2[int(secondsBeforeSpread*fs_plot+ 45*fs_plot):(int((secondsBeforeSpread+60)*fs_plot)),10].reshape(-1,1)
unknown01 = data_scalerDS2[int(secondsBeforeSpread*fs_plot+ 25*fs_plot):(int((secondsBeforeSpread+40)*fs_plot)),10].reshape(-1,1)

interictal02 = data_scalerIIDS2[fs_plot*0:fs_plot*15,1].reshape(-1,1)
ictal02 = data_scalerDS2[int(secondsBeforeSpread*fs_plot+ 20*fs_plot):(int((secondsBeforeSpread+35)*fs_plot)),1].reshape(-1,1)
unknown02 = data_scalerDS2[int(secondsBeforeSpread*fs_plot+ 2*fs_plot):(int((secondsBeforeSpread+17)*fs_plot)),1].reshape(-1,1)

interictal03 = data_scalerIIDS2[fs_plot*0:fs_plot*15,30].reshape(-1,1)
ictal03 = data_scalerDS2[int(secondsBeforeSpread*fs_plot+ 55*fs_plot):(int((secondsBeforeSpread+70)*fs_plot)),30].reshape(-1,1)
unknown03 = data_scalerDS2[int(secondsBeforeSpread*fs_plot+ 40*fs_plot):(int((secondsBeforeSpread+55)*fs_plot)),30].reshape(-1,1)

examplesPlot = np.concatenate([ interictal01, interictal02,interictal03, ictal01, ictal02  , ictal03,unknown01, unknown02, unknown03 ], axis = 1)

DataJson.plot_eeg(examplesPlot[fs_plot*0:fs_plot*5,:], fs_plot, nchan = None, dpi = 300, aspect=7, height=2, hspace=-0.4, lw = 7, savefig = False, pathFig = fnameFig)  

#%%generate example multiple channels EEG
fnameFig = join(pathFigureOverview, "exampleEEGspread_multipleChannels.png")
DataJson.plot_eeg(data_scalerDS2[(int(secondsBeforeSpread*fs_plot- 0*fs_plot)):(int((secondsBeforeSpread+60)*fs_plot+ 0*fs_plot)),:], fs_plot, nchan = None, 
                  dpi = 300, aspect=90, height=0.1, hspace=-0.9, lw = 1, savefig = False, pathFig = fnameFig)  




#%%making single probability plot

seizurePattern.plot_singleProbability(probWN, fsds, skip, channel=3, startInd = 0, stopInd = None, smoothing = 20, vmin = -1.0, vmax = 1.5)

fnameFig = join(pathFigureOverview, "exampleProbability.png")
#plt.savefig( fnameFig, transparent=True)

#%% Heatmap Power
seizurePattern.plot_heatmapSingle(powerNorm[:,:], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = -0.0, vmax = 1.0, cmapName="Greens")
fnameFig = join(pathFigureOverview, "power.png")
#plt.savefig( fnameFig, transparent=True)



#%% Heatmap Line length 
seizurePattern.plot_heatmapSingle(linelengthNormFilt[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = -0, vmax = 1.0, cmapName="Purples")
fnameFig = join(pathFigureOverview, "linelength.png")
#plt.savefig( fnameFig, transparent=True)



#%% Heatmap Degree
seizurePattern.plot_heatmapSingle(degreeNorm[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 10, vmin = 0.0, vmax = 1, cmapName="Blues")
seizurePattern.plot_heatmapSingle(degreePosNorm[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = 0.0, vmax = 1, cmapName="Blues")
seizurePattern.plot_heatmapSingle(degreeNegNorm[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = 0.0, vmax = 1, cmapName="Blues")
seizurePattern.plot_heatmapSingle(strengthNorm[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = 0.0, vmax = 1, cmapName="Blues")
seizurePattern.plot_heatmapSingle(strengthPosNorm[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = 0.0, vmax = 1, cmapName="Blues")
seizurePattern.plot_heatmapSingle(strengthNegNorm[220:575,c3], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 1, vmin = 0, vmax = 1, cmapName="Blues")


fnameFig = join(pathFigureOverview, "strengthNeg.png")
plt.savefig( fnameFig, transparent=True)







#%%proabilities, threshold, and onset

probabilityArray = probWN

smoothing = 20
threshold = 0.5
windows, nchan = probabilityArray.shape
thresholded = copy.deepcopy(probabilityArray)
thresholded[thresholded>threshold] = 1; thresholded[thresholded<=threshold] = 0
w = int(smoothing*fsds/skip)
mvgAvg = np.zeros(shape = (windows - w + 1, nchan))
for c in range(nchan):
    mvgAvg[:,c] =  echobase.movingaverage(probabilityArray[:,c], w)  
mvgAvgThreshold = copy.deepcopy(mvgAvg)
mvgAvgThreshold[mvgAvgThreshold>threshold] = 1
mvgAvgThreshold[mvgAvgThreshold<=threshold] = 0

    
seizure_start =int((secondsBeforeSpread-20)/skipWindow)
seizure_stop = int((secondsBeforeSpread + 20)/skipWindow)

mvgAvgThreshold_seizure = mvgAvgThreshold[seizure_start:,:]
spread_start = np.argmax(mvgAvgThreshold_seizure == 1, axis = 0)

for c in range(nchan): #if the channel never starts seizing, then np.argmax returns the index as 0. This is obviously wrong, so fixing this
    if np.all( mvgAvgThreshold_seizure[:,c] == 0  ) == True:
        spread_start[c] = len(mvgAvgThreshold_seizure)


spread_start_loc = ( (spread_start + seizure_start)  *skipWindow*fs).astype(int)
channel_order = np.argsort(spread_start)

print(np.array(df.columns)[channel_order])


channel_orderPlot = channel_order[[0,2,3,11, 15]]
channel_orderPlot = channel_order[np.array(range(15)).tolist()]

startIndPlot =  int((  seizure_start  *skipWindow*fs)) + int(0*fs)
stopIndPlot =   int((  seizure_stop  *skipWindow*fs))+ int(20*fs) 

markers = spread_start_loc - startIndPlot

data_plotFiltDS = DataJson.downsample(data_plotFilt, fs,fs)

data_plotFiltDS.shape

fnameFig = join(pathFigureOverview, "ONSET_withMarkers.png")
DataJson.plot_eeg(data_plotFilt[startIndPlot:stopIndPlot,channel_orderPlot], fs, markers = markers[channel_orderPlot], savefig = False, pathFig = fnameFig, nchan = len(channel_orderPlot), dpi = 300, aspect=10, height=1, hspace = -0.6, lw = 0.5)    
fnameFig = join(pathFigureOverview, "ONSET_WITHOUT_markers.png")
DataJson.plot_eeg(data_plotFilt[startIndPlot:stopIndPlot,channel_orderPlot], fs, savefig = False, pathFig = fnameFig, nchan = len(channel_orderPlot), dpi = 300, aspect=10, height=1, hspace = -0.6, lw = 3)    







seizurePattern.plot_heatmapSingle(probWN[0:,range(nchan)], fsds, skip, figsize = (5,2.5), smoothing = 20,  cmapName="Greens")
seizurePattern.plot_heatmapSingle(probCNN[0:,range(nchan)], fsds, skip, figsize = (5,2.5), smoothing = 20,  cmapName="Greens")
seizurePattern.plot_heatmapSingle(probLSTM[0:,range(nchan)], fsds, skip, figsize = (5,2.5), smoothing = 20,  cmapName="Greens")



seizurePattern.plot_heatmapSingleThreshold(probWN[0:,range(nchan)], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 20,  cmapName="Greens")
seizurePattern.plot_heatmapSingleThreshold(probCNN[0:,range(nchan)], fsds, skip, figsize = (5,2.5), threshold = 0.9, smoothing = 20,  cmapName="Greens")
seizurePattern.plot_heatmapSingleThreshold(probLSTM[0:,range(nchan)], fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 20,  cmapName="Greens")





















#%%window to fit model
data_scalerDS_X = echomodel.overlapping_windows(data_scalerDS, time_step, skip)
windows, _, _ = data_scalerDS_X.shape

probWN = np.zeros(shape = (windows, nchan))  
probCNN = np.zeros(shape = (windows, nchan))  
probLSTM = np.zeros(shape = (windows, nchan))  
    


  
#%%


for c in range(nchan):
    ch_pred =     data_scalerDS_X[:,:,c].reshape(windows, time_step, 1    )
    probWN[:,c] =  modelWN.predict(ch_pred, verbose=1)[:,1]
    probCNN[:,c] =  modelCNN.predict(ch_pred, verbose=1)[:,1]
    probLSTM[:,c] =  modelLSTM.predict(ch_pred, verbose=1)[:,1]
        


#%%

seizurePattern.plot_probheatmaps(probWN, fsds, skip, threshold=0.99)
seizurePattern.plot_probheatmaps(probCNN, fsds, skip, threshold=0.99)
seizurePattern.plot_probheatmaps(probLSTM, fsds, skip, threshold=0.9)


#%% subgeline

DataJson.plot_eeg(probability_arr_movingAvg, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    
DataJson.plot_eeg(probability_arr_movingAvg_threshold, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    


#%%getting start times

seizure_start =int((secondsBeforeSpread-20)/skipWindow)
seizure_stop = int((secondsBeforeSpread + 20)/skipWindow)

probability_arr_movingAvg_threshold_seizure = probability_arr_movingAvg_threshold[seizure_start:,:]
spread_start = np.argmax(probability_arr_movingAvg_threshold_seizure == 1, axis = 0)

for c in range(nchan): #if the channel never starts seizing, then np.argmax returns the index as 0. This is obviously wrong, so fixing this
    if np.all( probability_arr_movingAvg_threshold_seizure[:,c] == 0  ) == True:
        spread_start[c] = len(probability_arr_movingAvg_threshold_seizure)


spread_start_loc = ( (spread_start + seizure_start)  *skipWindow*fsds).astype(int)
markers = spread_start_loc
channel_order = np.argsort(spread_start)

print(np.array(df.columns)[channel_order])
#%%

DataJson.plot_eeg(probability_arr_movingAvg_threshold[:,channel_order], fsds, nchan = 50, dpi = 300, fill=True, aspect=200, height=0.05)    
#%%
DataJson.plot_eeg(data_scalerDS, fsds, markers = spread_start_loc, nchan = 10, dpi = 300, aspect=200, height=0.1)    


DataJson.plot_eeg(data_scalerDS[:,channel_order], fsds, markers = spread_start_loc[channel_order], nchan = nchan, dpi = 300, aspect=200, height=0.05)    


 
DataJson.plot_eeg(data_avgref, fs, markers =  ( (spread_start + seizure_start)  *skipWindow*fs).astype(int)[0:5], nchan = 5, dpi = 300, aspect=20, height=0.5)    


#%% Visualize


DataJson.plot_eeg(probability_arr_movingAvg, fsds, nchan = 5, dpi = 300, fill=True, aspect=35, height=0.3)    
DataJson.plot_eeg(data, fs, nchan = 5, dpi = 300, aspect=35)    
DataJson.plot_eeg(data_scalerDS, fsds, nchan = 5, dpi = 300, aspect=35)    

echobase.show_eeg_compare(data, data, int(fsds), channel=2)  

    

    
#%%diffusion model






sfc_datapath = join("/media","arevell","sharedSSD","linux","papers","paper005", "data", "data_processed", "aggregated_data")
sub = "sub0278"
iEEG_filename = "HUP138_phaseII"

start_time_usec = 416023190000
stop_time_usec = 416112890000 
fname = join(sfc_datapath,   f"sub-{sub}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )

os.path.exists(fname)

if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = DataClassSfc.sfc( **all_data  ) 
atlas = sfc_data.get_atlas_names()[1]   



st =int((secondsBeforeSpread-10)/skipWindow)
stp = int((secondsBeforeSpread +20)/skipWindow)

SC, SC_regions = sfc_data.get_structure(atlas)
spread = copy.deepcopy(probability_arr_movingAvg[st:stp,:])
spread_regions = np.array(df.columns )   
electrodeLocalization = sfc_data.get_electrodeLocalization(atlas)

#SC= SC_distInv

#%%
#null
#SC_null = bct.randmio_und(SC, 30)[0]

#sns.heatmap(SC, square=True)   
#sns.heatmap(SC_null, square=True)   

#SC = SC_null
#%%

import diffusionModels as DM
dm_data = DM.diffusionModels(SC, SC_regions, spread, spread_regions, electrodeLocalization)    

SC_hat = dm_data.preprocess_SC(SC, 0.4)    
    
def get_LTM(SC, threshold=0.1, time_steps = 6):
    corrs = np.zeros(shape = (len(SC)))
        
    for r in range(len(corrs)):
        node_state = dm_data.LTM(SC_hat, seed=r, threshold=0.1, time_steps = 6)
        #equalizing node state and spread
        el = copy.deepcopy(electrodeLocalization)
        #remove any electrodes in spread with no localization data
        ind = np.intersect1d(spread_regions, el["electrode_name"], return_indices=True)
        spreadMod = spread[:,ind[1]]
        el_mod = el.iloc[ind[2],:]
        el_mod_regions = np.array(el_mod[atlas + "_region_number"]).astype('str')
        #remove structure with no electrodes
        ind2 = np.intersect1d(SC_regions, el_mod_regions, return_indices=True)
        node_state_mod = node_state[:,ind2[1]]
        #select a random electrode from a region with multiple electrodes 
        SC_mod_region_names = ind2[0]
        ind3 = []
        #np.random.seed(41)
        for s in range(len(SC_mod_region_names)):
            ind3.append(np.random.choice(np.where(SC_mod_region_names[s] == el_mod_regions)[0],1)[0])
        ind3 = np.array(ind3)
        spreadMod_state = spreadMod[:, ind3]   
        #el_mod_sc = el_mod.iloc[ind3,:]
        #electrode_names = np.array(el_mod_sc["electrode_name"])
        #interpolate node state
        node_state_mod_resample = signal.resample(node_state_mod, spreadMod_state.shape[0])
        #sns.heatmap(node_state_mod_resample.T, cbar=False)   
        #sns.heatmap(spreadMod_state.T, cbar=False)   
        if (node_state_mod_resample == 0).all():#if entire prediction is zero, make the correlation zero
            corrs[r] = 0
        else:
            corrs[r] = spearmanr(node_state_mod_resample.flatten(), spreadMod_state.flatten())[0]
        print (corrs[r])
    return corrs

corrs = get_LTM(SC_hat)
Nnull = 2
corrs_null = np.zeros(shape = (Nnull, len(corrs)))
for nu in range(Nnull):
    print(nu)
    SC_null = bct.randmio_und(SC, 30)[0]
    dm_data = DM.diffusionModels(SC, SC_regions, spread, spread_regions, electrodeLocalization) 
    SC_hat = dm_data.preprocess_SC(SC_null, 0.4)  
    corrs_null[nu,:] = get_LTM(SC_hat)




ind = np.argsort(corrs)[::-1]

corrs_order = corrs[ind]
corrs_null_order = corrs_null[:,ind]
corrs_order_null_df = pd.DataFrame(corrs_null_order).melt()

fig, ax = plt.subplots(1,1, figsize = (5,5), dpi = 300)    
sns.lineplot(x = range(len(corrs_order)), y = corrs_order, ax = ax)
sns.lineplot(x="variable", y="value", data=corrs_order_null_df,ax = ax, color = "red")
ax.set_xlabel("Brain Region Number")
ax.set_ylabel("Spearman Rank Correlation")
ax.set_title("LTM: Brain regions most correlated to measured seizure pattern")

print(SC_regions[ind][0:5])
print(corrs_order[0:5])



#%%


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


centroidsPath = join(path, "data", "raw","atlases", "atlasCentroids", "AAL2_centroid.csv")
    
    
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
    
    
    