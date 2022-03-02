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
skipWindow = 0.25#Next window is skipped over in seconds
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

#%% Get files and relevant patient information to train model

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = dataclass_iEEG_metadata.dataclass_iEEG_metadata(jsonFile)




#%%Get data

patientsWithseizures = DataJson.get_patientsWithSeizuresAndInterictal()


i = 19 #RID0309: 33-43 ; RID0278:19
RID = np.array(patientsWithseizures["subject"])[i]
idKey = np.array(patientsWithseizures["idKey"])[i]
AssociatedInterictal = np.array(patientsWithseizures["AssociatedInterictal"])[i]
df, fs, ictalStartIndex, ictalStopIndex = DataJson.get_precitalIctalPostictal(RID, "Ictal", idKey, username, password,BIDS = BIDS, dataset= datasetiEEG, session = session, secondsBefore = secondsBefore, secondsAfter = secondsAfter)
df_interictal, _ = DataJson.get_iEEGData(RID, "Interictal", AssociatedInterictal, username, password, BIDS = BIDS, dataset= datasetiEEG, session = session, startKey = "Start")

                   
print("preprocessing")
dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels = DataJson.preprocessNormalizeDownsample(df, df_interictal, fs, fsds)


data, data_avgref, data_ar, data_filt, channels = echobase.preprocess(df, fs, fsds)
data_scalerDSDS = DataJson.downsample(data_scaler, fs, 16)
data_arDSDS = DataJson.downsample(data_ar, fs, 16)

nsamp, nchan = data_filt.shape


DataJson.plot_eeg(data_scalerDSDS, 16, markers = [secondsBefore*16], dpi = 25, aspect=aspect*2, height=nchan/aspect/2)  


# %%

#%%Meaure seizure spread
version = 11
fpath_wavenet = join(deepLearningModelsPath, f"wavenet/v{version:03d}.hdf5")
fpath_1dCNN = join(deepLearningModelsPath, f"1dCNN/v{version:03d}.hdf5")
fpath_lstm = join(deepLearningModelsPath, f"lstm/v{version:03d}.hdf5")


modelWN = load_model(fpath_wavenet)
modelCNN= load_model(fpath_1dCNN)
modelLSTM = load_model(fpath_lstm)

#%%window to fit model
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
        


#%%

seizurePattern.plot_probheatmaps(probWN, fsds, skip, threshold=0.9)
seizurePattern.plot_probheatmaps(probCNN, fsds, skip, threshold=0.9)
seizurePattern.plot_probheatmaps(probLSTM, fsds, skip, threshold=0.7)

#%%

    

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

# %%
THRESHOLD = 0.65
SMOOTHING = 20 #in seconds
prob_array= probCNN


probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(probWN, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
sns.heatmap( probability_arr_movingAvg.T )      
sns.heatmap( probability_arr_threshold.T )    
    
    
    
#%% Ridgeline

###
DataJson.plot_eeg(probability_arr_movingAvg, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    
DataJson.plot_eeg(probability_arr_threshold, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    
###

#%%getting start times

seizure_start =int((secondsBefore-20)/skipWindow)
seizure_stop = int((secondsBefore + 20)/skipWindow)

probability_arr_movingAvg_threshold_seizure = probability_arr_threshold[seizure_start:,:]
spread_start = np.argmax(probability_arr_movingAvg_threshold_seizure == 1, axis = 0)

for c in range(nchan): #if the channel never starts seizing, then np.argmax returns the index as 0. This is obviously wrong, so fixing this
    if np.all( probability_arr_movingAvg_threshold_seizure[:,c] == 0  ) == True:
        spread_start[c] = len(probability_arr_movingAvg_threshold_seizure)


spread_start_loc = ( (spread_start + seizure_start)  *skipWindow*fsds).astype(int)
markers = spread_start_loc
channel_order = np.argsort(spread_start)

print(np.array(df.columns)[channel_order])
#%%

DataJson.plot_eeg(probability_arr_threshold[:,channel_order], fsds, nchan = 50, dpi = 300, fill=True, aspect=200, height=0.05)    
#%%
DataJson.plot_eeg(data_scalerDS, fsds, markers = spread_start_loc, nchan = 10, dpi = 300, aspect=200, height=0.1)    


#DataJson.plot_eeg(data_scalerDS[:,channel_order], fsds, markers = spread_start_loc[channel_order], nchan = nchan, dpi = 300, aspect=200, height=0.05)    


 
DataJson.plot_eeg(data_avgref, fs, markers =  ( (spread_start + seizure_start)  *skipWindow*fs).astype(int)[0:5], nchan = 5, dpi = 300, aspect=20, height=0.5)    


#%% Visualize


DataJson.plot_eeg(probability_arr_movingAvg, fsds, nchan = 5, dpi = 300, fill=True, aspect=35, height=0.3)    
DataJson.plot_eeg(data, fs, nchan = 5, dpi = 300, aspect=35)    
DataJson.plot_eeg(data_scalerDS, fsds, nchan = 5, dpi = 300, aspect=35)    




    
#%%diffusion model






sfc_datapath = "/media/arevell/sharedSSD/linux/papers/brain_atlases/data/data_processed/aggregated_data"
RID ="RID0278"
iEEG_filename =   "HUP138_phaseII"

start_time_usec = 416023190000
stop_time_usec = 416112890000 
fname = os.path.join(sfc_datapath,   f"sub-{RID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )
fname = os.path.join(sfc_datapath,   f"sub-RID0320_HUP140_phaseII_D02_331979921071_332335094986_data.pickle" )


os.path.exists(fname)

if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = dataclass_SFC.dataclass_sfc( **all_data  ) 



#%%
atlas = sfc_data.get_atlas_names()[6]   
st =int((secondsBefore-10)/skipWindow)
stp = int((secondsBefore +20)/skipWindow)

SC, SC_regions = sfc_data.get_structure(atlas)
spread = copy.deepcopy(probability_arr_movingAvg[st:stp,:])
spread_regions = copy.deepcopy(channels)  
electrodeLocalization = sfc_data.get_electrodeLocalization(atlas)

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
"""
#%%
#null
#SC_null = bct.randmio_und(SC, 30)[0]

#sns.heatmap(SC, square=True)   
#sns.heatmap(SC_null, square=True)   

#SC = SC_null
#%%

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
        print(f"\rtop: {SC_regions[np.where(corrs == np.max(corrs))[0][0]]}; {np.round( np.max(corrs),2)}; {np.round((r+1)/len(corrs)*100,2)}%  ", end = "\r")
        
        SC_regions[np.where(corrs == np.max(corrs))[0][0]]
    return corrs


corrs = get_LTM(SC_hat)
Nnull = 2
corrs_null = np.zeros(shape = (Nnull, len(corrs)))
"""
for nu in range(Nnull):
    print(f"\r{np.round((nu+1)/Nnull*100,2)}", end = "\r")
    SC_null = bct.randmio_und(SC, 30)[0]
    dm_data = DM.diffusionModels(SC, SC_regions, spread, spread_regions, electrodeLocalization) 
    SC_hat = dm_data.preprocess_SC(SC_null, 0.4)  
    corrs_null[nu,:] = get_LTM(SC_hat)

"""


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

aaaa = np.vstack([SC_regions[ind],corrs_order ]).T


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
    
    
    