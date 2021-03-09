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
from  matplotlib import colors
from scipy.interpolate import interp1d
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
path = os.path.join("/media","arevell","sharedSSD","linux","papers","paper002") #Parent directory of project
sys.path.append(os.path.join(path, "seizure_spread", "code", "tools"))
sys.path.append(os.path.join(path, "seizure_spread/tools/ieegpy"))
import echobase
import echomodel 
import echoimage
import download_iEEG_data as downloadiEEG
import DataClassSfc
import DataClassJson


#%%


#%%

def plot_probheatmaps(probabilityArray, fsds, skip, threshold = 0.5, smoothing = 20, channel1 = 0, channel2 = 1):
    fig, ax = plt.subplots(3,2, figsize=(10,10), dpi =300)
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
        
    sns.lineplot( x = range(windows),  y= probabilityArray[:,channel1],    ci=None, ax = ax[0,0])    
    sns.lineplot( x = range(windows),  y= probabilityArray[:,channel2],    ci=None, ax = ax[0,1])    
    sns.heatmap( probabilityArray.T , ax = ax[1,0])    
    sns.heatmap( thresholded.T,  ax = ax[1,1] )    
    sns.heatmap( mvgAvg.T , ax = ax[2,0] )       
    sns.heatmap( mvgAvgThreshold.T,  ax = ax[2,1] )      



def plot_singleProbability(probabilityArray, fsds, skip, channel=1, startInd = 0, stopInd = None, smoothing = 20, vmin = -0.2, vmax = 1.2):
    windows, nchan = probabilityArray.shape
    if stopInd == None:
        stopInd = windows
    
    smoothing = smoothing
    w = int(smoothing*fsds/skip)
    mvgAvg = np.zeros(shape = (windows - w + 1, nchan))
    for c in range(nchan):
        mvgAvg[:,c] =  echobase.movingaverage(probabilityArray[:,c], w)  
 
    y=  mvgAvg[startInd:stopInd,channel]
    x = range(len(y))
    
    y2 = -0.05
    fig, ax = plt.subplots(1,1, figsize=(10,2), dpi =300)
    sns.lineplot( x = x,  y= y,    ci=None, ax = ax, alpha = 0)    
    
    
    interpolate = interp1d(np.array(x),np.array(x))
    xnew = interpolate(np.arange( 0, len(x)-1, 1 ))
    interpolate = interp1d(np.array(x),y)
    ynew = interpolate(np.arange( 0, len(x)-1, 1 ))
    
    Z = np.linspace(0, 1, len(xnew))
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap('Blues')
    df = pd.DataFrame({'x': xnew, 'y1': ynew, 'z': Z})
    xcolor = df['x'].values
    y1color = df['y1'].values
    zcolor = df['z'].values
    for ii in range(len(df['x'].values)-1):
        plt.fill_between( x = [ xcolor[ii] , xcolor[(ii+1)]] , y1 = [ y1color[ii], y1color[ii+1] ], y2=y2, color=cmap(normalize(zcolor[ii]))   )

    ax.set(yticks=[])
    ax.set(xticks=[])
    ax.set_ylim([y2,1])
    sns.despine(bottom=True, left=True)    

def plot_eegGradient(probabilityArray, fsds,  startInd = None, stopInd = None, nchan = None, markers = [],vminA = [-3,-0.2], vmaxA = [0.8,1.2], aspect = 45, height = 1.2, hspace = -0.9, dpi = 300, lw=10, fill = True, savefig = False, pathFig = None):
        if stopInd == None:
            stopInd = len(probabilityArray)/fsds
        if startInd == None:
            startInd = 0
        if nchan == None:
            nchan = probabilityArray.shape[1]
        
        df_wide = pd.DataFrame(probabilityArray[startInd:stopInd,:]   )#520
        df_long = pd.melt(df_wide, var_name = "channel", ignore_index = False)
        df_long["index"] = df_long.index
        
        vminArr = np.linspace(vminA[0], vminA[1], nchan)
        vmaxArr = np.linspace(vmaxA[0], vmaxA[1], nchan)
        if fill == True:
            pal = sns.cubehelix_palette(nchan, rot=-.25, light=.7)
        else:
            pal = sns.cubehelix_palette(nchan, rot=-.25, light=0)
            
        sns.set(rc={"figure.dpi":dpi})
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(df_long, row="channel", hue="channel", aspect=aspect, height=height, palette=pal)
        if fill == True:
            g.map(sns.lineplot,"index", "value", clip_on=False, color=(1, 1, 1), lw=lw)
            axes = g.axes  
            for ch in range(len(axes)):
                ynew = np.array(df_wide.iloc[:,ch])
                xnew = np.array(range(len(ynew)))
                Z = np.linspace(0, 1, len(df_wide))
                vmin = vminArr[ch];vmax = vmaxArr[ch]
                normalize = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.get_cmap('Blues')    
                for ii in range(len(xnew)-1):
                    axes[ch][0].fill_between( x = [ xnew[ii] , xnew[(ii+1)]] , y1 = [ ynew[ii], ynew[ii+1] ], y2=0, color=cmap(normalize(Z[ii]))   )
            #g.map(plt.fill_between,  "index", "value")
                
        else:
            g.map(sns.lineplot, "index", "value", clip_on=False, alpha=1, linewidth=lw)
        if len(markers) > 0:    
            axes = g.axes  
            if len(markers) > 1:
                for c in range(len(axes)):
                    axes[c][0].axvline(x=markers[c])
            else:
                for c in range(len(axes)):
                    axes[c][0].axvline(x=markers[0])
        g.fig.subplots_adjust(hspace=hspace)
        g.set_titles("")
        g.set(yticks=[])
        g.set(xticks=[])
        g.set_axis_labels("", "")
        g.despine(bottom=True, left=True)        
        
        if savefig:
            if pathFig == None:
                print("Must provide figure path and filename to save")
            else: plt.savefig(pathFig, transparent=True)

    


def plot_heatmapSingle(probabilityArray, fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 20, vmin = 0.1, vmax = 1.1, cmapName = "Blues"):

    fig, ax = plt.subplots(1,1, figsize=figsize, dpi =300)
    windows, nchan = probabilityArray.shape
    w = int(smoothing*fsds/skip)
    mvgAvg = np.zeros(shape = (windows - w + 1, nchan))
    for c in range(nchan):
        mvgAvg[:,c] =  echobase.movingaverage(probabilityArray[:,c], w)  

  
    cmap = plt.cm.get_cmap(cmapName)
    sns.heatmap( mvgAvg.T , cmap = cmap , ax = ax, cbar = False, xticklabels =  False, yticklabels = False, vmin = vmin, vmax = vmax)           
    
    

def plot_heatmapSingleThreshold(probabilityArray, fsds, skip, figsize = (5,2.5), threshold = 0.5, smoothing = 20, vmax = 1.1, cmapName = "Blues"):

    fig, ax = plt.subplots(1,1, figsize=figsize, dpi =300)
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
  
    cmap = plt.cm.get_cmap(cmapName)    
    sns.heatmap( mvgAvgThreshold.T , cmap = cmap , ax = ax, cbar = False, xticklabels =  False, yticklabels = False , vmax = vmax)       
  
    
    

"""

#%% Ridgeline

DataJson.plot_eeg(probabilityArray_movingAvg, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    
DataJson.plot_eeg(probabilityArray_movingAvg_threshold, fsds, nchan = nchan, dpi = 300, fill=True, aspect=200, height=0.05)    


#%%getting start times

seizure_start =int((secondsBeforeSpread-20)/skipWindow)
seizure_stop = int((secondsBeforeSpread + 20)/skipWindow)

probabilityArray_movingAvg_threshold_seizure = probabilityArray_movingAvg_threshold[seizure_start:,:]
spread_start = np.argmax(probabilityArray_movingAvg_threshold_seizure == 1, axis = 0)

for c in range(nchan): #if the channel never starts seizing, then np.argmax returns the index as 0. This is obviously wrong, so fixing this
    if np.all( probabilityArray_movingAvg_threshold_seizure[:,c] == 0  ) == True:
        spread_start[c] = len(probabilityArray_movingAvg_threshold_seizure)


spread_start_loc = ( (spread_start + seizure_start)  *skipWindow*fsds).astype(int)
markers = spread_start_loc
channel_order = np.argsort(spread_start)

print(np.array(df.columns)[channel_order])
#%%

DataJson.plot_eeg(probabilityArray_movingAvg_threshold[:,channel_order], fsds, nchan = 50, dpi = 300, fill=True, aspect=200, height=0.05)    
#%%
DataJson.plot_eeg(data_scalerDS, fsds, markers = spread_start_loc, nchan = 10, dpi = 300, aspect=200, height=0.1)    


DataJson.plot_eeg(data_scalerDS[:,channel_order], fsds, markers = spread_start_loc[channel_order], nchan = nchan, dpi = 300, aspect=200, height=0.05)    


 
DataJson.plot_eeg(data_avgref, fs, markers =  ( (spread_start + seizure_start)  *skipWindow*fs).astype(int)[0:5], nchan = 5, dpi = 300, aspect=20, height=0.5)    


#%% Visualize


DataJson.plot_eeg(probabilityArray_movingAvg, fsds, nchan = 5, dpi = 300, fill=True, aspect=35, height=0.3)    
DataJson.plot_eeg(data, fs, nchan = 5, dpi = 300, aspect=35)    
DataJson.plot_eeg(data_scalerDS, fsds, nchan = 5, dpi = 300, aspect=35)    

echobase.show_eeg_compare(data, data, int(fsds), channel=2)  

    

    
#%%diffusion model






sfc_datapath = os.path.join("/media","arevell","sharedSSD","linux","papers","paper005", "data", "data_processed", "aggregated_data")
RID = "RID0278"
iEEG_filename = "HUP138_phaseII"

start_time_usec = 416023190000
stop_time_usec = 416112890000 
fname = os.path.join(sfc_datapath,   f"sub-{RID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )

os.path.exists(fname)

if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = DataClassSfc.sfc( **all_data  ) 
atlas = sfc_data.get_atlas_names()[1]   



st =int((secondsBeforeSpread-10)/skipWindow)
stp = int((secondsBeforeSpread +20)/skipWindow)

SC, SC_regions = sfc_data.get_structure(atlas)
spread = copy.deepcopy(probabilityArray_movingAvg[st:stp,:])
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
    
 """ 
    