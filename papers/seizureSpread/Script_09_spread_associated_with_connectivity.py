#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:26:36 2022

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
import re
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
import statsmodels.api as sm
from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename

import matplotlib.colors

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

def pull_patient_localization(file_path):
    patient_localization = loadmat(file_path)['patient_localization']
    patients = [i[0] for i in np.squeeze(patient_localization['patient'])]

    labels = []
    for row in patient_localization['labels'][0, :]:
        labels.append([i[0][0] for i in row])

    ignore = np.squeeze(patient_localization['ignore'])
    resect = np.squeeze(patient_localization['resect'])
    gm_wm = np.squeeze(patient_localization['gm_wm'])
    coords = np.squeeze(patient_localization['coords'])
    region = np.squeeze(patient_localization['region'])

    soz = np.squeeze(patient_localization['soz'])

    return patients, labels, ignore, resect, gm_wm, coords, region, soz

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
project_folder = "derivatives/seizure_spread"
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

with open(paths.ATLAS_FILES_PATH) as f: atlas_files = json.load(f)

fname_patinet_localization = join(metadataDir, "patient_localization_final.mat")
RID_HUP = join(metadataDir, "RID_HUP.csv")
outcomes_fname = join(metadataDir, "patient_cohort_all_atlas.csv")

#% 03 Project parameters
version = 11

if version == 11:
    fsds = 128 
    window = 1 #window of eeg for training/testing. In seconds
    skipWindow = 0.1#Next window is skipped over in seconds
if version == 14:    
    fsds = 128 *2
    window = 1 #window of eeg for training/testing. In seconds
    skipWindow = 0.1#Next window is skipped over in seconds
annotationLayerName = "seizureChannelBipolar"
#annotationLayerName = "seizure_spread"
secondsBefore = 180
secondsAfter = 180

#window = 1 #window of eeg for training/testing. In seconds
#skipWindow = 0.1#Next window is skipped over in seconds
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

patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(fname_patinet_localization)
RID_HUP = pd.read_csv(RID_HUP)
with open(fnameJSON) as f: jsonFile = json.load(f)


#% Get files and relevant patient information to train model

#turning jsonFile to a @dataclass to make easier to extract info and data from it
DataJson = dataclass_iEEG_metadata.dataclass_iEEG_metadata(jsonFile)
patientsWithseizures = DataJson.get_patientsWithSeizuresAndInterictal()
np.unique(patientsWithseizures["subject"])
unique_patients = np.array(patientsWithseizures.subject)
v = {}
for i, x in enumerate(unique_patients):
    if x not in v:
        v[x] = i
        
indexes = np.array([v.get(x) for x in list(v.keys()) ])


outcomes = pd.read_csv(outcomes_fname, sep = "\t"); outcomes.rename(columns = {'RID':'subject'}, inplace = True)
#changing outcome scores to 1,2,3,4
for k in range(len(outcomes)):
    outcomes_list = ["Engel_6_mo", "Engel_12_mo","Engel_24_mo"]
    for o in range(len(outcomes_list)):
        value = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list[o]])[0]
        if not np.isnan(value):
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list[o]] =int(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list[o]] ) 
        
  
outcomes["Engel_6_mo_binary"] = np.nan
outcomes["Engel_12_mo_binary"] = np.nan
outcomes["Engel_24_mo_binary"] = np.nan
outcome_threshold = 1
for k in range(len(outcomes)):
    
    outcomes_list = ["Engel_6_mo", "Engel_12_mo","Engel_24_mo"]
    outcomes_list2 = ["Engel_6_mo_binary", "Engel_12_mo_binary","Engel_24_mo_binary"]
    for o in range(len(outcomes_list)):
        value = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list[o]])[0]
        if value > outcome_threshold:
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "poor"
        else:
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "good"
            
        if np.isnan(value):
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "unknown"
    
for k in range(len(outcomes)): #if poor outcome at 6 or 12 month, then propagate that thru if unknown
      outcomes_list2 = ["Engel_6_mo_binary", "Engel_12_mo_binary","Engel_24_mo_binary"]
      for o in [1,2]:
          value = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o]])[0]
          if value =="unknown":
              value_previous = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o-1]])[0]
              if value_previous == "poor":
                  outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "poor"
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
########################################################################            
                  

#%%


  
tanh = False
model_IDs = ["WN","CNN","LSTM" , "absolute_slope", "line_length", "power_broadband"]
by = 0.01


full_analysis_location = join(BIDS, project_folder, f"full_analysis_save")
full_analysis_location_file_basename = f"soz_overlap_tanh_{tanh}_by_{by}_model_{version}.pickle"
full_analysis_location_file = join(full_analysis_location, full_analysis_location_file_basename)


with open(full_analysis_location_file, 'rb') as f: [soz_overlap, percent_active, tanh, seconds_active, by, thresholds, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)


#%%

#get connectivity matrices

i=137
m = 0
thr = [0.69, 0.96, 0.58, 0.08, 0.11, 0.01]
thr = [0.72, 0.96, 0.58, 0.08, 0.11, 0.01]
sc_vs_quickness = pd.DataFrame(columns =[ "subject", "seizure", "quickenss", "sc_LR_mean", "sc_temporal_mean" ])

for i in range(len(patientsWithseizures)):
    print(i)
    RID = np.array(patientsWithseizures["subject"])[i]
    seizure = np.array(patientsWithseizures["idKey"])[i]
    seizure_length = patientsWithseizures.length[i]
    
    #atlas
    atlas = "BN_Atlas_246_1mm"
    atlas = "AAL3v1_1mm"
    #atlas = "HarvardOxford-sub-ONLY_maxprob-thr25-1mm"
    
    
    atlas_names_short =  list(atlas_files["STANDARD"].keys() )
    atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
    ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
    atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
    
    atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
    atlas_label_names = np.array(atlas_label.iloc[1:,1])
    atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
    
    temporal_regions = ["Hippocampus"]; names = "Hippocampus"
    temporal_regions = ["Temporal"]; names = "Temporal"
    #temporal_regions = ["Frontal"]; names = "Frontal"
    temporal_regions = ["Parietal"]; names = "Parietal"
    #temporal_regions = ["Amygdala"]; names = "Amygdala"
    #temporal_regions = ["Hippocampus", "Temporal", "Amygdala","Fusiform", "Insula"]; names = "Hippocampus, Temporal, Amygdala, Fusiform, Insula"
    #temporal_regions = ["Insula", "Hippocampus", "ParaHippocampal", "Fusiform" ,"Heschl", "Temporal"]; names = "Insula, Hippocampus, ParaHippocampal, Fusiform ,Heschl, Temporal"

    
    temporal_region_inds = []
    temporal_region_inds_L = []
    temporal_region_inds_R = []
    for r in range(len(atlas_label_names)):
        bools = []
        for k in range(len(temporal_regions)):
            bools.append(temporal_regions[k] in atlas_label_names[r] )
        if any(bools):
            temporal_region_inds.append(r)
        
        bools_L = []
        bools_R = []
        for k in range(len(temporal_regions)):
            if temporal_regions[k] in atlas_label_names[r] :
                if "_L" in atlas_label_names[r]:
                    bools_L.append(temporal_regions[k] in atlas_label_names[r] )
                elif "_R" in atlas_label_names[r]:
                    bools_R.append(temporal_regions[k] in atlas_label_names[r] )
            
        if any(bools):
            temporal_region_inds.append(r)
        if any(bools_L):
            temporal_region_inds_L.append(r)
        if any(bools_R):
            temporal_region_inds_R.append(r)
            
    temporal_region_numbers = atlas_label_region_numbers[temporal_region_inds].astype(int)
    temporal_region_numbers_L = atlas_label_region_numbers[temporal_region_inds_L].astype(int)
    temporal_region_numbers_R = atlas_label_region_numbers[temporal_region_inds_R].astype(int)

    connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"sub-{RID}.{atlas}.count.pass.connectogram.txt")
    
    connectivity_loc_glob = glob.glob( connectivity_loc  )
    
    #CHECKING IF FILES EXIST
    if len(connectivity_loc_glob) > 0:
        connectivity_loc_path = connectivity_loc_glob[0]

        
        sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)
        sc = sc/sc.max()
        
        #sc=utils.log_normalize_adj(sc)
        sc_region_labels = utils.read_DSI_studio_Txt_files_SC_return_regions(connectivity_loc_path, atlas).astype(ind)

        sc_temporal_region_ind = []
        for r in range(len(temporal_region_numbers)):
            ind = np.where(temporal_region_numbers[r] == sc_region_labels)[0][0]
            sc_temporal_region_ind.append(ind)
        sc_temporal_region_ind = np.array(sc_temporal_region_ind)
        
        
        sc_temporal_region_ind_L = []
        for r in range(len(temporal_region_numbers_L)):
            ind = np.where(temporal_region_numbers_L[r] == sc_region_labels)[0][0]
            sc_temporal_region_ind_L.append(ind)
        sc_temporal_region_ind_L = np.array(sc_temporal_region_ind_L)
        
        
        
        sc_temporal_region_ind_R = []
        for r in range(len(temporal_region_numbers_R)):
            ind = np.where(temporal_region_numbers_R[r] == sc_region_labels)[0][0]
            sc_temporal_region_ind_R.append(ind)
        sc_temporal_region_ind_R = np.array(sc_temporal_region_ind_R)
        
    
        sc_temporal = sc[sc_temporal_region_ind[:,None], sc_temporal_region_ind[None,:]]
        sc_temporal_L_R = sc[sc_temporal_region_ind_L[:,None], sc_temporal_region_ind_R[None,:]]
        
        #sns.heatmap(sc)
        #sns.heatmap(sc_temporal)
        #sns.heatmap(sc_temporal_L_R)
        sc_temporal_mean = np.nansum(sc_temporal)
        sc_LR_mean = np.nansum(sc_temporal_L_R)
        #get quickness
        
        spread_quick = soz_overlap[(soz_overlap["model"] == model_IDs[m])  & (soz_overlap["subject"] == RID)  & (soz_overlap["threshold"] == thr[m])  & (soz_overlap["seizure"] == seizure)]
        quickenss = np.array(abs(spread_quick["hipp_left_mean"] - spread_quick["hipp_right_mean"] ))[0]



        
        sc_vs_quickness = sc_vs_quickness.append(dict( subject =RID , seizure = seizure , quickenss = quickenss, sc_LR_mean = sc_LR_mean, sc_temporal_mean = sc_temporal_mean) , ignore_index=True )



        
#%
sc_vs_quickness      


ind1 = list(np.where((sc_vs_quickness["subject"] == "RID0454"  ) &(sc_vs_quickness["seizure"] == "1"  ) )[0])
ind2 = list(np.where((sc_vs_quickness["subject"] == "RID0278"  ) &(sc_vs_quickness["seizure"] == "1"  ) )[0])
ind3 = list(np.where((sc_vs_quickness["subject"] == "RID0365"  ))[0])
ind4 = list(np.where((sc_vs_quickness["subject"] == "RID0522"  ))[0])


inds = ind1 + ind2 + ind3 

sc_vs_quickness_filt = sc_vs_quickness.drop(inds)  


sc_vs_quickness_filt["inverse_quickness"] = 1/sc_vs_quickness_filt["quickenss"]

sc_vs_quickness_filt_group = sc_vs_quickness_filt.groupby(["subject"], as_index=False).median()

sc_vs_quickness_filt_fill= sc_vs_quickness_filt.fillna(0)
sc_vs_quickness_group_fill= sc_vs_quickness_filt_group.fillna(0)

#%

"""
fig, axes = utils.plot_make(size_length=10)
g = sns.regplot(data = sc_vs_quickness_filt_fill, x = "sc_LR_mean", y= "inverse_quickness", scatter_kws = dict( linewidth=0, s=100))
x = sc_vs_quickness_filt_fill["sc_LR_mean"]
y = sc_vs_quickness_filt_fill["quickenss"]
y_nanremoved = y[~np.isnan(y)]
x_nanremoved = x[~np.isnan(y)]
corr = spearmanr(x_nanremoved,y_nanremoved)
#axes.set(yscale='log') 



corr_r = np.round(corr[0], 2)
corr_p = np.round(corr[1], 8)
axes.set_title(f"{corr_r}, p = {corr_p}")

#plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", "sc_vs_spread_time_all_seizures.pdf"), bbox_inches='tight')     
"""

#%
fig, axes = utils.plot_make(size_length=5)
g = sns.regplot(data = sc_vs_quickness_group_fill, x = "sc_LR_mean", y= "inverse_quickness", scatter_kws = dict( linewidth=0, s=100), ci = None, line_kws=dict(lw = 7))
x = sc_vs_quickness_group_fill["sc_LR_mean"]
y = sc_vs_quickness_group_fill["quickenss"]
y_nanremoved = y[~np.isnan(y)]
x_nanremoved = x[~np.isnan(y)]

corr =spearmanr(x_nanremoved,y_nanremoved)
corr =pearsonr(x_nanremoved,y_nanremoved)
corr_r = np.round(corr[0], 2)
corr_p = np.round(corr[1], 8)
axes.set_title(f"{corr_r}, p = {corr_p}\n{names}")
axes.set_ylim([-0.033,0.2])
for i, tick in enumerate(axes.xaxis.get_major_ticks()):
    tick.label.set_fontsize(6)        
axes.tick_params(width=4) 
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

#%%
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", f"sc_vs_spread_time_patients_{names}.pdf"), bbox_inches='tight')     


utils.fdr2([0.0174, 0.028, 0.91])

utils.fdr2([0.029, 0.0389, 0.91])


utils.fdr2([0.009, 0.0129, 0.91, 0.56, 0.63, 0.007, 0.025])





#%%%
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
#Plots for paper


i=137

sc_vs_quickness = pd.DataFrame(columns =[ "subject", "seizure", "quickenss", "sc_LR_mean", "sc_temporal_mean" ])

RID = np.array(patientsWithseizures["subject"])[i]
seizure = np.array(patientsWithseizures["idKey"])[i]
seizure_length = patientsWithseizures.length[i]

#atlas
atlas = "AAL3v1_1mm"

atlas_names_short =  list(atlas_files["STANDARD"].keys() )
atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]

atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
atlas_label_names = np.array(atlas_label.iloc[1:,1])
atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
#temporal_regions = ["Insula", "Hippocampus", "ParaHippocampal", "Amygdala", "Fusiform" ,"Heschl", "Temporal"]
#temporal_regions = ["Hippocampus", "ParaHippocampal"]
#temporal_regions = ["Hippocampus"]

temporal_region_inds = []
temporal_region_inds_L = []
temporal_region_inds_R = []
for r in range(len(atlas_label_names)):
    bools = []
    for k in range(len(temporal_regions)):
        bools.append(temporal_regions[k] in atlas_label_names[r] )
    if any(bools):
        temporal_region_inds.append(r)
    
    bools_L = []
    bools_R = []
    for k in range(len(temporal_regions)):
        if temporal_regions[k] in atlas_label_names[r] :
            if "_L" in atlas_label_names[r]:
                bools_L.append(temporal_regions[k] in atlas_label_names[r] )
            elif "_R" in atlas_label_names[r]:
                bools_R.append(temporal_regions[k] in atlas_label_names[r] )
        
    if any(bools):
        temporal_region_inds.append(r)
    if any(bools_L):
        temporal_region_inds_L.append(r)
    if any(bools_R):
        temporal_region_inds_R.append(r)
        
temporal_region_numbers = atlas_label_region_numbers[temporal_region_inds].astype(int)
temporal_region_numbers_L = atlas_label_region_numbers[temporal_region_inds_L].astype(int)
temporal_region_numbers_R = atlas_label_region_numbers[temporal_region_inds_R].astype(int)

connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"sub-{RID}.{atlas}.count.pass.connectogram.txt")

connectivity_loc_glob = glob.glob( connectivity_loc  )

#CHECKING IF FILES EXIST
if len(connectivity_loc_glob) > 0:
    connectivity_loc_path = connectivity_loc_glob[0]

    
    sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)

    sc_region_labels = utils.read_DSI_studio_Txt_files_SC_return_regions(connectivity_loc_path, atlas).astype(ind)

    sc_temporal_region_ind = []
    for r in range(len(temporal_region_numbers)):
        ind = np.where(temporal_region_numbers[r] == sc_region_labels)[0][0]
        sc_temporal_region_ind.append(ind)
    sc_temporal_region_ind = np.array(sc_temporal_region_ind)
    
    
    sc_temporal_region_ind_L = []
    for r in range(len(temporal_region_numbers_L)):
        ind = np.where(temporal_region_numbers_L[r] == sc_region_labels)[0][0]
        sc_temporal_region_ind_L.append(ind)
    sc_temporal_region_ind_L = np.array(sc_temporal_region_ind_L)
    
    
    
    sc_temporal_region_ind_R = []
    for r in range(len(temporal_region_numbers_R)):
        ind = np.where(temporal_region_numbers_R[r] == sc_region_labels)[0][0]
        sc_temporal_region_ind_R.append(ind)
    sc_temporal_region_ind_R = np.array(sc_temporal_region_ind_R)
    

    sc_temporal = sc[sc_temporal_region_ind[:,None], sc_temporal_region_ind[None,:]]
    sc_temporal_L_R = sc[sc_temporal_region_ind_L[:,None], sc_temporal_region_ind_R[None,:]]
    







#%%

#sc = sc/sc.max()

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff" ,"#9e2626", "#6161d9", "#1376d8","#1376d8","#0d4f90" , "#173c60" ])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b0ceec","#1376d8","#0d4f90" ,"#0d4f90" , "#173c60" ])

masked_array = np.ma.masked_where(sc <0, sc)
#cmap = copy.copy(plt.get_cmap(cmap_name))
#cmap.set_bad(color='white')
#cmap.set_under('white')

vmin = 0

vmax = 5e3

fig, axes = utils.plot_make()
axes.imshow(masked_array, cmap=cmap, origin="upper", vmin = vmin , vmax = vmax)
axes.set_xticklabels([])
axes.set_yticklabels([])
axes.set_xticks([])
axes.set_yticks([])
axes.axis("off")
pos = axes.imshow(masked_array, cmap=cmap, origin="upper", vmin = vmin , vmax = vmax)
fig.colorbar(pos, ax=axes)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", "heatmap_sc.pdf"), bbox_inches='tight')       

#%%

end = len(sc_temporal)

new_order = np.array(list(np.arange(0,end,2) ) + list(np.arange(1,end,2) ))

sc_temporal_LR =sc_temporal[new_order[:,None], new_order[None,:]]

len(sc_temporal_LR)

fig, axes = utils.plot_make()
axes.imshow(sc_temporal, cmap=cmap, origin="upper", vmin = vmin , vmax = vmax)
axes.set_xticklabels([])
axes.set_yticklabels([])
axes.set_xticks([])
axes.set_yticks([])
axes.axis("off")


pos = axes.imshow(sc_temporal, cmap=cmap, origin="upper", vmin = vmin , vmax = vmax)
fig.colorbar(pos, ax=axes)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", "heatmap_temporal.pdf"), bbox_inches='tight')       

#%%


fig, axes = utils.plot_make()
axes.imshow(sc_temporal_L_R, cmap=cmap, origin="upper", vmin = vmin , vmax = vmax)
axes.set_xticklabels([])
axes.set_yticklabels([])
axes.set_xticks([])
axes.set_yticks([])
axes.axis("off")
pos = axes.imshow(sc_temporal_L_R, cmap=cmap, origin="upper", vmin = vmin , vmax = vmax)
fig.colorbar(pos, ax=axes)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", "heatmap_temporal_LR.pdf"), bbox_inches='tight')       
