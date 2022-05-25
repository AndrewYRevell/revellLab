#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:43:38 2022

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
    #print(np.array(channels)[channel_order])
    return spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels



def calculate_how_many_channels_overlap(soz_channel_names, channel_order_labels, num_to_look_modifier= 1):
    #num_to_look_modifier the multiplier for how many channels to caluclate in top SOZ
    num_to_look  = len(soz_channel_names) * num_to_look_modifier
    if num_to_look > len(channel_order_labels):
        num_to_look = len(channel_order_labels)
    #calculate how many are in the top
    top = 0
    for ch in range(num_to_look):
        channel_to_look = channel_order_labels[ch]
        if channel_to_look in soz_channel_names:
            top = top +1 
    denominator = len(soz_channel_names)
    percentage = top/len(soz_channel_names)
    return top, denominator,  percentage


def remove_EGG_and_ref(channel_order_labels):
    channel_names_new= [x.replace('EEG ', '').replace('-Ref', '').replace(' ', '') for x in channel_order_labels]
    #channel_names_new= echobase.channel2std(channel_names_new)
    
    return np.array(channel_names_new)

def channel2std_ECoG(channel_names_old):
    channel_names__new = []
    for ch in range(len(channel_names_old)):
        txt = channel_names_old[ch]
        
        numbers = re.findall(r'\d+', txt)
        if len(numbers)>0:
            num_txt = re.findall(r'\d+', txt)[0]
            num = int(num_txt)
            pos = txt.find(f"{num_txt}")
            new_num = f"{num:02d}"
            new_txt = f"{txt[:pos] + new_num}"
            channel_names__new.append(new_txt)
    return np.array(channel_names__new)
#%
#% 02 Paths and files
fnameiEEGusernamePassword = paths.IEEG_USERNAME_PASSWORD
metadataDir =  paths.METADATA
fnameJSON = join(metadataDir, "iEEGdataRevell_seizure_severity_joined.json")
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
len(np.unique(patientsWithseizures["subject"]))
print(f"number of patients: {len(np.unique(patientsWithseizures['subject']))}")
print(f"number of seizures: {len(patientsWithseizures)}")            

#%%


  
tanh = True
model_IDs = ["WN","CNN","LSTM" , "absolute_slope", "line_length", "power_broadband"]
by = 0.01


#%%

#get connectivity matrices

i=137
m = 3
thr = [0.69, 0.96, 0.58, 0.08, 0.11, 0.01]
thr = [0.7, 0.7, 0.6, 0.21, 0.5, 0.5]

sc_vs_quickness = pd.DataFrame(columns =[ "subject", "seizure", "quickenss", "sc_LR_mean", "sc_temporal_mean" ])

model_ID = model_IDs[m]

for i in range(len(patientsWithseizures)):
    print(i)
    RID = np.array(patientsWithseizures["subject"])[i]
    seizure = np.array(patientsWithseizures["idKey"])[i]
    idKey = np.array(patientsWithseizures["idKey"])[i]
    seizure_length = patientsWithseizures.length[i]
    
    #atlas
    atlas = "BN_Atlas_246_1mm"
    atlas = "AAL3v1_1mm"
    atlas = "AAL2"
    #atlas = "HarvardOxford-combined"
    
    
    atlas_names_short =  list(atlas_files["STANDARD"].keys() )
    atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
    ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
    atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
    
    atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
    atlas_label_names = np.array(atlas_label.iloc[1:,1])
    atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
    
    #STRUCTURAL CONNECTIVITY
    temporal_regions = ["Hippocampus"]; names = "Hippocampus"
    #temporal_regions = ["Hippocampus", "Temporal"]; names = "Hippocampus, Temporal"
    #temporal_regions = ["Temporal"]; names = "Temporal"
    #temporal_regions = ["Frontal"]; names = "Frontal"
    temporal_regions = ["Parietal"]; names = "Parietal"
    #temporal_regions = ["Amygdala"]; names = "Amygdala"
    #temporal_regions = ["Hippocampus", "Temporal", "Amygdala","Fusiform", "Insula"]; names = "All temporal"
    #temporal_regions = ["Insula", "Hippocampus", "ParaHippocampal", "Fusiform" ,"Heschl", "Temporal"]; names = "All temporal"
    
    #iEEG
    temporal_regions_spread = ["Hippocampus"]
    temporal_regions_spread = ["Hippocampus", "Temporal"]
    #temporal_regions_spread = ["Temporal"]
    #temporal_regions_spread = ["Frontal"]
    #temporal_regions_spread = ["Parietal"]
    #temporal_regions_spread = ["Amygdala"]
    #temporal_regions_spread = ["Hippocampus", "Temporal", "Amygdala","Fusiform", "Insula"]
    #temporal_regions_spread = ["Hippocampus",  "Temporal", "ParaHippocampal", "Fusiform" ,"Heschl" , "Insula"]
    #temporal_regions_spread = ["Hippocampus", "Temporal", "Amygdala", "Insula", "ParaHippocampal", "Fusiform" ,"Heschl"]
    #temporal_regions_spread = ["Hippocampus", "Amygdala","ParaHippocampal"]
    
    #temporal_regions_spread = ["Parietal"]
    
    
    
    #for BNA
    #temporal_regions = ["Hipp", "ITG", "STG", "MTG"]; names = "Hipp, ITG, STG, MTG"
    #temporal_regions = ["Hipp"]; names = "Hipp"
    
    #temporal_regions_spread = ["Hipp", "ITG", "STG", "MTG"]
    
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
                elif "Left" in atlas_label_names[r]:
                    bools_L.append(temporal_regions[k] in atlas_label_names[r] )
                elif "Right" in atlas_label_names[r]:
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
        #sc = sc/sc.max()
        
        sc=utils.log_normalize_adj(sc)
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
        
        
        
        #get the spread quickness
        
       
 
            
        #CHECKING IF SPREAD FILES EXIST
    
        fname = DataJson.get_fname_ictal(RID, "Ictal", idKey, dataset= datasetiEEG, session = session, startUsec = None, stopUsec= None, startKey = "EEC", secondsBefore = secondsBefore, secondsAfter = secondsAfter )
        
        spread_location = join(BIDS, datasetiEEG_spread, f"v{version:03d}", f"sub-{RID}" )
        spread_location_file_basename = f"{splitext(fname)[0]}_spread.pickle"
        spread_location_file = join(spread_location, spread_location_file_basename)
        
        
        feature_name = "absolute_slope"
        location_feature = join(BIDS, datasetiEEG_spread, "single_features", f"sub-{RID}" )
        location_abs_slope_basename = f"{splitext(fname)[0]}_{feature_name}.pickle"
        location_abs_slope_file = join(location_feature, location_abs_slope_basename)
        
        feature_name = "line_length"
        location_line_length_basename = f"{splitext(fname)[0]}_{feature_name}.pickle"
        location_line_length_file = join(location_feature, location_line_length_basename)
        
        feature_name = "power_broadband"
        location_power_broadband_basename = f"{splitext(fname)[0]}_{feature_name}.pickle"
        location_power_broadband = join(location_feature, location_power_broadband_basename)
        
       
        if utils.checkIfFileExists( spread_location_file , printBOOL=False) and utils.checkIfFileExists( location_abs_slope_file , printBOOL=False):
            #print("\n\n\n\nSPREAD FILE EXISTS\n\n\n\n")
        
            if model_ID == "WN" or model_ID == "CNN" or model_ID == "LSTM":
                with open(spread_location_file, 'rb') as f:[probWN, probCNN, probLSTM, data_scalerDS, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
                
            
            if model_ID == "WN":
                #print(model_ID)
                prob_array= probWN
            elif model_ID == "CNN":
                #print(model_ID)
                prob_array= probCNN
            elif model_ID == "LSTM":
                #print(model_ID)
                prob_array= probLSTM
            elif model_ID == "absolute_slope":
                if utils.checkIfFileExists(location_abs_slope_file, printBOOL=False):
                    with open(location_abs_slope_file, 'rb') as f:[abs_slope_normalized, abs_slope_normalized_tanh, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
                    if not tanh:
                        #abs_slope_normalized = utils.apply_arctanh(abs_slope_normalized_tanh)/1e-1 
                        abs_slope_normalized/np.max(abs_slope_normalized)
                        abs_slope_normalized = abs_slope_normalized/np.max(abs_slope_normalized)
                        prob_array=  abs_slope_normalized
                    else:
                        prob_array= abs_slope_normalized_tanh
                else: 
                    print(f"{i} {RID} file does not exist {location_abs_slope_file}\n")
             
            elif model_ID == "line_length":
                if utils.checkIfFileExists(location_line_length_file, printBOOL=False):
                    with open(location_line_length_file, 'rb') as f:[probLL, probLL_tanh, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
                    if not tanh:
                        probLL = probLL/np.max(probLL)
                        prob_array= probLL
                    else:
                        prob_array= probLL_tanh
                else: 
                    print(f"{i} {RID} file does not exist {location_line_length_file}\n")
                   
            elif model_ID == "power_broadband":
                if utils.checkIfFileExists(location_power_broadband, printBOOL=False):
                    with open(location_power_broadband, 'rb') as f:[power_total, power_total_tanh, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
                    if not tanh:
                        #power_total = utils.apply_arctanh(power_total_tanh)/7e-2  
                        power_total = power_total/np.max(power_total)
                        prob_array=  power_total
                        
                    else:
                        prob_array= power_total_tanh
                
                else: 
                    print(f"{i} {RID} file does not exist {location_power_broadband}\n")
            
            else:
                print("model ID not recognized. Using Wavenet")
                prob_array= probWN
            
            #####
            seizure_start = int((secondsBefore-0)/skipWindow)
            seizure_stop = int((secondsBefore + seizure_length)/skipWindow)
            
            
            THRESHOLD = thr[m]
            SMOOTHING = 20
            
            
            probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
            #sns.heatmap( probability_arr_movingAvg.T )      
            #sns.heatmap( probability_arr_threshold.T)    
            spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
       
            
            channel_order_labels = remove_EGG_and_ref(channel_order_labels)
            channels2 = remove_EGG_and_ref(channels)
            
            channel_order_labels = channel2std_ECoG(channel_order_labels)
            channels2 = channel2std_ECoG(channels2)
            
            
            atlas_localization_path = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-{session}", f"sub-{RID}_ses-{session}_desc-atlasLocalization.csv")
            if utils.checkIfFileExists(atlas_localization_path, printBOOL=False):
                atlas_localization = pd.read_csv(atlas_localization_path)
                
                temporal_regions
                
                atlas_localization.channel = channel2std_ECoG(atlas_localization.channel)
                #get channels in hipp
                channels_in_hippocampus = np.zeros(len(atlas_localization))    
                channels_in_hippocampus_label = []
                for r in range(len(atlas_localization)):
                    
                    reg_AAL = atlas_localization[f"{atlas}_label"][r]
                    
                    reg_AAL = atlas_localization.AAL_label[r]
                    reg_BNA = atlas_localization.BN_Atlas_246_1mm_label[r]
                    reg_HO = atlas_localization["HarvardOxford-combined_label"][r]
                    
                    if any([ x in reg_AAL for x in temporal_regions_spread]): 
                        channels_in_hippocampus[r] = 1
                        #left or right
                        if "_L" in reg_AAL or "_L" in reg_BNA or "Left" in reg_HO:
                            channels_in_hippocampus_label.append("left")
                        elif "_R" in reg_AAL or "_R" in reg_BNA or "Right" in reg_HO:
                            channels_in_hippocampus_label.append("right")
                
                
                channel_hipp_ind = np.where(channels_in_hippocampus == 1)[0]
                channel_hipp_label = np.array(atlas_localization.channel[channel_hipp_ind])
                
                
                channel_hipp_index_order = []
                channel_hipp_index_order_start_time = []
                ch_hipp = 1
                for ch_hipp in range(len(channel_hipp_label)):
                    if any(channel_hipp_label[ch_hipp] == channel_order_labels ):
                        channel_hipp_index_order.append(np.where(channel_hipp_label[ch_hipp] == channel_order_labels )[0][0])
                        channel_hipp_index_order_start_time.append(spread_start[channel_order][channel_hipp_index_order[ch_hipp]]*skipWindow )
                        if channel_hipp_index_order_start_time[ch_hipp] > seizure_length: #if the start time is longer than the seizure length, then spread never happened, so set it to nan
                            channel_hipp_index_order_start_time[ch_hipp] = np.nan
                    else:
                        channel_hipp_index_order.append("NONE")
                        channel_hipp_index_order_start_time.append(np.nan)
                        
                  
                channels_in_hippocampus_label_left = []
                channels_in_hippocampus_label_right = []
                for ch_hipp in range(len(channels_in_hippocampus_label)):
                    if "left" in channels_in_hippocampus_label[ch_hipp]:
                        channels_in_hippocampus_label_left.append(ch_hipp)
                    elif "right" in channels_in_hippocampus_label[ch_hipp]:
                        channels_in_hippocampus_label_right.append(ch_hipp)
                        
                hipp_left_mean = np.nanmean( np.array(channel_hipp_index_order_start_time)[channels_in_hippocampus_label_left]  )
                hipp_right_mean = np.nanmean( np.array(channel_hipp_index_order_start_time)[channels_in_hippocampus_label_right]  )
                
                hipp_abs_diff = abs(hipp_left_mean - hipp_right_mean )
            
            
                sc_vs_quickness = sc_vs_quickness.append(dict( subject =RID , seizure = seizure , quickenss = hipp_abs_diff, sc_LR_mean = sc_LR_mean, sc_temporal_mean = sc_temporal_mean) , ignore_index=True )

            
            
            
            
        
        
#%
np.unique(patientsWithseizures["subject"])
"""
sc_vs_quickness = sc_vs_quickness.append(dict( subject ="RID0646" , seizure = "1" , quickenss = 30, sc_LR_mean = 6.222475562257545, sc_temporal_mean = 230.98383312397252) , ignore_index=True )
sc_vs_quickness = sc_vs_quickness.append(dict( subject ="RID0679" , seizure = "1" , quickenss = np.nan, sc_LR_mean = 7.838260215653406, sc_temporal_mean = 238.8341151214537) , ignore_index=True )
sc_vs_quickness = sc_vs_quickness.append(dict( subject ="RID0566" , seizure = "1" , quickenss = np.nan, sc_LR_mean = 7.657777157996711, sc_temporal_mean = 228.95856120168494) , ignore_index=True )
sc_vs_quickness = sc_vs_quickness.append(dict( subject ="RID0529" , seizure = "1" , quickenss = np.nan, sc_LR_mean = 9.619899228344838, sc_temporal_mean = 251.02754366959104) , ignore_index=True )
sc_vs_quickness = sc_vs_quickness.append(dict( subject ="RID0520" , seizure = "1" , quickenss = np.nan, sc_LR_mean = 6.274218047654939, sc_temporal_mean = 223.95286464225373) , ignore_index=True )

sc_vs_quickness = sc_vs_quickness.append(dict( subject ="RID0394" , seizure = "1" , quickenss = 5, sc_LR_mean = 11.08268999550216, sc_temporal_mean = 261.9815342752608) , ignore_index=True )
"""
aaaaaaaaaaa_copy = copy.deepcopy(sc_vs_quickness)
sc_vs_quickness      

unilateral_ids = ["RID0522", "RID0508" , "RID0572" , "RID0583"  , "RID0596" , "RID0648" , "RID0679" , "RID0566" , "RID0529" , "RID0520" , ]


ind1 = list(np.where((sc_vs_quickness["subject"] == "RID0278"  ) &(sc_vs_quickness["seizure"] == "1"  ) )[0]) #artifact seizure
ind2 = list(np.where((sc_vs_quickness["subject"] == "RID0365"  )  )[0]) #artifact seizure

sc_vs_quickness.loc[(sc_vs_quickness["subject"] == "RID0454"  ) &(sc_vs_quickness["seizure"] == "1"  ), "quickenss"] = 70 #artifact seizure #453 asctually took 70 seconds to spread to right side. It was a very long seizure and a typical

#ind4 = list(np.where((sc_vs_quickness["subject"] == "RID0522"  ))[0]) #not bilateral implant
#ind5 = list(np.where((sc_vs_quickness["subject"] == "RID0508"  ))[0]) #not bilateral implant
#ind6 = list(np.where((sc_vs_quickness["subject"] == "RID0572"  ))[0]) #not bilateral implant
#ind7 = list(np.where((sc_vs_quickness["subject"] == "RID0583"  ))[0]) #not bilateral implant
#ind8 = list(np.where((sc_vs_quickness["subject"] == "RID0595"  ))[0]) #not bilateral implant
#ind9 = list(np.where((sc_vs_quickness["subject"] == "RID0596"  ))[0]) #not bilateral implant
#ind10 = list(np.where((sc_vs_quickness["subject"] == "RID0648"  ))[0]) #not bilateral implant

#ind11 = list(np.where((sc_vs_quickness["subject"] == "RID0440"  ))[0]) #not bilateral implant
#ind12 = list(np.where((sc_vs_quickness["subject"] == "RID0679"  ))[0]) #not bilateral implant
#ind13 = list(np.where((sc_vs_quickness["subject"] == "RID0566"  ))[0]) #not bilateral implant
#ind14 = list(np.where((sc_vs_quickness["subject"] == "RID0529"  ))[0]) #not bilateral implant
#ind15 = list(np.where((sc_vs_quickness["subject"] == "RID0520"  ))[0]) #not bilateral implant

#ind16 = list(np.where((sc_vs_quickness["subject"] == "RID0646"  ))[0]) #not bilateral implant
#ind3 = list(np.where((sc_vs_quickness["subject"] == "RID0365"  ))[0]) #a very clear outlier
#ind12 = list(np.where((sc_vs_quickness["subject"] == "RID0502"  ))[0]) 


sc_vs_quickness_unilateral = sc_vs_quickness[sc_vs_quickness["subject"].isin(unilateral_ids)     ]

inds = ind1
#inds = ind2 + ind4 +ind5 +ind6 +ind7  +ind9 +ind10  + ind11 + ind12 + ind13 + ind14 + ind15 + ind16
#inds = ind1 +ind2 + ind4 +ind5 +ind6 +ind7  +ind9 +ind10  + ind11 + ind12 + ind13 + ind14 + ind15 + ind16

sc_vs_quickness_filt = copy.deepcopy(sc_vs_quickness)
sc_vs_quickness_filt = sc_vs_quickness.drop(inds)  
sc_vs_quickness_filt.drop(sc_vs_quickness_filt[sc_vs_quickness_filt["subject"].isin(unilateral_ids)].index , inplace = True)

#sc_vs_quickness_filt = sc_vs_quickness.drop(inds)  
aaaaaa = copy.deepcopy(sc_vs_quickness_filt)



sc_vs_quickness_filt["inverse_quickness"] = 1/sc_vs_quickness_filt["quickenss"]


sc_vs_quickness_filt_group = sc_vs_quickness_filt.groupby(["subject"], as_index=False).median()

aaaaa_group = copy.deepcopy(sc_vs_quickness_filt_group)

sc_vs_quickness_filt_fill= sc_vs_quickness_filt.fillna(0)
sc_vs_quickness_group_fill= sc_vs_quickness_filt_group.fillna(0)

"""
fig, axes = utils.plot_make(size_length=10)
g = sns.regplot(data = sc_vs_quickness_filt_group, x = "sc_LR_mean", y= "quickenss", scatter_kws = dict( linewidth=0, s=100))
x = sc_vs_quickness_filt_group["sc_LR_mean"]
y = sc_vs_quickness_filt_group["quickenss"]
y_nanremoved = y[~np.isnan(y)]
x_nanremoved = x[~np.isnan(y)]
corr = spearmanr(x_nanremoved,y_nanremoved)
#axes.set(yscale='log') 

corr_r = np.round(corr[0], 2)
corr_p = np.round(corr[1], 8)
axes.set_title(f"{corr_r}, p = {corr_p}")




"""

non_spreader_ind = np.where(np.isnan( sc_vs_quickness_filt_group["quickenss"])  )[0]
spreader_ind = np.where(~np.isnan( sc_vs_quickness_filt_group["quickenss"])  )[0]

non_spreaders = sc_vs_quickness_filt_group["sc_LR_mean"][non_spreader_ind]
spreaders = sc_vs_quickness_filt_group["sc_LR_mean"][spreader_ind]
non_spreaders_vs_spreaders =  stats.mannwhitneyu(  non_spreaders, spreaders)
print(f"\n\n\nnon spreaders vs spreaders = \n{non_spreaders_vs_spreaders[1]}\n\n")

df_non_spreaders_vs_spreaders = copy.deepcopy(sc_vs_quickness_filt_group)
df_non_spreaders_vs_spreaders["spreaders"] = np.nan

df_non_spreaders_vs_spreaders.loc[non_spreader_ind,"spreaders"]  = "non_spreaders"
df_non_spreaders_vs_spreaders.loc[spreader_ind,"spreaders"]  = "spreaders"
sc_vs_quickness_unilateral_group = sc_vs_quickness_unilateral.groupby(["subject"], as_index=False).median()
sc_vs_quickness_unilateral_group["spreaders"] = "unilateral implant"

df_non_spreaders_vs_spreaders_unilateral = pd.concat(  [df_non_spreaders_vs_spreaders, sc_vs_quickness_unilateral_group])

fig, axes = utils.plot_make(c = 3, size_length=15)
sns.boxplot(ax = axes[0], data = df_non_spreaders_vs_spreaders_unilateral, x = "spreaders", y =  "sc_LR_mean", order= ["spreaders", "non_spreaders", "unilateral implant"] )
sns.swarmplot(ax = axes[0],data = df_non_spreaders_vs_spreaders_unilateral, x = "spreaders", y =  "sc_LR_mean", order= ["spreaders", "non_spreaders", "unilateral implant"] )

unilateral_sc = sc_vs_quickness_unilateral_group["sc_LR_mean"]
non_spreaders_vs_unilateral =  stats.mannwhitneyu(  non_spreaders, unilateral_sc)
spreaders_vs_unilateral =  stats.mannwhitneyu(  spreaders, unilateral_sc)

utils.fdr2([non_spreaders_vs_spreaders[1] , non_spreaders_vs_unilateral[1],  spreaders_vs_unilateral[1] ])
axes[0].set_title(f"SvsNS={non_spreaders_vs_spreaders[1]}\nNSvsUN={non_spreaders_vs_unilateral[1]}\nSvsUN={spreaders_vs_unilateral[1]}")

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
sns.regplot(ax = axes[1], data = sc_vs_quickness_group_fill, x = "sc_LR_mean", y= "inverse_quickness", scatter_kws = dict( linewidth=0, s=100), ci = None, line_kws=dict(lw = 7))

#g = sns.regplot(data = sc_vs_quickness_group_fill[~np.isnan(sc_vs_quickness_filt_group["inverse_quickness"])], x = "sc_LR_mean", y= "inverse_quickness", scatter_kws = dict( linewidth=0, s=100), ci = None, line_kws=dict(lw = 7))
#g = sns.scatterplot(data = sc_vs_quickness_group_fill, x = "sc_LR_mean", y= "inverse_quickness", linewidth=0, s=100)


sc_vs_quickness_group_fill_spreaders_only = sc_vs_quickness_group_fill[~np.isnan(sc_vs_quickness_filt_group["inverse_quickness"])]

x = sc_vs_quickness_group_fill["sc_LR_mean"]
y = sc_vs_quickness_group_fill["inverse_quickness"]
y_nanremoved = y[~np.isnan(y)]
x_nanremoved = x[~np.isnan(y)]

corr = spearmanr(x_nanremoved,y_nanremoved)
corr = pearsonr(x_nanremoved,y_nanremoved)
corr_r = np.round(corr[0], 2)
corr_p = np.round(corr[1], 8)

slope, intercept, r_value, p_value, std_err  = stats.linregress(x, y)
r_value_round = np.round(r_value, 2)
p_value_round = np.round(p_value, 5)


axes[1].set_title(f"r: {r_value_round}, p = {p_value_round}\n{names}")
#axes.set_ylim([-0.033,0.2])
for i, tick in enumerate(axes[1].xaxis.get_major_ticks()):
    tick.label.set_fontsize(6)        
axes[1].tick_params(width=4) 
# change all spines
for axis in ['top','bottom','left','right']:
    axes[1].spines[axis].set_linewidth(6)



#%

POWER = 1.1
from sklearn.linear_model import TweedieRegressor
X = np.array(x).reshape(-1,1)
Y = np.array(y)


pr = TweedieRegressor(power = POWER, alpha=0, fit_intercept=True)
X_new = np.linspace(X.min(), X.max()).reshape(-1,1)
y_pred_pr = pr.fit(X, Y).predict(np.linspace(X.min(), X.max()).reshape(-1,1))




sns.scatterplot(ax = axes[2], data = sc_vs_quickness_group_fill, x = "sc_LR_mean", y= "inverse_quickness", linewidth=0, s=100)
sns.lineplot(ax = axes[2],x = X_new.flatten(), y = y_pred_pr, lw = 5)

pr.score(X, Y)

X2 = sm.add_constant(X)
glm = sm.GLM(Y, X2, family=sm.families.Tweedie(var_power = POWER))
glm_fit = glm.fit()
print(glm_fit.summary())

axes[2].set_title(f"D = {pr.score(X, Y)}\np={glm_fit.pvalues[1]}")
glm_fit.pvalues

#Y2 = glm.predict(glm_fit.params)
#fig, axes = utils.plot_make(size_length=5)
#sns.scatterplot(data = sc_vs_quickness_group_fill, x = "sc_LR_mean", y= "inverse_quickness", linewidth=0, s=100)
#sns.lineplot(x = X.flatten(), y = Y2)





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
