#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:11:06 2022

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


fname_patinet_localization = join(metadataDir, "patient_localization_final.mat")
RID_HUP = join(metadataDir, "RID_HUP.csv")
outcomes_fname = join(metadataDir, "patient_cohort_all_atlas.csv")

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

patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(fname_patinet_localization)
RID_HUP = pd.read_csv(RID_HUP)
with open(fnameJSON) as f: jsonFile = json.load(f)


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
                  
         


#%%

i=0
type_of_overlap = "soz"
override_soz= True
def calculate_mean_rank_deep_learning(i, patientsWithseizures, version=11, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, type_of_overlap = "soz", override_soz = False, seconds_active  = None):
    #override_soz if True, then if there are no soz marking, then use the resection markings and assume those are SOZ contacts
    RID = np.array(patientsWithseizures["subject"])[i]
    idKey = np.array(patientsWithseizures["idKey"])[i]
    seizure_length = patientsWithseizures.length[i]
    
    
    #CHECKING IF SPREAD FILES EXIST

    fname = DataJson.get_fname_ictal(RID, "Ictal", idKey, dataset= datasetiEEG, session = session, startUsec = None, stopUsec= None, startKey = "EEC", secondsBefore = secondsBefore, secondsAfter = secondsAfter )
    
    spread_location = join(BIDS, datasetiEEG_spread, f"v{version:03d}", f"sub-{RID}" )

    
    spread_location_file_basename = f"{splitext(fname)[0]}_spread.pickle"
    spread_location_file = join(spread_location, spread_location_file_basename)
    
    if utils.checkIfFileExists( spread_location_file , printBOOL=False):
        #print("\n\n\n\nSPREAD FILE EXISTS\n\n\n\n")
        with open(spread_location_file, 'rb') as f:[probWN, probCNN, probLSTM, data_scalerDS, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
    
    
    
        #Getting SOZ labels
        RID_keys =  list(jsonFile["SUBJECTS"].keys() )
        hup_num_all = [jsonFile["SUBJECTS"][x]["HUP"]  for  x   in  RID_keys]
        
        hup_int = hup_num_all[RID_keys.index(RID)]
        hup_int_pad = f"{hup_int:03d}" 
        
        i_patient = patients.index(f"HUP{hup_int_pad}")
        HUP = patients[i_patient]
        hup = int(HUP[3:])
        
    
        
        channel_names = labels[i_patient]
        soz_ind = np.where(soz[i_patient] == 1)[0]
        soz_channel_names = np.array(channel_names)[soz_ind]
        
        resected_ind = np.where(resect[i_patient] == 1)[0]
        resected_channel_names = np.array(channel_names)[resected_ind]
        
        ignore_ind = np.where(ignore[i_patient] == 1)[0]
        ignore__channel_names = np.array(channel_names)[ignore_ind]
        
        soz_channel_names = echobase.channel2std(soz_channel_names)
        resected_channel_names = echobase.channel2std(resected_channel_names)
        #ignore__channel_names = echobase.channel2std(ignore__channel_names)
        
        
       
     
        soz_channel_names = channel2std_ECoG(soz_channel_names)
        resected_channel_names = channel2std_ECoG(resected_channel_names)
        ignore__channel_names = channel2std_ECoG(ignore__channel_names)
        #####
        seizure_start = int((secondsBefore-0)/skipWindow)
        seizure_stop = int((secondsBefore + seizure_length)/skipWindow)
        #%
        THRESHOLD = threshold
        SMOOTHING = smoothing #in seconds
        
        if model_ID == "WN":
            #print(model_ID)
            prob_array= probWN
        elif model_ID == "CNN":
            #print(model_ID)
            prob_array= probCNN
        elif model_ID == "LSTM":
            #print(model_ID)
            prob_array= probLSTM
        else:
            print("model ID not recognized. Using Wavenet")
            prob_array= probWN
        
        
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        #sns.heatmap( probability_arr_movingAvg.T , cbar=False)      
        #sns.heatmap( probability_arr_threshold.T , cbar=False)    
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
   
        
        channel_order_labels = remove_EGG_and_ref(channel_order_labels)
        channels2 = remove_EGG_and_ref(channels)
        
        channel_order_labels = channel2std_ECoG(channel_order_labels)
        channels2 = channel2std_ECoG(channels2)
        
        #print(soz_channel_names)
        #print(resected_channel_names)
        #print(channel_order_labels)
    
    
        #remove ignore electrodes from channel_order_labels
        ignore_index = np.intersect1d(  channel_order_labels, ignore__channel_names, return_indices=True)
        channel_order_labels[-ignore_index[1]]
        channel_order_labels = np.delete(channel_order_labels, ignore_index[1])
        
        
        
        #calculate ranks of SOZ
    
        if type_of_overlap == "soz":
            mean_rank = np.mean(np.intersect1d(  soz_channel_names, channel_order_labels, return_indices=True)[2])
            median_rank = np.median(np.intersect1d(  soz_channel_names, channel_order_labels, return_indices=True)[2])
            mean_rank_percent = mean_rank/len(channel_order_labels)
            median_rank_percent = median_rank/len(channel_order_labels)
            
        
            if len(soz_channel_names) <=0 and override_soz:
                mean_rank = np.mean(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
                median_rank = np.median(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
                mean_rank_percent = mean_rank/len(channel_order_labels)
                median_rank_percent = median_rank/len(channel_order_labels)
            
            if len(np.intersect1d(  soz_channel_names, channel_order_labels)) ==0:
                mean_rank = np.mean(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
                median_rank = np.median(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
                mean_rank_percent = mean_rank/len(channel_order_labels)
                median_rank_percent = median_rank/len(channel_order_labels)
                #print(f"\n\n\n{RID}:  {soz_channel_names} FUCKKKKKK")
            
            
            #print(f"\n SOZ:{i} {RID} {HUP} {median_rank_percent}")
        elif type_of_overlap == "resection":
           
 
            mean_rank = np.mean(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
            median_rank = np.median(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
            mean_rank_percent = mean_rank/len(channel_order_labels)
            median_rank_percent = median_rank/len(channel_order_labels)
            
            if len(resected_channel_names) <=0 and override_soz:
                mean_rank = np.mean(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
                median_rank = np.median(np.intersect1d(  resected_channel_names, channel_order_labels, return_indices=True)[2])
                mean_rank_percent = mean_rank/len(channel_order_labels)
                median_rank_percent = median_rank/len(channel_order_labels)
            #print(f"\n Resection: {i} {RID} {HUP} {median_rank_percent}")
            
        else:
            print(f"type_of_overlap not recognized: {type_of_overlap}\nMust be 'soz' or 'resection'")
        
        ####################################
        
        
        #calculate active at t time
        
        if seconds_active is None:
            seconds = np.arange(0,60*2+1,1)
        else:
            seconds = seconds_active
        percent_active_vec = np.zeros(len(seconds))
        percent_active_vec[:] = np.nan
        active_during_seizure = probability_arr_threshold[seizure_start:seizure_stop,:]
        for s in range(len(seconds)):
            sec = seconds[s]
            sec= int(sec/skipWindow)
            if sec < len(active_during_seizure):
                percent_active_vec[s] = sum(active_during_seizure[sec,:] == 1)/active_during_seizure.shape[1]
            
        #sns.lineplot(x = seconds, y=percent_active_vec)
        
        ###############################
        ###############################
        ###############################
        #calculate quickness of spread between both hippocampus
        
        ###############################
        
        
        patients, labels, ignore, resect, gm_wm, coords, region, soz
        
        regions_patient = region[i_patient]

        channels_in_hippocampus = np.zeros(len(channel_names))    
    
        r=4
        len(channel_names)
        channels_in_hippocampus_label = []
        for r in range(len(regions_patient)):
            if len(regions_patient[r][0])>0:
                reg = regions_patient[r][0][0]
                if "Hippocampus" in reg or "hippocamp" in reg or "temporal" in reg: #also include parahippocampus
                    channels_in_hippocampus[r] = 1
                    channels_in_hippocampus_label.append(reg)
            
        
        channel_hipp_ind = np.where(channels_in_hippocampus == 1)[0]
        
        
        #find the average start time for left and right hippocampus each (since multiple contacts can be in each).
        
        channel_hipp_label = np.array(channel_names)[channel_hipp_ind]
        channel_hipp_label = channel2std_ECoG(channel_hipp_label)
        
        
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
            if "Left" in channels_in_hippocampus_label[ch_hipp]:
                channels_in_hippocampus_label_left.append(ch_hipp)
            elif "Right" in channels_in_hippocampus_label[ch_hipp]:
                channels_in_hippocampus_label_right.append(ch_hipp)
                
        hipp_left_mean = np.nanmean( np.array(channel_hipp_index_order_start_time)[channels_in_hippocampus_label_left]  )
        hipp_right_mean = np.nanmean( np.array(channel_hipp_index_order_start_time)[channels_in_hippocampus_label_right]  )

        
        
        
        #Override becuase the localization data above is garbage in some patients. Getting my own localization data, and use that because it is more accurate
        if True:
            atlas_localization_path = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-{session}", f"sub-{RID}_ses-{session}_desc-atlasLocalization.csv")
            if utils.checkIfFileExists(atlas_localization_path, printBOOL=False):
                atlas_localization = pd.read_csv(atlas_localization_path)
                
                
                atlas_localization.channel = channel2std_ECoG(atlas_localization.channel)
                #get channels in hipp
                channels_in_hippocampus = np.zeros(len(atlas_localization))    
                channels_in_hippocampus_label = []
                for r in range(len(atlas_localization)):
                    reg_AAL = atlas_localization.AAL_label[r]
                    reg_BNA = atlas_localization.BN_Atlas_246_1mm_label[r]
                    reg_HO = atlas_localization["HarvardOxford-combined_label"][r]
            
                    
                    if "Hippocamp" in reg_AAL or "hippocamp" in reg_BNA or "Hippocampus" in reg_HO or "hippocamp" in reg_HO or "Temporal" in reg_AAL or "MTG" in reg_BNA or "ITG" in reg_BNA or "STG" in reg_BNA or "Temporal" in reg_HO: #this also includes parrahippocampus
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
        
            
            
        
        
        
        
        return RID, idKey, len(soz_channel_names), len(channel_order_labels), mean_rank, median_rank, mean_rank_percent, median_rank_percent, seconds, percent_active_vec, hipp_left_mean,  hipp_right_mean
    
#%%    
model_ID="WN"
threshold = 0.65

soz_overlap_single = pd.DataFrame(columns = ["subject", "seizure","number_channels_soz","number_channels", "mean_rank", "median_rank", "mean_rank_percent", "median_rank_percent",  "hipp_left_mean",  "hipp_right_mean"])

seconds_active = np.arange(0,60*2+1,1)
percent_active = pd.DataFrame(data = seconds_active, columns = ["time"])

columns = ["subject", "seizure"]
columns + list(seconds_active)
percent_active = pd.DataFrame(columns = columns + list(seconds_active))

for i in range(len(patientsWithseizures)):
    print(f"\r{i}   { np.round(   (i+1)/len(patientsWithseizures)*100   ,2)    }                   ", end = "\r")
    RID, idKey, number_channels_soz,number_channels, mean_rank, median_rank, mean_rank_percent, median_rank_percent, seconds, percent_active_vec,  hipp_left_mean,  hipp_right_mean = calculate_mean_rank_deep_learning(i, patientsWithseizures, version=11, threshold=threshold, smoothing = 20, model_ID=model_ID, secondsAfter=180, secondsBefore=180, type_of_overlap = "soz", override_soz = True, seconds_active = seconds_active)
    
    
    
    soz_overlap_single= soz_overlap_single.append(dict(subject=RID, 
                                                       seizure = idKey, 
                                                       number_channels_soz = number_channels_soz, 
                                                       number_channels = number_channels, 
                                                       mean_rank=mean_rank, 
                                                       median_rank= median_rank, 
                                                       mean_rank_percent = mean_rank_percent, 
                                                       median_rank_percent = median_rank_percent,
                                                       hipp_left_mean = hipp_left_mean,  
                                                       hipp_right_mean = hipp_right_mean ), ignore_index=True)

    percent_active = percent_active.append(dict(subject=RID, seizure = idKey), ignore_index=True)
    percent_active.iloc[i,2:] = percent_active_vec




soz_overlap_single_median = soz_overlap_single.groupby(["subject","number_channels"], as_index=False).median()

soz_overlap_single_outcomes = pd.merge(soz_overlap_single, outcomes, on='subject')
soz_overlap_single_median_outcomes = pd.merge(soz_overlap_single_median, outcomes, on='subject')
#%%






np.nanmean(soz_overlap_single_outcomes.mean_rank_percent )
np.nanmean(soz_overlap_single_outcomes.median_rank_percent )


sns.swarmplot(data = soz_overlap_single_outcomes, x = "Engel_24_mo_binary", y = "mean_rank_percent", hue= "Gender")

sns.swarmplot(data = soz_overlap_single_outcomes, y = "median_rank_percent")


sns.swarmplot(data = soz_overlap_single_median_outcomes, y = "mean_rank_percent", x = "Engel_24_mo_binary")
sns.swarmplot(data = soz_overlap_single_median_outcomes, y = "median_rank_percent", x = "Engel_24_mo_binary")

sns.regplot(data = soz_overlap_single_median_outcomes, x = "Engel_24_mo_binary", y = "median_rank_percent" )

sns.swarmplot(data = soz_overlap_single_median_outcomes, x = "Gender", y = "mean_rank_percent", hue= "Engel_24_mo_binary")

np.nanmean(soz_overlap_single_median_outcomes.mean_rank_percent )
np.nanmean(soz_overlap_single_median_outcomes.median_rank_percent )


np.nanmedian(soz_overlap_single_median_outcomes.mean_rank_percent )
np.nanmedian(soz_overlap_single_median_outcomes.median_rank_percent )



#%% Outcomes vs percent active at T time 

df = percent_active.melt( id_vars = ["subject", "seizure"], var_name = "time" , value_name= "percent_active")
df = df.fillna(np.nan)
df = df.groupby(["subject","time"], as_index=False).median()


df = pd.merge(df, outcomes, on='subject')
df.fillna(np.nan)

category = "Engel_24_mo_binary"
df_filtered = df[(df[category] != "unknown")]
df_filtered_90s = df[(df[category] != "unknown") & (df["time"] <= 90)]

ax = sns.lineplot(data = df_filtered_90s.fillna(np.nan), x = "time", y = "percent_active", hue = "Engel_24_mo_binary" , ci = 68)
plt.show()
cohensd = np.zeros(len(seconds_active))
outcomes_stats = np.zeros(len(seconds_active))

for tt in range(len(seconds_active)):
    
    
    tmp = df_filtered[df_filtered["time"] == seconds_active[tt]]
    tmp=tmp.dropna()
    tmp_good = tmp[tmp["Engel_24_mo_binary"] == "good"]
    tmp_poor = tmp[tmp["Engel_24_mo_binary"] == "poor"]
    
    v1 = tmp_good["percent_active"]
    v2 = tmp_poor["percent_active"]
    print(f" {tt}  {stats.mannwhitneyu(v1, v2)[1]},                {stats.ttest_ind(v1, v2)[1]}")
    
    outcomes_stats[tt] = stats.mannwhitneyu(v1, v2)[1]
    
    cohensd[tt] = utils.cohend(v1, v2)
    cohensd[tt] = utils.cohend2(v1, v2)
    
    #sns.boxplot(data = tmp, x = "Engel_24_mo_binary", y= "percent_active")
    #sns.swarmplot(data = tmp, x = "Engel_24_mo_binary", y= "percent_active")


sns.lineplot(x = seconds_active[0:90], y = cohensd[0:90])
plt.show()

print(outcomes_stats[30])
print(outcomes_stats[31])
print(outcomes_stats[32])


#%% How quickly spread from one hipp to another

#count how many have hippocampus spread < X seconds

for co  in np.arange(5,70,5):
    threshold_hippocampus_spread = co
    
    hipp_spread_time = abs(soz_overlap_single_median_outcomes.hipp_left_mean - soz_overlap_single_median_outcomes.hipp_right_mean)
    
    hipp_spread = copy.deepcopy(soz_overlap_single_median_outcomes)
    
    hipp_spread["hipp_spread_time"] = hipp_spread_time
    
    tmp1 = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") ]
    tmp2 =hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") ]
    
    good_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good")])
    poor_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor")])
    
    hipp_spread_good = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread)]
    hipp_spread_poor = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread)]
    
    hipp_spread_good_len = len(hipp_spread_good)
    hipp_spread_poor_len = len(hipp_spread_poor)
    
    """
    Contingency table
          Hipp spread less than threshold   |    Never Spread
    good   hipp_spread_good_len             |    good_len - hipp_spread_good_len
    poor   hipp_spread_poor_len             |      poor_len - hipp_spread_poor_len
    """
    
    
    
    contingency_table = [ [hipp_spread_good_len, good_len - hipp_spread_good_len] , [hipp_spread_poor_len,  poor_len - hipp_spread_poor_len]    ]
    
    pval1 = stats.chi2_contingency(contingency_table, correction=True)[1]
    
    
    
    
    #Only consider temporal lobe epilepsy
    good_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))])
    poor_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))])
    
    tmp1 = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    tmp2 =hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    
    hipp_spread_good = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread) & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    hipp_spread_poor = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread) & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    
    contingency_table = [ [hipp_spread_good_len, good_len - hipp_spread_good_len] , [hipp_spread_poor_len,  poor_len - hipp_spread_poor_len]    ]
    
    pval2 = stats.chi2_contingency(contingency_table, correction=True)[1]

    print(f"{co}: {pval1}, {pval2}")



#%% Compare over thresholds


thresholds_median = pd.DataFrame(columns = ["model","version", "threshold", "mean", "median", "sd"])
soz_overlap = pd.DataFrame(columns = ["model","version", "threshold", "subject", "seizure", "number_channels_soz","number_channels", "mean_rank", "median_rank", "mean_rank_percent", "median_rank_percent",  "hipp_left_mean",  "hipp_right_mean"])

pd.DataFrame(columns = ["model","version", "threshold", "subject", "seizure", "number_channels", "mean_rank", "median_rank", "mean_rank_percent", "median_rank_percent"])

seconds_active = np.arange(0,60*2+1,1)
percent_active = pd.DataFrame(data = seconds_active, columns = ["time"])

columns = ["model","version", "threshold", "subject", "seizure"]
columns + list(seconds_active)
percent_active = pd.DataFrame(columns = columns + list(seconds_active))


model_ID = "WN"
version = 11

by = 0.01
np.linspace(0.5,1,10, endpoint=False)
thresholds = np.arange(0.4,1 + by, by)
thresholds = np.round(thresholds,2)

model_IDs = ["WN","CNN","LSTM" ]
count = 0
for m in range(len(model_IDs)):
    model_ID =model_IDs[m]
    version = 11
    
    for t in thresholds:
        print(f"\n\n{model_ID}: {t}      ")
        
        for i in range(len(patientsWithseizures)):
            print(f"\r{i}   { np.round(   (i+1)/len(patientsWithseizures)*100   ,2)    }%                   ", end = "\r")
            RID, idKey, number_channels_soz,number_channels, mean_rank, median_rank, mean_rank_percent, median_rank_percent, seconds, percent_active_vec,  hipp_left_mean,  hipp_right_mean = calculate_mean_rank_deep_learning(i, patientsWithseizures, version=version, threshold=t, smoothing = 20, model_ID=model_ID, secondsAfter=180, secondsBefore=180, type_of_overlap = "soz", override_soz = True)
            
            soz_overlap= soz_overlap.append(dict(model = model_ID , version = version, threshold = t, subject=RID, seizure = idKey, number_channels = number_channels, mean_rank=mean_rank, median_rank= median_rank, mean_rank_percent = mean_rank_percent, median_rank_percent = median_rank_percent, hipp_left_mean = hipp_left_mean,  hipp_right_mean = hipp_right_mean), ignore_index=True)
            
            percent_active = percent_active.append(dict(model = model_ID , version = version, threshold = t, subject=RID, seizure = idKey), ignore_index=True)
            percent_active.iloc[count,5:] = percent_active_vec
            count = count +1
            

##########################
##########################
##########################

#%%
soz_overlap_median = soz_overlap.groupby(['model', 'subject', "threshold"], as_index=False).median()


soz_overlap_outcomes = pd.merge(soz_overlap, outcomes, on='subject')
soz_overlap_median_outcomes = pd.merge(soz_overlap_median, outcomes, on='subject')

fig, axes = utils.plot_make(size_length=10)
sns.lineplot(data = soz_overlap_median, x = "threshold", y = "median_rank_percent", hue="model", ci=95, estimator=np.median, ax = axes, hue_order=["WN", "CNN", "LSTM"])
#sns.lineplot(data = soz_overlap_median, x = "threshold", y = "median_rank_percent", hue="model", err_style="bars", ci=95, estimator=np.median, ax = axes)
axes.set_ylim([0,0.65])


find_lowest = soz_overlap_median.groupby(["model", "threshold"], as_index=False).median()
for m in range(len(model_IDs)):
    model_ID = model_IDs[m]
    lowest = np.min(find_lowest[find_lowest["model"] == model_ID]["median_rank_percent"])
    lowest_ind = np.where(find_lowest[find_lowest["model"] == model_ID]["median_rank_percent"] == lowest)[0][0]
    lowest_threshold = np.array(find_lowest[find_lowest["model"] == model_ID]["threshold"])[lowest_ind]
    print(lowest_threshold)
#%%

#round threshold value cause python is so stupid with 0.5 will sometime be 0.50000001. wtf
for t in range(len(percent_active)):
    percent_active["threshold"][t] = int(np.round(percent_active["threshold"][t],2)*100)/100


"""
tmp = copy.deepcopy(soz_overlap_median_outcomes)
tmp.columns


model_ID = "LSTM"


tmp = tmp.loc[(tmp['model'] ==model_ID) & (tmp['threshold'] == thresholds[np.round(thresholds, 2) == 0.7][0])]


tmp = tmp.loc[(tmp['model'] ==model_ID)]
tmp = tmp.loc[(tmp['threshold'] >= 0.39)]


sns.lineplot(data = tmp, x = "threshold", y = "median_rank_percent" , hue = "Engel_24_mo")

"""

#%%




#%% Outcomes vs percent active at T time 

df = percent_active.melt( id_vars = ["model","version", "subject", "seizure", "threshold"], var_name = "time" , value_name= "percent_active")
df = df.drop(["version"], axis = 1)
df = df.fillna(np.nan)
df = df.groupby(["model", "subject","time", "threshold"], as_index=False).median()


sns.lineplot(x = seconds,y = percent_active.iloc[400,5:])

df = pd.merge(df, outcomes, on='subject')
df = df.fillna(np.nan)



thresh = 0.95
m=1
model_ID = model_IDs[m]

category = "Engel_24_mo_binary"

df_thresh = df[( df["threshold"]== thresh )]
df_model =  df_thresh[( df_thresh["model"] ==model_ID )]

df_filtered = df_model[(df_model[category] != "unknown")]
df_filtered_90s = df_model[(df_model[category] != "unknown") & (df_model["time"] <= 90)]



ax = sns.lineplot(data = df_filtered_90s.fillna(np.nan), x = "time", y = "percent_active", hue = "Engel_24_mo_binary" , ci = 68)
plt.show()
cohensd = np.zeros(len(seconds_active))
outcomes_stats = np.zeros(len(seconds_active))

for tt in range(len(seconds_active)):
    
    
    tmp = df_filtered[df_filtered["time"] == seconds_active[tt]]
    tmp=tmp.dropna()
    tmp_good = tmp[tmp["Engel_24_mo_binary"] == "good"]
    tmp_poor = tmp[tmp["Engel_24_mo_binary"] == "poor"]
    
    v1 = tmp_good["percent_active"]
    v2 = tmp_poor["percent_active"]
    
    if np.round(stats.mannwhitneyu(v1, v2)[1],2) <0.05:
        extra = "****"
    else:
        extra = ""
    print(f" {seconds_active[tt]}  {np.round(stats.mannwhitneyu(v1, v2)[1],2)},     {np.round(stats.ttest_ind(v1, v2)[1],2)}    {extra}")
    
    outcomes_stats[tt] = stats.mannwhitneyu(v1, v2)[1]
    
    cohensd[tt] = utils.cohend(v1, v2)
    cohensd[tt] = utils.cohend2(v1, v2)
    
    #sns.boxplot(data = tmp, x = "Engel_24_mo_binary", y= "percent_active")
    #sns.swarmplot(data = tmp, x = "Engel_24_mo_binary", y= "percent_active")


sns.lineplot(x = seconds_active[0:90], y = cohensd[0:90])
plt.show()

print(outcomes_stats[30])
print(outcomes_stats[31])
print(outcomes_stats[32])


#%% How quickly spread from one hipp to another

#count how many have hippocampus spread < X seconds

for co  in np.arange(5,70,5):
    threshold_hippocampus_spread = co
    
    hipp_spread_time = abs(soz_overlap_single_median_outcomes.hipp_left_mean - soz_overlap_single_median_outcomes.hipp_right_mean)
    
    hipp_spread = copy.deepcopy(soz_overlap_single_median_outcomes)
    
    hipp_spread["hipp_spread_time"] = hipp_spread_time
    
    tmp1 = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") ]
    tmp2 =hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") ]
    
    good_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good")])
    poor_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor")])
    
    hipp_spread_good = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread)]
    hipp_spread_poor = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread)]
    
    hipp_spread_good_len = len(hipp_spread_good)
    hipp_spread_poor_len = len(hipp_spread_poor)
    
    """
    Contingency table
          Hipp spread less than threshold   |    Never Spread
    good   hipp_spread_good_len             |    good_len - hipp_spread_good_len
    poor   hipp_spread_poor_len             |      poor_len - hipp_spread_poor_len
    """
    
    
    
    contingency_table = [ [hipp_spread_good_len, good_len - hipp_spread_good_len] , [hipp_spread_poor_len,  poor_len - hipp_spread_poor_len]    ]
    
    pval1 = stats.chi2_contingency(contingency_table, correction=True)[1]
    
    
    
    
    #Only consider temporal lobe epilepsy
    good_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))])
    poor_len = len(hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))])
    
    tmp1 = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    tmp2 =hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    
    hipp_spread_good = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "good") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread) & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    hipp_spread_poor = hipp_spread[(hipp_spread["Engel_24_mo_binary"] == "poor") & (hipp_spread["hipp_spread_time"] <= threshold_hippocampus_spread) & ((hipp_spread["Target"] == "MTL") | (hipp_spread["Target"] == "Temporal"))]
    
    contingency_table = [ [hipp_spread_good_len, good_len - hipp_spread_good_len] , [hipp_spread_poor_len,  poor_len - hipp_spread_poor_len]    ]
    
    pval2 = stats.chi2_contingency(contingency_table, correction=True)[1]

    print(f"{co}: {pval1}, {pval2}")

