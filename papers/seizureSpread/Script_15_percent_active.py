#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:18:03 2022

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
import matplotlib
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from fuzzywuzzy import fuzz, process
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

from plotting import pparams 

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
        else:
            channel_names__new.append(txt)
    return np.array(channel_names__new)


def replace_region_name(region_name):
    region_name = region_name.replace("[", "")  
    region_name = region_name.replace("]", "")  
    region_name = region_name.replace("'", "")  
    region_name = region_name.replace(" Left", "Left")  
    region_name = region_name.replace(" Right", "Right")  
    
    region_name = region_name.replace('"SUB"', "")  
    region_name = region_name.replace('"DG"', "")  
    region_name = region_name.replace('"CA1"', "")  
    region_name = region_name.replace('"CA3"', "")  
    region_name = region_name.replace('"PHC"', "")  
    region_name = region_name.replace('"BA36"', "")  
    region_name = region_name.replace('"BA35"', "")  
    region_name = region_name.replace('"ERC"', "")  
    region_name = region_name.replace('"misc"', "")  
    region_name = region_name.replace('Left FuG fusiform gyrus/"sulcus"', "Left FuG fusiform gyrus")  
    region_name = region_name.replace('/', "")  
    
    if region_name == "Brain Stem" or region_name == "Left Cerebral White Matter" or region_name == "Left Lateral Ventricle" or region_name == "Left Inf Lat Vent" or region_name == "Right Cerebral White Matter" or region_name == "Right Lateral Ventricle" or region_name == "Right Inf Lat Vent":
        region_name = ""
    
    
    return region_name





#%

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


fname_patinet_localization = join(metadataDir, "patient_localization_final.mat")
RID_HUP = join(metadataDir, "RID_HUP.csv")
outcomes_fname = join(metadataDir, "patient_cohort_all_atlas_update.csv")

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

paths.ATLAS_LABELS
with open(paths.ATLAS_FILES_PATH) as f: atlas_files = json.load(f)
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
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "NA"
    
for k in range(len(outcomes)): #if poor outcome at 6 or 12 month, then propagate that thru if unknown
      outcomes_list2 = ["Engel_6_mo_binary", "Engel_12_mo_binary","Engel_24_mo_binary"]
      for o in [1,2]:
          value = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o]])[0]
          if value =="NA":
              value_previous = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o-1]])[0]
              if value_previous == "poor":
                  outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "poor"
                  


len(np.unique(patientsWithseizures["subject"]))


unique_patients = np.unique(patientsWithseizures["subject"])

bilateral = []
unilateral = []

for p in range(len(unique_patients)):
    
    RID = unique_patients[p]
    
    #get atlas localization, if none, then go with region file
    atlas_localization_path = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-{session}", f"sub-{RID}_ses-{session}_desc-atlasLocalization.csv")

        
    if utils.checkIfFileExists(atlas_localization_path, printBOOL=False):
        atlas_localization = pd.read_csv(atlas_localization_path)
        atlas_localization.channel = channel2std_ECoG(atlas_localization.channel)
        region_list =  list(atlas_localization["AAL2_label"])
        lefts = []
        rights = []
        for k in range(len(region_list)):
            r = region_list[k]
            if "_L" in r:
                lefts.append(1)
            if "_R" in r:
                rights.append(1)
        left_sum = np.sum(np.array(lefts))
        right_sum = np.sum(np.array(rights))
        if left_sum > 0 and right_sum >0:
            bilateral.append(RID)
        else:
            unilateral.append(RID)

    else:
        hup = RID_HUP["hupsubjno"][np.where(int(RID[3:]) ==  RID_HUP["record_id"])[0][0]]
        
        if f"HUP{hup:03d}" in patients:

            hup_ind = np.where(f"HUP{hup:03d}" == np.asarray(patients))[0][0]
            regs = region[hup_ind]
        
            lefts = []
            rights = []
            for k in range(len(regs)):
                r = regs[k][0]
                if len( r) > 0:
                    if "Left" in r[0]:
                        lefts.append(1)
                    if "Right" in r[0]:
                        rights.append(1)
            left_sum = np.sum(np.array(lefts))
            right_sum = np.sum(np.array(rights))
    
            if left_sum > 0 and right_sum >0:
                bilateral.append(RID)
            else:
                unilateral.append(RID)
bilaterality = pd.DataFrame(np.vstack(  [np.array([bilateral, np.repeat("bilateral", len(bilateral))]).T, np.array([unilateral, np.repeat("unilateral", len(unilateral))]).T  ]  ), columns = ["subject", "bilaterality"])                
print(f"number of patients: {len(np.unique(patientsWithseizures['subject']))}")
print(f"number of seizures: {len(patientsWithseizures)}")  
print(f"number of bilateral: {len(bilateral)}")  
print(f"number of unilateral: {len(unilateral)}")  
#%%
i=132
type_of_overlap = "soz"
threshold=0.69
smoothing = 20
model_ID="WN"
tanh = False
atlas_name_to_use = "AAL2"
def calculate_percent_spread(i, patientsWithseizures, version, atlas_name_to_use, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, tanh = False, seconds_active = None):
    #override_soz if True, then if there are no soz marking, then use the resection markings and assume those are SOZ contacts
    RID = np.array(patientsWithseizures["subject"])[i]
    idKey = np.array(patientsWithseizures["idKey"])[i]
    seizure_length = patientsWithseizures.length[i]
    
    
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
    

        THRESHOLD = threshold
        SMOOTHING = smoothing #in seconds
        
    
        
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
        
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = THRESHOLD, smoothing = SMOOTHING)
        #sns.heatmap( probability_arr_movingAvg.T )      
        #sns.heatmap( probability_arr_threshold.T)    
        spread_start, seizure_start, spread_start_loc, channel_order, channel_order_labels = get_start_times(secondsBefore, skipWindow, fsds, channels, 0, seizure_length, probability_arr_threshold)
   
        
        channel_order_labels = remove_EGG_and_ref(channel_order_labels)
        channels2 = remove_EGG_and_ref(channels)
        if "RT13-1" in channels2:
            channels2 = np.array([ch.replace("-", "") for ch in channels2 ])
        
        channel_order_labels = channel2std_ECoG(channel_order_labels)
        channels2 = channel2std_ECoG(channels2)
        
        #print(soz_channel_names)
        #print(resected_channel_names)
        #print(channel_order_labels)
    
    
        #remove ignore electrodes from channel_order_labels
        #ignore_index = np.intersect1d(  channel_order_labels, ignore__channel_names, return_indices=True)
        #channel_order_labels[-ignore_index[1]]
        #channel_order_labels = np.delete(channel_order_labels, ignore_index[1])
        
        
        
        
        
        
        ################################################################
        ################################################################
        ################################################################
        ################################################################
        ################################################################
        #Analysis on clustering of regions active
        ################################################################
        ################################################################
        ################################################################
        ################################################################
        ################################################################
        
        
        #build array of unique regions
        
    
        
        #find time at which that region became active as a percentage of seizure length
        
             
        
        #atlas
        #atlas = "BN_Atlas_246_1mm"
        #atlas = "AAL3v1_1mm"
        atlas = atlas_name_to_use
        #atlas = "HarvardOxford-sub-ONLY_maxprob-thr25-1mm"

        atlas_names_short =  list(atlas_files["STANDARD"].keys() )
        atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
        ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
        atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
        atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
        atlas_label_names = np.array(atlas_label.iloc[1:,1])
        atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
        atlas_label_region_numbers = atlas_label_region_numbers.astype(int)
        
        
        region_activation = pd.DataFrame(columns = atlas_label_names )

        atlas_localization_path = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-{session}", f"sub-{RID}_ses-{session}_desc-atlasLocalization.csv")

            
        if not utils.checkIfFileExists(atlas_localization_path, printBOOL=False) or RID == 'RID0018':
            print(f"{RID}: No atlas localization file.                       Doing own localization")
            
            #channel_names = labels[i_patient]
            #regions_patient = region[i_patient]
            #regions_to_use = regions_unique
            #Getting SOZ labels
            
            RID_keys =  list(jsonFile["SUBJECTS"].keys() )
            hup_num_all = [jsonFile["SUBJECTS"][x]["HUP"]  for  x   in  RID_keys]
            
            hup_int = hup_num_all[RID_keys.index(RID)]
            hup_int_pad = f"{hup_int:03d}" 
            
            if f"HUP{hup_int_pad}" in patients:
                i_patient = patients.index(f"HUP{hup_int_pad}")
                
                coordinates = coords[i_patient]
                channel_names = labels[i_patient]
                if "RT13-1" in channel_names:
                    channel_names = np.array([ch.replace("-", "") for ch in channel_names ])
                
                ignore_ind = np.where(ignore[i_patient] == 1)[0]
                
                
                channel_names_std = channel2std_ECoG(channel_names)
                
                region[i_patient]
               
                ignore_channel_names = np.array(channel_names)[ignore_ind]
                ignore_channel_names_std = channel2std_ECoG(ignore_channel_names)
                
                #creating new localization
                
                atlas_localization = pd.DataFrame(columns = ["channel", f"{atlas}_region_number", f"{atlas}_label"])
                atlas_localization[f"channel"]= channel_names_std
                
                atlas_path = join(paths.ATLASES, atlas_files["STANDARD"][atlas_names_short[ind]]["name"])
                img = nib.load(atlas_path)
                #utils.show_slices(img, data_type = "img")
                img_data = img.get_fdata()
                affine = img.affine
                shape = img_data.shape
                
                coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(img.affine), coordinates)
                coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel
                coordinates_voxels = coordinates_voxels.astype(int)
                
                try:
                    img_ROI = img_data[coordinates_voxels[:,0], coordinates_voxels[:,1], coordinates_voxels[:,2]]
                except: #checking to make sure coordinates are in the atlas. This happens usually for electrodes on the edge of the SEEG. For example, RID0420 electrodes LE11 and LE12 are outside the brain/skull, and thus are outside even the normal MNI space of 181x218x181 voxel dimensions
                    img_ROI = np.zeros((coordinates_voxels.shape[0],))
                    for i in range(0,coordinates_voxels.shape[0]):
                        if((coordinates_voxels[i,0]>img_data.shape[0]) or (coordinates_voxels[i,0]<1)):
                            img_ROI[i] = 0
                            print(f'{channel_names[i]} is outside image space: setting to zero')
                        elif((coordinates_voxels[i,1]>img_data.shape[1]) or (coordinates_voxels[i,1]<1)):
                            img_ROI[i] = 0
                            print(f'{channel_names[i]} is outside image space: setting to zero')
                        elif((coordinates_voxels[i,2]>img_data.shape[2]) or (coordinates_voxels[i,2]<1)):
                            img_ROI[i] = 0
                            print(f'{channel_names[i]} is outside image space: setting to zero')
                        else:
                            img_ROI[i]
                atlas_localization[f"{atlas}_region_number"]= img_ROI.astype(int)
                #get region label
                for kk in range(len(atlas_localization)):
                    atl_name = atlas_label_names[np.where(atlas_localization[f"{atlas}_region_number"][kk] == atlas_label_region_numbers )]
                    if len(atl_name) >0:
                        atlas_localization.loc[atlas_localization["channel"] == atlas_localization["channel"][kk], "AAL2_label"]  = atlas_label_names[np.where(atlas_localization[f"{atlas}_region_number"][kk] == atlas_label_region_numbers )][0]
                    else:
                        atlas_localization.loc[atlas_localization["channel"] == atlas_localization["channel"][kk], "AAL2_label"]  = "NotInAtlas"
            else: #if there are no atlas localization files, and there are no coordinates for this patient, then can't do anything
                print(f"\n\n{RID}: NO COORDINATE FILES OR ATLAS LOCALIZATION FILES\n")
                reg_act_time = np.zeros(shape = (len(atlas_label_names)))
                reg_act_time[:] = np.nan
                region_activation = region_activation.append(pd.DataFrame(reg_act_time.reshape(1,-1), columns=list(region_activation)), ignore_index=True)
                time_bins = np.round(np.arange(0, 120,1),0)
                percent_channels_active = pd.DataFrame(columns = time_bins)
                percent_channels_active.loc[percent_channels_active.shape[0]] = [np.nan]*len(time_bins)
                
                percent_channels_active = pd.DataFrame(columns = time_bins)
                percent_channels_active.loc[percent_channels_active.shape[0]] = [np.nan]*len(time_bins)
                
                percent_regions_total_active = pd.DataFrame(columns = time_bins)
                percent_regions_total_active.loc[percent_regions_total_active.shape[0]] = [np.nan]*len(time_bins)
                
                percent_regions_implanted_active = pd.DataFrame(columns = time_bins)
                percent_regions_implanted_active.loc[percent_regions_implanted_active.shape[0]] = [np.nan]*len(time_bins)
                if seconds_active is None:
                    seconds = np.round(np.arange(0,60*2+1,1),0)
                else:
                    seconds = seconds_active
                pa_chan = np.zeros(len(seconds))
                pa_reg = np.zeros(len(seconds))
                pa_reg_total = np.zeros(len(seconds))
                pa_chan[:] = np.nan
                pa_reg[:] = np.nan
                pa_reg_total[:] = np.nan
                num_regions = np.nan
                num_chans = np.nan
                
                return pa_reg, pa_reg_total, pa_chan,num_regions,num_chans
                #return percent_regions_implanted_active, percent_regions_total_active, percent_channels_active
                    
        else:
            print(f"{RID}: Localization file exists...loading")
            atlas_localization = pd.read_csv(atlas_localization_path)
            atlas_localization.channel = channel2std_ECoG(atlas_localization.channel)
            
            
        channel_names = list(atlas_localization.channel)
        regions_patient = atlas_localization[f"{atlas}_label"]
        channels_region_index_label = []
        for r in range(len(regions_patient)):
            reg = str(regions_patient[r])

            reg_index = np.where( reg ==  atlas_label_names  )[0]
            if len(reg_index) == 0:
                ind = -1
            else:
                ind = reg_index[0]
            channels_region_index_label.append(ind)
        channels_region_index_label = np.asarray(channels_region_index_label)
       
        channels_region_index_label = np.asarray(channels_region_index_label)

        
        channel_activation_time = pd.DataFrame(columns = ["channel", "region_num", "activation_time"])
        
        
        
        #calculate active at t time
        
        if seconds_active is None:
            seconds = np.round(np.arange(0,60*2+1,1),0)
        else:
            seconds = seconds_active
        pa_chan = np.zeros(len(seconds))
        pa_reg = np.zeros(len(seconds))
        pa_reg_total = np.zeros(len(seconds))
        pa_chan[:] = np.nan
        pa_reg[:] = np.nan
        pa_reg_total[:] = np.nan
        skip_before = 10
        active_during_seizure = probability_arr_threshold[int(seizure_start-skip_before/skipWindow):,:]
        for s in range(len(seconds)):
            sec = seconds[s]
            sec= int(sec/skipWindow)
            if sec < len(active_during_seizure):
                pa_chan[s] = sum(active_during_seizure[sec + int(skip_before/skipWindow),:] == 1)/active_during_seizure.shape[1]
        
        #sns.lineplot(x = seconds , y =pa_chan)
        
        ac_df = pd.DataFrame(probability_arr_movingAvg[int(seizure_start-10/skipWindow):,:].T, index= channels2)
        
        
        channel_names = channel2std_ECoG(channel_names)
        channel_activation_time["channel"] = channel_names
        channel_activation_time["region_num"] = channels_region_index_label
        
        
        inter = np.intersect1d(channels2, channel_names, return_indices=True)

        
        reg_num_of_activity = channel_activation_time["region_num"][inter[2]]
        
        ac_df = ac_df.loc[inter[0],:]
        ac_df["region_num"] = np.array(reg_num_of_activity)
        
        ac_df_mean = ac_df.groupby(["region_num"]).mean()
        ac_df_mean[ac_df_mean >= THRESHOLD] = 1
        ac_df_mean[ac_df_mean < THRESHOLD] = 0
        
        ac_df_mean = ac_df_mean.reset_index()
        #ac_df_mean_long = ac_df_mean.melt(id_vars = "region_num",var_name = "time", value_name = "probability")
        #sns.lineplot(data = ac_df_mean_long, x = "time" , y ="probability", hue ="region_num" )
        ac_df_thresh = ac_df_mean.iloc[1:, :]
        
        #if RID == 'RID0014' or RID == 'RID0179'  or RID == 'RID0259' or RID == "RID0033"  or RID == "RID0042": #weird localization cause ECoG
        #    pa_reg = pa_reg
  
        for s in range(len(seconds)):
            sec = seconds[s]
            sec= int(sec/skipWindow)
            if ac_df_thresh.shape[0] >0:
                pa_reg[s] = sum(ac_df_thresh.iloc[1:, sec + int(skip_before/skipWindow)] == 1)/ac_df_thresh.shape[0]
                pa_reg_total[s] =  sum(ac_df_thresh.iloc[1:, sec + int(skip_before/skipWindow)] == 1)/len(atlas_label_names)
            
        #sns.lineplot(x = seconds , y =pa_reg)
        #sns.lineplot(x = seconds , y =pa_reg_total)   
        num_regions = ac_df_thresh.shape[0]
        num_chans = active_during_seizure.shape[1]
        return pa_reg, pa_reg_total, pa_chan, num_regions, num_chans
        
        
        """
        
        
        
        
        #get activation time
        for ch in range(len(channel_activation_time)):
            chan = channel_activation_time["channel"][ch]
            ind_overlap = np.where(chan == channels2  )[0]
            
            if len(ind_overlap) > 0:
                ind_chan = np.where(chan == channels2  )[0][0]
                chan_start = spread_start[ind_chan] * skipWindow
                chan_start_percent = chan_start
                if chan_start_percent > seizure_length:
                    chan_start_percent = 10000 #putting in large number so to know which regions were still implanted but had no activation
                    
            else:
                chan_start_percent = np.nan
            channel_activation_time.loc[ch, 'activation_time'] = chan_start_percent

     
        channel_activation_time["activation_time"] = channel_activation_time["activation_time"].astype(float)
        
    
        channel_activation_time_only_times = channel_activation_time.drop("channel", axis= 1)
        
        channel_activation_time_only_times_tmp = channel_activation_time_only_times[channel_activation_time_only_times.activation_time != 10000]
        channel_activation_time_only_times_1000 = channel_activation_time_only_times[channel_activation_time_only_times.activation_time == 10000]
        
        channel_activation_time_only_times_tmp= channel_activation_time_only_times_tmp.astype('float')
        channel_activation_time_only_times_1000= channel_activation_time_only_times_1000.astype('float')
        region_activation_time = channel_activation_time_only_times_tmp.groupby(["region_num"], as_index=False).mean()
        region_activation_time_1000 = channel_activation_time_only_times_1000.groupby(["region_num"], as_index=False).mean()
        for kkkk in range(len(region_activation_time_1000)):
            reggg = region_activation_time_1000["region_num"][kkkk]
            if not reggg in list(region_activation_time["region_num"]):
                region_activation_time = region_activation_time.append(region_activation_time_1000.iloc[kkkk], ignore_index = True)
        region_activation_time = region_activation_time[region_activation_time.activation_time >= 0]
        """   
        #reg_act_time = np.zeros(shape = (len(atlas_label_names)))
        #reg_act_time[:] = np.nan
        #for rrr in range(len(region_activation_time)):
        #    reg_ind = region_activation_time["region_num"][rrr]
        #    if not reg_ind == -1:
        #        reg_act_time[int(reg_ind)] = region_activation_time["activation_time"][rrr]
        
        #region_activation = region_activation.append(pd.DataFrame(reg_act_time.reshape(1,-1), columns=list(region_activation)), ignore_index=True)
        """
        #get percent of channels active at certain time
        time_bins = np.round(np.arange(0, 120,1),0)
        percent_channels_active = pd.DataFrame(columns = time_bins)
        percent_channels_active.loc[percent_channels_active.shape[0]] = [None]*len(time_bins)
       
        percent_regions_implanted_active = pd.DataFrame(columns = time_bins)
        percent_regions_implanted_active.loc[percent_regions_implanted_active.shape[0]] = [None]*len(time_bins)
        
        percent_regions_total_active = pd.DataFrame(columns = time_bins)
        percent_regions_total_active.loc[percent_regions_total_active.shape[0]] = [None]*len(time_bins)
        
        
  
        for tt in range(len(time_bins)):
            times = np.array(channel_activation_time["activation_time"]).astype(float)
            percent = len(np.where(times < time_bins[tt])[0])/len(channels2)
            percent_channels_active.loc[0,time_bins[tt]] = percent
            
            times = np.array(region_activation_time["activation_time"]).astype(float)
            if len(region_activation_time) > 0:
                percent = len(np.where(times < time_bins[tt])[0])/len(region_activation_time)
            else:
                percent = 0
            percent_regions_implanted_active.loc[0,time_bins[tt]] = percent
        
            percent = len(np.where(times < time_bins[tt])[0])/len(atlas_label_names)
            percent_regions_total_active.loc[0,time_bins[tt]] = percent
        
        #sns.lineplot(x = range(len(time_bins)), y = np.array(percent_channels_active)[0])
        #sns.lineplot(x = range(len(time_bins)), y = np.array(percent_regions_implanted_active)[0])
        #sns.lineplot(x = range(len(time_bins)), y = np.array(percent_regions_total_active)[0])
        
        return percent_regions_implanted_active, percent_regions_total_active, percent_channels_active


    """
#%%

use_atlas = False   
tanh = True
model_IDs = ["WN","CNN","LSTM" , "absolute_slope", "line_length", "power_broadband"]
m=0
model_ID= model_IDs[m]
threshold = 0.6

seconds_active = np.round(np.arange(0,60*2+1,1),0)
#pd.DataFrame(columns = ["subject", "seizure"] +list( regions_unique) )
#atlas

#atlas = "AAL3v1_1mm"

atlas = "BN_Atlas_246_1mm"
atlas = "AAL2"
#atlas = "AAL3v1_1mm"
#atlas = "HarvardOxford-combined"
#atlas = "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2"
atlas_name_to_use = atlas

atlas_names_short =  list(atlas_files["STANDARD"].keys() )
atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
atlas_label_names = np.array(atlas_label.iloc[1:,1])

percent_active_regions_implanted = pd.DataFrame(columns =seconds_active )
percent_active_regions_total = pd.DataFrame(columns = seconds_active )
percent_active_channel = pd.DataFrame(columns = seconds_active )

regions_implanted = []
total_channels = []

for i in range(len(patientsWithseizures)):
    print(f"\r{i}   { np.round(   (i+1)/len(patientsWithseizures)*100   ,2)}%                  ", end = "\r")
    percent_regions_implanted_active, percent_regions_total_active, percent_channels_active, num_regions, num_chans = calculate_percent_spread(i, patientsWithseizures, version, atlas_name_to_use = atlas_name_to_use, threshold=threshold, smoothing = 20, model_ID=model_ID, secondsAfter=180, secondsBefore=180, tanh = tanh, seconds_active = seconds_active)
    
    percent_active_regions_implanted.loc[i] = percent_regions_implanted_active
    percent_active_regions_total.loc[i] = percent_regions_total_active
    percent_active_channel.loc[i] = percent_channels_active

    regions_implanted.append(num_regions)
    total_channels.append(num_chans)

#%
outcome_bins = ["Engel_6_mo_binary","Engel_12_mo_binary","Engel_24_mo_binary"]
#%%

percent_active_regions_implanted_about = copy.deepcopy(percent_active_regions_implanted)
percent_active_regions_total_about = copy.deepcopy(percent_active_regions_total)
percent_active_channel_about = copy.deepcopy(percent_active_channel)

percent_active_regions_implanted_about = percent_active_regions_implanted_about.astype(float)
percent_active_regions_total_about = percent_active_regions_total_about.astype(float)
percent_active_channel_about = percent_active_channel_about.astype(float)

percent_active_regions_implanted_about["subject"] = patientsWithseizures["subject"]
percent_active_regions_total_about["subject"] = patientsWithseizures["subject"]
percent_active_channel_about["subject"] = patientsWithseizures["subject"]

percent_active_regions_implanted_about["seizure"] = patientsWithseizures["idKey"]
percent_active_regions_total_about["seizure"] = patientsWithseizures["idKey"]
percent_active_channel_about["seizure"] = patientsWithseizures["idKey"]

percent_active_regions_implanted_about["regions_implanted"] = regions_implanted
percent_active_regions_implanted_about["total_channels"] = total_channels
percent_active_regions_total_about["regions_implanted"] = regions_implanted
percent_active_regions_total_about["total_channels"] = total_channels
percent_active_channel_about["regions_implanted"] = regions_implanted
percent_active_channel_about["total_channels"] = total_channels


types = ["by_regions_implanted",  "by_regions_total",  "by_channels_implanted"]
percent_active_regions_implanted_about["type"] = "by_regions_implanted"
percent_active_regions_total_about["type"] = "by_regions_total"
percent_active_channel_about["type"] = "by_channels_implanted"

df = pd.concat([percent_active_regions_implanted_about,percent_active_regions_total_about , percent_active_channel_about], axis = 0)


df_long = df.melt(id_vars =[ "subject", "seizure" ,"type", "regions_implanted", "total_channels"],var_name = "time", value_name = "percent_active")

df_long_mean = df_long.groupby(["subject", "time", "type", "regions_implanted", "total_channels"] , as_index=False).median()


df_long_bilaterality = pd.merge(df_long_mean, bilaterality, on='subject')
df_long_outcome = pd.merge(df_long_bilaterality, outcomes, on='subject')


#%%Plot single patient sieuzre

ty=types[0]
RID = "RID0309"
for t, ty in enumerate(types):
    df_plot_single = df_long[(df_long["subject"] == RID) & (df_long["type"] == ty)]
    
    ind = np.argsort(df_plot_single[df_plot_single["time" ]== 30]["percent_active"]  )
    ind = ind.reset_index()
    sz_order = np.array(df_plot_single.loc[ind["index"]].iloc[ ind["percent_active"]]["seizure"])
    
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b0ceec","#1376d8","#0d4f90" ,"#0d4f90" , "#173c60" ])
    
    fig, ax = utils.plot_make(size_length = 7)
    sns.lineplot(data = df_plot_single, x = "time", y = "percent_active", hue = "seizure"  , palette= "Blues" , hue_order= sz_order)
    utils.fix_axes(ax)
    ax.get_legend().remove()
    plt.show()
    fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"single_seizure_{ty}_{RID}.pdf")
    plt.savefig(fname,  bbox_inches='tight')


#%%Get number of regions and channels implanted

analyses = ["regions_implanted", "total_channels"]
an = 0

fig, ax = utils.plot_make(size_length = 2)
whis = 1.5
sns.boxplot(ax = ax, data = df_long_outcome[(df_long_outcome["time"] == 30) & (df_long_outcome["type"] == types[1])], x = "Engel_24_mo_binary", y = analyses[an], order= ["good", "poor"],showfliers = False,whis = whis,palette=pparams.PALETTE_OUTCOME)


sns.swarmplot(ax = ax, data = df_long_outcome[(df_long_outcome["time"] == 30) & (df_long_outcome["type"] == types[1])], x = "Engel_24_mo_binary", y = analyses[an], order= ["good", "poor"], palette=pparams.PALETTE_OUTCOME_DARK, dodge = True)

v1 = df_long_outcome[(df_long_outcome["time"] == 30) & (df_long_outcome["type"] == types[1]) & (df_long_outcome[outcome_bins[o]] == "poor")][analyses[an]]
v2 = df_long_outcome[(df_long_outcome["time"] == 30) & (df_long_outcome["type"] == types[1]) & (df_long_outcome[outcome_bins[o]] == "good")][analyses[an]]
mwu = stats.mannwhitneyu(v1, v2)

ax.set_title(f"{mwu[1]:.4f}\n{analyses[an]}")
utils.reformat_boxplot(ax)
fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"implantes_{analyses[an]}_outcome_{o}.pdf")
plt.savefig(fname,  bbox_inches='tight')


#%%Get percent active over time

tyt = 0
o=2
for tyt in range(len(types)):
    df_lineplot = df_long_outcome[df_long_outcome["type"] == types[tyt]]
    fig, ax = utils.plot_make(size_length = 4)
    sns.lineplot(ax = ax, data = df_lineplot, x = "time", y = "percent_active", hue = outcome_bins[o], hue_order= ["good", "poor"], ci = 68, lw = 5)
    utils.fix_axes(ax)
    plt.axvline(x=30, color='k', linestyle='--', lw = 5)
    
    fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"percent_over_time_{types[tyt]}_outcome_{o}.pdf")
    plt.savefig(fname,  bbox_inches='tight')



#%%bootstrap
"""
t=30

dat = df_long_outcome[(df_long_outcome["time"] == t)]
o = 2
tyt =0
df_plot =  dat[(dat["type"] == types[tyt]) & (dat[outcome_bins[o]] != "NA")]

poor = df_plot[(df_plot[outcome_bins[o]] == "poor")]
good = df_plot[(df_plot[outcome_bins[o]] == "good")]

tstats = pd.DataFrame(columns = ["B", "real", "null"]) 

for B in range(100):
    count = len(tstats)
    bs_poor_sub = random.choices(  np.unique(poor["subject"]), k=len(np.unique(poor["subject"])))
    bs_good_sub = random.choices(  np.unique(good["subject"]), k=len(np.unique(good["subject"])))

    null_good = np.array(random.sample(list(np.unique(poor["subject"])) + list(np.unique(good["subject"])), k = len(np.unique(good["subject"]))))
    null_poor = np.setxor1d(null_good,list(np.unique(poor["subject"])) + list(np.unique(good["subject"])))
    null_poor_sub = random.choices(  null_good, k=len(null_poor))
    null_good_sub = random.choices(  null_poor, k=len(null_good))

    #build dataframes
    bs_poor = pd.DataFrame(columns = poor.columns)
    nulll_poor = pd.DataFrame(columns = poor.columns)
    for b in range(len(bs_poor_sub)):
        bs_poor = bs_poor.append(poor[poor["subject"] == bs_poor_sub[b]], ignore_index= True)
        nulll_poor = nulll_poor.append(df_plot[df_plot["subject"] == null_poor_sub[b]], ignore_index= True)
    
    bs_good = pd.DataFrame(columns = good.columns)
    nulll_good = pd.DataFrame(columns = good.columns)
    for b in range(len(bs_good_sub)):
        bs_good = bs_good.append(good[good["subject"] == bs_good_sub[b]], ignore_index= True)
        nulll_good = nulll_good.append(df_plot[df_plot["subject"] == null_good_sub[b]], ignore_index= True)
        
    v1 = np.array(bs_good["percent_active"])
    v2 = np.array(bs_poor["percent_active"])
    
    v1_null = np.array(nulll_good["percent_active"])
    v2_null = np.array(nulll_poor["percent_active"])
    

    mwu1 = stats.mannwhitneyu(v2, v1)
    mwu2 = stats.mannwhitneyu(v2_null, v1_null)
    
    print(f"{B}   {mwu1[1]:.3f},      { mwu2[1]:.3f}")
    tstats = tstats.append(dict(B = count, real = mwu1[0], null = mwu2[0]), ignore_index=True)
    
    
tstats_long = tstats.melt(id_vars="B", var_name = "boot", value_name = "tstat")
sns.histplot( data = tstats_long, x = "tstat", hue = "boot")

pval = len(np.where(tstats["null"] > tstats["real"].mean())[0]) /len(tstats)
"""

#%
#%%
t=30

dat = df_long_outcome[(df_long_outcome["time"] == t)]
outcome_bins = ["Engel_6_mo_binary","Engel_12_mo_binary","Engel_24_mo_binary"]
o = 2
tyt =2
df_plot =  dat[(dat["type"] == types[tyt]) & (dat[outcome_bins[o]] != "NA")]

if tyt ==1:
    ylim = [-0.01,0.3]
else:
    ylim = [-0.05,1.1]

poor = df_plot[(df_plot[outcome_bins[o]] == "poor")]
good = df_plot[(df_plot[outcome_bins[o]] == "good")]
v1 = np.array(poor["percent_active"])
v2 = np.array(good["percent_active"])
mwu = stats.mannwhitneyu(v1, v2)

print(t, mwu[1])


whis = 1
fig, ax = utils.plot_make(size_length = 2)
sns.boxplot(ax = ax, data = df_plot, y = "percent_active", x = outcome_bins[o], order= ["good", "poor"], showfliers = False,whis = whis,palette=pparams.PALETTE_OUTCOME)
sns.swarmplot(ax = ax,data = df_plot, y = "percent_active", x = outcome_bins[o], dodge = True, order= ["good", "poor"], palette=pparams.PALETTE_OUTCOME_DARK)
ax.set_ylim(ylim)
ax.set_title(f"{mwu[1]:.4f}\n{types[tyt]}")
utils.reformat_boxplot(ax)
fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"extent_{types[tyt]}_outcome_{o}_t_{t:02d}_all.pdf")
plt.savefig(fname,  bbox_inches='tight')




fig, ax = utils.plot_make(size_length = 2)
sns.boxplot(ax = ax, data = df_plot[df_plot["bilaterality"]=="unilateral"], x = outcome_bins[o], y = "percent_active", order= ["good", "poor"], showfliers = False,whis = whis,palette=pparams.PALETTE_OUTCOME)
sns.swarmplot(ax = ax,data = df_plot[df_plot["bilaterality"]=="unilateral"], y = "percent_active", x = outcome_bins[o], dodge = True, order= ["good", "poor"], palette=pparams.PALETTE_OUTCOME_DARK)
ax.set_ylim(ylim)
mwu = stats.mannwhitneyu(df_plot[(df_plot["bilaterality"]=="unilateral") & (df_plot[outcome_bins[o]] == "poor")]["percent_active"], df_plot[(df_plot["bilaterality"]=="unilateral") & (df_plot[outcome_bins[o]] == "good")]["percent_active"])
ax.set_title(f"{mwu[1]:.4f}\n{types[tyt]}")
#ax.spines["left"].set_visible(False)
#ax.set_ylabel("")
ax.tick_params(left = False)
utils.reformat_boxplot(ax)
fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"extent_{types[tyt]}_outcome_{o}_t_{t:02d}_unilateral.pdf")
plt.savefig(fname,  bbox_inches='tight')


        


fig, ax = utils.plot_make(size_length = 2)
sns.boxplot(ax = ax, data = df_plot[df_plot["bilaterality"]=="bilateral"], x = outcome_bins[o], y = "percent_active", order= ["good", "poor"], showfliers = False,whis = whis,palette=pparams.PALETTE_OUTCOME)
sns.swarmplot(ax = ax,data = df_plot[df_plot["bilaterality"]=="bilateral"], y = "percent_active", x = outcome_bins[o], dodge = True, order= ["good", "poor"], palette=pparams.PALETTE_OUTCOME_DARK)
ax.set_ylim(ylim)
mwu = stats.mannwhitneyu(df_plot[(df_plot["bilaterality"]=="bilateral") & (df_plot[outcome_bins[o]] == "poor")]["percent_active"], df_plot[(df_plot["bilaterality"]=="bilateral") & (df_plot[outcome_bins[o]] == "good")]["percent_active"])
ax.set_title(f"{mwu[1]:.4f}\n{types[tyt]}")
#ax.spines["left"].set_visible(False)
#ax.set_ylabel("")
ax.tick_params(left = False)
utils.reformat_boxplot(ax)
fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"extent_{types[tyt]}_outcome_{o}_t_{t:02d}_bilateral.pdf")
plt.savefig(fname,  bbox_inches='tight')


        



utils.adjust_box_widths(fig, 0.8)
#%%
cohensd = pd.DataFrame(columns = ["type_imp", "t", "cohensd"])
#get effect sizes


for t in range(60):
    
    for tyt in range(len(types)):
    
        dat = df_long_outcome[(df_long_outcome["time"] == t)]
        d1 = dat[(dat["type"] == types[tyt])]
       
        outcome_bins = ["Engel_6_mo_binary","Engel_12_mo_binary","Engel_24_mo_binary"]
        o = 2
        poor = d1[(d1[outcome_bins[o]] == "poor")]
        good = d1[(d1[outcome_bins[o]] == "good")]
        
    
        v1 = np.array(poor["percent_active"])
        v2 = np.array(good["percent_active"])
        
        cd = utils.cohend(v2, v1)
        
        cohensd = cohensd.append(dict(type_imp = types[tyt], t = t, cohensd = cd ), ignore_index=True)
        
        print(t, scipy.stats.mannwhitneyu(v1, v2)[1])


for tyt in range(len(types)):
    fig, ax = utils.plot_make()
    sns.lineplot(ax = ax, data = cohensd[(cohensd["t"] > 1) & (cohensd["type_imp"] == types[tyt])], x = "t", y = "cohensd", palette="Set2" , lw = 5)
    utils.fix_axes(ax)
    ax.set_ylim([0,0.7])
    plt.axvline(x=30, color='k', linestyle='--', lw =5)
    fname = join(paths.SEIZURE_SPREAD_FIGURES, "extent_of_spread", f"cohensd_{types[tyt]}_outcome_{o}.pdf")
    plt.savefig(fname,  bbox_inches='tight')
    
    




######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
#%%

tanh = True
model_IDs = ["WN","CNN","LSTM" , "absolute_slope", "line_length", "power_broadband"]
by = 0.01


full_analysis_location = join(BIDS, project_folder, f"full_analysis_save")
full_analysis_location_file_basename = f"soz_overlap_tanh_{tanh}_by_{by}_model_{version}.pickle"
full_analysis_location_file = join(full_analysis_location, full_analysis_location_file_basename)


with open(full_analysis_location_file, 'rb') as f: [soz_overlap, percent_active, tanh, seconds_active, by, thresholds, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)

#%%
palette = {"WN": "#1d5e9e", "CNN": "#73ace5", "LSTM": "#7373e5", "absolute_slope": "#961c1d", "line_length": "#d16a6a" , "power_broadband": "#d19e6a" }
palette_dark = {"WN": "#0a2036", "CNN": "#1e60a1", "LSTM": "#3737b3", "absolute_slope": "#250b0b", "line_length": "#5b1c1c" , "power_broadband": "#5b3c1c" }



#%%
soz_overlap_median = soz_overlap.groupby(['model', 'subject', "threshold"], as_index=False).median()


soz_overlap_outcomes = pd.merge(soz_overlap, outcomes, on='subject')
soz_overlap_median_outcomes = pd.merge(soz_overlap_median, outcomes, on='subject')





m = 0
thr = [0.69, 0.96, 0.58, 0.08, 0.11, 0.01]
thr = [0.59, 0.75, 0.58, 0.6, 0.6, .3]
co = 60

thresh = thr[m]


model_ID = model_IDs[m]

threshold_hippocampus_spread = co #cutoff in seconds for time to spread

hipp_spread = copy.deepcopy(soz_overlap_median_outcomes)
hipp_spread_time = abs(soz_overlap_median_outcomes.hipp_left_mean - soz_overlap_median_outcomes.hipp_right_mean)
hipp_spread["hipp_spread_time"] = hipp_spread_time

hipp_spread_thresh = hipp_spread[( hipp_spread["threshold"]== thresh )]
hipp_spread_model =  hipp_spread_thresh[( hipp_spread_thresh["model"] ==model_ID )]

hipp_spread_model = hipp_spread_model[pd.DataFrame(hipp_spread_model.subject.tolist()).isin(bilateral).any(1).values]
#hipp_spread_model = hipp_spread_model[pd.DataFrame(hipp_spread_model.subject.tolist()).isin(unilateral).any(1).values]

good_len = len(hipp_spread_model[(hipp_spread_model["Engel_24_mo_binary"] == "good")])
poor_len = len(hipp_spread_model[(hipp_spread_model["Engel_24_mo_binary"] == "poor")])

hipp_spread_model_good = hipp_spread_model[(hipp_spread_model["Engel_24_mo_binary"] == "good") & (hipp_spread_model["hipp_spread_time"] <= threshold_hippocampus_spread)]
hipp_spread_model_poor = hipp_spread_model[(hipp_spread_model["Engel_24_mo_binary"] == "poor") & (hipp_spread_model["hipp_spread_time"] <= threshold_hippocampus_spread)]

hipp_spread_model_good_len = len(hipp_spread_model_good)
hipp_spread_model_poor_len = len(hipp_spread_model_poor)

"""
Contingency table
      Hipp spread less than threshold   |    Never Spread
good   hipp_spread_model_good_len             |    good_len - hipp_spread_model_good_len
poor   hipp_spread_model_poor_len             |      poor_len - hipp_spread_model_poor_len




                                    poor                             /     good
Hipp spread less than threshold     hipp_spread_model_poor_len      /              hipp_spread_model_good_len  
Never Spread                      poor_len - hipp_spread_model_poor_len        good_len - hipp_spread_model_good_len       



"""


sensitivity = (hipp_spread_model_poor_len)/(hipp_spread_model_poor_len +(poor_len - hipp_spread_model_poor_len ) )
specificty = (good_len - hipp_spread_model_good_len)/ (hipp_spread_model_good_len   +   good_len - hipp_spread_model_good_len  )

ppv = (hipp_spread_model_poor_len) /(hipp_spread_model_poor_len + (hipp_spread_model_good_len ))
npv = (good_len - hipp_spread_model_good_len )/ ((poor_len - hipp_spread_model_poor_len )+ (good_len - hipp_spread_model_good_len ))

contingency_table = [ [hipp_spread_model_good_len, good_len - hipp_spread_model_good_len] , [hipp_spread_model_poor_len,  poor_len - hipp_spread_model_poor_len]    ]

correction = False
N1 = sum(sum(np.array(contingency_table)))
if hipp_spread_model_good_len == 0 and hipp_spread_model_poor_len ==0:
    pval1 = 1
    cramers_V1 = 0
    odds_ratio1=1
else:
    chi2 = stats.chi2_contingency(contingency_table, correction=correction)[0]
    pval1 = stats.chi2_contingency(contingency_table, correction=correction)[1]
    cramers_V1= np.sqrt((stats.chi2_contingency(contingency_table, correction=correction)[0])/N1)
    if contingency_table[0][1] == 0 or contingency_table[1][0] == 0:
        odds_ratio1=1
    else:
        odds_ratio1 = (contingency_table[0][0]*contingency_table[1][1]) / (contingency_table[0][1]*contingency_table[1][0] )


contingency_table = [ [ hipp_spread_model_poor_len, hipp_spread_model_good_len  ] , [ poor_len - hipp_spread_model_poor_len  , good_len - hipp_spread_model_good_len ]    ]


print(f"contingency_table:          {contingency_table}")
print(f"chi2:                       {chi2}")
print( f"pval1:                      {pval1}")
print(f"cramers_V1:                 {cramers_V1}")

print(f"sensitivity:           {sensitivity}")
print(f"specificty:            {specificty}")
print(f"ppv:                   {ppv}")
print(f"npv:                   {npv}")

#utils.fdr2([0.0238, 0.0142, 0.0082, 0.04169, 0.22928])

#utils.fdr2([0.03398, 0.0508, 0.0400, 0.1817, 0.777])

utils.fdr2([0.01222, 0.02060, 0.0169, 0.095068, 0.5347])


utils.fdr2([0.01222020006938264, 0.020604039682107254, 0.009411442857254457, 0.01690219083632435, 0.09506890415800442, 0.534771721989973])

















































