#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:59:33 2022

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
from PIL import Image
from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr


from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
        
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
                  
regions_all = []

for r in range(len(region)):
    region_patient = region[r]
    for kk in range(len(region_patient)):
        reg = str(np.unique(region_patient[kk].flatten())[0]) 
        reg = replace_region_name(reg)
        
        

        if reg == "": reg
        else:
            regions_all.append(reg)
regions_unique = np.unique(regions_all)



#%%


i=137
type_of_overlap = "soz"
threshold=0.6
smoothing = 20
model_ID="WN"
tanh = False

def calculate_mean_rank_deep_learning(i, patientsWithseizures, version, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, tanh = False):
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
        #%
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
        
        channel_order_labels = channel2std_ECoG(channel_order_labels)
        channels2 = channel2std_ECoG(channels2)
        
        #print(soz_channel_names)
        #print(resected_channel_names)
        #print(channel_order_labels)
    
    
        #remove ignore electrodes from channel_order_labels
        #ignore_index = np.intersect1d(  channel_order_labels, ignore__channel_names, return_indices=True)
        #channel_order_labels[-ignore_index[1]]
        #channel_order_labels = np.delete(channel_order_labels, ignore_index[1])
        
        
        atlas_localization_path = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-{session}", f"sub-{RID}_ses-{session}_desc-atlasLocalization.csv")
        if utils.checkIfFileExists(atlas_localization_path, printBOOL=False):
            atlas_localization = pd.read_csv(atlas_localization_path)
            
            
            atlas_localization.channel = channel2std_ECoG(atlas_localization.channel)
            
            
            for r in range(len(atlas_localization)):
                reg_AAL = atlas_localization.AAL_label[r]
                reg_BNA = atlas_localization.BN_Atlas_246_1mm_label[r]
                reg_HO = atlas_localization["HarvardOxford-combined_label"][r]
                
                
        coord_start_times = pd.DataFrame(columns = ["channel", "x", "y", "z", "start_time"])
        
        coord_start_times["channel"] = channels2
        
        for ch in range(len(coord_start_times)):
            x = np.array(atlas_localization[channels2[ch]  == atlas_localization.channel]["x"])[0]
            y = np.array(atlas_localization[channels2[ch]  == atlas_localization.channel]["y"])[0]
            z = np.array(atlas_localization[channels2[ch]  == atlas_localization.channel]["z"])[0]
            
            x = np.array(atlas_localization[channels2[ch]  == atlas_localization.channel]["x"])[0]
            y = np.array(atlas_localization[channels2[ch]  == atlas_localization.channel]["y"])[0]
            z = np.array(atlas_localization[channels2[ch]  == atlas_localization.channel]["z"])[0]
            
            
            coord_start_times.loc[coord_start_times["channel"] == coord_start_times["channel"][ch]   ,    "x"] = x
            coord_start_times.loc[coord_start_times["channel"] == coord_start_times["channel"][ch]   ,    "y"] = y
            coord_start_times.loc[coord_start_times["channel"] == coord_start_times["channel"][ch]   ,    "z"] = z
            
            spread_start
            
            channel_start = spread_start[coord_start_times["channel"][ch] == channels2 ] * skipWindow
            if len(channel_start) > 0:
                channel_start = channel_start[0]
                if channel_start > seizure_length:
                    channel_start = np.nan
            else:
                channel_start = np.nan
            coord_start_times.loc[coord_start_times["channel"] == coord_start_times["channel"][ch]   ,    "start_time"] = channel_start
          
            
        t1_image = glob.glob(join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-implant01", "tmp", "orig_nu_std.nii.gz" ))[0]
        t1_image_brain = glob.glob(join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-implant01", "tmp", "brain_std.nii.gz" ))[0]
        img = nib.load(t1_image)
        img_brain = nib.load(t1_image_brain)
        #utils.show_slices(img, data_type = "img")
        img_data = img.get_fdata()
        brain_data = img_brain.get_fdata()
        affine = img.affine
        shape = img_data.shape
        img_data_total = copy.deepcopy(img_data)
        img_data_total[  np.where(img_data_total != 0)  ] = 0
        
        img_data_N = copy.deepcopy(img_data)
        img_data_N[  np.where(img_data_N != 0)  ] = 0
        
        
       
        for ch in range(len(coord_start_times)):
            print(f"\r{ch}/{len(coord_start_times)}    ", end = "\r")
            coord = coord_start_times.iloc[ch]
            radius = 40
            
            img_data_sphere = copy.deepcopy(img_data)
            img_data_sphere[  np.where(img_data_sphere != 0)  ] = 0
            
            coordinates = np.array(coord[["x", "y", "z"]]).astype(float)
            coordinates_voxels = utils.transform_coordinates_to_voxel(coordinates, affine)
            x,y,z = coordinates_voxels[0],coordinates_voxels[1],coordinates_voxels[2]
            sphere = utils.make_sphere_from_point(img_data_sphere, x,   y,  z, radius = radius)
            if not np.isnan(coord_start_times.start_time[ch]):
                img_data_N = img_data_N + sphere
                sphere[np.where(sphere >0)] = coord_start_times.start_time[ch]
                img_data_total = img_data_total + sphere
            
            #utils.show_slices(sphere, data_type = "data")
            
        utils.show_slices(img_data_N, data_type = "data")
        utils.show_slices(img_data_total, data_type = "data", cmap = "mako")
        
        
        utils.show_slices(brain_data, data_type = "data")
        
    
    
        img_data_avg = img_data_total/img_data_N
        img_data_avg[np.where(brain_data <= 0)] = np.nan
        
        utils.show_slices(img, data_type = "img")
        utils.show_slices(img_data_avg, data_type = "data", cmap = "mako")
        
        #img_data_avg[np.isnan(img_data_avg)] = seizure_length
        
        low, middle, high = 0.33,0.48,0.7
        slices_t1 = [   img_data[:, int((img_data.shape[1]*low)), : ] , img_data[:, int(img_data.shape[1]*middle), :] , img_data[:, int(img_data.shape[1]*high), :]   ]
        slices_heat = [   img_data_avg[:, int((img_data_avg.shape[1]*low)), : ] , img_data_avg[:, int(img_data_avg.shape[1]*middle), :] , img_data_avg[:, int(img_data_avg.shape[1]*high), :]   ]
        slices_brain = [   brain_data[:, int((brain_data.shape[1]*low)), : ] , brain_data[:, int(brain_data.shape[1]*middle), :] , brain_data[:, int(brain_data.shape[1]*high), :]   ]
        
        cmap1 = "gray"
        cmap2 = "Wistia_r"
        """
        fig, axes = utils.plot_make()
        #sns.heatmap(slices_t1[1], cmap=cmap1, ax = axes, square = True)
        axes.imshow(slices_t1[1].T, cmap=cmap1, origin="lower")
        pos = axes.imshow(slices_heat[1].T, cmap=cmap2, origin="lower")
        fig.colorbar(pos, ax=axes)
        """
        slice_image = slices_heat[1]
        
        mask = np.where(~np.isnan(slice_image))
        interp = interpolate.NearestNDInterpolator(np.transpose(mask), slice_image[mask])
        filled_data = interp(*np.indices(slice_image.shape))
        
        filled_data_copy_gaussian = scipy.ndimage.gaussian_filter(filled_data, sigma = 2)
        
        filled_data_copy = copy.deepcopy(filled_data_copy_gaussian)
        filled_data_copy[np.where(slices_brain[1] <= 0)] = np.nan

        plt.style.use('default')
        cmap1 = "gray"
        cmap2 = "Spectral"
        fig, axes = utils.plot_make()
        axes.imshow(slices_t1[1].T, cmap=cmap1, origin="lower")
        pos = axes.imshow(filled_data_copy.T, cmap=cmap2, origin="lower")
        fig.colorbar(pos, ax=axes)
    

        
        plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES, "spread_by_coordinates", "spread_by_coordinates2.pdf"))











