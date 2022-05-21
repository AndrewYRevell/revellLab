

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
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "unknown"
    
for k in range(len(outcomes)): #if poor outcome at 6 or 12 month, then propagate that thru if unknown
      outcomes_list2 = ["Engel_6_mo_binary", "Engel_12_mo_binary","Engel_24_mo_binary"]
      for o in [1,2]:
          value = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o]])[0]
          if value =="unknown":
              value_previous = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o-1]])[0]
              if value_previous == "poor":
                  outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "poor"
                  
         


with open(paths.ATLAS_FILES_PATH) as f: atlas_files = json.load(f)

#%%
tanh = False

i=0
type_of_overlap = "soz"
override_soz= True
threshold=0.6
threshold=0.71
smoothing = 20
model_ID="WN"
seconds_active = np.arange(0,60*2+1,1)

def calculate_mean_rank_deep_learning(i, patientsWithseizures, version, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, type_of_overlap = "soz", override_soz = False, seconds_active  = None, tanh = False):

    #%%
    RID = np.array(patientsWithseizures["subject"])[i]
    idKey = np.array(patientsWithseizures["idKey"])[i]
    seizure_length = patientsWithseizures.length[i]
    
    print(RID)
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
    
    if seconds_active is None:
        seconds = np.arange(0,60*2+1,1)
    else:
        seconds = seconds_active
    percent_active_vec = np.zeros(len(seconds))
    percent_active_vec[:] = np.nan
    
    if utils.checkIfFileExists( spread_location_file , printBOOL=False) and utils.checkIfFileExists( location_abs_slope_file , printBOOL=False):
        #print("\n\n\n\nSPREAD FILE EXISTS\n\n\n\n")
    
    
    
        #Getting SOZ labels
        RID_keys =  list(jsonFile["SUBJECTS"].keys() )
        hup_num_all = [jsonFile["SUBJECTS"][x]["HUP"]  for  x   in  RID_keys]
        
        hup_int = hup_num_all[RID_keys.index(RID)]
        hup_int_pad = f"{hup_int:03d}" 
        
        #i_patient = patients.index(f"HUP{hup_int_pad}")
        #HUP = patients[i_patient]
        #hup = int(HUP[3:])
        
    
        
        #channel_names = labels[i_patient]
        #soz_ind = np.where(soz[i_patient] == 1)[0]
        #soz_channel_names = np.array(channel_names)[soz_ind]
        
        #resected_ind = np.where(resect[i_patient] == 1)[0]
        #resected_channel_names = np.array(channel_names)[resected_ind]
        
        #ignore_ind = np.where(ignore[i_patient] == 1)[0]
        #ignore__channel_names = np.array(channel_names)[ignore_ind]
        
        #soz_channel_names = echobase.channel2std(soz_channel_names)
        #resected_channel_names = echobase.channel2std(resected_channel_names)
        #ignore__channel_names = echobase.channel2std(ignore__channel_names)
        

        #soz_channel_names = channel2std_ECoG(soz_channel_names)
        #resected_channel_names = channel2std_ECoG(resected_channel_names)
        #ignore__channel_names = channel2std_ECoG(ignore__channel_names)
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
        
        
        
        
        
        print(i)
        RID = np.array(patientsWithseizures["subject"])[i]
        seizure = np.array(patientsWithseizures["idKey"])[i]
        seizure_length = patientsWithseizures.length[i]
        
        #atlas
        atlas = "BN_Atlas_246_1mm"
        atlas = "AAL3v1_1mm"
        #atlas = "AAL2"
        #atlas = "HarvardOxford-sub-ONLY_maxprob-thr25-1mm"
        
        
        atlas_names_short =  list(atlas_files["STANDARD"].keys() )
        atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
        ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
        atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
        
        atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
        atlas_label_names = np.array(atlas_label.iloc[1:,1])
        atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
    
        
        connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"sub-{RID}.{atlas}.count.pass.connectogram.txt")
        
        connectivity_loc_glob = glob.glob( connectivity_loc  )
        
        
        if len(connectivity_loc_glob) > 0:
            connectivity_loc_path = connectivity_loc_glob[0]

            
            sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)
            sc = sc/sc.max()
            #sc = utils.log_normalize_adj(sc)
            #sc=utils.log_normalize_adj(sc)
            sc_region_labels = utils.read_DSI_studio_Txt_files_SC_return_regions(connectivity_loc_path, atlas).astype(ind)

        
        
        
            
            atlas_localization_path = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{RID}", f"ses-{session}", f"sub-{RID}_ses-{session}_desc-atlasLocalization.csv")
            if utils.checkIfFileExists(atlas_localization_path, printBOOL=False):
                atlas_localization = pd.read_csv(atlas_localization_path)
                
                
                atlas_localization.channel = channel2std_ECoG(atlas_localization.channel)
                #get channels in hipp
                
                
                
                
                
                ##############################################################
                ##############################################################
                ##############################################################
                ##############################################################
                ##############################################################
                ##############################################################
                #find the activation time between channels
                
                spread_start
                channels2
                
                sc_vs_time = pd.DataFrame(columns= ["ch1", "ch2", "time", "sc"])
                
                ch1 = 1
                ch2 = 10
                """
                for ch1 in range(len(channels2)):
                    for ch2 in range(len(channels2)):
                        time_between = abs(spread_start[ch1] - spread_start[ch2])*skipWindow
                        if spread_start[ch1]*skipWindow > seizure_length:
                            time_between = np.nan
                            
                        if spread_start[ch2]*skipWindow > seizure_length:
                            time_between = np.nan
                        
                        ch1_name = channels2[ch1]
                        ch2_name = channels2[ch2]
                        ch1_ind_atlas_loc_ind = np.where(ch1_name == atlas_localization.channel )[0]
                        ch2_ind_atlas_loc_ind = np.where(ch2_name == atlas_localization.channel )[0]
                        if len(ch1_ind_atlas_loc_ind) >0:
                            if len(ch2_ind_atlas_loc_ind)>0:
                                region_num1 = atlas_localization[f"{atlas}_region_number"][ch1_ind_atlas_loc_ind[0]]
                                region_num2 = atlas_localization[f"{atlas}_region_number"][ch2_ind_atlas_loc_ind[0]]
                                
                                
                                
                                if region_num1 in sc_region_labels:
                                    if region_num2 in sc_region_labels:
                                        if not region_num2 == region_num1:
                                            ch1_sc_ind = np.where(region_num1 == sc_region_labels)[0][0]
                                            ch2_sc_ind = np.where(region_num2 == sc_region_labels)[0][0]
                                            connectivity = sc[ch1_sc_ind, ch2_sc_ind]
                                            
                                            sc_vs_time = sc_vs_time.append( dict(ch1 = ch1_name, ch2 = ch2_name , time = time_between, sc = connectivity) , ignore_index=True)
                        
                sc_vs_time["inverse"] = 1/sc_vs_time["time"]
                fig, axes = utils.plot_make()
                sns.regplot(data = sc_vs_time, x = "sc", y = "time", ax = axes, scatter_kws=dict(s = 1))
                
                sc_vs_time.loc[sc_vs_time["time"] == 0, "inverse"] = 0
                
                sc_vs_time_nanfill = sc_vs_time.fillna(0)
                
                axes.set_ylim([0,1])            
                axes.set_xlim([0,0.5])            
                
                """
                #get the average time each region was active
                region_times = pd.DataFrame(columns = ["region", "time"])
                for r in range(len(sc_region_labels)):
                    reg_num = sc_region_labels[r]
                    channels_in_reg = np.where(reg_num == atlas_localization[f"{atlas}_region_number"])[0]
                    
                    reg_starts = []
                    if len(channels_in_reg) >0:
                        for ch in range(len(channels_in_reg)):
                            ch_name = atlas_localization.channel[channels_in_reg[ch]] 
                            ch_in_spread = np.where(ch_name == channels2)[0]
                            if len(ch_in_spread)>0:
                                if spread_start[ch_in_spread[0]]*skipWindow  > seizure_length:
                                    reg_starts.append(np.nan)
                                else:
                                    reg_starts.append(spread_start[ch_in_spread[0]]*skipWindow )
                        reg_mean = np.nanmean(reg_starts)
                    else:
                        reg_mean = np.nan
                    region_times = region_times.append(dict(region = reg_num, time = reg_mean), ignore_index=True)
                    
                sc_vs_time_reg =  pd.DataFrame(columns= ["reg1", "reg2", "time", "sc"])
                
                for r1 in range(len(sc)):
                    for r2 in range(r1+1, len(sc)):
                        connectvity = sc[r1, r2]
                        time_diff = abs(region_times.iloc[r1, 1] - region_times.iloc[r2, 1])
                        sc_vs_time_reg = sc_vs_time_reg.append(dict(reg1 = sc_region_labels[r1],  reg2 = sc_region_labels[r2], time = time_diff, sc = connectvity), ignore_index=True)
                
                
                #fig, axes = utils.plot_make()
                #sns.scatterplot(data = sc_vs_time_reg, x = "sc", y = "time", linewidth = 0, s=5)
                
                fig, axes = utils.plot_make(size_length=5)
                g = sns.regplot(data = sc_vs_time_reg, x = "sc", y= "time", scatter_kws = dict( linewidth=0, s=50), ci = None, line_kws=dict(lw = 7, color = "black"))
           
                x = sc_vs_time_reg["sc"]
                y = sc_vs_time_reg["time"]
                y_nanremoved = y[~np.isnan(y)]
                x_nanremoved = x[~np.isnan(y)]
                corr = spearmanr(x_nanremoved,y_nanremoved)
                corr = pearsonr(x_nanremoved,y_nanremoved)
                corr_r = np.round(corr[0], 3)
                corr_p = np.round(corr[1], 8)
    
                axes.set_title(f"r = {corr_r}, p = {corr_p}")
                #axes.set_ylim([-0.033,0.2])
                for l, tick in enumerate(axes.xaxis.get_major_ticks()):
                    tick.label.set_fontsize(6)        
                axes.tick_params(width=4) 
                # change all spines
                for axis in ['top','bottom','left','right']:
                    axes.spines[axis].set_linewidth(6)
                
   #%%         
            plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", f"sc_vs_spread_time_SINGLE_PATIENT_{RID}_01.pdf"), bbox_inches='tight')     

            
            
            
            
            #sns.regplot(data = sc_vs_time_reg, x = "time", y = "sc", scatter_kws=dict(s = 1))
            
                       
            sc_vs_time_reg["inverse"] = 1/sc_vs_time_reg["time"]
            #sc_vs_time_reg.loc[sc_vs_time["time"] == 0, "inverse"] = 0
        
            sc_vs_time_reg_fill = copy.deepcopy(sc_vs_time_reg)
            sc_vs_time_reg_fill = sc_vs_time_reg_fill.fillna(0)
            
            sns.scatterplot(data = sc_vs_time_reg_fill, x = "sc", y = "inverse", linewidth = 0, s=5)
            
            
            
            fig, axes = utils.plot_make(size_length=5)
            g = sns.regplot(data = sc_vs_time_reg_fill, x = "sc", y= "time", scatter_kws = dict( linewidth=0, s=20), ci = None, line_kws=dict(lw = 5, color = "black"))
       
            x = sc_vs_time_reg_fill["sc"]
            y = sc_vs_time_reg_fill["time"]
            y_nanremoved = y[~np.isnan(y)]
            x_nanremoved = x[~np.isnan(y)]
            corr = spearmanr(x_nanremoved,y_nanremoved)
            corr = pearsonr(x_nanremoved,y_nanremoved)
            corr_r = np.round(corr[0], 2)
            corr_p = np.round(corr[1], 10)

            axes.set_title(f"{corr_r}, p = {corr_p}")
            #axes.set_ylim([-0.033,0.2])
            for i, tick in enumerate(axes.xaxis.get_major_ticks()):
                tick.label.set_fontsize(6)        
            axes.tick_params(width=4) 
            # change all spines
            for axis in ['top','bottom','left','right']:
                axes.spines[axis].set_linewidth(6)
            
            
               
            plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"connectivity", f"sc_vs_spread_time_SINGLE_PATIENT_{RID}_02.pdf"), bbox_inches='tight')     

            
            
            
            
            
            
            
          
        