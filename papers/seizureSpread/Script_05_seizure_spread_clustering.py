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
            outcomes.loc[outcomes["subject"] == outcomes["subject"][k], outcomes_list2[o]] = "unknown"
    
for k in range(len(outcomes)): #if poor outcome at 6 or 12 month, then propagate that thru if unknown
      outcomes_list2 = ["Engel_6_mo_binary", "Engel_12_mo_binary","Engel_24_mo_binary"]
      for o in [1,2]:
          value = np.array(outcomes.loc[outcomes["subject"] == outcomes["subject"][k]][outcomes_list2[o]])[0]
          if value =="unknown":
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
i=0
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
        
    
        
        
        region_activation = pd.DataFrame(columns = regions_unique )
        
        #find time at which that region became active as a percentage of seizure length
        
        channel_names = labels[i_patient]
        regions_patient = region[i_patient]



        len(channel_names)
        channels_region_index_label = []
        for r in range(len(regions_patient)):
            
            reg = str(np.unique(regions_patient[r].flatten())[0])
            
            reg = replace_region_name(reg)
            
            reg_index = np.where( reg ==  regions_unique  )[0]
            if len(reg_index) == 0:
                ind = -1
            else:
                ind = reg_index[0]
            channels_region_index_label.append(ind)
        channels_region_index_label = np.asarray(channels_region_index_label)
  
        channels_region_index_label.shape
        
        
        channel_activation_time = pd.DataFrame(columns = ["channel", "region_num", "activation_time"])
        
        len(channel_names)
        len(channel2std_ECoG(channel_names))
        np.array(channel_names)
        channel2std_ECoG(channel_names)
        
        channel_names = channel2std_ECoG(channel_names)
        channel_activation_time["channel"] = channel_names
        channel_activation_time["region_num"] = channels_region_index_label
        
        #get activation time
        
        for ch in range(len(channel_activation_time)):
            chan = channel_activation_time["channel"][ch]
            ind_overlap = np.where(chan == channels2  )[0]
            
            if len(ind_overlap) > 0:
                ind_chan = np.where(chan == channels2  )[0][0]
                chan_start = spread_start[ind_chan]
                chan_start_percent = chan_start/seizure_length
                if chan_start_percent > 1:
                    chan_start_percent = np.nan
                    
            else:
                chan_start_percent = np.nan
            channel_activation_time.loc[ch, 'activation_time'] = chan_start_percent

        
        channel_activation_time["activation_time"] = channel_activation_time["activation_time"].astype(float)
        
        
        channel_activation_time_only_times = channel_activation_time.drop("channel", axis= 1)
        region_activation_time = channel_activation_time_only_times.groupby(["region_num"], as_index=False).mean()
        
        reg_act_time = np.zeros(shape = (len(regions_unique)))
        reg_act_time[:] = 1
        for rrr in range(len(region_activation_time)):
            reg_ind = region_activation_time["region_num"][rrr]
            if not reg_ind == -1:
                reg_act_time[reg_ind] = region_activation_time["activation_time"][rrr]
        
    
        
        region_activation = region_activation.append(pd.DataFrame(reg_act_time.reshape(1,-1), columns=list(region_activation)), ignore_index=True)
        
        return region_activation
        
        
#%%  
   
tanh = False
model_IDs = ["WN","CNN","LSTM" , "absolute_slope", "line_length", "power_broadband"]
m=0
model_ID= model_IDs[m]
threshold = 0.6
#pd.DataFrame(columns = ["subject", "seizure"] +list( regions_unique) )
region_activation =pd.DataFrame(columns = list( regions_unique) )

seconds_active = np.arange(0,60*2+1,1)
percent_active = pd.DataFrame(data = seconds_active, columns = ["time"])


for i in range(len(patientsWithseizures)):
    print(f"\r{i}   { np.round(   (i+1)/len(patientsWithseizures)*100   ,2)}%                  ", end = "\r")
    region_activation_patient = calculate_mean_rank_deep_learning(i, patientsWithseizures, version, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, tanh = False)
    
    region_activation= region_activation.append(region_activation_patient, ignore_index=True)


     
        
#%%
region_activation_fillna = region_activation.fillna(1)
x_data = np.array(region_activation_fillna)

samples, nfeature = x_data.shape
#find where column are all the same
ind_same = []
for c in range(nfeature):
    if len(np.unique(x_data[:, c])) == 0:
        ind_same.append(c)
print(ind_same)
#%%
        
        

n_clusters = 3
        
pca_activation = PCA(n_components=2)        
        
        
principalComponents= pca_activation.fit_transform(x_data)        
        
        
pca_activation.explained_variance_ratio_

df = pd.DataFrame(principalComponents, columns = ["PC1", "PC2"])




kmeans = KMeans(init="random",n_clusters=n_clusters,n_init=10,max_iter=300,random_state=42)

kmeans.fit(x_data)

kmeans.inertia_
kmeans.n_iter_
kmeans.cluster_centers_

df["cluster"] = kmeans.labels_

df["cluster"] = df["cluster"].astype(str)

#sns.scatterplot(data = df, x = "PC1", y = "PC2", s = 5, hue = "cluster" , palette = {"0":"#9b59b6", "1":"#3498db",   "2":"#95a5a6"}, linewidth=0)
sns.scatterplot(data = df, x = "PC1", y = "PC2", s = 12, hue = "cluster" , palette = "tab10" , linewidth=0)

#%%
df["subject"] = patientsWithseizures["subject"]
df["seizure"] = patientsWithseizures["idKey"]

subject_categories = []
subjects_to_plot = ["RID0442", "RID0309", "RID0278", "RID0267"]
for s in range(len(df)):
    sub = patientsWithseizures["subject"][s]
    if sub in subjects_to_plot:
        sub = sub
    else:
        sub = "other"
    
    subject_categories.append(sub)
    
df["subject_category"] = subject_categories
sns.scatterplot(data = df.sort_values(by =["subject_category"], ascending = False), x = "PC1", y = "PC2", s = 20, hue = "subject_category" , palette = "Spectral" , linewidth=0, hue_order= ["other"] + subjects_to_plot)

df.sort_values(by =["subject_category"], ascending = False)
sns.color_palette("Set2")
#%%

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
sse = []
kmeans_clusters_max = 10
for k in range(1, kmeans_clusters_max):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_data)
    sse.append(kmeans.inertia_)

plt.plot(range(1, kmeans_clusters_max), sse)
#plt.xticks(range(1, kmeans_clusters_max))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()















       
        