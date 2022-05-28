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
import scipy.cluster.hierarchy as shc

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
import matplotlib
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
print(f"number of patients: {len(np.unique(patientsWithseizures['subject']))}")
print(f"number of seizures: {len(patientsWithseizures)}")  
#%%
i=132
type_of_overlap = "soz"
threshold=0.69
smoothing = 20
model_ID="WN"
tanh = False

def calculate_mean_rank_deep_learning(i, patientsWithseizures, version, atlas_name_to_use, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, tanh = False, use_atlas = False):
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

            
        if not utils.checkIfFileExists(atlas_localization_path, printBOOL=False):
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
                return region_activation
                    
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
        
        
        channel_names = channel2std_ECoG(channel_names)
        channel_activation_time["channel"] = channel_names
        channel_activation_time["region_num"] = channels_region_index_label
        
        #get activation time
        for ch in range(len(channel_activation_time)):
            chan = channel_activation_time["channel"][ch]
            ind_overlap = np.where(chan == channels2  )[0]
            
            if len(ind_overlap) > 0:
                ind_chan = np.where(chan == channels2  )[0][0]
                chan_start = spread_start[ind_chan] * skipWindow
                chan_start_percent = chan_start/seizure_length
                if chan_start_percent > 1:
                    chan_start_percent = np.nan
                    
            else:
                chan_start_percent = np.nan
            channel_activation_time.loc[ch, 'activation_time'] = chan_start_percent

     
        channel_activation_time["activation_time"] = channel_activation_time["activation_time"].astype(float)
        
    
        channel_activation_time_only_times = channel_activation_time.drop("channel", axis= 1)
        
        channel_activation_time_only_times= channel_activation_time_only_times.astype('float')
        region_activation_time = channel_activation_time_only_times.groupby(["region_num"], as_index=False).mean()
        
        reg_act_time = np.zeros(shape = (len(atlas_label_names)))
        reg_act_time[:] = np.nan
        for rrr in range(len(region_activation_time)):
            reg_ind = region_activation_time["region_num"][rrr]
            if not reg_ind == -1:
                reg_act_time[int(reg_ind)] = region_activation_time["activation_time"][rrr]
        
        region_activation = region_activation.append(pd.DataFrame(reg_act_time.reshape(1,-1), columns=list(region_activation)), ignore_index=True)
    

        
        return region_activation
        
        
#%%  
use_atlas = False   
tanh = True
model_IDs = ["WN","CNN","LSTM" , "absolute_slope", "line_length", "power_broadband"]
m=0
model_ID= model_IDs[m]
threshold = 0.69
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

region_activation =pd.DataFrame(columns = list( atlas_label_names) )

for i in range(len(patientsWithseizures)):
    print(f"\r{i}   { np.round(   (i+1)/len(patientsWithseizures)*100   ,2)}%                  ", end = "\r")
    region_activation_patient = calculate_mean_rank_deep_learning(i, patientsWithseizures, version, atlas_name_to_use = atlas_name_to_use, threshold=threshold, smoothing = 20, model_ID=model_ID, secondsAfter=180, secondsBefore=180, tanh = tanh, use_atlas = use_atlas)
    
    region_activation= region_activation.append(region_activation_patient, ignore_index=True)

#%%
region_activation_fillna = copy.deepcopy(region_activation.fillna(1))

#remove redundant or rarely implanted regions that provide little utility
if atlas == "AAL3v1_1mm":
    indexes = [16,17, 94,95,96,97] + list(range(94,169))
    region_activation_fillna = region_activation_fillna.drop(region_activation.columns[indexes], axis = 1)
    mask = np.ones(len(atlas_label_names), bool)
    mask[indexes] = 0
    atlas_label_names_new = atlas_label_names[mask]
    
if atlas == "AAL2":
    indexes = [16,17] + list(range(94,120))
    region_activation_fillna = region_activation_fillna.drop(region_activation.columns[indexes], axis = 1)
    mask = np.ones(len(atlas_label_names), bool)
    mask[indexes] = 0
    atlas_label_names_new = atlas_label_names[mask]
    

x_data = np.array(region_activation_fillna)

"""

samples, nfeature = x_data.shape
#find where column are all the same
ind_same = []
for c in range(nfeature):
    if len(np.unique(x_data[:, c])) == 0:
        ind_same.append(c)
print(ind_same)
"""

#%%
SIZE = 300

n_clusters = 5
pca_nclusters = 30
pca_activation = PCA(n_components=pca_nclusters)        
        
        
principalComponents= pca_activation.fit_transform(x_data)        
        
        
pca_activation.explained_variance_ratio_

df = pd.DataFrame(principalComponents[:, 0:3], columns = ["PC1", "PC2", "PC3"])


palette = sns.color_palette("tab20")
palette[0] = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
palette[2] = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
palette[3] =  (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
palette1 = palette[0:n_clusters] 
palette2 = {"0": "#D62728", "1": "#9567BE", "2": "#5b1111", "3": "#1F78B5", "4": "#84b2ec" , "5": "#00ff00", "6": "#ff0000", "7": "#ff00ff"}
palette2 = {"0": "#5b1111", "1": "#9567BE", "2": "#c94849", "3": "#1F78B5", "4": "#84b2ec", "5": "#00ff00" , "6": "#ff0000", "7": "#ffff00"}


"""
kmeans = KMeans(init="random",n_clusters=n_clusters, n_init=10,max_iter=300,random_state=6)

kmeans.fit(x_data)

kmeans.inertia_
kmeans.n_iter_
kmeans.cluster_centers_
df["cluster"] = kmeans.labels_

df["cluster"] = df["cluster"].astype(str)
"""








#%%

palette2 = {"1": "#5b1111", "3": "#9567BE", "2": "#c94849", "4": "#1F78B5", "0": "#84b2ec", "5": "#00ff00" , "6": "#ff00ff", "7": "#ffff00", "8": "#666666"}
#for sdc in range(0,20):
Z = shc.linkage(principalComponents[:,0:6], method ='complete', optimal_ordering = True)
fc_cluster = shc.fcluster(Z, t=5, criterion='maxclust' )-1
df["cluster"] = fc_cluster
df["cluster"] = df["cluster"].astype(str)


palette2_light = {"1": "#9dbde3", "3": "#da8384", "2": "#2c94db", "4": "#be2323", "0": "#ba9dd5" }
palette2_clusters = {"1": "#629ce6", "3": "#c94849", "2": "#1c6da5", "4": "#5b1111", "0": "#9567BE" }



fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(ax = axes,data = df, x = "PC1", y = "PC2", s = SIZE, hue = "cluster" , palette = palette2_clusters , linewidth=0)

#axes.set_title(sdc)

#sns.scatterplot(data = df, x = "PC1", y = "PC2", s = 5, hue = "cluster" , palette = {"0":"#9b59b6", "1":"#3498db",   "2":"#95a5a6"}, linewidth=0)
#fig, axes = utils.plot_make(size_length=10, size_height=6)
#sns.scatterplot(ax = axes,data = df, x = "PC1", y = "PC2", s = SIZE, hue = "cluster" , palette = palette2 , linewidth=0)


axes.set_yticks(np.arange(-1, 4, 1))
#axes.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(5))
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#plt.show()
#plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_minifig_big.pdf"), bbox_inches='tight')
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_clusters_big.pdf"), bbox_inches='tight')
#%%Dendrogram

#turn patient RIDS to subject numbers 
shc.set_link_color_palette(['#9567BE', '#629ce6', '#1c6da5', '#c94849', "#5b1111"])

subs_unique = np.unique(patientsWithseizures["subject"])
subs_ids = np.array(range(len(subs_unique)))+1

subs_ids_vanilla = []
for s in range(len(patientsWithseizures["subject"])):
    su = patientsWithseizures["subject"][s]
    ind = np.where(subs_unique ==su )[0][0]
    subs_ids_vanilla.append(subs_ids[ind])
    
    
    
fig, ax = utils.plot_make(size_length=12, size_height=8)
dend = shc.dendrogram(Z,  labels = np.array(subs_ids_vanilla), above_threshold_color='#000000', leaf_rotation = 270, leaf_font_size = 3)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0)
ax.set_yticks([])

#%%PCA explained variance


fig, axes = utils.plot_make(size_length=20, size_height=6)
sns.lineplot(x = range(1,pca_nclusters+1), y = pca_activation.explained_variance_ratio_, lw = 8)
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
axes.set_title("explained_variance_ratio")
axes.set_ylabel("explained_variance_ratio")
axes.set_xlabel("n components")
axes.tick_params(width=4)
axes.set_xlim([0,30])
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "explained_variance_ratio.pdf"), bbox_inches='tight')




last = Z[-20:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)






dend = shc.dendrogram(Z,   truncate_mode = "level", p = 5, show_leaf_counts = True)



fc_cluster[168]
fc_cluster[60]
fc_cluster[150]
fc_cluster[183]

fc_cluster[205]





#%%

region_activation_class_avg = copy.deepcopy(region_activation_fillna)
region_activation_class_avg = region_activation_class_avg.replace(1, np.nan)
region_activation_class_avg["cluster"] =  fc_cluster#kmeans.labels_


region_activation_fillna_about = copy.deepcopy(region_activation_fillna)
region_activation_fillna_about["seizure"] = patientsWithseizures["idKey"]
region_activation_fillna_about["subject"] = patientsWithseizures["subject"]
region_activation_fillna_about["cluster"] =  fc_cluster#kmeans.labels_

region_activation_fillna_about.to_csv(join(paths.SEIZURE_SPREAD_DERIVATIVES_CLUSTERING, "Seizure_clusters.csv"))
#%%

############################################
############################################
############################################
############################################
#DO NOT RUN DO NOT RUN

############################################
############################################
############################################
############################################
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

linked = linkage(x_data)
dendrogram(linked, orientation='top',distance_sort='descending',)

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")

dend = shc.dendrogram(shc.linkage(x_data, method='complete'))


from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters = 5 , linkage = "single")
model.fit_predict(x_data)

df["cluster_ag"] = model.fit_predict(x_data)
df["cluster_ag"] = df["cluster_ag"].astype(str)

#sns.scatterplot(data = df, x = "PC1", y = "PC2", s = 5, hue = "cluster" , palette = {"0":"#9b59b6", "1":"#3498db",   "2":"#95a5a6"}, linewidth=0)
fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(ax = axes,data = df, x = "PC1", y = "PC2", s = SIZE, hue = "cluster_ag" , palette = palette2 , linewidth=0)

Dendrogram = shc.dendrogram((shc.linkage(x_data, method ='ward')))


############################################
############################################
############################################
############################################
############################################
############################################
fc_cluster = fcluster(shc.linkage(principalComponents[:,0:5], method ='ward', optimal_ordering = True), t=5, criterion='maxclust' )-1
df["cluster"] = fcluster(shc.linkage(principalComponents[:,0:5], method ='ward', optimal_ordering = True), t=5, criterion='maxclust' )-1
df["cluster"] = df["cluster"].astype(str)
fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(ax = axes,data = df, x = "PC1", y = "PC2", s = SIZE, hue = "cluster" , palette = palette2 , linewidth=0)

############################################
############################################
############################################
############################################
############################################
############################################
############################################




def extract_levels(row_clusters, labels):
    clusters = {}
    for row in range(row_clusters.shape[0]):
        cluster_n = row + len(labels)
        # which clusters / labels are present in this row
        glob1, glob2 = row_clusters[row, 0], row_clusters[row, 1]

        # if this is a cluster, pull the cluster
        this_clust = []
        for glob in [glob1, glob2]:
            if glob > (len(labels)-1):
                this_clust += clusters[glob]
            # if it isn't, add the label to this cluster
            else:
                this_clust.append(glob)

        clusters[cluster_n] = this_clust
    return clusters

tmp =extract_levels(linkage(x_data, method='ward')   , range(len(x_data)))


ac2 = AgglomerativeClustering(n_clusters = 6)







fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(df["PC1"], df["PC2"], df["PC3"], cmap = palette2)


my_cmap = plt.get_cmap('hsv')

#%%
palette = sns.color_palette("Set2")
palette=  [ (0.8,0.8,0.8)] + palette 
df["subject"] = patientsWithseizures["subject"]
df["seizure"] = patientsWithseizures["idKey"]
#palette[0:len(subjects_to_plot)+1] 
palette2 = {"RID0472": "#5b1111", "RID0278": "#9567BE", "RID0238": "#c94849", "RID0522": "#1F78B5", "RID0060": "#AEC8E9" , "other": "#CDCDCD" }
palette2 = {"RID0472": "#9567BE", "RID0278": "#c94849", "RID0238": "#1F78B5", "RID0522": "#84b2ec", "RID0060": "#c98848" , "other": "#CDCDCD" }
palette2 = {"RID0472": "#9567BE", "RID0278": "#c94849", "RID0238": "#1F78B5", "RID0522": "#84b2ec", "RID0060": "#c98848" , "other": "#CDCDCD" }

np.unique(patientsWithseizures["subject"])
len(np.unique(patientsWithseizures["subject"]))

subject_categories = []
subjects_to_plot = ["RID0472", "RID0278", "RID0238", "RID0522", "RID0060" ]
subjects_to_plot = ["RID0472", "RID0278", "RID0238", "RID0522", "RID0060" ]
#subjects_to_plot = ["RID0442", "RID0309", "RID0365", "RID0472"]
#subjects_to_plot = ["RID0055", "RID0024", "RID0021", "RID0020", "RID0014" ]
for s in range(len(df)):
    sub = patientsWithseizures["subject"][s]
    if sub in subjects_to_plot:
        sub = sub
    else:
        sub = "other"
    
    subject_categories.append(sub)
    
df["subject_category"] = subject_categories

df_ordered = df.sort_values(by =["subject_category"], ascending = False)

fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_ordered, x = "PC1", y = "PC2", s = SIZE, hue = "subject_category" , palette = palette2 , linewidth=0, hue_order= ["other"] + subjects_to_plot)


# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_patients_big.pdf"), bbox_inches='tight')

#%%
df_outcome = pd.merge(df, outcomes, on='subject')

df_outcome_drop = copy.deepcopy(df_outcome)
df_outcome_drop.drop(df_outcome_drop[df_outcome_drop['Engel_24_mo_binary']  == "NA"].index, inplace = True)

fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_outcome_drop, x = "PC1", y = "PC2", s = SIZE, hue = "Engel_24_mo_binary"  , linewidth=0, hue_order=["good", "poor"], palette= dict(good = "#420067", poor = "#c17d00") )

# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_outcome_big.pdf"), bbox_inches='tight')

#%%
target_categories = []
palette_target = {"Temporal": "#c94849", "Frontal": "#1F78B5", "Parietal": "#9567BE", "RID0522": "#AEC8E9", "RID0060": "#c98848" , "other": "#CDCDCD" }

for s in range(len(df_outcome)):
    tar = np.array(df_outcome["Target"])[s]
    if tar == "Insular":
        tar = "Temporal"
    elif tar == "MTL":
        tar = "Temporal"
    elif tar == "MFL":
        tar = "Frontal"
    elif tar == "FP":
        tar = "Frontal"
    target_categories.append(tar)
    
    
df_outcome["target_category"] = target_categories

fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_outcome, x = "PC1", y = "PC2", s = SIZE, hue = "target_category"  , linewidth=0, palette= palette_target)




# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_target_category_big.pdf"), bbox_inches='tight')
#%%
fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_outcome, x = "PC1", y = "PC2", s = SIZE, hue = "Laterality"  , linewidth=0, palette= dict(L = "#a53132", R = "#151515", LR = "#a53132") )

# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_Laterality_big.pdf"), bbox_inches='tight')
#%%
palette_implant = {"SEEG": "#d75253", "ECoG": "#458fd5" }
palette_Lesion_status = {"Non-Lesional": "#2669a8", "Lesional": "#d58b45" }
palette_gender = {"M": "#669dd2", "F": "#d2669d" }

fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_outcome, x = "PC1", y = "PC2", s = SIZE, hue = "Implant"  , linewidth=0, palette=palette_implant)
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_Implant_big.pdf"), bbox_inches='tight')



fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_outcome, x = "PC1", y = "PC2", s = SIZE, hue = "Lesion_status"  , linewidth=0, palette=palette_Lesion_status)
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_Lesion_status_big.pdf"), bbox_inches='tight')




fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_outcome, x = "PC1", y = "PC2", s = SIZE, hue = "Gender"  , linewidth=0, palette=palette_gender)
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_Gender_big.pdf"), bbox_inches='tight')






tmp = region_activation_fillna_about[['subject', 'cluster']]

#%% Plot on length
time_cutoffs = [30,60,120]
subject_cutoffs = []
for s in range(len(df)):
    length = patientsWithseizures["length"][s]
    
    cutoff_ind = np.where(length<time_cutoffs)[0]
    if len(cutoff_ind) > 0:
        cutoff = f"<{time_cutoffs[cutoff_ind[0]]}"
    else:
        cutoff = f">{time_cutoffs[-1]}"
    subject_cutoffs.append(cutoff)

df["seizure_length"] = subject_cutoffs
fig, axes = utils.plot_make(size_length=10, size_height=6)

sns.scatterplot(data = df, x = "PC1", y = "PC2", s = SIZE, hue = "seizure_length"  , linewidth=0, hue_order= ["<30", "<60", "<120", ">120"], palette={"<30": "#ec9b9c", "<60": "#ac2124", "<120": "#267bd0", ">120": "#14406c" }   )
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_seizure_length_big.pdf"), bbox_inches='tight')

#%%

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
sse = []
kmeans_clusters_max = 30
for k in range(1, kmeans_clusters_max):
    kmeans_total = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans_total.fit(x_data)
    sse.append(kmeans_total.inertia_)


fig, axes = utils.plot_make(size_length=20, size_height=6)
sns.lineplot(x = range(1,kmeans_clusters_max), y = sse, lw = 8, color = "#8586e8")
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
axes.set_title("k means clustering")
axes.set_xlabel("Number of Clusters")
axes.set_ylabel("SSE")
axes.tick_params(width=4)
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "kmeans_clustering.pdf"), bbox_inches='tight')


#%%
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
#Number of regions active for each seizure cluster

region_activation_class_avg

num_regions_active = []

for r in range(len(region_activation_class_avg)):
    sample = region_activation_class_avg.iloc[r,:-1]
    sample[np.isnan(sample)] = -1
    num_regions_active.append(len(np.where(sample >= 0)[0]))


region_activation_class_avg_regions_active = copy.deepcopy(region_activation_class_avg)


region_activation_class_avg_regions_active["num_regions"] = np.array(num_regions_active)

region_activation_class_avg_regions_active["cluster"] = region_activation_class_avg_regions_active["cluster"].astype(str)

#%%
palette2_light = {"1": "#9dbde3", "3": "#da8384", "2": "#2c94db", "4": "#be2323", "0": "#ba9dd5" }
palette2_clusters = {"1": "#629ce6", "3": "#c94849", "2": "#1c6da5", "4": "#5b1111", "0": "#9567BE" }

{"1": "#c94849", "3": "#5b1111", "2": "#1f78b5", "4": "#84b2ec", "0": "#9567BE" }
{"1": "#be2323", "3": "#ba9dd5", "2": "#da8384", "4": "#2c94db", "0": "#bcd2ec" }


cluster_order = ["2", "1", "0", "4", "3"]

fig, axes = utils.plot_make(size_length=30)
sns.boxplot(data = region_activation_class_avg_regions_active, x = "cluster", y = "num_regions", palette= palette2_light, order=cluster_order, width=0.5)
sns.swarmplot(data = region_activation_class_avg_regions_active, x = "cluster", y = "num_regions", palette= palette2_clusters, order=cluster_order, s = 5)

for i,artist in enumerate(axes.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    #artist.set_facecolor('None')

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    for j in range(i*6,i*6+6):
        line = axes.lines[j]
        #line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

utils.fix_axes(axes)
utils.reformat_boxplot(axes)

    
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "kmeans_clustering_number_of_regions_long4.pdf"), bbox_inches='tight')       
#%%

#Plot averages of atlas and clusters

#%
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################

###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################



     
        
#%

#creating new localization
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
atlas_path = join(paths.ATLASES, atlas_files["STANDARD"][atlas_names_short[ind]]["name"])
img = nib.load(atlas_path)
#utils.show_slices(img, data_type = "img")
img_data = img.get_fdata()
affine = img.affine
shape = img_data.shape


region_activation_class_avg = copy.deepcopy(region_activation_fillna)
region_activation_class_avg = copy.deepcopy(region_activation)
region_activation_class_avg = region_activation_class_avg.replace(1, np.nan)
region_activation_class_avg["cluster"] = fc_cluster #kmeans.labels_ ####################################################################


region_activation_class_avg_group = region_activation_class_avg.groupby(["cluster"], as_index=False).mean()
region_activation_class_avg_group = region_activation_class_avg_group.fillna(1)

region_activation_class_avg_more_than_one_n = copy.deepcopy(region_activation_class_avg)
#only take mean if more than one patient with sampling in any given region
for c in range(len(region_activation_class_avg_group)):
    
    reg_cluster = region_activation_class_avg_more_than_one_n[region_activation_class_avg_more_than_one_n["cluster"] == c]
    for col in range(reg_cluster.shape[1]-1):
        
        number_nan = np.where(np.isnan(reg_cluster.iloc[:, col ]  )    )[0]
        if len(number_nan) == len(reg_cluster)-1:
            col_name = reg_cluster.iloc[:,col ].name
            region_activation_class_avg_more_than_one_n.loc[region_activation_class_avg_more_than_one_n["cluster"] == c, col_name] = np.nan

a5= reg_cluster

region_activation_class_avg_group = region_activation_class_avg_more_than_one_n.groupby(["cluster"], as_index=False).mean()
region_activation_class_avg_group = region_activation_class_avg_group.fillna(1)
a6= region_activation_class_avg_group
#a7= region_activation_class_avg_group2

activation_all_clusters = []
for cluster in range(n_clusters):
    print(f"\n {cluster}")
    activation = pd.DataFrame(columns =[ "region", "activation"])
    activation["region"] = atlas_label_names
    activation["activation"] = 0
    activation["activation"] =   np.array(region_activation_class_avg_group.iloc[cluster,1:] )

    activation_all_clusters.append(activation)

    
a1 = region_activation_fillna_about[["subject", "cluster"]]
a2 = region_activation_fillna_about
a3 = region_activation_class_avg_group

region_activation_class_avg_group.to_csv(join(paths.SEIZURE_SPREAD_DERIVATIVES_CLUSTERING, "Seizure_clusters_average.csv"))

#%%
for cluster in range(5):
    print(f"\n {cluster}")
    
    img_data_activation = copy.deepcopy(img_data)
    region_nums = np.array(atlas_label.iloc[1:,0])
    for a in range(len(region_nums)):
        region_num = float(region_nums[a])  
        act = activation_all_clusters[cluster]["activation"][a]
        #if np.isnan(act):
        #    act = np.nanmax(activation_aal_all_clusters[cluster]["activation"])
        img_data_activation[np.where(region_num == img_data)] = act
    
    img_data_activation[np.where(img_data_activation == 0)] = np.nan
    
    #utils.show_slices(img_data_activation, data_type = "data", cmap = "Spectral")
    
    utils.save_nib_img_data(img_data_activation, img, join(BIDS, project_folder, f"atlases", f"more_patients_cluster_{cluster}.nii.gz" ) )
    



#%%

cluster = 3
for cluster in range(5):
    
    cluster_path = join(BIDS, project_folder, f"atlases", f"more_patients_cluster_{cluster}.nii.gz" )
    img = nib.load(cluster_path)
    #utils.show_slices(img, data_type = "img")
    img_data = img.get_fdata()
    img_data[np.where(img_data == 0)] = -1
    
    #Right mesial  0.00, 0.8 , 110
    #lateral temporal  0, 0.6, 110
    #early bilateral   0.07, 0.5, 110
    #left mesial  0.05, 0.6, 110
    #FOCAL   -0.6, 0.6, 96
    
    if cluster == 0: 
        vmin, vmax, slice_num = 0.05, 0.6, 110
    if cluster == 1: 
        vmin, vmax, slice_num  =  0.07, 0.5, 110
    if cluster == 2: 
        vmin, vmax, slice_num =  -0.1, 0.6, 110
    if cluster == 3: 
        vmin, vmax, slice_num =  0, 0.6, 110
    if cluster == 4:  #
        vmin, vmax, slice_num = 0.00, 0.8 , 110
    
    #for slice_num in range(50,140):
    print(slice_num)
    slice_num = int(slice_num)
    img_data.shape
    
    slice1 = utils.get_slice(img_data, slice_ind = 1, slice_num = slice_num, data_type = "data")
    
    cmap = copy.copy(plt.get_cmap("magma"))
    masked_array = np.ma.masked_where(slice1 >=0.99, slice1)
    cmap.set_bad(color='#cdcdcd')
    cmap.set_under('white')
    fig, axes = utils.plot_make(size_length=15, size_height=15)
    axes.imshow(masked_array, cmap=cmap, origin="lower", vmin = vmin, vmax = vmax)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_xticks([])
    axes.set_yticks([])
    axes.axis("off")
    pos = axes.imshow(masked_array, cmap=cmap, origin="lower", vmin = vmin, vmax = vmax)
    fig.colorbar(pos, ax=axes)
    axes.set_title(f"Cluster {cluster}    {slice_num}")
    
    
    
    plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", f"MORE_PATIENTS_COLOR_cluster_atlas_{cluster}_slice_{slice_num}.pdf"), bbox_inches='tight')
    plt.show()
    
    
    
    
#%%
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#Structural connectivity as a predictor of cluster  Focal vs not
i=33



FOCAL_CLUSTER = 4 #MAY CHANGE IF RUN ON DIFFERENT K MEANS CLUSTERING NUMBER, ETC.
subject_cluster_mode = region_activation_fillna_about.groupby(['subject'])['cluster'].agg(pd.Series.mode)
patients_unique = np.unique(patientsWithseizures['subject'])

sc_list = []
sc_clusters = []

atlas_names
atlas = "AAL2"
#atlas = "AAL3v1_1mm"


atlas_names_short =  list(atlas_files["STANDARD"].keys() )
atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
atlas_label_names = np.array(atlas_label.iloc[1:,1])
atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
atlas_label_region_numbers = atlas_label_region_numbers.astype(int)
atlas_path = join(paths.ATLASES, atlas_files["STANDARD"][atlas_names_short[ind]]["name"])

patinets_list = []

for i in range(len(patients_unique)):
    
    RID = patients_unique[i]
    print(f"{i} {RID}")
    cluster =subject_cluster_mode[RID]
    
    np.asarray(cluster)
    if FOCAL_CLUSTER in np.asarray(cluster):
        cluster = 1
    else:
        cluster = 0

    #if cluster.size > 1:
    #    cluster = cluster[0]

    
    connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"sub-{RID}.{atlas}.count.pass.connectogram.txt")
    
    connectivity_loc_glob = glob.glob( connectivity_loc  )
    
    
    if len(connectivity_loc_glob) > 0:
    
            connectivity_loc_path = connectivity_loc_glob[0]
        
            
            sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)
            sc = sc/sc.max()
            #sc = utils.log_normalize_adj(sc)
      
            sc_region_labels = utils.read_DSI_studio_Txt_files_SC_return_regions(connectivity_loc_path, atlas).astype(ind)
            sc_list.append(sc)
            sc_clusters.append(cluster)
            
            patinets_list.append(RID)
len(patinets_list)     
sc_clusters = np.array(sc_clusters)

sc_clusters[np.where(sc_clusters == 2)] = 1 
sc_clusters[np.where(sc_clusters == 3)] = 2 
sc_clusters[np.where(sc_clusters == 4)] = 3 

a = np.array(sc_clusters)
b = np.zeros((a.size, a.max()+1))
b[np.arange(a.size),a] = 1

clusters_one_hot = b

#%%


#sc_indexes_to_get = np.array([40,41,84,85])
sc_indexes_to_get = np.array(range(0,93))
#sc_indexes_to_get = np.array(range(len(sc_list[0])))

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
sc_subset = sc_list[0][ sc_indexes_to_get[:, None] , sc_indexes_to_get[None, :]  ]
sc_array = utils.getUpperTriangle(sc_subset)
for i in range(1, len(sc_list)):
    sc_subset = sc_list[i][ sc_indexes_to_get[:, None] , sc_indexes_to_get[None, :]  ]
    #utils.plot_adj_heatmap(sc_subset)
    sc_array = np.vstack([sc_array, utils.getUpperTriangle(sc_subset) ])


#%%

X_train, X_test, y_train, y_test = train_test_split(sc_array, sc_clusters, test_size =0.5)

clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)
accuracy  = metrics.accuracy_score(y_test, y_pred)
f1  = metrics.f1_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

pred_prob[:,0]

metrics.accuracy_score(sc_clusters, np.repeat(0, len(sc_clusters)) )

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob[:,0], pos_label=0)

print(f"AUC: {metrics.roc_auc_score(y_test, pred_prob[:,1])}\nf1: {f1}\nprecision:{precision}\nrecall : {recall}")

fig, axes = utils.plot_make()
axes.plot(fpr,tpr)
axes.set_title(f"AUC: {metrics.roc_auc_score(y_test, pred_prob[:,1])}")

#sns.scatterplot(ax = axes, x = fpr, y =tpr)
sns.lineplot(x = [0,1], y = [0,1], ls = "--")

#%%

def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=100):
    auc_values = []
    for b in range(nsamples):
        print(f"\r{b+1}/{nsamples}. {(b+1)/nsamples*100:.1f}%     ", end = "\r")
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = metrics.roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))

def permutation_test(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    idx1 = np.arange(X_train.shape[0])
    idx2 = np.arange(X_test.shape[0])
    auc_values = np.empty(nsamples)
    for b in range(nsamples):
        print(f"\r{b+1}/{nsamples}. {(b+1)/nsamples*100:.1f}%    ", end = "\r")
        np.random.shuffle(idx1)  # Shuffles in-place
        np.random.shuffle(idx2)
        clf.fit(X_train, y_train[idx1])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = metrics.roc_auc_score(y_test[idx2].ravel(), pred.ravel())
        auc_values[b] = roc_auc
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    roc_auc = metrics.roc_auc_score(y_test.ravel(), pred.ravel())
    return roc_auc, np.mean(auc_values >= roc_auc)

clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
X_train, X_test, y_train, y_test = train_test_split(sc_array, sc_clusters, test_size =0.25)
ci_auc = bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=50)

ci_auc_permute= permutation_test(clf, X_train, y_train, X_test, y_test, nsamples=50)
#%%
#metrics.roc_auc_score(y_test, pred_prob)

from sklearn.model_selection import cross_val_score   
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
scoring="roc_auc"
scores = cross_val_score(clf, sc_array, sc_clusters, cv=10,  scoring="roc_auc")   

fig, axes = utils.plot_make()
sns.boxplot(ax = axes, y = scores)
sns.swarmplot(ax = axes, y = scores)
axes.set_ylim([0,1])

axes.set_title(f"Mean AUC: {scores.mean()}")


print(f"{scores.mean():.2f} {scoring} with a standard deviation of {scores.std():.2f}" )  
print(scores)

print(stats.ttest_1samp(scores, 1/2, axis=0 )  )  


#stats.wilcoxon(scores - (1/4))
stats.wilcoxon(scores - (1/4), alternative='greater')
#%%

    
feature_names = []
for a in range(len(atlas_label_names[sc_indexes_to_get])):
    for b in range(a+1, len(atlas_label_names[sc_indexes_to_get])):
        feature_names.append(f"{atlas_label_names[sc_indexes_to_get][a]} -- {atlas_label_names[sc_indexes_to_get][b]}")

clf.fit(sc_array, sc_clusters)
feature_importance = pd.DataFrame(columns = ["feature", "importance"])
for name, score in zip(feature_names, clf.feature_importances_):
    feature_importance = feature_importance.append(dict(feature = name, importance = score),ignore_index=True)
    #print(name, score)
    
    
sorted_feature_importance = np.array(np.argsort( np.array(feature_importance["importance"] ) ) )
sorted_feature_importance_rev = sorted_feature_importance[::-1]

feature_importance_sorted = feature_importance.iloc[sorted_feature_importance_rev]

print(feature_importance_sorted)
    
  #%%  
    
SIZE = 300

k_clusters_sc = 3
pca_activation_sc = PCA(n_components=22)        
principalComponents_sc = pca_activation_sc.fit_transform(sc_array)        
        
        
pca_activation_sc.explained_variance_ratio_

df_sc = pd.DataFrame(principalComponents_sc[:, 0:3], columns = ["PC1", "PC2", "PC3"])


palette_sc = {"1": "#5b1111", "3": "#9567BE", "2": "#c94849", "4": "#1F78B5", "0": "#84b2ec", "5": "#00ff00" , "6": "#ff0000", "7": "#ff00ff"}

kmeans_sc = KMeans(init="random",n_clusters=k_clusters_sc,n_init=10,max_iter=300,random_state=42)

kmeans_sc.fit(sc_array)

kmeans_sc.inertia_
kmeans_sc.n_iter_
kmeans_sc.cluster_centers_
sc_clusters
df_sc["cluster"] = kmeans_sc.labels_

df_sc["cluster"] = df_sc["cluster"].astype(str)

#sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = 5, hue = "cluster" , palette = {"0":"#9b59b6", "1":"#3498db",   "2":"#95a5a6"}, linewidth=0)
fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = SIZE, hue = "cluster" , palette = palette_sc , linewidth=0)

# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_minifig_big.pdf"), bbox_inches='tight')
#plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_clusters_big.pdf"), bbox_inches='tight')

#%%
   
sc_array_pc = np.array(pd.DataFrame(principalComponents_sc[:, 0:100]))
sc_array_pc.shape

from sklearn.model_selection import ShuffleSplit
n_samples = sc_array_pc.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.33, random_state=0)
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
scores = cross_val_score(clf, sc_array_pc, sc_clusters, cv=3)    
    
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))  
    
stats.ttest_1samp(scores, 1/4, axis=0)    





sc_array_pc = np.array(pd.DataFrame(principalComponents_sc[:, 0:150]))
clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
scoring="roc_auc"
scores = cross_val_score(clf, sc_array_pc, sc_clusters, cv=5,  scoring="roc_auc")   

fig, axes = utils.plot_make()
sns.boxplot(ax = axes, y = scores)
sns.swarmplot(ax = axes, y = scores)
axes.set_ylim([0,1])

axes.set_title(f"Mean AUC: {scores.mean():.5f}")


print(f"{scores.mean():.2f} {scoring} with a standard deviation of {scores.std():.2f}" )  
print(scores)

print(stats.ttest_1samp(scores, 1/2, axis=0 )  )  

#stats.wilcoxon(scores - (1/4))
stats.wilcoxon(scores - (1/4), alternative='greater')


clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
X_train, X_test, y_train, y_test = train_test_split(sc_array_pc, sc_clusters, test_size =0.25)
ci_auc = bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=50)



###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
#%% Getting all SC



sc_list = []


atlas_names
atlas = "AAL2"


atlas_names_short =  list(atlas_files["STANDARD"].keys() )
atlas_names = [atlas_files["STANDARD"][x]["name"] for x in atlas_names_short ]
ind = np.where(f"{atlas}.nii.gz"  == np.array(atlas_names))[0][0]
atlas_label_name = atlas_files["STANDARD"][atlas_names_short[ind]]["label"]
atlas_label = pd.read_csv(join(paths.ATLAS_LABELS, atlas_label_name))
atlas_label_names = np.array(atlas_label.iloc[1:,1])
atlas_label_region_numbers = np.array(atlas_label.iloc[1:,0])
atlas_label_region_numbers = atlas_label_region_numbers.astype(int)
atlas_path = join(paths.ATLASES, atlas_files["STANDARD"][atlas_names_short[ind]]["name"])

sc_people = os.listdir(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES))
sc_RIDS = []
sc_list_people = []
for i in range(len(sc_people)):
    
    RID = sc_people[i]
    print(f"{i} {RID}")


    connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"{RID}.{atlas}.count.pass.connectogram.txt")
    
    connectivity_loc_glob = glob.glob( connectivity_loc  )
    
    
    if len(connectivity_loc_glob) > 0:
        connectivity_loc_path = connectivity_loc_glob[0]
        sc_RIDS.append(RID)
        
        sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)
        sc = sc/sc.max()
        #sc = utils.log_normalize_adj(sc)
        sc_list_people.append(sc)

len(sc_list_people)

sc_indexes_to_get = np.array(range(len(sc_list_people[0])))

sc_subset = sc_list_people[0][ sc_indexes_to_get[:, None] , sc_indexes_to_get[None, :]  ]
sc_array_people = utils.getUpperTriangle(sc_subset)
for i in range(1, len(sc_list_people)):
    sc_subset = sc_list_people[i][ sc_indexes_to_get[:, None] , sc_indexes_to_get[None, :]  ]
    #utils.plot_adj_heatmap(sc_subset)
    sc_array_people = np.vstack([sc_array_people, utils.getUpperTriangle(sc_subset) ])



#%%
SIZE = 300

k_clusters_sc = 4
pca_activation_sc = PCA(n_components=100)        
sc_array_people.shape
        
principalComponents_sc = pca_activation_sc.fit_transform(sc_array_people)        
        
        
pca_activation_sc.explained_variance_ratio_

df_sc = pd.DataFrame(principalComponents_sc[:, 0:5], columns = ["PC1", "PC2", "PC3", "PC4", "PC5"])


palette_sc = {"1": "#5b1111", "3": "#9567BE", "2": "#c94849", "4": "#1F78B5", "0": "#84b2ec", "5": "#00ff00" , "6": "#ff0000", "7": "#ff00ff"}

kmeans_sc = KMeans(init="random",n_clusters=k_clusters_sc,n_init=10,max_iter=300,random_state=42)

kmeans_sc.fit(sc_array_people)

kmeans_sc.inertia_
kmeans_sc.n_iter_
kmeans_sc.cluster_centers_
sc_clusters
df_sc["cluster"] = kmeans_sc.labels_
df_sc["subject"] = sc_RIDS

df_sc["cluster"] = df_sc["cluster"].astype(str)

#sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = 5, hue = "cluster" , palette = {"0":"#9b59b6", "1":"#3498db",   "2":"#95a5a6"}, linewidth=0)
fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = SIZE, hue = "cluster" , palette = palette_sc , linewidth=0)

# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_minifig_big.pdf"), bbox_inches='tight')
#plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering_update", "clustering_clusters_big.pdf"), bbox_inches='tight')


controls = [285, 286, 287, 288, 289, 290, 292,292,297,505,599,600,602,603,604,615,682,683,815,816,817,818,819,820,826,827,830,833,854,855]

#controls = [505,599,600,602,603,604,615,682,683,815,816,817,818,819,820,826,827,830,833,854,855]



#%%

df_sc["control"] = "patient"
for p in range(len(df_sc)):
    if int(df_sc["subject"][p][7:]) in controls:
        df_sc.loc[   df_sc["subject"]    == df_sc["subject"][p] , "control"  ] = "control"
        
        

fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = SIZE, hue = "control"  , linewidth=0)


#%%%
df_sc["subjects_spread"] = "no"
df_sc["cluster"] = "NA"
#% Get PC of just patients with iEEG and SC
sc_array_pc_list = []
for p in range(len(df_sc)):
    if df_sc["subject"][p][4:] in patinets_list:
        if p ==0:
            sc_array_pc = principalComponents_sc[p,:]
        else:
            sc_array_pc = np.vstack([sc_array_pc,  principalComponents_sc[p,:] ])
        df_sc.loc[   df_sc["subject"]    == df_sc["subject"][p] , "subjects_spread"  ] = "subjects_spread"
        
        cluster_ind = np.where( df_sc["subject"][p][4:] == np.array(patinets_list))[0][0]
        df_sc.loc[   df_sc["subject"]    == df_sc["subject"][p] , "cluster"  ]  = sc_clusters[cluster_ind]

fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = SIZE, hue = "subjects_spread"  , linewidth=0)


fig, axes = utils.plot_make(size_length=10, size_height=6)
sns.scatterplot(data = df_sc, x = "PC1", y = "PC2", s = SIZE/2, hue = "cluster"  , linewidth=0)






#%%


clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
scores = cross_val_score(clf, sc_array_pc, sc_clusters, cv=5)    
    
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))  
    
stats.ttest_1samp(scores, 1/4, axis=0)    





clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
scoring="roc_auc"

scores = cross_val_score(clf, sc_array_pc, sc_clusters, cv=5,  scoring="roc_auc")   

fig, axes = utils.plot_make()
sns.boxplot(ax = axes, y = scores)
sns.swarmplot(ax = axes, y = scores)
axes.set_ylim([0,1])

axes.set_title(f"Mean AUC: {scores.mean():.5f}")
stats.ttest_1samp(scores, 1/2, axis=0)  



clf = RandomForestClassifier(n_estimators = 100, n_jobs=-1) 
X_train, X_test, y_train, y_test = train_test_split(sc_array_pc, sc_clusters, test_size =0.25)
ci_auc = bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=50)

