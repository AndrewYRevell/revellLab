#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:13:33 2022

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
from packages.diffusionModels import functions as dmf 

from revellLab.packages.diffusionModels import diffusionModels as DM
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

unique_patients= np.unique(patientsWithseizures["subject"])


bilateral = []
unilateral = []

for p in range(len(patients)):
    
    pt = patients[p]
    regs = region[p]
    
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
    
    rid = RID_HUP["record_id"][np.where(int(pt[3:]) ==  RID_HUP["hupsubjno"])[0][0]]
    RID = f"RID{rid:04d}"
    if left_sum > 0 and right_sum >0:
        bilateral.append(RID)
    else:
        unilateral.append(RID)


#%%

region_activation_fillna_about = pd.read_csv(join(paths.SEIZURE_SPREAD_DERIVATIVES_CLUSTERING, "Seizure_clusters.csv"), index_col=0)

    
CLUSTER_FOCAL = 0
CLUSTER_LEFT_MESIAL = 3
CLUSTER_RIGHT_MESIAL = 1
CLUSTER_BILATERAL = 4
CLUSTER_LATERAL_TEMPORAL = 2 


subject_cluster_mode = region_activation_fillna_about.groupby(['subject'])['cluster'].agg(pd.Series.mode)
patients_unique = np.unique(patientsWithseizures['subject'])

sc_list = []
sc_clusters = []

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


    #if cluster.size > 1:
    #    cluster = cluster[0]

    
    connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"sub-{RID}.{atlas}.count.pass.connectogram.txt")
    
    connectivity_loc_glob = glob.glob( connectivity_loc  )
    
    
    if len(connectivity_loc_glob) > 0:
    
            connectivity_loc_path = connectivity_loc_glob[0]
        
            
            sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)
            #sc = sc/sc.max()
            #sc = utils.log_normalize_adj(sc)
      
            sc_region_labels = utils.read_DSI_studio_Txt_files_SC_return_regions(connectivity_loc_path, atlas).astype(ind)
            sc_list.append(sc)
            sc_clusters.append(cluster)
            
            patinets_list.append(RID)
len(patinets_list)     
sc_clusters = np.array(sc_clusters)
sc_array = np.asarray(sc_list)
atlas_centroids = pd.read_csv(paths.AAL2_CENTROIDS)
distance_coords = np.asarray(atlas_centroids[["x", "y", "z"]])
distance_matrix =  utils.get_pariwise_distances(distance_coords)
distance_matrix = 1/distance_matrix
distance_matrix[np.where(distance_matrix == np.inf)] = 0

#%%
CLUSTER_FOCAL = 0
CLUSTER_LEFT_MESIAL = 3
CLUSTER_RIGHT_MESIAL = 1
CLUSTER_BILATERAL = 4
CLUSTER_LATERAL_TEMPORAL = 2 


df_all =pd.DataFrame(columns =  ["patient", "seizure", "cluster", "region_num", "region", "sc_type", "corr"])
#%%
sub = "RID0278"
i=0
len(patinets_list)
for i in range(13,22):
    
    sub = patinets_list[i]
    print(f"\n\n\n\n\n{sub}\n{sub}\n{sub}\n{sub}\n")
    ind = np.where(np.array(patinets_list) == sub)[0][0]
    sc_clusters[ind]
    sthreshold=0.1
    adj = sc_array[ind]
    #adj = utils.log_normalize_adj(SC)
    adj_hat = adj/adj.max()
    adj_hat_threshold = copy.deepcopy(adj_hat)
    adj_hat_threshold_select = sthreshold < adj_hat_threshold
    adj_hat_threshold[~adj_hat_threshold_select]=0
    
    
    
    SC_regions = sc_region_labels
    el_path = glob.glob( join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{sub}",  f"ses-{session}", "*atlasLocalization.csv"))[0]
    eloc = pd.read_csv(el_path)
    
    eloc["channel"] = utils.channel2std(np.array(eloc["channel"]) )
    electrodeLocalization = eloc[["channel", f"{atlas}_region_number", f"{atlas}_label"]]
    
    # Get atlas label data
    atlas_label_path = glob.glob( join(paths.ATLAS_LABELS, f"{atlas}.csv"))
    if len(atlas_label_path) > 0:
        atlas_label2 = atlas_label_path[0]
        atlas_label2 = pd.read_csv(atlas_label2, header=1)
    else:
        atlas_label2 = pd.DataFrame(  dict( region= SC_regions.astype(int), label =  SC_regions))
    
    
    dm_data = DM.diffusionModels(adj_hat_threshold, 0, 0, 0, 0)  
    
    seziures_total = len(np.where(patientsWithseizures["subject"] == sub)[0])
    seizure_num = 0
    for sz in range(seziures_total):
        seizure_num = sz
   #if sub == "RID0278": seizure_num=1
        patient_ind = np.where(patientsWithseizures["subject"] == sub)[0][seizure_num]
        
        cluster_id = region_activation_fillna_about.iloc[patient_ind]["cluster"]
        idKey = np.array(patientsWithseizures["idKey"])[patient_ind]
        seizure_length = patientsWithseizures.length[patient_ind]
        
        print(f"\n\n\n\n\n{sub}\n{sub}\n{sub}\n{sub}\nseizure: {idKey}\ncluster: {cluster}")
        #CHECKING IF SPREAD FILES EXIST
        
  
        if sub =="RID0278" and int(idKey) >2:
            continue
        if sub =="RID0309" and int(idKey) >2:
            continue
        if sub =="RID0440" and int(idKey) >2:
            continue
        if sub =="RID0490" and int(idKey) >2:
            continue
        if int(idKey) >2:
            continue
   
        fname = DataJson.get_fname_ictal(RID, "Ictal", idKey, dataset= datasetiEEG, session = session, startUsec = None, stopUsec= None, startKey = "EEC", secondsBefore = secondsBefore, secondsAfter = secondsAfter )
        
        spread_location = join(BIDS, datasetiEEG_spread, f"v{version:03d}", f"sub-{RID}" )
        spread_location_file_basename = f"{splitext(fname)[0]}_spread.pickle"
        spread_location_file = join(spread_location, spread_location_file_basename)
        
        
        
        with open(spread_location_file, 'rb') as f:[probWN, probCNN, probLSTM, data_scalerDS, channels, window, skipWindow, secondsBefore, secondsAfter] = pickle.load(f)
        prob_array= probWN
        probability_arr_movingAvg, probability_arr_threshold = prob_threshold_moving_avg(prob_array, fsds, skip, threshold = 0.69, smoothing = 20)
        channels_spread = copy.deepcopy(channels)  
        
        
        
        spread_before=30
        spread_after=10
        st =int((secondsBefore-spread_before)/skipWindow)
        stp = int((secondsBefore +spread_after)/skipWindow)
        spread = copy.deepcopy(probability_arr_movingAvg[st:stp,:])
        
        #%
        dm_data = DM.diffusionModels(adj_hat_threshold, SC_regions, spread, channels_spread, electrodeLocalization)    
        
        
        
        
        #create random
        G = nx.newman_watts_strogatz_graph(len(adj_hat_threshold), 25, p=0.5)
        G = dmf.add_edge_weights(G, max_weight = 5) #add edges
        adj_rand = nx.to_numpy_array(G)
        adj_rand = adj_rand/adj_rand.max()
        
        ######################################
        ######################################
        ######################################
        sns.heatmap(adj_hat_threshold);plt.show()
        sns.heatmap(distance_matrix);plt.show()
        sns.heatmap(adj_rand);plt.show()
        
        
        
        
        diffusion_model_type= 0
        threshold_dm = 0.05
        time_steps = 9
        gradient = 0.01
        """
        diffusion_model_type= 1
        threshold_dm = 0.001
        time_steps = 9
        gradient = 0.5
        
        diffusion_model_type= 2
        threshold_dm = 1
        time_steps = 9
        gradient = 0.3
        """
        
        corrs, sig = DM.get_diffusion_model_correlations(adj_hat_threshold, SC_regions, spread, channels_spread, electrodeLocalization, atlas, atlas_label2, dm_data, diffusion_model_type = diffusion_model_type, threshold=threshold_dm, time_steps = time_steps, gradient = gradient, visualize = True, r_to_visualize = SC_regions[40])
        
        
        sns.lineplot( x = range(len(corrs)), y =  np.sort(corrs)[::-1] )
        plt.show()
        np.sort(corrs)[::-1]
        print(atlas_label_names[0:93][np.argsort(corrs[0:93])[::-1]])
        
        
        
        
        
        
        
        
        corrs_dist, sig_dist = DM.get_diffusion_model_correlations(distance_matrix, SC_regions, spread, channels_spread, electrodeLocalization, atlas, atlas_label2, dm_data, diffusion_model_type = diffusion_model_type, threshold=threshold_dm/2, time_steps = time_steps, gradient = gradient, visualize = True, r_to_visualize = 4101)
        
        
     
        corrs_rand, sig_rand = DM.get_diffusion_model_correlations(adj_rand, SC_regions, spread, channels_spread, electrodeLocalization, atlas, atlas_label2, dm_data, diffusion_model_type = diffusion_model_type, threshold=threshold_dm/10, time_steps = time_steps, gradient = gradient, visualize = True, r_to_visualize = 4101)
        
        
        
        dfdf = pd.DataFrame(columns = ["region", "sc", "dist","rand"])
        dfdf["region"] = atlas_label_names[0:93][np.argsort(corrs[0:93])[::-1]]
        dfdf["sc"] = corrs[0:93][np.argsort(corrs[0:93])[::-1]]
        dfdf["dist"] = corrs_dist[0:93][np.argsort(corrs[0:93])[::-1]]
        #dfdf["null"] = corrs_null[0:93][np.argsort(corrs[0:93])[::-1]]
        dfdf["rand"] = corrs_rand[0:93][np.argsort(corrs[0:93])[::-1]]
        dfdf = dfdf.reset_index()
        
        df_long1 = pd.melt(dfdf , id_vars= ["index", "region"] , var_name= "sc_type", value_name = "corr")
        df_long1= df_long1.rename({"index": "region_num"}, axis=1 )
        sns.lineplot(data = df_long1, x = "region_num" , y =  "corr"  , hue = "sc_type", palette="Set1" )
        
        
        #####NULL MODEL
        df_null = pd.DataFrame(columns = ["index", "region", "sc_type", "corr"])
        for n in range(1):
            adj_null, eff = bct.null_model_und_sign(adj_hat)
            adj_null_threshold = copy.deepcopy(adj_null)
            adj_null_threshold_select = sthreshold < adj_null_threshold
            adj_null_threshold[~adj_null_threshold_select]=0
            corrs_null, sig_null = DM.get_diffusion_model_correlations(adj_null_threshold, SC_regions, spread, channels_spread, electrodeLocalization, atlas, atlas_label2, dm_data, diffusion_model_type = diffusion_model_type, threshold=threshold_dm, time_steps = time_steps, gradient = gradient, visualize = False, r_to_visualize = 4101)
            null_corr = pd.DataFrame(columns =  ["region", "sc_type", "corr"])
            null_corr["region"] = atlas_label_names[0:93][np.argsort(corrs[0:93])[::-1]]
            null_corr["sc_type"] = "null"
            null_corr["corr"] = corrs_null[0:93][np.argsort(corrs[0:93])[::-1]]
            null_corr = null_corr.reset_index()
            df_null= df_null.append(null_corr, ignore_index=True  )
        
        
        sns.heatmap(adj_null_threshold);plt.show()
        sns.heatmap(adj_null);plt.show()
        
        
        
        
        df_long = pd.melt(dfdf , id_vars= ["index", "region"] , var_name= "sc_type", value_name = "corr")
        df_long= df_long.append(df_null, ignore_index=True  )
        
        
        
        df_long= df_long.rename({"index": "region_num"}, axis=1 )
        sns.lineplot(data = df_long, x = "region_num" , y =  "corr"  , hue = "sc_type", palette="Set1" )
        plt.show()
        
        df_long_append = copy.deepcopy(df_long)
        df_long_append["patient"] = sub
        df_long_append["seizure"] = idKey
        df_long_append["cluster"] = cluster_id
        
        
        
        df_all = df_all.append(df_long_append, ignore_index=True )
    



#%%

CLUSTER_FOCAL = 0
CLUSTER_LEFT_MESIAL = 3
CLUSTER_RIGHT_MESIAL = 1
CLUSTER_BILATERAL = 4
CLUSTER_LATERAL_TEMPORAL = 2 

np.unique(df_all["cluster"])
#hipp_L = df_all[(df_all["region"] == "Hippocampus_L") | (df_all["region"] == "Cingulate_Ant_L") | (df_all["region"] == "Cingulate_Mid_L") | (df_all["region"] == "Cingulate_Post_L")| (df_all["region"] == "Insula_L")| (df_all["region"] == "ParaHippocampal_L")| (df_all["region"] == "Temporal_Inf_L")| (df_all["region"] == "Temporal_Mid_L") | (df_all["region"] == "Temporal_Pole_Mid_L")| (df_all["region"] == "Temporal_Pole_Sup_L")| (df_all["region"] == "Temporal_Sup_L")]

hipp_L = df_all[(df_all["region"] == "Hippocampus_L") | (df_all["region"] == "Temporal_Inf_L")| (df_all["region"] == "Temporal_Mid_L") | (df_all["region"] == "Temporal_Sup_L")]

hipp_R = df_all[(df_all["region"] == "Hippocampus_R") | (df_all["region"] == "Cingulate_Ant_R") | (df_all["region"] == "Cingulate_Mid_R") | (df_all["region"] == "Cingulate_Post_R")| (df_all["region"] == "Insula_R")| (df_all["region"] == "ParaHippocampal_R")| (df_all["region"] == "Temporal_Inf_R")| (df_all["region"] == "Temporal_Mid_R") | (df_all["region"] == "Temporal_Pole_Mid_R")| (df_all["region"] == "Temporal_Pole_Sup_R")| (df_all["region"] == "Temporal_Sup_R")]



sns.boxplot(data =hipp_L , x = "cluster", y = "corr", hue = "sc_type", showfliers=False )
sns.swarmplot(data =hipp_L , x = "cluster", y = "corr", hue = "sc_type", dodge = True, s = 2)



sns.boxplot(data =hipp_R , x = "cluster", y = "corr", hue = "sc_type", showfliers=False )
sns.swarmplot(data =hipp_R , x = "cluster", y = "corr", hue = "sc_type", dodge = True, s = 2)


v1 = hipp_L[(hipp_L["sc_type"] == "sc") & (hipp_L["cluster"] == 3)]["corr"]
v2 = hipp_L[(hipp_L["sc_type"] == "dist") & (hipp_L["cluster"] == 3)]["corr"]

stats.wilcoxon(v1, v2)

v1 = hipp_R[(hipp_R["sc_type"] == "sc") & (hipp_R["cluster"] == 1)]["corr"]
v2 = hipp_R[(hipp_R["sc_type"] == "dist") & (hipp_R["cluster"] == 1)]["corr"]

stats.wilcoxon(v1, v2)





hipp_L_average = hipp_L.groupby(["patient", "cluster", "sc_type", "seizure"], as_index=False).mean()
hipp_R_average = hipp_R.groupby(["patient", "cluster", "sc_type", "seizure"], as_index=False).mean()

sns.boxplot(data =hipp_L_average , x = "cluster", y = "corr", hue = "sc_type", showfliers=False )
sns.swarmplot(data =hipp_L_average , x = "cluster", y = "corr", hue = "sc_type", dodge = True, s = 3)

sns.boxplot(data =hipp_R_average , x = "cluster", y = "corr", hue = "sc_type", showfliers=False )
sns.swarmplot(data =hipp_R_average , x = "cluster", y = "corr", hue = "sc_type", dodge = True, s = 3)

v1 = hipp_L_average[(hipp_L_average["sc_type"] == "sc") & (hipp_L_average["cluster"] == 3)]["corr"]
v2 = hipp_L_average[(hipp_L_average["sc_type"] == "dist") & (hipp_L_average["cluster"] == 3)]["corr"]

stats.wilcoxon(v1, v2)

v1 = hipp_R_average[(hipp_R_average["sc_type"] == "sc") & (hipp_R_average["cluster"] == 3)]["corr"]
v2 = hipp_R_average[(hipp_R_average["sc_type"] == "dist") & (hipp_R_average["cluster"] == 3)]["corr"]

stats.wilcoxon(v1, v2)



"Amygdala_L"
"Cingulate_Ant_L"
"Cingulate_Mid_L"
"Cingulate_Post_L"
"Insula_L"
"ParaHippocampal_L"
"Temporal_Inf_L"
"Temporal_Mid_L"
"Temporal_Pole_Mid_L"
"Temporal_Pole_Sup_L"
"Temporal_Sup_L"

"Amygdala_R"
"Cingulate_Ant_R"
"Cingulate_Mid_R"
"Cingulate_Post_R"
"Insula_R"
"ParaHippocampal_R"
"Temporal_Mid_R"
"Temporal_Pole_Mid_R"
"Temporal_Pole_Sup_R"
"Temporal_Sup_R"




#%%#creating new localization

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

img_data_sc_3 = copy.deepcopy(img_data)
img_data_sc_3[np.where(img_data_sc_3 >-100)]=0
img_data_sc_1 = copy.deepcopy(img_data)
img_data_sc_1[np.where(img_data_sc_1 >-100)]=0

img_data_dist_3 = copy.deepcopy(img_data)
img_data_dist_3[np.where(img_data_dist_3 >-100)]=0
img_data_dist_1 = copy.deepcopy(img_data)
img_data_dist_1[np.where(img_data_dist_1 >-100)]=0



N_sc_3 = 0
N_sc_1 = 0

N_dist_3 = 0
N_dist_1 = 0
for r in range(len(df_all)):
    print(f"\r{r}: {r/len(df_all)*100:.2f}    ", end = "\r")
    reg = df_all.loc[:,"region"][r]
    reg_sc_type = df_all.loc[:,"sc_type"][r]
    reg_ind = np.where(atlas_label_names == reg)[0][0]
    reg_num = atlas_label_region_numbers[reg_ind]
    clust = df_all.loc[:,"cluster"][r]
    reg_corr = df_all.loc[:,"corr"][r]
    if reg_sc_type == "sc":
        if clust == 3:
            img_data_sc_3[np.where(img_data == reg_num)] = img_data_sc_3[np.where(img_data == reg_num)] + reg_corr
            N_sc_3 = N_sc_3+1
        if clust == 1:
            img_data_sc_1[np.where(img_data == reg_num)] = img_data_sc_1[np.where(img_data == reg_num)] + reg_corr
            N_sc_1 = N_sc_1+1
    if reg_sc_type == "dist":
        if clust == 3:
            img_data_dist_3[np.where(img_data == reg_num)] = img_data_dist_3[np.where(img_data == reg_num)] + reg_corr
            N_dist_3 = N_dist_3+1
        if clust == 1:
            img_data_dist_1[np.where(img_data == reg_num)] = img_data_dist_1[np.where(img_data == reg_num)] + reg_corr
            N_dist_1 = N_dist_1+1


        
utils.show_slices(img_data_sc_3, data_type = "data", cmap = "rainbow")
utils.show_slices(img_data_dist_3, data_type = "data", cmap = "Reds")


utils.show_slices(img_data_sc_1, data_type = "data", cmap = "Reds")
utils.show_slices(img_data_dist_1, data_type = "data", cmap = "Reds")

utils.save_nib_img_data(img_data_sc_3/N_sc_3, img, join(BIDS, project_folder, f"atlases", f"cluster_3_corr_sc.nii.gz" ) )
utils.save_nib_img_data(img_data_dist_3/N_dist_3, img, join(BIDS, project_folder, f"atlases", f"cluster_3_corr_dist.nii.gz" ) )


img_data_sc_3_tmp = img_data_sc_3/N_sc_3
img_data_dist_3_tmp = img_data_dist_3/N_dist_3



cluster_img = nib.load( join(BIDS, project_folder, f"atlases", f"more_patients_cluster_3.nii.gz" ) )
cluster_3_data = cluster_img.get_fdata()

utils.show_slices(cluster_3_data, data_type = "data")


xx = cluster_3_data[np.where(img_data>0) and (np.where(cluster_3_data<0.99))]
yy = img_data_sc_3_tmp[np.where(img_data>0) and (np.where(cluster_3_data<0.99))]
zz = img_data_dist_3_tmp[np.where(img_data>0) and (np.where(cluster_3_data<0.99))]


sns.scatterplot(x = xx, y =yy)
sns.scatterplot(x = xx, y =zz)


#%%calculate diffusion model simualtion
normalize = True
logs = False
sthreshold=0
time_steps=200
threshold_dm = 0.5
gradient= 0.01
def diffusion_model_simulation(sc_array, i , patinets_list, sthreshold, threshold_dm, sc_clusters, distance_matrix, time_steps, gradient):
    
    RID = patinets_list[i]
    adj = sc_array[i]
    cluster = sc_clusters[i]
    
    adj_distance = distance_matrix
    
    if RID == "RID0278":
        cluster= 4
    
    if cluster == 4 or cluster == 1 or cluster == 0:
        seed = 40
    else:
        seed = 41
        
    

    if normalize:
        adj_hat = adj/adj.max()
        adj_distance_hat = adj_distance/adj_distance.max()
    elif logs:
        adj_hat = utils.log_normalize_adj(adj)
        adj_distance_hat = utils.log_normalize_adj(adj_distance)
    else:
        adj_hat = adj
        adj_distance_hat = adj_distance
    
    adj_hat_threshold = copy.deepcopy(adj_hat)
    adj_hat_threshold_select = sthreshold < adj_hat_threshold
    adj_hat_threshold[~adj_hat_threshold_select]=0
    
    #adj_null, eff = bct.randmio_und_connected(adj_hat_threshold, 100)
    

    am = dmf.gradient_LTM(adj_hat_threshold, seed = seed, time_steps = time_steps, threshold =threshold_dm, gradient = gradient)
    #sns.heatmap(am)
    
    #am_null = dmf.gradient_LTM(adj_null, seed = seed, time_steps = time_steps, threshold =threshold_dm, gradient = gradient)
    #sns.heatmap(am_null)

    am_distance = dmf.gradient_LTM(adj_distance, seed = seed, time_steps = time_steps, threshold =threshold_dm, gradient = gradient*2)
    #sns.heatmap(am_distance)
    return am, am_distance

def diffusion_model_simulation_null(sc_array, i , patinets_list, sthreshold, threshold_dm, sc_clusters, distance_matrix, time_steps, gradient, null_iteration=20):
    
    RID = patinets_list[i]
    adj = sc_array[i]
    cluster = sc_clusters[i]
    
    adj_distance = distance_matrix
    
    if RID == "RID0278":
        cluster= 4
    
    if cluster == 4 or cluster == 1 or cluster == 0:
        seed = 40
    else:
        seed = 41
        
    
    if normalize:
        adj_hat = adj/adj.max()
        adj_distance_hat = adj_distance/adj_distance.max()
    elif logs:
        adj_hat = utils.log_normalize_adj(adj)
        adj_distance_hat = utils.log_normalize_adj(adj_distance)
    else:
        adj_hat = adj
        adj_distance_hat = adj_distance
    
    adj_hat_threshold = copy.deepcopy(adj_hat)
    adj_hat_threshold_select = sthreshold < adj_hat_threshold
    adj_hat_threshold[~adj_hat_threshold_select]=0
    
    adj_null, eff = bct.randmio_und_connected(adj_hat_threshold, null_iteration)
    

    am = dmf.gradient_LTM(adj_hat_threshold, seed = seed, time_steps = time_steps, threshold =threshold_dm, gradient = gradient)
    #sns.heatmap(am)
    am_null = dmf.gradient_LTM(adj_null, seed = seed, time_steps = time_steps, threshold =threshold_dm, gradient = gradient)
    
    return am, am_null



#%%

i=132
type_of_overlap = "soz"
threshold=0.69
smoothing = 20
model_ID="WN"
tanh = False
atlas_name_to_use= atlas
def calculate_mean_rank_deep_learning(i, patientsWithseizures, version, atlas_name_to_use, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, tanh = False, use_atlas = False):
    #override_soz if True, then if there are no soz marking, then use the resection markings and assume those are SOZ contacts
#%
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
        
        
        channels_region_index_label.shape
        

        #% get probability_arr_movingAvg in same region
        channel_names = np.asarray(channel2std_ECoG(channel_names)  )      
        channels2
        
        
        probability_arr_movingAvg.shape
        channels2.shape
        
        channel_names.shape
        channels_region_index_label.shape
        
        _,_,channel_names_ind_overlap = np.intersect1d(channels2, channel_names, return_indices=True)
        channels_region_index_label_overlap = channels_region_index_label[channel_names_ind_overlap]
        
        df = pd.DataFrame(probability_arr_movingAvg.T)
        df["region" ] = channels_region_index_label_overlap
        df_mean = df.groupby(["region"], as_index=False).mean()
        
        sns.heatmap(np.asarray(df_mean.iloc[:,1800:2900]))
        
        return df_mean, seizure_start, skipWindow, seizure_length

#%%

sc_array[1]


DM.get_diffusion_model_correlations(SC, SC_regions, spread, channels_spread, electrodeLocalization, atlas_name, atlas_label, dm_data, diffusion_model_type = 0, threshold=None, time_steps = None, gradient = None, visualize = False, r_to_visualize = None)





df_mean, seizure_start, skipWindow, seizure_length  = calculate_mean_rank_deep_learning(132, patientsWithseizures, version, atlas_name_to_use, threshold=0.6, smoothing = 20, model_ID="WN", secondsAfter=180, secondsBefore=180, tanh = False, use_atlas = False)




RID = "RID0278"

atlas_label_names

normalize = True
logs = False
sthreshold=0
time_steps=300
threshold_dm = 0.5
gradient= 0.01

np.where(RID == np.array(patinets_list) )[0][0]

am, am_distance = diffusion_model_simulation(sc_array, np.where(RID == np.array(patinets_list) )[0][0] , patinets_list, sthreshold, threshold_dm, sc_clusters, distance_matrix, time_steps, gradient)



sns.heatmap(am    )

df_mean["region"][1:]

sns.heatmap(am[:150,df_mean["region"][1:]]    )




































#%%
RID = "RID0278"
region_activation_subject = copy.deepcopy(region_activation_fillna_about)
region_activation_subject_mean = region_activation_subject.groupby(["subject"], as_index=False).mean()

region_activation_subject_mean = region_activation_subject[(region_activation_subject["subject"] == "RID0278") & (region_activation_subject["seizure"] == 3)]

region_subject = region_activation_subject_mean[region_activation_subject_mean["subject"] == RID].iloc[:,1:-3]
region_subject_names = region_subject.columns
activity_spread = np.array(region_subject)[0]

activity_spread.shape

atlas_label_names

normalize = True
logs = False
sthreshold=0
time_steps=300
threshold_dm = 0.5
gradient= 0.01

np.where(RID == np.array(patinets_list) )[0][0]

am, am_distance = diffusion_model_simulation(sc_array, np.where(RID == np.array(patinets_list) )[0][0] , patinets_list, sthreshold, threshold_dm, sc_clusters, distance_matrix, time_steps, gradient)

am_order = copy.deepcopy(am)
am_order[np.where(am_order <1)]  = 0


sns.heatmap(am)
sns.heatmap(am_order)


am_order[-1,np.where(am_order[-1,:] == 0)[0]] =1

tmp,_,atlas_label_names_ind = np.intersect1d(region_subject_names, atlas_label_names, return_indices=True)



am_order_equal = am_order[:, atlas_label_names_ind]
sns.heatmap(am_order_equal)


dm_spread_order = np.argsort(np.argmax(am_order_equal, axis = 0))

atlas_label_names[atlas_label_names_ind][dm_spread_order]


activity_spread_order = np.argsort(activity_spread)

activity_spread[activity_spread_order]

atlas_label_names[activity_spread_order]



order_both = pd.DataFrame(columns = ["regions", "eeg_order", "dm_order"])
order_both["regions"] = region_subject_names


atlas_label_names_equal = atlas_label_names[atlas_label_names_ind]
for r in range(len(region_subject_names)):
    region = region_subject_names[r]
    
    region_ind_dm = np.where(region == atlas_label_names_equal)[0][0]
    dm_ord = np.where(dm_spread_order == region_ind_dm)[0][0]
    
    region_ind_sp = np.where(region == region_subject_names)[0][0]
    sp_ord = np.where(activity_spread_order == region_ind_sp)[0][0]
    
    #do not record if there was not an implant in that region
    if activity_spread[region_ind_sp] >= 0.99:
        sp_ord = -1
    
    order_both.loc[order_both["regions"]  == region, "eeg_order" ] = sp_ord
    order_both.loc[order_both["regions"]  == region, "dm_order" ] = dm_ord
    
aaa =   order_both
    
    
sns.scatterplot(data = order_both, x = "dm_order", y = "eeg_order")

v1 = np.array(order_both["eeg_order"]).astype(int)
v2 = np.array(order_both["dm_order"]).astype(int)


spearmanr(v1[~v1 < 0],v2[~v1 < 0])



region_subject_names["eeg_order"] = np.argsort(activity_spread)
np.argmax(am_order[:, :93], axis = 0)
