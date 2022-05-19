#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:58:00 2022

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
palette = {"WN": "#1d5e9e", "CNN": "#73ace5", "LSTM": "#7373e5", "absolute_slope": "#961c1d", "line_length": "#d16a6a" , "power_broadband": "#d19e6a" }
palette_dark = {"WN": "#0a2036", "CNN": "#1e60a1", "LSTM": "#3737b3", "absolute_slope": "#250b0b", "line_length": "#5b1c1c" , "power_broadband": "#5b3c1c" }



#%%
soz_overlap_median = soz_overlap.groupby(['model', 'subject', "threshold"], as_index=False).median()


soz_overlap_outcomes = pd.merge(soz_overlap, outcomes, on='subject')
soz_overlap_median_outcomes = pd.merge(soz_overlap_median, outcomes, on='subject')

fig, axes = utils.plot_make(size_length=24, size_height=10)
sns.lineplot(data = soz_overlap_median, x = "threshold", y = "median_rank_percent", hue="model", ci=95, estimator=np.mean, ax = axes, hue_order=model_IDs, palette=palette, lw=6 , err_kws = dict(alpha = 0.075))

# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
axes.set_xlim([0,1])

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "over_thresholds.pdf"), bbox_inches='tight')

find_lowest = soz_overlap_median.groupby(["model", "threshold"], as_index=False).median()
for m in range(len(model_IDs)):
    model_ID = model_IDs[m]
    lowest = np.min(find_lowest[find_lowest["model"] == model_ID]["median_rank_percent"])
    lowest_ind = np.where(find_lowest[find_lowest["model"] == model_ID]["median_rank_percent"] == lowest)[0][0]
    lowest_threshold = np.array(find_lowest[find_lowest["model"] == model_ID]["threshold"])[lowest_ind]
    print(f"{lowest_threshold} {lowest}")
"""
0.69 0.08024691358024691
0.94 0.13414634146341464
0.58 0.103125
0.08 0.17557251908396948
0.11 0.20707070707070707
0.01 0.2625

tanh = True
0.69 0.08024691358024691
0.94 0.13414634146341464
0.58 0.103125
0.26 0.07142857142857142
0.43 0.06944444444444445
0.19 0.0898876404494382
"""
#%% Max box plots
thr = [0.69, 0.96, 0.58, 0.26, 0.43, 0.19]
thr = [0.69, 0.96, 0.58, 0.08, 0.11, 0.01]

model_IDs_label = ["WN", "CNN", "LSTM", "Abs Slope", "LL", "Pwr"]

    
df = soz_overlap_median[   ( (soz_overlap_median["model"] == "WN") & (soz_overlap_median["threshold"] == thr[0])  ) | ( (soz_overlap_median["model"] == "CNN") & (soz_overlap_median["threshold"] == thr[1])  ) | ( (soz_overlap_median["model"] == "LSTM") & (soz_overlap_median["threshold"] == thr[2])  ) | ( (soz_overlap_median["model"] == "absolute_slope") & (soz_overlap_median["threshold"] == thr[3])  ) | ( (soz_overlap_median["model"] == "line_length") & (soz_overlap_median["threshold"] == thr[4])  ) | ( (soz_overlap_median["model"] == "power_broadband") & (soz_overlap_median["threshold"] == thr[5])  )  ]




fig, axes = utils.plot_make(size_length=5)
sns.boxplot(ax = axes, data = df, x = "model", y = "median_rank_percent" , order=model_IDs, fliersize=0, palette=palette, medianprops=dict(color="black", lw = 4))

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
        
for i, tick in enumerate(axes.xaxis.get_major_ticks()):
    tick.label.set_fontsize(6)        
    tick.label.set_label(model_IDs_label[i])        
        
axes.set_xlabel("model",fontsize=20)

sns.swarmplot(ax = axes,data = df, x = "model", y = "median_rank_percent" , order=model_IDs, palette=palette_dark)


plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "boxplot_soz.pdf"), bbox_inches='tight')



pvals = pd.DataFrame(columns = ["model1", "model2", "pval", "cohend"])
for m in range(len(model_IDs)):
    start = m+1
    for n in range(start,len(model_IDs)):
        v1 = np.array(df[ (df["model"] == model_IDs[m])]["median_rank_percent"])
        v2 = np.array(df[ (df["model"] == model_IDs[n])]["median_rank_percent"])
        pval = stats.mannwhitneyu(v1,v2)[1]
        cohend = utils.cohend(v1, v2)
        pvals = pvals.append(dict(model1 = model_IDs[m], model2 = model_IDs[n], pval = pval, cohend = cohend), ignore_index=True)
        
        #pvals.append(pval)
        print(f"{model_IDs[m]},    {model_IDs[n]}     {np.round(pval,2)}  ")
        
pvals["fdr"] = utils.fdr2(pvals["pval"])        
print(pvals)
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

spread_at_time_save_location = join(BIDS, project_folder, f"full_analysis_save")
spread_at_time_save_location_file_basename = f"spread_at_time_tanh_{tanh}_by_{by}_model_{version}_2.pickle"
spread_at_time_save_location_file = join(spread_at_time_save_location, spread_at_time_save_location_file_basename)

with open(spread_at_time_save_location_file, 'rb') as f:  [finding_max, finding_max_seconds, tanh] = pickle.load(f)



#%%
df = copy.deepcopy(percent_active)
df = df.drop(["seizure"], axis = 1)
df = percent_active.melt( id_vars = ["model","version", "subject", "seizure", "threshold"], var_name = "time" , value_name= "percent_active")
df = df.drop(["version"], axis = 1)
df = df.fillna(np.nan)
df = df.groupby(["model", "subject", "seizure", "time", "threshold"], as_index=False).mean()
#df = df.groupby(["model", "subject", "time", "threshold"], as_index=False).mean()

df = pd.merge(df, outcomes, on='subject')
df = df.fillna(np.nan)

aaaaa = df.columns
#%% Time series of a single seizure

time_imit = 80

seconds = np.unique(df["time"])
sub_to_plot = "RID0309" #424,309,24,     51 ,64, 222          ,112,    70

fig, axes = utils.plot_make(size_length=40, c= len(model_IDs))
for m in range(len(model_IDs)):
    patinet_to_plot = percent_active[  (percent_active["model"] == model_IDs[m]) & (percent_active["threshold"] == thr[m]) & (percent_active["subject"] == sub_to_plot) ]
    patinet_to_plot_long = patinet_to_plot.melt( id_vars = ["model", "version", "threshold", "subject", "seizure"], var_name = "time" , value_name= "percent_active")
    
    
    ind = np.array(np.argsort(patinet_to_plot.iloc[:,30]))
    patinet_to_plot.iloc[ind]["seizure"]
    

    sns.lineplot(data = patinet_to_plot_long, x ="time",y ="percent_active", hue = "seizure"  , palette= "Blues", hue_order= patinet_to_plot.iloc[ind]["seizure"], lw = 5, ax = axes[m])
    # change all spines
    for axis in ['top','bottom','left','right']:
        axes[m].spines[axis].set_linewidth(6)
    
    # increase tick width
    axes[m].tick_params(width=4)
    #axes[m].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    axes[m].set_xlim([0,time_imit])
    axes[m].set_ylim([0,1])
    axes[m].get_legend().remove()
    if m > 0:
        axes[m].set_ylabel("")
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "spread_over_time.pdf"), bbox_inches='tight')

#%%
palette_outcome = dict(good = "#4d36da", poor = "#e89c14") 
time_imit = 80
fig, axes = utils.plot_make(size_length=40, c= len(model_IDs))
for m in range(len(model_IDs)):
    utils.printProgressBar(m+1, len(model_IDs), suffix= model_IDs[m] )

    model_ID = model_IDs[m]
    thresh = np.round(thr[m],2)
    
    
    category = "Engel_24_mo_binary"

    df_thresh = df[( df["threshold"]== thresh )]
    df_model =  df_thresh[( df_thresh["model"] ==model_ID )]
    
    df_filtered =     df_model[(df_model[category] != "unknown") & (df_model["time"] <= time_imit)]
    df_filtered_sec = df_model[(df_model[category] != "unknown") & (df_model["time"] <= time_imit)]
    ax = sns.lineplot(data = df_filtered_sec.fillna(np.nan), x = "time", y = "percent_active", hue = "Engel_24_mo_binary" , ci = 68, lw = 5, ax = axes[m], palette=palette_outcome)
    axes[m].get_legend().remove()
    axes[m].set_xlim([0,time_imit])
    axes[m].axvline(x=30, ls = '--', color = "#555555", lw = 4)
    for axis in ['top','bottom','left','right']:
        axes[m].spines[axis].set_linewidth(6)
    
    # increase tick width
    axes[m].tick_params(width=4)
    if m > 0:
        axes[m].set_ylabel("")
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "spread_over_time_good_vs_bad.pdf"), bbox_inches='tight')










#%%
s = 30
thr2 = [0.45, 0.6, 0.49, 0.08, 0.11, 0.01]
thr2 = [0.49, 0.58, 0.5, 0.1, 0.15, 0.01]
th = thr
m=0
for s in range(30,31):
    sec = df[   ( (df["model"] == "WN") & (df["threshold"] == th[0]) & (df["time"] == s)   ) | ( (df["model"] == "CNN") & (df["threshold"] == th[1])  & (df["time"] == s)  ) | ( (df["model"] == "LSTM") & (df["threshold"] == th[2]) & (df["time"] == s)  ) | ( (df["model"] == "absolute_slope") & (df["threshold"] == th[3]) & (df["time"] == s)   ) | ( (df["model"] == "line_length") & (df["threshold"] == th[4])& (df["time"] == s)   ) | ( (df["model"] == "power_broadband") & (df["threshold"] == th[5]) & (df["time"] == s)  )  ]
    
    sec = sec.replace("unknown", np.nan)
    
    pvals = pd.DataFrame(columns = ["model", "pval", "cohend"])
    for m in range(len(model_IDs)):
    
        v1 = np.array(sec[ (sec["model"] == model_IDs[m])  & (sec["Engel_24_mo_binary"] =="good")]["percent_active"])
        v2 = np.array(sec[ (sec["model"] == model_IDs[m])  & (sec["Engel_24_mo_binary"] =="poor")]["percent_active"])
        
        v1= v1[~np.isnan(v1)]
        v2 = v2[~np.isnan(v2)]
        pval = stats.mannwhitneyu(v1,v2)[1]
        cohend = utils.cohend(v1, v2)
        pvals = pvals.append(dict(model = model_IDs[m], pval = pval, cohend = cohend), ignore_index=True)
        
        #pvals.append(pval)
        #print(f"{model_IDs[m]},    {model_IDs[n]}     {np.round(pval,3)}  ")
        
    pvals["fdr"] = utils.fdr2(pvals["pval"])        
    print(s)
    print(pvals)
    print("\n\n")


palette_outcome = dict(good = "#4d36da", poor = "#e89c14") 
palette_outcome_dark = dict(good = "#2c1b93", poor = "#8e5f0c") 


fig, axes = utils.plot_make(size_length=10)
sns.boxplot(ax = axes, data = sec, x = "model", y = "percent_active" , hue = "Engel_24_mo_binary" , order=model_IDs, fliersize=0, palette=palette_outcome, medianprops=dict(color="black", lw = 4))
utils.adjust_box_widths(fig, 0.8)

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
        
for i, tick in enumerate(axes.xaxis.get_major_ticks()):
    tick.label.set_fontsize(6)        
    tick.label.set_label(model_IDs_label[i])        
        
axes.set_xlabel("model",fontsize=20)

sns.swarmplot(ax = axes,data = sec, x = "model", y = "percent_active" , hue = "Engel_24_mo_binary"  , order=model_IDs, palette=palette_outcome_dark, dodge=True, s=3)

# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)

# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "spread_at_030s_good_vs_bad.pdf"), bbox_inches='tight')
#%% Bootstrap

s = 30
thr2 = [0.45, 0.6, 0.49, 0.08, 0.11, 0.01]
thr2 = [0.49, 0.58, 0.5, 0.1, 0.15, 0.01]
th = thr

sec = df[   ( (df["model"] == "WN") & (df["threshold"] == th[0]) & (df["time"] == s)   ) | ( (df["model"] == "CNN") & (df["threshold"] == th[1])  & (df["time"] == s)  ) | ( (df["model"] == "LSTM") & (df["threshold"] == th[2]) & (df["time"] == s)  ) | ( (df["model"] == "absolute_slope") & (df["threshold"] == th[3]) & (df["time"] == s)   ) | ( (df["model"] == "line_length") & (df["threshold"] == th[4])& (df["time"] == s)   ) | ( (df["model"] == "power_broadband") & (df["threshold"] == th[5]) & (df["time"] == s)  )  ]
    
effect_sizes = pd.DataFrame(columns = ["model", "iteration", "cohend", "test_stat_ttest", "test_stat_mwu"])

N=10000
for m in range(len(model_IDs)):
    print(m)
    for it in range(N):
        #randomly pull patients
        
        pts_good = np.unique(sec[sec["Engel_24_mo_binary"] =="good"]["subject"])
        pts_poor = np.unique(sec[sec["Engel_24_mo_binary"] =="poor"]["subject"])
        pts_good_bs = random.choices(pts_good, k = len(pts_good))
        pts_poor_bs = random.choices(pts_poor, k = len(pts_poor))
        
        sec.loc[sec['subject'].isin(pts_good_bs)]
        
        good = sec.loc[sec['subject'].isin(pts_good_bs)]
        poor = sec.loc[sec['subject'].isin(pts_poor_bs)]
        
        
        v1 = good[good["model"] == model_IDs[m]]["percent_active"]
        v2 = poor[poor["model"] == model_IDs[m]]["percent_active"]
        
        v1= v1[~np.isnan(v1)]
        v2 = v2[~np.isnan(v2)]
        
        #v1_rand = random.choices(v1, k=len(v1))
        #v2_rand = random.choices(v2, k=len(v2))
        test_stat_ttest = stats.ttest_ind(v1,v2)[0]
        test_stat_mwu = stats.mannwhitneyu(v1,v2)[0]
        #cohend = utils.cohend(v1_rand, v2_rand)
        cohend = utils.cohend(v1, v2)
        effect_sizes = effect_sizes.append(dict(model = model_IDs[m],  iteration = it, cohend = cohend, test_stat_ttest =test_stat_ttest , test_stat_mwu = test_stat_mwu), ignore_index=True)
        

#%%
model_IDs_reversed = [ele for ele in reversed(model_IDs)]
model_IDs_reversed = [ "LSTM","power_broadband", "WN", "CNN",  "absolute_slope", "line_length"]
plt.rcParams['patch.edgecolor'] = 'none'
fig, axes = utils.plot_make(size_length=10)
#sns.histplot(data = effect_sizes, x = "cohend", hue = "model", kde=True, palette=palette,line_kws=dict(lw=5) , binwidth=0.05, alpha=0.1)
sns.kdeplot(data = effect_sizes, x = "cohend", hue = "model", palette=palette , fill = True, alpha=0.5, lw = 0, hue_order=model_IDs_reversed )
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0)

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "spread_at_030s_good_vs_bad_cohensd_bootstrap.pdf"), bbox_inches='tight')


plt.rcParams['patch.edgecolor'] = 'none'
fig, axes = utils.plot_make(size_length=10)
#sns.histplot(data = effect_sizes, x = "test_stat_ttest", hue = "model", kde=True, palette=palette,line_kws=dict(lw=5) )
sns.kdeplot(data = effect_sizes, x = "test_stat_ttest", hue = "model", palette=palette , fill = True, alpha=0.5, lw = 0, hue_order=model_IDs_reversed )
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0)

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "spread_at_030s_good_vs_bad_teststat_ttest_bootstrap.pdf"), bbox_inches='tight')

plt.rcParams['patch.edgecolor'] = 'none'
fig, axes = utils.plot_make(size_length=10)
#sns.histplot(data = effect_sizes, x = "test_stat_mwu", hue = "model", kde=True, palette=palette,line_kws=dict(lw=5) )
sns.kdeplot(data = effect_sizes, x = "test_stat_mwu", hue = "model", palette=palette , fill = True, alpha=0.5, lw = 0, hue_order=model_IDs_reversed )
# change all spines
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(6)
# increase tick width
axes.tick_params(width=4)
axes.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0)

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "spread_at_030s_good_vs_bad_teststat_mwu_bootstrap.pdf"), bbox_inches='tight')

#%%
pvals = pd.DataFrame(columns = ["model1", "model2", "pval_compare", "pval_zero"])
for m in range(len(model_IDs)-1):
    start = m+1
    for n in range(start,len(model_IDs)):
        
        model_values = effect_sizes[   (effect_sizes["model"] == model_IDs[m])]["cohend"] 
        mean_of_model = np.nanmean(model_values)
        model_to_test_against = effect_sizes[   (effect_sizes["model"] == model_IDs[n])]["cohend"]
        
        #sns.kdeplot( x = model_values, fill = True, alpha=0.5, lw = 0 )
        #sns.kdeplot( x = model_to_test_against, fill = True, alpha=0.5, lw = 0 )
        
        len_model = len(model_to_test_against)

        pval = len(np.where(mean_of_model < model_to_test_against  )[0])/len_model
        pval_zero = len(np.where(0 > model_values )[0])/len_model
        
        pvals = pvals.append(dict(model1 = model_IDs[m], model2 = model_IDs[n], pval_compare = pval, pval_zero =pval_zero ), ignore_index=True)
        
        #pvals.append(pval)
        print(f"{model_IDs[m]}, {np.round(pval_zero,3)}  ")
        #print(f"{model_IDs[m]},    {model_IDs[n]}     {np.round(pval,3)}  ")
        
pvals["fdr"] = utils.fdr2(pvals["pval_compare"])        
pvals["fdr2"] = utils.fdr2(pvals["pval_zero"])   

pvals.loc[:, "pval_zero"]
     
print(pvals)



#%%
#plot heatmap of cohensd 
fig, axes = utils.plot_make(r = 1, c = len(model_IDs), size_length=  len(model_IDs)*9 )
cohensd_max =  np.nanmax(np.asarray(finding_max_seconds["cohensd"]))
for m in range(len(model_IDs)):

    model_ID = model_IDs[m]
    cohensd_plot = finding_max_seconds[(finding_max_seconds["model"] == model_ID)]
    cohensd_plot_matrix = cohensd_plot.pivot(index='second', columns='threshold',values='cohensd')
    
    cohensd_plot_matrix_nanfill = np.nan_to_num(np.asarray(cohensd_plot_matrix))

    
    cohensd_plot_matrix_nanfill = scipy.ndimage.gaussian_filter(cohensd_plot_matrix_nanfill, sigma = 1.5)
    
    
    sns.heatmap(cohensd_plot_matrix_nanfill.T, ax = axes[m], vmin= 0, vmax = cohensd_max)
    
    axes[m].set_title(model_ID)
    axes[m].set_title(model_ID)
    if m == 0:
        axes[m].set_ylim([43,100])
    if m == 1:
        axes[m].set_ylim([47,100])
    if m == 2:
        axes[m].set_ylim([48,100])
    axes[m].set_xlim([0,time_imit])
 

plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "cohensd_across_all.pdf"), bbox_inches='tight')
    
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



quickness_analysis_save_location = join(BIDS, project_folder, f"full_analysis_save")
quickness_analysis_save_location_file_basename = f"quickness_analysis_tanh_{tanh}_by_{by}_model_{version}.pickle"
quickness_analysis_save_location_file = join(quickness_analysis_save_location, quickness_analysis_save_location_file_basename)


with open(quickness_analysis_save_location_file, 'rb') as f: [finding_max_quickness, tanh] = pickle.load(f)
#%%
#plot heatmap of cramers v for paper



fig, axes = utils.plot_make(r = 1, c = len(model_IDs), size_length=  len(model_IDs)*9 )
#axes = axes.flatten()

cramers_V1_max = np.nanmax(np.array(finding_max_quickness.cramers_V1))

for m in range(len(model_IDs)):

    model_ID = model_IDs[m]
    cramersV_plot = finding_max_quickness[(finding_max_quickness["model"] == model_ID)]
    cramersV_plot_matrix1 = cramersV_plot.pivot(index='cutoff', columns='threshold',values='cramers_V1')

    plt1 = np.array(np.array(cramersV_plot_matrix1).astype("float"))
    sigma = 1.5
    plt1 = scipy.ndimage.gaussian_filter(plt1, sigma = sigma)

    sns.heatmap(plt1.T, ax = axes[m], vmin = 0, vmax = cramers_V1_max)

    print(np.max(np.max(cramersV_plot_matrix1)))
    print(np.min(np.min(cramersV_plot.pivot(index='cutoff', columns='threshold',values='pval_everyone'))))

    print("\n")
    if m == 0:
        axes[m].set_ylim([40,100])
    if m == 1:
        axes[m].set_ylim([40,100])
    if m == 2:
        axes[m].set_ylim([40,100])
    if m == 3:
        axes[m].set_ylim([0,70])
    if m == 4:
        axes[m].set_ylim([0,70])
    if m == 5:
        axes[m].set_ylim([0,50])
    axes[m].set_xlim([0,60])
    
    
    
for m in range(len(model_IDs)):
    axes[m].set_xlabel("")
    axes[m].set_ylabel("")
    axes[m].set_ylabel(f"{model_IDs[i]}", fontsize = 15)
    
    
plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"validation", "cramersV_across_all.pdf"), bbox_inches='tight')
#%%
#plot heatmap of cramers v



fig, axes = utils.plot_make(r = len(model_IDs), c = 4, size_length=  20 ,size_height=len(model_IDs)*2 )
#axes = axes.flatten()

cramers_V1_max = np.nanmax(np.array(finding_max_quickness.cramers_V1))
cramers_V2_max = np.nanmax(np.array(finding_max_quickness.cramers_V2))

for m in range(len(model_IDs)):

    model_ID = model_IDs[m]
    cramersV_plot = finding_max_quickness[(finding_max_quickness["model"] == model_ID)]
    
    cramersV_plot_matrix1 = cramersV_plot.pivot(index='cutoff', columns='threshold',values='cramers_V1')
    cramersV_plot_matrix2 = cramersV_plot.pivot(index='cutoff', columns='threshold',values='cramers_V2')
    
    #cramersV_plot_matrix1 = cramersV_plot.pivot(index='cutoff', columns='threshold',values='pval_TLE')
    #cramersV_plot_matrix2 = cramersV_plot.pivot(index='cutoff', columns='threshold',values='pval_everyone')
    
    cramersV_plot_matrix1_odds = cramersV_plot.pivot(index='cutoff', columns='threshold',values='odds_ratio1')
    cramersV_plot_matrix2_odds = cramersV_plot.pivot(index='cutoff', columns='threshold',values='odds_ratio2')
    
    
    np.array(cramersV_plot_matrix1)
    np.array(cramersV_plot_matrix2).astype("float")
    np.array(cramersV_plot_matrix2_odds).astype("float")
    
    plt1 = np.array(np.array(cramersV_plot_matrix1).astype("float"))
    plt2 = np.array(np.array(cramersV_plot_matrix2).astype("float"))
    plt3 = np.array(np.array(cramersV_plot_matrix1_odds).astype("float"))
    plt4 = np.array(cramersV_plot_matrix2_odds).astype("float")
    
    sigma = 1.5
    plt1 = scipy.ndimage.gaussian_filter(plt1, sigma = sigma)
    plt2 = scipy.ndimage.gaussian_filter(plt2, sigma = sigma)
    plt3 = scipy.ndimage.gaussian_filter(plt3, sigma = sigma)
    plt4 = scipy.ndimage.gaussian_filter(plt4, sigma = sigma)
    
    sns.heatmap(plt1.T, ax = axes[m][0], vmin = 0, vmax = cramers_V1_max)
    sns.heatmap(plt2.T, ax = axes[m][1], vmin = 0, vmax = cramers_V2_max)
    sns.heatmap(plt3.T, ax = axes[m][2], vmin = 0, vmax = 1, cmap = "mako_r")
    sns.heatmap(plt4.T, ax = axes[m][3], vmin = 0, vmax = 1, cmap = "mako_r")
    
    
    print(np.max(np.max(cramersV_plot_matrix1)))
    print(np.max(np.max(cramersV_plot_matrix2)))
    print(np.min(np.min(cramersV_plot.pivot(index='cutoff', columns='threshold',values='pval_everyone'))))
    print(np.min(np.min(cramersV_plot.pivot(index='cutoff', columns='threshold',values='pval_TLE'))))
    print("\n")
    if m == 0:
        axes[m][0].set_ylim([43,100])
    if m == 1:
        axes[m][0].set_ylim([47,100])
    if m == 2:
        axes[m][0].set_ylim([48,100])
    axes[m][0].set_xlim([0,100])
    
    
    
for i in range(len(model_IDs)):
    for j in range(4):
        axes[i][j].set_xlabel("")
        axes[i][j].set_ylabel("")
        if j == 0:
            axes[i][j].set_ylabel(f"{model_IDs[i]}", fontsize = 9)

"pval_TLE"
"pval_everyone"
#%%

finding_max_quickness_fdr = copy.deepcopy(finding_max_quickness)
finding_max_quickness_fdr["pval_everyone_fdr"] = utils.correct_pvalues_for_multiple_testing(finding_max_quickness_fdr["pval_everyone"])
finding_max_quickness_fdr["pval_TLE_fdr"] = utils.correct_pvalues_for_multiple_testing(finding_max_quickness_fdr["pval_TLE"])

thr3 = [0.6 , 0.7, 0.6, 0.3, 0.5, 0.2]
th = thr


cutoff = 20
quickness_comparison = finding_max_quickness_fdr[(finding_max_quickness_fdr["cutoff"] == cutoff)]


quickness_comparison_filtered = quickness_comparison[   ( (quickness_comparison["model"] == "WN") & (quickness_comparison["threshold"] == th[0])  ) | ( (quickness_comparison["model"] == "CNN") & (quickness_comparison["threshold"] == th[1])  ) | ( (quickness_comparison["model"] == "LSTM") & (quickness_comparison["threshold"] == th[2])  ) | ( (quickness_comparison["model"] == "absolute_slope") & (quickness_comparison["threshold"] == th[3])  ) | ( (quickness_comparison["model"] == "line_length") & (quickness_comparison["threshold"] == th[4])  ) | ( (quickness_comparison["model"] == "power_broadband") & (quickness_comparison["threshold"] == th[5])  )  ]



print(quickness_comparison_filtered.pval_everyone)
print(quickness_comparison_filtered.cramers_V1)
print(quickness_comparison_filtered.pval_TLE)

print(utils.correct_pvalues_for_multiple_testing(list(quickness_comparison_filtered.pval_everyone)))



utils.correct_pvalues_for_multiple_testing(list(quickness_comparison_filtered.pval_TLE) + [0.04])

utils.correct_pvalues_for_multiple_testing(list(quickness_comparison_filtered.pval_everyone) )
utils.correct_pvalues_for_multiple_testing(list(quickness_comparison_filtered.pval_TLE) )






#%% Caluclate specific contingency tables