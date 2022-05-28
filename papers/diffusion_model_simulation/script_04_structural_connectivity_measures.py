#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:58:29 2022

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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from scipy.stats import pearsonr, spearmanr
from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from os.path import join, splitext, basename

from pathos.multiprocessing import ProcessingPool as Pool #may need to do "pip install ipywidgets"

from packages.utilities import utils
from packages.diffusionModels import functions as dmf 
from paths import constants_paths as paths
from sklearn.linear_model import TweedieRegressor
#%

atlasPath = join(paths.TOOLS, "atlases", "atlases" )
atlasLabelsPath = join(paths.TOOLS, "atlases", "atlasLabels" )
atlasfilesPath = join(paths.TOOLS, "atlases", "atlasMetadata.json")

with open(paths.ATLAS_FILES_PATH) as f: atlas_files = json.load(f)

controls = [285, 286, 287, 288, 289, 290, 292,292,297,505,599,600,602,603,604,615,682,683,815,816,817,818,819,820,826,827,830,833,854,855]

#%
sc_list = []


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
atlas_numbers_of_regions = len(atlas_label_region_numbers)

sc_people = os.listdir(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES))
sc_RIDS = []
sc_list_subjects = []
for i in range(len(sc_people)):
    
    RID = sc_people[i]
    print(f"{i} {RID}")
    connectivity_loc = join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"{RID}" , "ses-research3Tv[0-9][0-9]" ,"matrices", f"{RID}.{atlas}.count.pass.connectogram.txt")
    connectivity_loc_glob = glob.glob( connectivity_loc  )
    
    
    if len(connectivity_loc_glob) > 0:
        connectivity_loc_path = connectivity_loc_glob[0]
        sc_RIDS.append(RID)
        sc = utils.read_DSI_studio_Txt_files_SC(connectivity_loc_path)
        #sc = sc/sc.max()
        #sc = utils.log_normalize_adj(sc)
        sc_list_subjects.append(sc)

number_of_subjects = len(sc_list_subjects)

#sc_indexes_to_get = np.array(range(len(sc_list_subjects[0])))
#sc_subset = sc_list_subjects[0][ sc_indexes_to_get[:, None] , sc_indexes_to_get[None, :]  ]
#sc_array_people = utils.getUpperTriangle(sc_subset)
#for i in range(1, len(sc_list_subjects)):
#    sc_subset = sc_list_subjects[i][ sc_indexes_to_get[:, None] , sc_indexes_to_get[None, :]  ]
#    #utils.plot_adj_heatmap(sc_subset)
#    sc_array_people = np.vstack([sc_array_people, utils.getUpperTriangle(sc_subset) ])

#%%
normalize = True
logs = False
#%%
#for all regions get highest tipping point 
sthreshold = 0
tp_max_array = np.zeros( shape = (number_of_subjects, atlas_numbers_of_regions ) )
tp_range = np.linspace( 0.0, 1, 30 )
number_of_tps = len(tp_range)
threshold_to_be_considered_no_spread = 0.5 * atlas_numbers_of_regions


for i in range(1,(number_of_subjects)):
    tp_max_per_region = np.zeros(shape = (number_of_tps, atlas_numbers_of_regions ))
    
    adj = sc_list_subjects[i]
    if normalize:
        adj_hat = adj/adj.max()
    if logs:
        adj_hat = utils.log_normalize_adj(adj)
    else:
        adj_hat = adj
    
    adj_hat_select = sthreshold < adj_hat
    adj_hat[~adj_hat_select]=0
    
    count = 1
    for r in range( atlas_numbers_of_regions ):
        for t in range(number_of_tps):
            print(f"\r{sc_RIDS[i]} {i}/{number_of_subjects}: {count/(atlas_numbers_of_regions*number_of_tps)*100:.1f}%      ", end = "\r")
            am = dmf.gradient_LTM(adj_hat, seed = r, time_steps = 20, threshold = tp_range[t])
            regions_with_spread = len(np.where(am[-1,:] == 1)[0])
            tp_max_per_region[t, r] = regions_with_spread
            count = count + 1
        #get max tp value
        tp_max = tp_range[np.argmax(tp_max_per_region[:,r] < threshold_to_be_considered_no_spread)   ]     
        tp_max_array[i, r] = tp_max

#%%
df = pd.DataFrame(tp_max_array, columns=  atlas_label_names, index = sc_RIDS)
tp_max_array.to_csv("max_tipping_point_norm_{normalize}_log_{logs}.csv" )


#%%

sns.histplot(tp_max_array[0,:], bins = 20)
sns.heatmap(tp_max_array) 

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#%%multiprocessing



def multiprocess_tp(i, sc_list_subjects, tp_range, sthreshold, threshold_percent_to_be_considered_no_spread, atlas_numbers_of_regions, time_steps, normalize, logs):
    number_of_tps = len(tp_range)
    threshold_to_be_considered_no_spread = threshold_percent_to_be_considered_no_spread * atlas_numbers_of_regions
    tp_max_per_region = np.zeros(shape = (number_of_tps, atlas_numbers_of_regions ))
    tp_max_array_id = np.zeros(atlas_numbers_of_regions)
    
    adj = sc_list_subjects[i]
    if normalize:
        adj_hat = adj/adj.max()
    elif logs:
        adj_hat = utils.log_normalize_adj(adj)
    else:
        adj_hat = adj
    
    adj_hat_threshold = copy.deepcopy(adj_hat)
    adj_hat_threshold_select = sthreshold < adj_hat_threshold
    adj_hat_threshold[~adj_hat_threshold_select]=0
    
    #sns.heatmap(adj)
    #sns.heatmap(adj_hat)
    
    count = 1
    for r in range( atlas_numbers_of_regions ):
        for t in range(number_of_tps):
            print(f"\r{sc_RIDS[i]} {i:02}: {count/(atlas_numbers_of_regions*number_of_tps)*100:.1f}%      ", end = "\r")
            am = dmf.LTM2(adj_hat_threshold, seed = r, time_steps = time_steps, threshold =tp_range[t])
            regions_with_spread = len(np.where(am[-1,:] == 1)[0])
            tp_max_per_region[t, r] = regions_with_spread
            count = count + 1
        #get max tp value
        tp_max = tp_range[np.argmax(tp_max_per_region[:,r] < threshold_to_be_considered_no_spread)   ]     
        tp_max_array_id[r] = tp_max
    return  tp_max_array_id


def multiprocess_tp_wrapper(cores, iterations, tp_max_array, sc_list_subjects, tp_range, sthreshold, threshold_percent_to_be_considered_no_spread, atlas_numbers_of_regions, time_steps, normalize, logs):


    p = Pool(cores)
    simulation = p.map(multiprocess_tp, iterations,
                                         repeat(sc_list_subjects),
                                         repeat(tp_range),
                                         repeat(sthreshold),
                                         repeat(threshold_percent_to_be_considered_no_spread),
                                         repeat(atlas_numbers_of_regions),
                                         repeat(time_steps),
                                         repeat(normalize),
                                         repeat(logs)
                                         )
    p.close()
    p.join()
    p.clear()
    
    for i,it in enumerate(iterations):
        print(i, it)
        tp_max_array[it, :] = simulation[i]

    return tp_max_array


def multiprocess_tp_gradient(i, sc_list_subjects, sthreshold, threshold, gradient, atlas_numbers_of_regions, time_steps, normalize, logs):
    

    
    region_percent_active_over_time = np.zeros(shape = (time_steps, atlas_numbers_of_regions ))

    
    adj = sc_list_subjects[i]
    if normalize:
        adj_hat = adj/adj.max()
    elif logs:
        adj_hat = utils.log_normalize_adj(adj)
    else:
        adj_hat = adj
    
    adj_hat_threshold = copy.deepcopy(adj_hat)
    adj_hat_threshold_select = sthreshold < adj_hat_threshold
    adj_hat_threshold[~adj_hat_threshold_select]=0
    
    #sns.heatmap(adj)

    for r in range( atlas_numbers_of_regions ):
        print(f"\r{sc_RIDS[i]} {i:02}: {r/(atlas_numbers_of_regions)*100:.1f}%      ", end = "\r")
        am = dmf.gradient_LTM(adj_hat_threshold, seed = r, time_steps = time_steps, threshold =threshold, gradient = gradient)
        percent_active = np.zeros(shape = (time_steps))
        for m in range(len(percent_active)):
            percent_active[m] = len(np.where(am[m,:] >=1)[0])/am.shape[1] *100
        region_percent_active_over_time[:,r] = percent_active
        
        #for r in range( atlas_numbers_of_regions ):
        #    sns.lineplot(x = range(time_steps), y=region_percent_active_over_time[:,r])

    return region_percent_active_over_time

def multiprocess_tp_gradient_wrapper(cores, iterations, tp_gradient_array, sc_list_subjects, sthreshold, threshold, gradient, atlas_numbers_of_regions, time_steps, normalize, logs):

    p = Pool(cores)
    simulation = p.map(multiprocess_tp_gradient, iterations,
                                         repeat(sc_list_subjects),
                                         repeat(sthreshold),
                                         repeat(threshold),
                                         repeat(gradient),
                                         repeat(atlas_numbers_of_regions),
                                         repeat(time_steps),
                                         repeat(normalize),
                                         repeat(logs)
                                         )
    p.close()
    p.join()
    p.clear()
    
    for i,it in enumerate(iterations):
        print(i, it)
        tp_gradient_array[it, :,:] = simulation[i]

    return tp_gradient_array
#%%

cores=12
normalize= True
logs = False


time_steps = 30
tp_max_array = np.zeros( shape = (number_of_subjects, atlas_numbers_of_regions ) )
tp_range = np.linspace( 0.0, 1, 5 )
threshold_percent_to_be_considered_no_spread = 0.50

sthreshold = 0.0
iterations = [0,1,2,4,5,6,7,9,10,11,20]
iterations = range(number_of_subjects)
tp_max_array = multiprocess_tp_wrapper(cores, iterations, tp_max_array, sc_list_subjects, tp_range, sthreshold, threshold_percent_to_be_considered_no_spread, atlas_numbers_of_regions, time_steps, normalize, logs)


sns.heatmap(tp_max_array) 
sns.histplot(tp_max_array[0,:], bins = 20)

#%%

df = pd.DataFrame(tp_max_array, columns=  atlas_label_names, index = sc_RIDS)
df.to_csv(f"max_tipping_point_norm_{normalize}_log_{logs}_sthreshold_{sthreshold}.csv" )


#%% Gradient

cores=12
normalize= True
logs = False


time_steps=250
threshold=0.5
gradient=0.01
tp_gradient_array= np.zeros(shape = (number_of_subjects, time_steps, atlas_numbers_of_regions ))

sthreshold = 0.0
iterations = [0,1,2,4,5,6,7,9,10,11,20]
iterations = range(number_of_subjects)

tp_gradient_array = multiprocess_tp_gradient_wrapper(cores, iterations, tp_gradient_array, sc_list_subjects, sthreshold, threshold, gradient, atlas_numbers_of_regions, time_steps, normalize, logs)


fname = join(paths.SEIZURE_SPREAD_FIGURES, f"percent_spread_norm_{normalize}_log_{logs}_sthreshold_{sthreshold}.pickle" )
#with open(fname, 'wb') as f: pickle.dump(tp_gradient_array, f)


with open(fname, 'rb') as f:tp_gradient_array = pickle.load(f)
#%%
for p in range(number_of_subjects):
    sns.lineplot(  x = range(time_steps), y= tp_gradient_array[p,:,40])



#get percent active at percent threshold



percent_active= np.zeros(shape = (number_of_subjects, atlas_numbers_of_regions ))

percent_threshold = 50
for p in range(number_of_subjects):
    for r in range(atlas_numbers_of_regions):
        percent_active[p,r] = np.argmax(tp_gradient_array[p,:,r] >= percent_threshold)


#%%

sns.histplot(percent_active[:,40])


control_RIDS = [f"sub-RID{x:04d}" for x in controls]


control_indexes = np.intersect1d( np.array(control_RIDS), np.array(sc_RIDS), return_indices=True)[2]


controls_spread_time_hipp = percent_active[:,40][control_indexes]




patient_indexes = []
for s in range(len(sc_RIDS)):
    if not sc_RIDS[s] in control_RIDS:
        patient_indexes.append(s)
patient_indexes = np.asarray(patient_indexes)
    


patients_spread_time_hipp = percent_active[:,40][patient_indexes]



sns.histplot(controls_spread_time_hipp,color= "red")
sns.histplot(patients_spread_time_hipp,color= "blue")


colors = np.repeat( "#ccccaa", number_of_subjects )

colors[control_indexes] = "#dd2222"

#%%

hipp_l = "Hippocampus_L"
hipp_r = "Hippocampus_R"

hipp_l_ind = np.where(atlas_label_names == hipp_l)[0][0]
hipp_r_ind = np.where(atlas_label_names == hipp_r)[0][0]

sc_array  = np.asarray(sc_list_subjects)

#%%

sc_array_norm = np.asarray([x/x.max() for x in sc_list_subjects])
sc_array_norm_threshold = copy.deepcopy(sc_array_norm)
sc_array_norm_threshold_select = sthreshold < sc_array_norm_threshold
sc_array_norm_threshold[~sc_array_norm_threshold_select]=0
#%%Compare hipp to max tp
sc_array_to_use = copy.deepcopy(sc_array_norm_threshold)


strength_bl_hipp = sc_array_to_use[:, hipp_l_ind, hipp_r_ind]


sns.regplot(x = strength_bl_hipp,  y = percent_active[:,40] )
spearmanr( strength_bl_hipp ,percent_active[:,40]  )
pearsonr( strength_bl_hipp ,percent_active[:,40]  )


#%%
#compare all SC of hipp to max tp

indexes =np.array( [hipp_l_ind, hipp_r_ind])
sc_hipp_connections = sc_array_to_use[:, indexes, :]

sc_hipp_connections_to_all_others = np.sum(np.sum(sc_hipp_connections, axis=1), axis=1)


sns.regplot(x = sc_hipp_connections_to_all_others,  y = percent_active[:,40] )

atlas_label_names
spearmanr( sc_hipp_connections_to_all_others ,percent_active[:,40]  )
pearsonr( sc_hipp_connections_to_all_others ,percent_active[:,40]  )
#%%
sns.regplot(x = strength_bl_hipp,  y = percent_active[:,40] )

#%%
sns.scatterplot(x = sc_hipp_connections_to_all_others,  y = percent_active[:,40], c = colors)
#%%
POWER = 1.1
total_network_weight = np.sum(np.sum(sc_array_to_use, axis=1), axis=1)


x = sc_hipp_connections_to_all_others.reshape(-1,1)
y = percent_active[:,40]


pr = TweedieRegressor(power = POWER, alpha=0, fit_intercept=True)
x_new = np.linspace(x.min(), x.max()).reshape(-1,1)
y_pred_pr = pr.fit(x, y).predict(x_new)



fig, axes = utils.plot_make(r = 10, c = 10,dpi=100)
axes = axes.flatten()
for a in range(100):
    
    x_orig = total_network_weight
    x = x_orig.reshape(-1,1)
    y = percent_active[:,a]
    
    invy = 1/y
    invy[np.where(invy == np.inf)[0]]=0
    
    pr = TweedieRegressor(power = 0, alpha=0, fit_intercept=True)
    x_new = np.linspace(x.min(), x.max()).reshape(-1,1)
    y_pred_pr = pr.fit(x, invy).predict(x_new)
    
    sns.scatterplot(ax = axes[a],  x = x_orig, y= invy, linewidth=0, s=50)
    sns.lineplot(ax = axes[a], x = x_new.flatten(), y = y_pred_pr, lw = 5, color = "black")
    #axes[a].set_ylim([0,250])





#%%
control_indexes
patient_indexes
for p in range(number_of_subjects):
    if p in control_indexes:
        color = "#ff0000"
    elif p in patient_indexes:
        color = "#0000ff"
    sns.lineplot(  x = range(time_steps), y= tp_gradient_array[p,:,40], color=color)







