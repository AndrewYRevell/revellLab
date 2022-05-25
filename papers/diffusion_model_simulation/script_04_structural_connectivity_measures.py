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

from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from os.path import join, splitext, basename

from pathos.multiprocessing import ProcessingPool as Pool #may need to do "pip install ipywidgets"

from packages.utilities import utils
from packages.diffusionModels import functions as dmf 
from paths import constants_paths as paths

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
            am = dmf.gradient_LTM(adj_hat_threshold, seed = r, time_steps = time_steps, threshold = tp_range[t])
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

#%%

cores=12
normalize= True
logs = False


time_steps = 50
tp_max_array = np.zeros( shape = (number_of_subjects, atlas_numbers_of_regions ) )
tp_range = np.linspace( 0.0, 1, 30 )
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






