#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:01:49 2022

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
from scipy.stats import pearsonr, spearmanr
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

normalize = True
logs = False
sc_array  = np.asarray(sc_list_subjects)

#%%
sthreshold = 0.0
sc_array_norm = np.asarray([x/x.max() for x in sc_list_subjects])
sc_array_norm_threshold = copy.deepcopy(sc_array_norm)
sc_array_norm_threshold_select = sthreshold < sc_array_norm_threshold
sc_array_norm_threshold[~sc_array_norm_threshold_select]=0
#%%
#for all regions get highest tipping point 



sthreshold=0.0
df = pd.read_csv(f"max_tipping_point_norm_{normalize}_log_{logs}_sthreshold_{sthreshold}.csv", header = 0 , index_col=0)

#get hippocampus SC

hipp_l = "Hippocampus_L"
hipp_r = "Hippocampus_R"
df[hipp_l]

hipp_l_ind = np.where(atlas_label_names == hipp_l)[0][0]
hipp_r_ind = np.where(atlas_label_names == hipp_r)[0][0]


#%%Compare hipp to max tp
tp_max_hipp = np.array(df[hipp_l])
strength_bl_hipp = sc_array[:, hipp_l_ind, hipp_r_ind]


sns.regplot(x = strength_bl_hipp,  y = tp_max_hipp )
spearmanr( strength_bl_hipp ,tp_max_hipp )
pearsonr( strength_bl_hipp ,tp_max_hipp )


#%%
#compare all SC of hipp to max tp

indexes =np.array( [hipp_l_ind, hipp_r_ind])
sc_hipp_connections = sc_array[:, indexes, :]

sc_hipp_connections_to_all_others = np.sum(np.sum(sc_hipp_connections, axis=1), axis=1)

sns.regplot(x = sc_hipp_connections_to_all_others,  y = tp_max_hipp )
spearmanr( sc_hipp_connections_to_all_others ,tp_max_hipp )
pearsonr( sc_hipp_connections_to_all_others ,tp_max_hipp )


#%% Separate out controls

control_RIDS = [f"sub-RID{x:04d}" for x in controls]


control_indexes = np.intersect1d( np.array(control_RIDS), np.array(sc_RIDS), return_indices=True)[2]


controls_sc_hipp = sc_array[control_indexes, hipp_l_ind, hipp_r_ind]
controls_tp_hipp = np.array(df.iloc[control_indexes][hipp_l])


sns.regplot(x = controls_sc_hipp,  y = controls_tp_hipp )
spearmanr( controls_sc_hipp ,controls_tp_hipp )
pearsonr( controls_sc_hipp ,controls_tp_hipp )



#%%Patients



patient_indexes = []
for s in range(len(sc_RIDS)):
    if not sc_RIDS[s] in control_RIDS:
        patient_indexes.append(s)
patient_indexes = np.asarray(patient_indexes)
    

patient_sc_hipp = sc_array[patient_indexes, hipp_l_ind, hipp_r_ind]
patient_tp_hipp = np.array(df.iloc[patient_indexes][hipp_l])


sns.regplot(x = patient_sc_hipp,  y = patient_tp_hipp )
spearmanr( patient_sc_hipp ,patient_tp_hipp )
pearsonr( patient_sc_hipp ,patient_tp_hipp )


















