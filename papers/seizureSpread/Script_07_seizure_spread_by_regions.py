#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:48:25 2022

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
from revellLab.packages.diffusionModels import diffusionModels as DM
#Plotting parameters
custom_params = {"axes.spines.right": False, "axes.spines.top": False, 'figure.dpi': 300,
                 "legend.frameon": False, "savefig.transparent": True}
sns.set_theme(style="ticks", rc=custom_params,  palette="pastel")
sns.set_context("talk")
aspect = 50
kde_kws = {"bw_adjust": 2}



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

#%%
cluster = 2
for cluster in range(5):
    cluster_path = join(BIDS, project_folder, f"atlases", f"cluster_{cluster}.nii.gz" )
    img = nib.load(cluster_path)
    #utils.show_slices(img, data_type = "img")
    img_data = img.get_fdata()
    img_data[np.where(img_data == 0)] = -1
    
    
    
    if cluster == 0:
        vmin, vmax, slice_num = 0, 0.6, 100
    if cluster == 1:
        slice_num=115
        vmin, vmax, slice_num  = 0, 0.6, 115
    if cluster == 2:
        vmin, vmax, slice_num = 0.08, 0.5, 110
    if cluster == 3:
        vmin, vmax, slice_num = 0, 0.3, 130
    if cluster == 4:
        vmin, vmax, slice_num = 0, 0.6, 96
    
    
    #for slice_num in range(50,140):
    print(slice_num)
    slice_num = int(slice_num)
    img_data.shape
    
    slice1 = utils.get_slice(img_data, slice_ind = 1, slice_num = slice_num, data_type = "data")
    
    cmap = copy.copy(plt.get_cmap("magma"))
    masked_array = np.ma.masked_where(slice1 >=1, slice1)
    cmap.set_bad(color='#cdcdcd')
    cmap.set_under('white')
    fig, axes = utils.plot_make()
    axes.imshow(masked_array, cmap=cmap, origin="lower", vmin = vmin, vmax = vmax)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_xticks([])
    axes.set_yticks([])
    axes.axis("off")
    pos = axes.imshow(masked_array, cmap=cmap, origin="lower", vmin = vmin, vmax = vmax)
    fig.colorbar(pos, ax=axes)
    axes.set_title(f"{slice_num}")


plt.savefig(join(paths.SEIZURE_SPREAD_FIGURES,"clustering", f"COLOR_cluster_atlas_{cluster}_slice_{slice_num}.pdf"), bbox_inches='tight')
    



"""    

cluster 0:     100
cluster 1:     115

"""



