#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:57:33 2022

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
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
import statsmodels.api as sm
from scipy import signal, stats
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename
from pathos.multiprocessing import ProcessingPool as Pool

#revellLab
#utilities, constants/parameters, and thesis helper functions
from revellLab.packages.utilities import utils
from revellLab.papers.white_matter_iEEG import constants_parameters as params
from revellLab.papers.white_matter_iEEG import constants_plotting as plot
from revellLab.paths import constants_paths as paths
from revellLab.papers.white_matter_iEEG.helpers import thesis_helpers as helper


# %%
ecog_sub = ["sub-RID0013", "sub-RID0015", "sub-RID0018", "sub-RID0020", "sub-RID0021", "sub-RID0024", "sub-RID0027", "sub-RID0030", "sub-RID0031", "sub-RID0032", "sub-RID0049", "sub-RID0050", "sub-RID0051", "sub-RID0054", "sub-RID0055", "sub-RID0058", "sub-RID0065", "sub-RID0068", "sub-RID0069", "sub-RID0101", "sub-RID0102", "sub-RID0106", "sub-RID0117", "sub-RID0143", "sub-RID0160", "sub-RID0171", "sub-RID0193", "sub-RID0213", "sub-RID0222", "sub-RID0295", "sub-RID0357", "sub-RID0520"]
subs = os.listdir(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION)

df_percent = pd.DataFrame(columns = ["sub", "gm_50", "wm_50", "gm_dist", "wm_dist", "gm_arg", "wm_arg"])
for s in range(len(subs)):
    sub = subs[s]
    if sub in ecog_sub:
        continue
    if os.path.exists(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION + f"/{sub}/ses-implant01/{sub}_ses-implant01_desc-atlasLocalization.csv"):
        df = pd.read_csv(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION + f"/{sub}/ses-implant01/{sub}_ses-implant01_desc-atlasLocalization.csv")
        
        df_inside = df[(df["tissue_numberArgMax"] > 1)]
        N = len(df_inside)
    
        gm_50 = len(df_inside[df_inside["percent_GM"] > 0.5])
        wm_50 = len(df_inside[df_inside["percent_WM"] > 0.5])
        
        gm_dist = len(df_inside[df_inside["distance_to_GM_millimeters"] <=0])
        wm_dist = len(df_inside[df_inside["distance_to_GM_millimeters"] > 2])
        
        
        gm_arg = len(df_inside[(df_inside["tissue_numberArgMax"] == 2)])
        wm_arg = len(df_inside[(df_inside["tissue_numberArgMax"] == 3)])
        
        percent_gm_50 = gm_50 /N
        percent_wm_50 = wm_50 /N
        percent_gm_dist = gm_dist /N
        percent_wm_dist = wm_dist /N
        percent_gm_arg = gm_arg /N
        percent_wm_arg = wm_arg /N
        
        
        df_percent = df_percent.append(dict(sub = sub, 
                                            gm_50 = percent_gm_50, 
                                            wm_50 = percent_wm_50, 
                                            gm_dist =  percent_gm_dist,
                                            wm_dist =  percent_wm_dist,
                                            gm_arg =  percent_gm_arg,
                                            wm_arg = percent_wm_arg
                                            
                                            
                                            ),ignore_index=True)
        print(f"{s}; {np.round(percent_wm_50, 2)}")



#%%
means = []
np.median(df_percent["gm_50"])
np.median(df_percent["gm_dist"])
np.median(df_percent["gm_arg"])

np.median(df_percent["wm_50"])
np.median(df_percent["wm_dist"])
np.median(df_percent["wm_arg"])



fig, axes = utils.plot_make(r = 3, c = 2, size_length = 15, sharex = False, sharey = True)
axes = axes.flatten()
sns.histplot( data = df_percent, x = "gm_50" , bins=20, kde=True, ax = axes[0], color = plot.COLORS_TISSUE_LIGHT_MED_DARK[1][0], lw = 0, line_kws = dict(lw = 5))
sns.histplot( data = df_percent, x = "gm_dist" , bins=20, kde=True, ax = axes[2], color = plot.COLORS_TISSUE_LIGHT_MED_DARK[1][0], lw = 0, line_kws = dict(lw = 5))
sns.histplot( data = df_percent, x = "gm_arg" , bins=20, kde=True, ax = axes[4], color = plot.COLORS_TISSUE_LIGHT_MED_DARK[1][0], lw = 0, line_kws = dict(lw = 5))
sns.histplot( data = df_percent, x = "wm_50" , bins=20, kde=True, ax = axes[1], color = plot.COLORS_TISSUE_LIGHT_MED_DARK[1][1], lw = 0, line_kws = dict(lw = 5))
sns.histplot( data = df_percent, x = "wm_dist" , bins=20, kde=True, ax = axes[3], color = plot.COLORS_TISSUE_LIGHT_MED_DARK[1][1], lw = 0, line_kws = dict(lw = 5))
sns.histplot( data = df_percent, x = "wm_arg" , bins=20, kde=True, ax = axes[5], color = plot.COLORS_TISSUE_LIGHT_MED_DARK[1][1], lw = 0, line_kws = dict(lw = 5, color = "red"))

axes[0].set_xlim([0.25,0.8])
axes[2].set_xlim([0.25,0.8])
axes[4].set_xlim([0.25,0.8])
axes[1].set_xlim([0.0,0.7])
axes[3].set_xlim([0.0,0.7])
axes[5].set_xlim([0.0,0.7])

for i in range(len(axes)):
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['top'].set_visible(False)


axes[0].axvline(np.median(df_percent["gm_50"]) ,ls='--', lw=5, color = plot.COLORS_TISSUE_LIGHT_MED_DARK[2][0])
axes[2].axvline(np.median(df_percent["gm_dist"]) ,ls='--', lw=5, color = plot.COLORS_TISSUE_LIGHT_MED_DARK[2][0])
axes[4].axvline(np.median(df_percent["gm_arg"]) ,ls='--', lw=5, color = plot.COLORS_TISSUE_LIGHT_MED_DARK[2][0])
axes[1].axvline(np.median(df_percent["wm_50"]) ,ls='--', lw=5, color = plot.COLORS_TISSUE_LIGHT_MED_DARK[2][1])
axes[3].axvline(np.median(df_percent["wm_dist"]) ,ls='--', lw=5, color = plot.COLORS_TISSUE_LIGHT_MED_DARK[2][1])
axes[5].axvline(np.median(df_percent["wm_arg"]) ,ls='--', lw=5, color = plot.COLORS_TISSUE_LIGHT_MED_DARK[2][1])

axes[0].set_ylim([0.0,16])
plt.savefig("/home/arevell/Documents/figures/white_matter_iEEG/contact_localization_distribution.pdf")



