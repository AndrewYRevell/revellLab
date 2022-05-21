#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:43:22 2022

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
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
from scipy import signal, stats
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename
import matplotlib.pyplot as plt
from revellLab.paths import constants_paths as paths
from revellLab.packages.utilities import utils

metadataDir =  paths.METADATA
fnameJSON_old1 = join(metadataDir, "iEEGdataRevell.json")
fnameJSON_old2 = join(metadataDir, "iEEGdataRevell_seizure_severity.json")
RID_HUP = join(metadataDir, "RID_HUP.csv")


#%%

with open(fnameJSON_old1) as f: jsonFile1 = json.load(f)
with open(fnameJSON_old2) as f: jsonFile2 = json.load(f)


#%%

patients1 =  np.sort(np.array(list(jsonFile1["SUBJECTS"].keys())   )).astype('U8')
patients2 = np.sort(np.array(list(jsonFile2["SUBJECTS"].keys()))).astype('U8')

patinets_to_merge = np.setdiff1d(patients1, patients2)


for i in range(len(patinets_to_merge)):
    events = jsonFile1["SUBJECTS"][patinets_to_merge[i]   ]["Events"]["Ictal"]
    event_keys = list(events.keys())
    
    
    if not jsonFile1["SUBJECTS"][patinets_to_merge[i]   ]["Events"]["Ictal"][event_keys[0]]["FILE"] == "missing":
        patient_to_add =  jsonFile1["SUBJECTS"][patinets_to_merge[i]   ]
        RID =  patient_to_add["RID"]
        patient_to_add_key =  f"RID{RID:04d}"
        jsonFile2["SUBJECTS"][patient_to_add_key] = patient_to_add



with open( join(metadataDir, "iEEGdataRevell_seizure_severity_joined.json"), 'w') as f: json.dump(jsonFile2, f,  sort_keys=False, indent=4)
