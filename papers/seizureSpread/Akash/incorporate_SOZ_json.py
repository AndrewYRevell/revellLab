#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:20:02 2022

@author: arevell
"""


import json
import copy
import pandas as pd
import numpy as np
from os.path import join, splitext, basename
from scipy.io import loadmat


from revellLab.paths import constants_paths as paths





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

#%%
#% 02 Paths and files

metadataDir =  paths.METADATA
fnameJSON = join(metadataDir, "iEEGdataRevell.json")
fnameJSON_seizure_severity = join(metadataDir, "iEEGdataRevell_seizure_severity.json")
fname_patinet_localization = join(metadataDir, "patient_localization_final.mat")
RID_HUP = join(metadataDir, "RID_HUP.csv")

file_path= fname_patinet_localization

#%% Red files
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(fname_patinet_localization)


RID_HUP = pd.read_csv(RID_HUP)


with open(fnameJSON_seizure_severity) as f: jsonFile = json.load(f)

#%%

jsonFile["SUBJECTS"]
RID_keys =  list(jsonFile["SUBJECTS"].keys() )
hup_num_all = [jsonFile["SUBJECTS"][x]["HUP"]  for  x   in  RID_keys]


for i, HUP in enumerate(patients):
    #i=57; HUP = patients[i]; patients.index(f"HUP{hup_num_all[RID_keys.index('RID0424')]}");
    hup = int(HUP[3:])
    
    loc2 = hup_num_all.index(hup)
    
    
    

    channel_names = labels[i]
    
    
    
    soz_ind = np.where(soz[i] == 1)[0]
    
    soz_channel_names = np.array(channel_names)[soz_ind]
    
    
    resected_ind = np.where(resect[i] == 1)[0]
    resected_channel_names = np.array(channel_names)[resected_ind]
    
    
    soz_channel_names
    resected_channel_names
