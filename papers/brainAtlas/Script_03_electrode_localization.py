"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 
    1. This is a wrapper script: Runs through meta-data to automatically calculate for all data
        Meta-data: data_raw/iEEG_times/EEG_times.xlsx
    2. Get electrode localization: Find which region each electrode is in for each atlas
    3. Calls electrode_localization.by_atlas in paper001/code/tools
        See this function for more detail

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Import appropriate tools 
    2. Get appropriate input and output paths and file names
    3. Setting appropriate parameters and preprocessing of data
    4. Get electrode localization
        1. Get electrode localization file for each patient
        2. For standard atlases:
            1. Get atlas image path
            2. Call electrode_localization.by_atlas
        3. For random atlases:
            1. For each random atlas permutation
                1. Get atlas image path
                2. Call electrode_localization.by_atlas
                

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    None. This is a wrapper scipt that automatically runs based on meta-data file
    Meta-data: data_raw/iEEG_times/EEG_times.xlsx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    Saves electrode localization by atlas for each patinet's electrode localization file 
    in appropriate directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python3.6 Script_03_electrode_localization.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import electrode_localization
import pandas as pd
import numpy as np

#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_electrode_localization = ospj( path, "data/data_raw/electrode_localization")
ifpath_atlases_standard = ospj( path, "data/data_raw/atlases/standard_atlases")
ifpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ofpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization_atlas_region")

#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)


#%% Get Electrode Localization. Find which region each electrode is in for each atlas
for i in range(len(sub_IDs_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print("\nSubject: {0}".format(sub_ID))
    #getting electrode localization file
    ifpath_electrode_localization_sub_ID = ospj(ifpath_electrode_localization, "sub-{0}".format(sub_ID))
    if not os.path.exists(ifpath_electrode_localization_sub_ID):
        print("Path does not exist: {0}".format(ifpath_electrode_localization_sub_ID))
    ifname_electrode_localization_sub_ID =[f for f in sorted(os.listdir(ifpath_electrode_localization_sub_ID))][0]    
    ifname_electrode_localization_sub_ID_fullpath = ospj(ifpath_electrode_localization_sub_ID, ifname_electrode_localization_sub_ID)
    #getting atlas names
    atlas_names_standard = [f for f in sorted(os.listdir(ifpath_atlases_standard))]
    atlas_names_random = [f for f in sorted(os.listdir(ifpath_atlases_random))]
    #Output electrode localization
    ofpath_electrode_localization_sub_ID = ospj(ofpath_electrode_localization, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofpath_electrode_localization_sub_ID)): os.mkdir(ofpath_electrode_localization_sub_ID)#if the path doesn't exists, then make the directory
    #standard atlases: getting electrode localization by region
    for a in range(len(atlas_names_standard)):
        ifname_atlases_standard = ospj(ifpath_atlases_standard, atlas_names_standard[a] )
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
        ofpath_electrode_localization_sub_ID_atlas = ospj(ofpath_electrode_localization_sub_ID, atlas_name)
        if not (os.path.isdir(ofpath_electrode_localization_sub_ID_atlas)): os.mkdir(ofpath_electrode_localization_sub_ID_atlas)
        ofname_electrode_localization = "{0}_{1}.csv".format( os.path.splitext(ifname_electrode_localization_sub_ID)[0],os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0])
        ofname_electrode_localization_fullpath = ospj(ofpath_electrode_localization_sub_ID_atlas, ofname_electrode_localization)
        print("Subject: {0}  Atlas: {1}".format(sub_ID, atlas_names_standard[a]))
        electrode_localization.by_atlas(ifname_electrode_localization_sub_ID_fullpath, ifname_atlases_standard, ofname_electrode_localization_fullpath)
    #random atlases: getting electrode localization by region
    for a in range(len(atlas_names_random)):
        versions = [f for f in sorted(os.listdir(ospj(ifpath_atlases_random, atlas_names_random[a])))]  
        for p in range(len(versions)):
           ifname_atlases_random = ospj(ifpath_atlases_random,atlas_names_random[a] , versions[p]  )
           atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
           ofpath_electrode_localization_sub_ID_atlas = ospj(ofpath_electrode_localization_sub_ID, atlas_name)
           if not (os.path.isdir(ofpath_electrode_localization_sub_ID_atlas)): os.mkdir(ofpath_electrode_localization_sub_ID_atlas)
           ofname_electrode_localization = "{0}_{1}.csv".format( os.path.splitext(ifname_electrode_localization_sub_ID)[0],os.path.splitext(os.path.splitext("{0}_v{1}.nii.gz".format(atlas_names_random[a], '{:04}'.format(p+1)))[0])[0])
           ofname_electrode_localization_fullpath = ospj(ofpath_electrode_localization_sub_ID_atlas, ofname_electrode_localization)
           print("Subject: {0}  Atlas: {1}".format(sub_ID,  "{0}_v{1}.nii.gz".format(atlas_names_random[a], '{:04}'.format(p+1))  ))
           electrode_localization.by_atlas(ifname_electrode_localization_sub_ID_fullpath, ifname_atlases_random, ofname_electrode_localization_fullpath)
      

   

#%%












