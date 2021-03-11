"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/media/arevell/sharedSSD/linux/papers/paper005" #Parent directory of project
import sys
import os
import pandas as pd
import numpy as np
from os.path import join as ospj

#import custom
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
import tissue_segmentation_atlas_registration as seg

#%% Input/Output Paths and File names
ifname_elec_loc_csv = ospj( path, "data/data_raw/electrode_localization/electrode_localization_subjects.csv")
ifpath_electrode_localization = ospj( path, "data/data_raw/electrode_localization")
ifname_MNI = ospj( path, "data/data_raw/MNI_brain_template/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182")
ifname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
ifpath_atlases = ospj( path, "data/data_raw/atlases/atlases")
ifpath_imaging = ospj( path, "data/data_raw/imaging")

ofpath_segmentation = ospj( path, "data/data_processed/GM_WM_segmentations")
ofpath_atlas_registration = ospj( path, "data/data_processed/atlas_registration/T00")

if not (os.path.isdir(ofpath_atlas_registration)): os.makedirs(ofpath_atlas_registration, exist_ok=True)
if not (os.path.isdir(ofpath_segmentation)): os.makedirs(ofpath_segmentation, exist_ok=True)

#%% Load Study Meta Data
data = pd.read_csv(ifname_elec_loc_csv)    
atlases = pd.read_csv(ifname_atlases_csv)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]


#%% Get Electrode Localization. Find which region each electrode is in for each atlas
for i in range(0,len(sub_IDs_unique)):
    sub_ID = sub_IDs_unique[i]
    print(f"\nSubject: {sub_ID}")
    #getting T00 images
    ifname_T00 = ospj(ifpath_electrode_localization, f"sub-{sub_ID}", f"sub-{sub_ID}_T00_mprage")
    ofpath_segmentation_sub_ID = ospj(ofpath_segmentation, f"sub-{sub_ID}")
    if not (os.path.isdir(ofpath_segmentation_sub_ID)): os.mkdir(ofpath_segmentation_sub_ID)

    #register atlases to T00 
    ofpath_atlas_registration_sub_ID = ospj(ofpath_atlas_registration, f"sub-{sub_ID}")
    if not (os.path.isdir(ofpath_atlas_registration_sub_ID)): os.mkdir(ofpath_atlas_registration_sub_ID)
    ofbase_flirt_MNI = ospj(ofpath_segmentation_sub_ID, "sub-{sub_ID}_MNI_to_T00_std_linear")
    ofbase_fnirt_MNI = ospj(ofpath_segmentation_sub_ID, "sub-{sub_ID}_MNI_to_T00_std_nonlinear")
    
    if not (os.path.exists(ofbase_fnirt_MNI + ".nii.gz")):
        seg.register_MNI_to_T00(ifname_T00,ifname_MNI, ofpath_segmentation_sub_ID, ofbase_flirt_MNI, ofbase_fnirt_MNI)
    
    #getting atlases
    for a in range(len(atlases)):
        ifname_atlas = ospj(ifpath_atlases, np.array(atlases["atlas_filename"])[a] )
        ofname_atlas_to_T00 = ospj(ofpath_atlas_registration_sub_ID, "sub-{sub_ID}_T00_{np.array(atlases['atlas_filename'])[a]}" )
        if not (os.path.exists( ofname_atlas_to_T00 )):
            print("\n")
            seg.applywarp_to_atlas(ifname_atlas, ifname_T00, ofbase_fnirt_MNI, ofpath_segmentation_sub_ID, ofname_atlas_to_T00)


#%%












