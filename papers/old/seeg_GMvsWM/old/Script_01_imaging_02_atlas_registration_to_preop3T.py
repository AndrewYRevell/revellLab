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
ifname_diffusion_imaging = ospj( path, "data/data_raw/iEEG_times/diffusion_imaging.csv")
ifname_MNI = ospj( path, "data/data_raw/MNI_brain_template/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182")
ifname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
ifpath_atlases = ospj( path, "data/data_raw/atlases/atlases")
ifpath_imaging_qsiprep = ospj( path, "data/data_processed/imaging/qsiprep/")

ofpath_atlas_registration = ospj( path, "data/data_processed/atlas_registration/preop3T")

if not (os.path.isdir(ofpath_atlas_registration)): os.makedirs(ofpath_atlas_registration, exist_ok=True)

#%% Load Study Meta Data
data = pd.read_csv(ifname_diffusion_imaging) 

atlases = pd.read_csv(ifname_atlases_csv)       
#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]


#%% Get Electrode Localization. Find which region each electrode is in for each atlas
for i in range(len(sub_IDs_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print(f"\nSubject: {sub_ID}")
    #getting preop3T and T00 images
    
    ifpath_imaging_qsiprep_subID = ospj(ifpath_imaging_qsiprep, f"sub-{sub_ID}/anat")
    ifname_3Tpreop = ospj(ifpath_imaging_qsiprep_subID, f"sub-{sub_ID}_desc-preproc_T1w")
    ifname_3Tpreop_mask = ospj(ifpath_imaging_qsiprep_subID, f"sub-{sub_ID}_desc-brain_mask")
    #register atlases to patient 
    ofpath_atlas_registration_sub_ID = ospj(ofpath_atlas_registration, f"sub-{sub_ID}")
    
    if not (os.path.isdir(ofpath_atlas_registration_sub_ID)): os.mkdir(ofpath_atlas_registration_sub_ID)
    
    ofbase_flirt_MNI = ospj(ofpath_atlas_registration_sub_ID, f"sub-{sub_ID}_MNI_to_preop3T_std_linear")
    ofbase_fnirt_MNI = ospj(ofpath_atlas_registration_sub_ID, f"sub-{sub_ID}_MNI_to_preop3T_std_nonlinear")

    if not (os.path.exists( ofbase_fnirt_MNI+ ".nii.gz" )):
        seg.register_MNI_to_3Tpreop(ifname_3Tpreop, ifname_3Tpreop_mask, ifname_MNI, ofpath_atlas_registration_sub_ID, ofbase_flirt_MNI, ofbase_fnirt_MNI)
    

    #register atlases
    for a in range(len(atlases)):
        ifname_atlas = ospj(ifpath_atlases, np.array(atlases["atlas_filename"])[a] )
        ofname_atlas_to_preop3T = ospj(ofpath_atlas_registration_sub_ID, f"sub-{sub_ID}_preop3T_{np.array(atlases['atlas_filename'])[a]}" )
        if not (os.path.exists( ofname_atlas_to_preop3T )):
            print("\n")
            seg.applywarp_to_atlas(ifname_atlas, ifname_3Tpreop, ofbase_fnirt_MNI, ofpath_atlas_registration_sub_ID, ofname_atlas_to_preop3T)

    
    




#%%












