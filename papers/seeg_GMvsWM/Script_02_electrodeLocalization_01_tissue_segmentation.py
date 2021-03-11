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
import combine_FIRST_and_FAST_images as combo
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
    #getting preop3T and T00 images
    ifname_preop3T = ospj(ifpath_imaging, f"sub-{sub_ID}", "ses-preop3T", "anat", f"sub-{sub_ID}_ses-preop3T_acq-3D_T1w")
    ifname_T00 = ospj(ifpath_electrode_localization, f"sub-{sub_ID}", f"sub-{sub_ID}_T00_mprage")
    ofpath_segmentation_sub_ID = ospj(ofpath_segmentation, f"sub-{sub_ID}")
    
    if not (os.path.isdir(ofpath_segmentation_sub_ID)): os.mkdir(ofpath_segmentation_sub_ID)
    
    ofbase_flirt = ospj(ofpath_segmentation_sub_ID, f"sub-{sub_ID}_preop3T_to_T00_std_linear")
    ofbase_fnirt = ospj(ofpath_segmentation_sub_ID, f"sub-{sub_ID}_preop3T_to_T00_std_nonlinear")
    
    ifname_first =  ospj(ofpath_segmentation_sub_ID, f"sub-{sub_ID}_ses-preop3T_acq-3D_T1w_std_bet_subcort_all_fast_firstseg.nii.gz")
    ifname_fast =  ospj(ofpath_segmentation_sub_ID, f"sub-{sub_ID}_ses-preop3T_acq-3D_T1w_std_bet_seg.nii.gz")
    
    ofname_FIRST_FAST_COMBINED = ospj(ofpath_segmentation_sub_ID, f"sub-{sub_ID}_ses-preop3T_acq-3D_T1w_std_bet_GM_WM_CSF.nii.gz")
    ofname_FIRST_FAST_COMBINED_to_T00 = ospj(ofpath_segmentation_sub_ID, f"sub-{sub_ID}_preop3T_to_T00_std_GM_WM_CSF.nii.gz")
    
    #First and Fast segmentation (first is subcortical structures, fast is cortical gray matter)
    if not (os.path.exists(ifname_fast)):
        seg.first_and_fast_segmentation(ifname_preop3T, ifname_T00, ofpath_segmentation_sub_ID)
        
    #Combine First and Fast images
    if not (os.path.exists(ofname_FIRST_FAST_COMBINED)):
        combo.combine_first_and_fast(ifname_first, ifname_fast, ofname_FIRST_FAST_COMBINED)
    
    #Register preop3T to T00
    if not (os.path.exists( f"{ofbase_flirt}.nii.gz" )):
        seg.register_preop3T_to_T00(ifname_preop3T, ifname_T00, ofpath_segmentation_sub_ID, ofbase_flirt, ofbase_fnirt)
        
    #Apply the warp registration to FIRST/FAST image
    if not (os.path.exists(ofname_FIRST_FAST_COMBINED_to_T00)):
        seg.applywarp_to_combined_first_fast(ofname_FIRST_FAST_COMBINED, ifname_T00, ofbase_flirt, ofpath_segmentation_sub_ID, ofname_FIRST_FAST_COMBINED_to_T00)
        
        

#%%












