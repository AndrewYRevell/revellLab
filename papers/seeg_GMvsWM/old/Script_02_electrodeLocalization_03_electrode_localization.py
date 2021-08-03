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
import electrode_localization

#%% Input/Output Paths and File names
ifname_elec_loc_csv = ospj( path, "data/data_raw/electrode_localization/electrode_localization_subjects.csv")
ifpath_electrode_localization = ospj( path, "data/data_raw/electrode_localization")
ifpath_segmentations = ospj( path, "data/data_processed/GM_WM_segmentations")
ifname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
ifpath_atlas_registration = ospj( path, "data/data_processed/atlas_registration/T00")
ifpath_atlas_labels = ospj( path, "data/data_raw/atlases/atlas_labels")

ofpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization")
if not (os.path.isdir(ofpath_electrode_localization)): os.makedirs(ofpath_electrode_localization, exist_ok=True)
#%% Load Study Meta Data
data = pd.read_csv(ifname_elec_loc_csv)    
atlases = pd.read_csv(ifname_atlases_csv)  
#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]


#%% Get Electrode Localization. Find which region each electrode is in for each atlas
for i in range(len(sub_IDs_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print(f"\nSubject: {sub_ID}")
   
    #getting electrode localization file
    ifpath_electrode_localization_sub_ID = ospj(ifpath_electrode_localization, f"sub-{sub_ID}")
    if not os.path.exists(ifpath_electrode_localization_sub_ID): print(f"Path does not exist: {ifpath_electrode_localization_sub_ID}")
    ifname_electrode_localization_sub_ID = ospj(ifpath_electrode_localization_sub_ID, f"sub-{sub_ID}_electrodenames_coordinates_native_and_T1.csv")
    os.path.isfile(ifname_electrode_localization_sub_ID)
    #getting atlas/segmentation files
    ifpath_seg_sub_ID = ospj(ifpath_segmentations, f"sub-{sub_ID}")
    if not os.path.exists(ifpath_seg_sub_ID): print(f"Path does not exist: {ifpath_seg_sub_ID}")
    ifname_seg_sub_ID = ospj(ifpath_seg_sub_ID, f"sub-{sub_ID}_preop3T_to_T00_std_GM_WM_CSF.nii.gz")
    ifpath_atlas_registration_sub_ID = ospj(ifpath_atlas_registration, f"sub-{sub_ID}")
    
    
    #Output names for electrode localization
    ofpath_electrode_localization_sub_ID = ospj(ofpath_electrode_localization, f"sub-{sub_ID}")
    if not (os.path.isdir(ofpath_electrode_localization_sub_ID)): os.mkdir(ofpath_electrode_localization_sub_ID)#if the path doesn't exists, then make the directory
    ofpath_localization_files = ospj(ofpath_electrode_localization_sub_ID, "individual_files")
    if not (os.path.isdir(ofpath_localization_files)): os.mkdir(ofpath_localization_files)#if the path doesn't exists, then make the directory
    ofname_electrode_localization_concatenated = ospj(ofpath_electrode_localization_sub_ID, f"sub-{sub_ID}_electrode_localization.csv")

    #Atlas input names
    ifname_atlas_path = ifname_seg_sub_ID
    ifname_atlas_labels_path = ospj( ifpath_atlas_labels, "tissue_segmentation.csv")
    
    #############
    #############
    #localization by region to tissue segmentation
    ofname = ospj(ofpath_localization_files, f"sub-{sub_ID}_00_GM_WM_CSF.csv")
    electrode_localization.by_region(ifname_electrode_localization_sub_ID, ifname_atlas_path, ifname_atlas_labels_path, ofname, description = "tissue_segmentation", noLabels=False)
    df = pd.read_csv(ofname, sep=",", header=0)
    for e in range(len( df  )):
        electrode_name = df.iloc[e]["electrode_name"]
        if (len(electrode_name) == 3): electrode_name = f"{electrode_name[0:2]}0{electrode_name[2]}"
        df.at[e, "electrode_name" ] = electrode_name
    pd.DataFrame.to_csv(df, ofname, header=True, index=False)
    
    
    #distance to tissue 
    ofname = ospj(ofpath_localization_files, f"sub-{sub_ID}_00_WM_distance.csv")
    if not os.path.exists(ofname):
        electrode_localization.distance_from_label(ifname_electrode_localization_sub_ID, ifname_atlas_path, 2,ifname_atlas_labels_path, ofname)
    df = pd.read_csv(ofname, sep=",", header=0)
    for e in range(len( df  )):
        electrode_name = df.iloc[e]["electrode_name"]
        if (len(electrode_name) == 3): electrode_name = f"{electrode_name[0:2]}0{electrode_name[2]}"
        df.at[e, "electrode_name" ] = electrode_name
    pd.DataFrame.to_csv(df, ofname, header=True, index=False)
    
    #All other atlases
    for a in range(len(atlases)):
        atlas_name =  np.array(atlases["atlas_filename"])[a]
        atlas_name_base = os.path.splitext(os.path.splitext(os.path.basename(atlas_name))[0])[0]
        print(atlas_name_base)
        ifname_atlas_path = ospj(ifpath_atlas_registration_sub_ID, f"sub-{sub_ID}_T00_{atlas_name}" )
        ifname_atlas_labels_path = ospj(ifpath_atlas_labels, f"{atlas_name_base}.csv" )
        if not os.path.exists(ifname_atlas_path): print(f"Path does not exist: {ifname_atlas_path}")
        if "RandomAtlas" in atlas_name_base: 
            print("Random Atlas")
            noLabels=True
        else:
            noLabels=False
        ofname = ospj(ofpath_localization_files, f"sub-{sub_ID}_{atlas_name_base}.csv")
        electrode_localization.by_region(ifname_electrode_localization_sub_ID, ifname_atlas_path, ifname_atlas_labels_path, ofname,description =atlas_name_base, noLabels=noLabels)

        #rename channels to standard 4 characters (2 letters, 2 numbers)
        df = pd.read_csv(ofname, sep=",", header=0)
        for e in range(len( df  )):
            electrode_name = df.iloc[e]["electrode_name"]
            if (len(electrode_name) == 3): electrode_name = f"{electrode_name[0:2]}0{electrode_name[2]}"
            df.at[e, "electrode_name" ] = electrode_name
        pd.DataFrame.to_csv(df, ofname, header=True, index=False)
        
    #Concatenate files into one
    data = pd.read_csv(ospj(ofpath_localization_files, f"sub-{sub_ID}_00_GM_WM_CSF.csv"), sep=",", header=0)
    data = pd.concat([data, pd.read_csv(ospj(ofpath_localization_files, f"sub-{sub_ID}_00_WM_distance.csv"), sep=",", header=0).iloc[:,4:] ]  , axis = 1)
    for a in range(len(atlases)):
        atlas_name =  np.array(atlases["atlas_filename"])[a]
        atlas_name_base = os.path.splitext(os.path.splitext(os.path.basename(atlas_name))[0])[0]
        print(atlas_name_base)
        ofname = ospj(ofpath_localization_files, f"sub-{sub_ID}_{atlas_name_base}.csv")
        if not os.path.exists(ofname): print(f"Path does not exist: {ofname}")
        data= pd.concat([data, pd.read_csv(ofname, sep=",", header=0).iloc[:,4:] ]  , axis = 1)
    pd.DataFrame.to_csv(data, ofname_electrode_localization_concatenated, header=True, index=False)
            




#%%












