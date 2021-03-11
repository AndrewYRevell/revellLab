"""
2020.05.12
Andy Revell and Alex Silva
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
    python3.6 Script_08_SFC.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
import pandas as pd
import numpy as np
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import structure_function_correlation

#%% Paths and File names

ifname_EEG_times = ospj(path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_atlases_standard = ospj( path, "data/data_raw/atlases/standard_atlases")
ifpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ifpath_SC = ospj( path, "data/data_processed/connectivity_matrices/structure/pass")
ifpath_FC = ospj( path, "data/data_processed/connectivity_matrices/function")     
ifpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization_atlas_region")        

ofpath_SFC = ospj( path, "data/data_processed/structure_function_correlation") 
#%%Load Data
data = pd.read_excel(ifname_EEG_times)    
sub_ID_unique = np.unique(data.RID)


#%%
#Get atlas names

atlas_names_standard = [f for f in sorted(os.listdir(ifpath_atlases_standard))]
atlas_names_random = [f for f in sorted(os.listdir(ifpath_atlases_random))]

#%%
for i in range(len(data)):
    #parsing data DataFrame to get information
    sub_ID = data.iloc[i].RID
    #Creating input and output names
    ifpath_SC_sub_ID = ospj( ifpath_SC, "sub-{0}".format(sub_ID)) 
    ifpath_FC_sub_ID = ospj( ifpath_FC, "sub-{0}".format(sub_ID)) 
    ifpath_electrode_localization_sub_ID = ospj( ifpath_electrode_localization, "sub-{0}".format(sub_ID)) 
    ofpath_SFC_sub_ID = ospj( ofpath_SFC, "sub-{0}".format(sub_ID)) 
    if not (os.path.isdir(ofpath_SFC_sub_ID)): os.mkdir(ofpath_SFC_sub_ID)
    #################
    #Getting Function
    #################
    iEEG_filename = data.iloc[i].file
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    ifname_FC = "{0}/sub-{1}_{2}_{3}_{4}_functionalConnectivity.pickle".format(ifpath_FC_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    if not (os.path.exists(ifname_FC)):#check if file exists
        print("File does not exist: {0}".format(ifname_FC))
    else:
        #Standard Atlases
        for a in range(len(atlas_names_standard)):
            atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
            ifpath_SC_sub_ID_atlas = ospj( ifpath_SC_sub_ID, atlas_name) 
            ifpath_electrode_localization_atlas = ospj( ifpath_electrode_localization_sub_ID, atlas_name) 
            ofpath_SFC_sub_ID_atlas = ospj( ofpath_SFC_sub_ID,  atlas_name ) 
            if not (os.path.isdir(ofpath_SFC_sub_ID_atlas)): os.mkdir(ofpath_SFC_sub_ID_atlas)
            #################
            #Getting Structure
            #################
            base_name = "sub-{0}_ses-preop3T_dwi-eddyMotionB0Corrected.{1}.count.pass.connectivity.mat".format( sub_ID, atlas_name )
            base_name2 = "sub-{0}_ses-preop3T_dwi-eddyMotionB0Corrected.{1}.count.pass.connectogram.txt".format( sub_ID, atlas_name )
            ifname_SC =  ospj( ifpath_SC_sub_ID_atlas, base_name) 
            ifname_SCtxt = ospj( ifpath_SC_sub_ID_atlas, base_name2) 
            if not (os.path.exists(ifname_SC)):#check if file exists
                print("File does not exist: {0}".format(ifname_SC))
            else:
                #################
                #Getting Electrode Localization
                #################
                ifname_electrode_localization = ospj( ifpath_electrode_localization_atlas, "sub-{0}_electrode_coordinates_mni_{1}.csv".format(sub_ID,atlas_name )) 
                if not (os.path.exists(ifname_electrode_localization)):#check if file exists
                    print("File does not exist: {0}".format(ifname_electrode_localization))
                else:
                    #Output filename 
                    ofname_SFC = "{0}/sub-{1}_{2}_{3}_{4}_{5}_SFC.pickle".format(ofpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, start_time_usec, stop_time_usec,atlas_name )
                    if not (os.path.exists(ofname_SFC)):#check if file exists
                        print("Calculating SFC: {0}".format(ofname_SFC))
                        structure_function_correlation.SFC(ifname_SC,ifname_SCtxt, ifname_FC,ifname_electrode_localization, ofname_SFC)
                    else:
                        print("File exists: {0}".format(ofname_SFC))
        #Random Atlases
        for a in range(len(atlas_names_random)):
            #Creating input and output names
            atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
            ifpath_SC_sub_ID_atlas = ospj( ifpath_SC_sub_ID, atlas_name) 
            ifpath_electrode_localization_atlas = ospj( ifpath_electrode_localization_sub_ID, atlas_name) 
            ofpath_SFC_sub_ID_atlas = ospj( ofpath_SFC_sub_ID,  atlas_name ) 
            if not (os.path.isdir(ofpath_SFC_sub_ID_atlas)): os.mkdir(ofpath_SFC_sub_ID_atlas)
            versions = [f for f in sorted(os.listdir(ospj(ifpath_SC_sub_ID, atlas_names_random[a])))]  
            extension = ".mat" #get only .mat files
            versions = [s for s in versions if extension.lower() in s.lower()]

            for p in range(len(versions)):
                #################
                #Getting Structure
                #################
                base_name = versions[p]
                ifname_SC =  ospj( ifpath_SC_sub_ID_atlas, base_name) 
                if not (os.path.exists(ifname_SC)):#check if file exists
                    print("File does not exist: {0}".format(ifname_SC))
                else:
                    #################
                    #Getting Electrode Localization
                    #################
                    ifname_electrode_localization = ospj( ifpath_electrode_localization_atlas, "sub-{0}_electrode_coordinates_mni_{1}_v{2}.csv".format(sub_ID,atlas_name , '{:04}'.format(p+1))) 
                    if not (os.path.exists(ifname_electrode_localization)):#check if file exists
                        print("File does not exist: {0}".format(ifname_electrode_localization))
                    else:
                        ofname_SFC = "{0}/sub-{1}_{2}_{3}_{4}_{5}_v{6}_SFC.pickle".format(ofpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, start_time_usec, stop_time_usec,atlas_name , '{:04}'.format(p+1))
                        if not (os.path.exists(ofname_SFC)):#check if file exists
                            print("Calculating SFC: {0}".format(ofname_SFC))
                            structure_function_correlation.SFC(ifname_SC,ifname_FC,ifname_electrode_localization, ofname_SFC)
                        else:
                            print("File exists: {0}".format(ofname_SFC))
        
            
            
            
    