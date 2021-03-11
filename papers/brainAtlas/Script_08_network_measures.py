"""
2020.01.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:
    python3.6 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import pandas as pd
import numpy as np
import network_measures

#%% Paths and File names

ifname_EEG_times = ospj(path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_tractography = ospj(path, "data/data_processed/tractography")
ifpath_atlases_standard = ospj( path, "data/data_raw/atlases/standard_atlases")
ifpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ifpath_SC = ospj( path, "data/data_processed/connectivity_matrices/structure/pass")
ofpath_network_measures = ospj( path, "data/data_processed/network_measures")


#%% Paramters

data = pd.read_excel(ifname_EEG_times)    
sub_ID_unique = np.unique(data.RID)


#%%


for i in range(len(sub_ID_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_ID_unique[i]
    print("\n\nSub-ID: {0}".format(sub_ID))
    atlas_names_standard = [f for f in sorted(os.listdir(ifpath_atlases_standard))]
    atlas_names_random = [f for f in sorted(os.listdir(ifpath_atlases_random))]
    ifname_base = "sub-{0}_ses-preop3T_dwi-eddyMotionB0Corrected".format(sub_ID) 
    ifpath_SC_sub_ID = ospj( ifpath_SC, "sub-{0}".format(sub_ID)) 
    #standard atlases: Calculating Connectivity per atlas
    for a in range(len(atlas_names_standard)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
        ifpath_SC_sub_ID_atlas = ospj( ifpath_SC_sub_ID, atlas_name) 

        #################
        #Getting Structure
        #################
        base_name = "sub-{0}_ses-preop3T_dwi-eddyMotionB0Corrected.{1}.count.pass.connectivity.mat".format( sub_ID, atlas_name )
        ifname_SC =  ospj( ifpath_SC_sub_ID_atlas, base_name) 
        
        #initializing
        if i == 0 and a == 0:
            colNames_standard =  ["RID", "atlas"]
            network_measures_names_standard = list(network_measures.get_network_measures(ifname_SC).columns)
            colNames_standard.extend(network_measures_names_standard)
            network_measures_standard = pd.DataFrame( columns=colNames_standard )
            print("initializing standard atlas network measures dataframe")
        #if Structural Connectivity file exists, compute network measures
        if os.path.exists(ifname_SC):
            data = [sub_ID, atlas_name]
            data.extend(np.array(network_measures.get_network_measures(ifname_SC))[0] )
            df = pd.DataFrame([data], columns = colNames_standard)
            network_measures_standard = pd.concat([network_measures_standard, df])
            print("Network Measures: {0}".format(base_name))
        else:
            print("File does not exist: {0}".format(ifname_SC))
        
    #random atlases: Calculating Connectivity per atlas
    for a in range(len(atlas_names_random)):
        versions = [f for f in sorted(os.listdir(ospj(ifpath_SC_sub_ID, atlas_names_random[a])))]  
        extension = ".mat" #get only .mat files
        versions = [s for s in versions if extension.lower() in s.lower()]
        for p in range(len(versions)):
            atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
            ifpath_SC_sub_ID_atlas = ospj( ifpath_SC_sub_ID, atlas_name) 
            #################
            #Getting Structure
            #################  
            base_name = versions[p]
            ifname_SC =  ospj( ifpath_SC_sub_ID_atlas, base_name) 
            #initializing
            if i == 0 and a == 0 and p == 0:
                colNames_random =  ["RID", "atlas", "version"]
                network_measures_names_random = list(network_measures.get_network_measures(ifname_SC).columns)
                colNames_random.extend(network_measures_names_random)
                network_measures_random = pd.DataFrame( columns=colNames_random )
                print("initializing random atlas network measures dataframe")
            #if Structural Connectivity file exists, compute network measures
            if os.path.exists(ifname_SC):
                data = [sub_ID, atlas_name, '{:04}'.format(p+1) ]
                data.extend(np.array(network_measures.get_network_measures(ifname_SC))[0] )
                df = pd.DataFrame([data], columns = colNames_random)
                network_measures_random = pd.concat([network_measures_random, df])
                print("Network Measures: {0}".format(base_name))
            else: 
                print("File does not exist: {0}".format(ifname_SC))
    
ofname_network_measures_standard = ospj( ofpath_network_measures, "network_measures_standard_atlas.csv")
ofname_network_measures_random  = ospj( ofpath_network_measures, "network_measures_random_atlas.csv")
pd.DataFrame.to_csv(network_measures_standard, ofname_network_measures_standard, header=True, index=False)                     
pd.DataFrame.to_csv(network_measures_random, ofname_network_measures_random, header=True, index=False)                     
