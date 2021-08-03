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


path = "/media/arevell/sharedSSD/linux/papers/paper005" 
#path = "/mnt"
import sys
import os
import pandas as pd
import numpy as np
from os.path import join as ospj
import seaborn as sns
import matplotlib.pyplot as plt
import time
import multiprocessing
from itertools import repeat
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
import network_measures

#% Input/Output Paths and File names
fname_diffusion_imaging = ospj( path, "data/data_raw/iEEG_times/diffusion_imaging.csv")
fname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
fpath_connectivity = ospj( path, "data/data_processed/connectivity_matrices/structure")
fpath_networkMeasures = ospj( path, "data/data_processed/network_measures/structure")
fname_network_measures = ospj( fpath_networkMeasures, "network_measures_structure.csv")

if not (os.path.isdir(fpath_networkMeasures)): os.makedirs(fpath_networkMeasures, exist_ok=True)

#% Load Study Meta Data
data = pd.read_csv(fname_diffusion_imaging) 
atlases = pd.read_csv(fname_atlases_csv)     
#% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]




#%%


for i in range(len(sub_IDs_unique)):
    t0_subid = time.time()
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print(f"\n\nSub-ID: {sub_ID}")

    fname_base = f"sub-{sub_ID}.sub-{sub_ID}_preop3T"
    fpath_SC_sub_ID = ospj( fpath_connectivity, f"sub-{sub_ID}".format(sub_ID)) 
    #standard atlases: Calculating Connectivity per atlas
    for a in range(len(atlases)):
        atlas_fname = os.path.splitext(os.path.splitext(  atlases.iloc[a]["atlas_filename"] )[0])[0]
        atlas_name = os.path.splitext(os.path.splitext(  atlases.iloc[a]["atlas_name"] )[0])[0]

        print(f"{sub_ID}; {atlas_name}")
        #################
        #Getting Structure
        #################
        t0 = time.time()
        fname_SC =  ospj( fpath_SC_sub_ID, f"{fname_base}_{atlas_fname}.count.pass.connectogram.txt") 
        os.path.exists(fname_SC)
        fname_connectivity = fname_SC
        #initializing
        if i == 0 and a == 0:
            colNames =  ["RID", "atlas"]
            network_measures_names = list(network_measures.get_network_measures(fname_SC).columns)
            colNames.extend(network_measures_names)
            network_measures_atlases = pd.DataFrame( columns=colNames )
            print("initializing atlas network measures dataframe")
        #if Structural Connectivity file exists, compute network measures
        if os.path.exists(fname_SC):
            if os.path.exists(fname_network_measures): #if file exists, pick up where left off
                last_network = pd.read_csv(fname_network_measures)
                all_subID  = np.array(last_network["RID"])
                if any(sub_ID == all_subID): #if any subIDs were done, find what atlases have already been computed
                    all_atlas = np.array(last_network[last_network["RID"] == sub_ID]["atlas"])
                    if any(atlas_fname==all_atlas): 1+1 #do nothing
                    else:
                        data = [sub_ID, atlas_fname]
                        data.extend(np.array(   network_measures.get_network_measures(fname_SC)     )[0] )
                        df = pd.DataFrame([data], columns = colNames)
                        network_measures_atlases = pd.concat([network_measures_atlases, df])
                        pd.DataFrame.to_csv(network_measures_atlases, fname_network_measures, header=True, index=False)      
                else:
                    data = [sub_ID, atlas_fname]
                    data.extend(np.array(   network_measures.get_network_measures(fname_SC)     )[0] )
                    df = pd.DataFrame([data], columns = colNames)
                    network_measures_atlases = pd.concat([network_measures_atlases, df])
                    pd.DataFrame.to_csv(network_measures_atlases, fname_network_measures, header=True, index=False)      
            else:
                data = [sub_ID, atlas_fname]
                data.extend(np.array(   network_measures.get_network_measures(fname_SC)     )[0] )
                df = pd.DataFrame([data], columns = colNames)
                network_measures_atlases = pd.concat([network_measures_atlases, df])
                pd.DataFrame.to_csv(network_measures_atlases, fname_network_measures, header=True, index=False)                     
        else:
            print("File does not exist: {0}".format(fname_SC))
        t1 = time.time(); td = np.round((t1-t0)/60,2); tr = np.round(  td*(len(atlases)-a-1) ,2); print(f"\ttime: {td} min; remain: {tr} min")
    t1 = time.time(); td = np.round((t1-t0_subid)/60, 2); tr = np.round(  td*(len(sub_IDs_unique)-a-1) ,2); print(f"\n{sub_ID}; time: {td} min; remain: {tr} min\n")
    

           


#%%
y = np.array(network_measures_atlases["Density"])[np.array(range(39,108,5))]

sns.scatterplot(   )


