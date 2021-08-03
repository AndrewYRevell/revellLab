"""
2020.08.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Averaging PSD vs Distance arrays of all patients

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

~~~~~~~
"""

path = "/media/arevell/sharedSSD1/linux/papers/paper005" #Parent directory of project
import pickle
import numpy as np
import os
import sys
from os.path import join as ospj
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
np.seterr(divide = 'ignore')

#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization")

ifpath_psdVsDist = ospj(path, "data/data_processed/PSDvsDistance/montage/referential/filtered/original")
ofpath_psdVsDist_avg = ospj(path, "data/data_processed/PSDvsDistance/montage/referential/filtered/averaged")
if not (os.path.isdir(ofpath_psdVsDist_avg)): os.makedirs(ofpath_psdVsDist_avg, exist_ok=True)
#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)

#%%
ofname_psdVsDist_avg = ospj(ofpath_psdVsDist_avg,"PSDvsDistance_All_patients.pickle")

PSDvsDistance_avg = np.zeros(shape = (201,1025, 4))

descriptors = ["interictal","preictal","ictal","postictal"]





descriptors = ["interictal","preictal","ictal","postictal"]
for per in range(len(descriptors)):
    print("{0}\n\n".format(per))
    count_sub = 0
    PSDvsDistance = np.empty(shape = (201,1025, len(sub_IDs_unique)))#second number is longest distance
    PSDvsDistance[:] = np.NaN
    for i in range(len(data)):
        #parsing data DataFrame to get iEEG information
        sub_ID = data.iloc[i].RID
        iEEG_filename = data.iloc[i].file
        ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
        start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
        stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
        descriptor = data.iloc[i].descriptor
        
        if (descriptor == descriptors[per]):
    
            print("{0}: {1}".format(sub_ID, descriptor)   )
            ifpath_psdVsDist_sub_ID = ospj(ifpath_psdVsDist, "sub-{0}".format(sub_ID))
            ifname_psdVsDist_sub_ID = ospj(ifpath_psdVsDist_sub_ID, "sub-{0}_{1}_{2}_{3}_PSDvsDistance_filtered.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))

            with open(ifname_psdVsDist_sub_ID, 'rb') as f: PSD_df, PSD_interplation_df = pickle.load(f)
            index = np.where(PSD_interplation_df.index == 50)[0][0]
            len_index = len(PSD_interplation_df.index)
            len_col = len(PSD_interplation_df.columns)
            print("{0}".format(len_index))
            print("{0}\n\n".format(len_col))
            if len_index > 50:
                end = len_index
                PSD_interplation_df_new = PSD_interplation_df.drop(PSD_interplation_df.index[range(index+1, end)])
            PSDvsDistance[:,np.array(range(0,len_col )),count_sub] = PSD_interplation_df_new
            count_sub = count_sub + 1
            

          
    PSDvsDistance_avg[:,:,per] = np.nanmean(PSDvsDistance, axis=2)


print("saving file {0}\n\n".format(ofname_psdVsDist_avg))
with open(ofname_psdVsDist_avg, 'wb') as f: pickle.dump(PSDvsDistance_avg, f)











