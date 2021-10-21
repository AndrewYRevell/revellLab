"""
2020.08.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compute SNR

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

interictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_394423190000_394512890000_EEG_filtered.pickle"
preictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_415933490000_416023190000_EEG_filtered.pickle"
ictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_416023190000_416112890000_EEG_filtered.pickle"
postictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_416112890000_416292890000_EEG_filtered.pickle"

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

ifpath_SNR = ospj(path, "data/data_processed/SNR/montage/referential/filtered/original")
ofpath_SNR_avg = ospj(path, "data/data_processed/SNR/montage/referential/filtered/averaged")
if not (os.path.isdir(ofpath_SNR_avg)): os.makedirs(ofpath_SNR_avg, exist_ok=True)
#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)

#%%
ofname_SNR_avg = ospj(ofpath_SNR_avg,"SNR_All_patients.pickle")



descriptors = ["interictal","preictal","ictal","postictal"]


#%%
count_ictal = 0
breaks = [0,3,6]
interpolation_length = 200
SNR_all_patients = np.zeros(shape=(interpolation_length, len(breaks)+1,4, len(sub_IDs_unique)))#4 = interictal, preictal, ictal, postictal

descriptors = ["interictal","preictal","ictal","postictal"]
for per in range(len(descriptors)):
    print("{0}\n\n".format(per))
    #PSDvsDistance = np.empty(shape = (201,1025, len(sub_IDs_unique)))#second number is longest distance
    #PSDvsDistance[:] = np.NaN
    count_patient = 0
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
            ifpath_SNR_sub_ID = ospj(ifpath_SNR, "sub-{0}".format(sub_ID))
            ifname_SNR_sub_ID = ospj(ifpath_SNR_sub_ID, "sub-{0}_{1}_{2}_{3}_SNR_filtered.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))

            with open(ifname_SNR_sub_ID, 'rb') as f: SNR, SNR_interpolation, SNR_avg_period, bins,bins_new, columns, dist_vec = pickle.load(f)
            dist_breaks = []
            for d in range(len(breaks)):
                if d == 0:
                    dist_breaks.append(np.where(dist_vec[:,0] == breaks[d]))
                if d > 0:
                    dist_breaks.append(np.where(np.logical_and(dist_vec[:,0]>breaks[d-1], dist_vec[:,0]<=breaks[d])))
            dist_breaks.append(np.where(dist_vec[:,0] > breaks[len(breaks)-1]))
            SNR_mean = np.zeros(shape=(len(bins), len(breaks)+1))
            for d in range(len(breaks)+1):
                if len(SNR[dist_breaks[d], :][0]) > 0:
                    SNR_mean[:,d]= np.nanmean(SNR[dist_breaks[d], :][0], axis=0)
                if len(SNR[dist_breaks[d], :][0]) == 0:
                    SNR_mean[:, d] = np.nan
            interp = interp1d(bins, SNR_mean, kind='linear', axis=0)
            bins_new = np.linspace(bins[0], bins[-1], num=interpolation_length, endpoint=True)
            SNR_mean_interplation = interp(bins_new)
            SNR_all_patients[:,:,count_ictal, count_patient] = SNR_mean_interplation
            count_patient = count_patient + 1
    count_ictal = count_ictal +1
            

          

SNR_all_patients_mean = np.nanmean(SNR_all_patients, axis=3)


ofname_SNR_avg = ospj(ofpath_SNR_avg, "SNR_all_patients_mean.pickle")
print("saving file {0}\n\n".format(ofname_SNR_avg))
with open(ofname_SNR_avg, 'wb') as f: pickle.dump([SNR_all_patients_mean, SNR_all_patients, bins_new, sub_IDs_unique, breaks], f)







tmp = SNR_all_patients[:,:,2,:]




