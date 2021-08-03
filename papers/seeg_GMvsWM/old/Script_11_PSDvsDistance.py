"""
2020.08.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compute spectrogram and interpolate each segment to a length of 200

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

ifpath_eeg = ospj(path, "data/data_processed/eeg/montage/referential/filtered")
ofpath_psdVsDist = ospj(path, "data/data_processed/PSDvsDistance/montage/referential/filtered/original")

if not (os.path.isdir(ofpath_psdVsDist)): os.makedirs(ofpath_psdVsDist, exist_ok=True)
#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)
#%%
for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    sub_RID = "sub-{0}".format(sub_ID)
    iEEG_filename = data.iloc[i].file
    ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    print( "\n\n{0}: {1}".format(sub_ID,descriptor) )
    
    #Inputs and OUtputs
    #input filename EEG
    ifpath_eeg_sub_ID = ospj(ifpath_eeg, sub_RID)
    ifpath_electrode_localization_sub_ID =  ospj(ifpath_electrode_localization, sub_RID)
    ifname_EEG_filtered = ospj(ifpath_eeg_sub_ID, "sub-{0}_{1}_{2}_{3}_EEG_filtered.pickle.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))

    ifname_electrode_localization = ospj(ifpath_electrode_localization_sub_ID, "sub-{0}_electrode_localization.csv".format(sub_ID))
    ofpath_psdVsDist_sub_ID = ospj(ofpath_psdVsDist, sub_RID)
    ofname_psdVsDist_sub_ID = ospj(ofpath_psdVsDist_sub_ID, "sub-{0}_{1}_{2}_{3}_PSDvsDistance_filtered.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
    if not (os.path.isdir(ofpath_psdVsDist_sub_ID)): os.mkdir(ofpath_psdVsDist_sub_ID)
    
    
    with open(ifname_EEG_filtered, 'rb') as f: eeg, fs = pickle.load(f)
    electrode_localization = pd.read_csv(ifname_electrode_localization)
    
    
    data_array = np.array(eeg)
    columns = eeg.columns
    win = 4 * fs

    freq_shape = np.shape(signal.welch(data_array[:,1], fs, nperseg=win))[1]
    PSD = np.zeros(shape=(freq_shape,1))
    Distance = np.zeros(shape=(1,1))
    for e in range(len(columns)):
        electrode_name = columns[e]
        if (len(electrode_name) == 3): electrode_name = "{0}{1}{2}".format(electrode_name[0:2], 0, electrode_name[2])
        if any(np.array(electrode_localization["electrode_name"] ) == electrode_name):#if any analyzed signals are not in the electrode localization file, dont' compute.
            loc = np.where(np.array(electrode_localization["electrode_name"] )  == electrode_name)[0][0]
            dist = np.reshape(electrode_localization["distances_label_2"][loc], newshape=(1,1))
            region = electrode_localization["region_number"][loc]
            if region >=2: #only consider signals inside the brain
                freqs, Pxx = signal.welch(data_array[:,e], fs, nperseg=win)
                Pxx = np.reshape(Pxx, newshape=(freq_shape, 1))
                PSD = np.append(PSD, Pxx, axis=1)
                Distance = np.append(Distance, dist, axis=1)

    PSD = np.delete(PSD, [0], axis=1)
    Distance = np.delete(Distance, [0], axis=1)
    Distance = Distance.flatten()
    #interpolation
    maximum = np.max(Distance)
    interp = interp1d(Distance, PSD, kind='nearest', axis=1)
    Distance_interpolated = np.arange(start=0, stop=maximum, step = 0.01)
    PSD_interplation = interp(Distance_interpolated)
    I = pd.Index(freqs, name="frequency")
    C = pd.Index(Distance, name="distance")
    PSD_df = pd.DataFrame(PSD, index=I, columns=C)
    I = pd.Index(freqs, name="frequency")
    C = pd.Index(np.round(Distance_interpolated, 2), name="distance")
    PSD_interplation_df = pd.DataFrame(PSD_interplation, index=I, columns=C)
    
    plot = np.delete(np.log10(np.array(PSD_interplation_df)), range(50, np.shape(PSD_interplation)[0]), axis=0)
    plot = np.delete(plot, 88, axis=1)
    sns.heatmap(plot, cmap=sns.color_palette("Spectral_r", n_colors=20, desat=0.6))
    plt.show()
    print("saving file {0}\n\n".format(ofname_psdVsDist_sub_ID))
    with open(ofname_psdVsDist_sub_ID, 'wb') as f: pickle.dump([PSD_df, PSD_interplation_df], f)


