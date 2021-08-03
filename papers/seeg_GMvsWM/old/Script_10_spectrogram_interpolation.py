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
np.seterr(divide = 'ignore')



#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_filtered_eeg = ospj(path, "data/data_processed/eeg/montage/referential/filtered")
ofpath_spectrogram = ospj(path, "data/data_processed/spectrogram/montage/referential/filtered/original")
ofpath_spectrogram_interpolated = ospj(path, "data/data_processed/spectrogram/montage/referential/filtered/interpolated")

if not (os.path.isdir(ofpath_spectrogram)): os.makedirs(ofpath_spectrogram, exist_ok=True)
if not (os.path.isdir(ofpath_spectrogram_interpolated)): os.makedirs(ofpath_spectrogram_interpolated, exist_ok=True)


#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)



#%%

for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    iEEG_filename = data.iloc[i].file
    ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    #input filename EEG
    ifpath_sub_ID_eeg = os.path.join(ifpath_filtered_eeg, "sub-{0}".format(sub_ID))
    
    


    ofpath_spectrogram_sub_ID = os.path.join(ofpath_spectrogram, "sub-{0}".format(sub_ID))
    ofpath_interpolated_sub_ID = os.path.join(ofpath_spectrogram_interpolated, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofpath_spectrogram_sub_ID)): os.mkdir(ofpath_spectrogram_sub_ID)
    if not (os.path.isdir(ofpath_interpolated_sub_ID)): os.mkdir(ofpath_interpolated_sub_ID)

    ifname_EEG_filtered = ospj(ifpath_sub_ID_eeg, "sub-{0}_{1}_{2}_{3}_EEG_filtered.pickle.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
    ofname_spectrogram = ospj(ofpath_spectrogram_sub_ID, "sub-{0}_{1}_{2}_{3}_spectrogram_filtered.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
    ofname_interpolated = ospj(ofpath_interpolated_sub_ID, "sub-{0}_{1}_{2}_{3}_spectrogram_filtered_interpolated.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
    
    
    if not (os.path.exists(ofname_interpolated)):
        print("reading file {0}".format(ifname_EEG_filtered))
        with open(ifname_EEG_filtered, 'rb') as f: eeg, fs = pickle.load(f)
        data_array = np.array(eeg)
        NFFT = int(256)
        noverlap = 128
        spec_shape = plt.specgram(x=data_array[:,0], Fs=fs, NFFT=NFFT, scale_by_freq=True, noverlap=noverlap)[0].shape
        interpolation_length = 200
        spectrogram = np.zeros(shape = (spec_shape[0], spec_shape[1], data_array.shape[1]) )
        spectrogram_interpolation = np.zeros(shape=(spec_shape[0], interpolation_length, data_array.shape[1]))
        for i in range(np.shape(data_array)[1]):
            x = data_array[:,i]
            Pxx, freqs, bins, im = plt.specgram(x=x, Fs=fs, NFFT=NFFT, scale_by_freq=True, cmap='rainbow', noverlap=noverlap)
            spectrogram[:,:,i] = Pxx
            #interpolation
            interp = interp1d(bins, Pxx, kind='nearest')
            xnew = np.linspace(bins[0], bins[-1], num=interpolation_length, endpoint=True)
            Pxx_interplation = interp(xnew)
            spectrogram_interpolation[:, :, i] = Pxx_interplation
        print("saving file {0}".format(ofname_spectrogram))
        with open(ofname_spectrogram, 'wb') as f: pickle.dump([spectrogram, freqs, bins, eeg.columns[:]], f)
        print("saving file {0}\n\n".format(ofname_interpolated))
        with open(ofname_interpolated, 'wb') as f: pickle.dump([spectrogram_interpolation, freqs, xnew, eeg.columns[:]], f)
