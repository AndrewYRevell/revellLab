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

ifpath_eeg = ospj(path, "data/data_processed/eeg/montage/referential/filtered")
ofpath_SNR= ospj(path, "data/data_processed/SNR/montage/referential/filtered/original")

if not (os.path.isdir(ofpath_SNR)): os.makedirs(ofpath_SNR, exist_ok=True)
#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)

#%%

descriptors = ["interictal","preictal","ictal","postictal"]
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
    ofpath_SNR_sub_ID = ospj(ofpath_SNR, sub_RID)
    ofname_SRN_sub_ID = ospj(ofpath_SNR_sub_ID, "sub-{0}_{1}_{2}_{3}_SNR_filtered.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
    if not (os.path.isdir(ofpath_SNR_sub_ID)): os.mkdir(ofpath_SNR_sub_ID)
    
    #Calculating SNR
    with open(ifname_EEG_filtered, 'rb') as f: eeg, fs = pickle.load(f)
    electrode_localization = pd.read_csv(ifname_electrode_localization)
    
    data_array = np.array(eeg)
    win = 4 * fs
    NFFT = int(fs*4)
    noverlap = NFFT/2
    columns = eeg.columns
    dist_vec = np.zeros(shape=(len(columns), 1))
    dist_vec[:] = np.NaN#ANY NaNs REPRESENT ELECTRODES EITHER (1) NOT IN GM OR WM TISSUE, OR NOT IN ELECTRODE LOCALIZATION FILES
    for e in range(len(columns)):
        electrode_name = columns[e]
        if (len(electrode_name) == 3): electrode_name = "{0}{1}{2}".format(electrode_name[0:2], 0, electrode_name[2])
        if any(np.array(electrode_localization["electrode_name"] ) == electrode_name):#if any analyzed signals are not in the electrode localization file, dont' compute.
            loc = np.where(np.array(electrode_localization["electrode_name"] )  == electrode_name)[0][0]
            dist = np.reshape(electrode_localization["distances_label_2"][loc], newshape=(1,1))
            region = electrode_localization["region_number"][loc]
            dist_vec[e,0] = dist
            #Freqs, Pxx = signal.welch(data_array[:, loc], fs, nperseg=win)
            #power = np.trapz(Pxx[range(2, 201)], dx=0.25)
            #sns.scatterplot(x = Freqs[range(2, 201)], y= Pxx[range(2, 201)])
            Pxx, Freqs, bins, im = plt.specgram(x=data_array[:, e], Fs=fs, NFFT=NFFT, scale_by_freq=True, cmap='rainbow', noverlap=noverlap); #plt.show()
            Pxx_mean = np.mean(Pxx, axis=1)
            power = np.trapz(Pxx_mean[range(2, np.where(Freqs==50)[0][0]+1  )], dx=Freqs[1]-Freqs[0])
            #sns.scatterplot(x=Freqs2[range(2, 201)], y=Pxx3[range(2, 201)])
            interpolation_length = 200
            if e == 0:
                #initialize
                SNR_time = np.zeros(shape=(len(columns),len(bins))); SNR_time[:] = np.NaN
                SNR = np.zeros(shape=(len(columns), len(bins))); SNR[:] = np.NaN
                SNR_interpolation = np.zeros(shape=(len(columns), interpolation_length)); SNR_interpolation[:] = np.NaN
                SNR_avg_period = np.zeros(shape=(len(columns),1)); SNR_avg_period[:] = np.NaN
                if descriptor == descriptors[0]:
                    power_interictal = np.zeros(shape=(len(columns), 1))
                    power_interictal[:] = np.NaN
            if region >=2: #only consider signals inside the brain
                if descriptor == descriptors[0]:
                    power_interictal[e, 0] = power
                SNR_time[e,:] = np.trapz(Pxx[range(2, np.where(Freqs==50)[0][0]+1  ),:], dx=Freqs[1]-Freqs[0], axis=0)/power_interictal[e,0]#/len(bins)
                interp = interp1d(bins, SNR_time, kind='nearest')
                bins_new = np.linspace(bins[0], bins[-1], num=interpolation_length, endpoint=True)
                SNR_time_interplation = interp(bins_new)
                SNR_avg_period[e,0] = power/power_interictal[e,0]
                #sns.scatterplot(x = bins2 , y = power3)
    #ANY NaNs REPRESENT ELECTRODES EITHER (1) NOT IN GM OR WM TISSUE, OR NOT IN ELECTRODE LOCALIZATION FILES
    SNR[:,:] = SNR_time #need to restructure data since ictal time != postictal time.
    SNR_interpolation[:,:] = SNR_time_interplation

    print("saving file {0}\n\n".format(ofname_SRN_sub_ID))
    with open(ofname_SRN_sub_ID, 'wb') as f: pickle.dump([SNR, SNR_interpolation, SNR_avg_period, bins,bins_new, columns, dist_vec], f)


    fig1 = sns.scatterplot(x = np.ndarray.flatten(dist_vec), y= np.ndarray.flatten(SNR_avg_period)  )
    fig1.set(xlabel='Dist from GM (mm)', ylabel='SNR', title = "{0} {1} SNR".format(sub_ID, descriptor))
    #plt.show()



















