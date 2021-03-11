"""
2020.08.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Averaging interpolated spectrogram of all patients

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
np.seterr(divide = 'ignore')



#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization")
ifpath_spectrogram = ospj(path, "data/data_processed/spectrogram/montage/referential/filtered/original")
ifpath_spectrogram_interpolated = ospj(path, "data/data_processed/spectrogram/montage/referential/filtered/interpolated")

ofpath_spectrogram_interpolated_avg = ospj(path, "data/data_processed/spectrogram/montage/referential/filtered/interpolated_avg")
ofpath_spectrogram_interpolated_avg_all = ospj(path, "data/data_processed/spectrogram/montage/referential/filtered/interpolated_all_patients_combined")


if not (os.path.isdir(ofpath_spectrogram_interpolated_avg)): os.makedirs(ofpath_spectrogram_interpolated_avg, exist_ok=True)
if not (os.path.isdir(ofpath_spectrogram_interpolated_avg_all)): os.makedirs(ofpath_spectrogram_interpolated_avg_all, exist_ok=True)


#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)
#%%
total_num_electrodes_GM = np.zeros(shape=[1, len(sub_IDs_unique)])
total_num_electrodes_WM = np.zeros(shape=[1, len(sub_IDs_unique)])
distance_considered_GM = 0
distance_considered_WM = 2
for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    print(sub_ID)
    iEEG_filename = data.iloc[i].file
    ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    
    #Inputs and OUtputs
    #input filename EEG
    ifpath_interpolated_sub_ID = os.path.join(ifpath_spectrogram_interpolated, "sub-{0}".format(sub_ID))
    ifpath_electrode_localization_sub_ID = os.path.join(ifpath_electrode_localization, "sub-{0}".format(sub_ID))
    
    ifname_interpolated = ospj(ifpath_interpolated_sub_ID, "sub-{0}_{1}_{2}_{3}_spectrogram_filtered_interpolated.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
    ifname_electrode_localization = ospj(ifpath_electrode_localization_sub_ID, "sub-{0}_electrode_localization.csv".format(sub_ID))
    
    ofpath_spectrogram_interpolated_avg_sub_ID = ospj(ofpath_spectrogram_interpolated_avg, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofpath_spectrogram_interpolated_avg_sub_ID)): os.mkdir(ofpath_spectrogram_interpolated_avg_sub_ID)
    
    ofname_spectrodram_interpolated_avg_sub_ID =  ospj(ofpath_spectrogram_interpolated_avg_sub_ID, "sub-{0}_{1}_{2}_{3}_spectrogram_filtered_interpolated_avg.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))

    #Calculating Averages of binary classification
    
    if (os.path.exists(ifname_electrode_localization)):
        electrode_localization = pd.read_csv(ifname_electrode_localization)
        print("reading file {0}".format(ifname_interpolated))
        with open(ifname_interpolated, 'rb') as f: spectrogram_interpolation, freqs, xnew, columns = pickle.load(f)
        
        np.logical_and( electrode_localization["distances_label_2"] > distance_considered_WM, electrode_localization["region_number"] >= 2 )
        num_GM = len(np.where(    np.logical_and( electrode_localization["distances_label_2"] <= distance_considered_GM, electrode_localization["region_number"] >= 2 )  )[0]   )  
        num_WM = len(np.where(    np.logical_and( electrode_localization["distances_label_2"] >  distance_considered_WM, electrode_localization["region_number"] >= 2 )  )[0]  )
        spectrogram_GM = np.zeros( shape=(spectrogram_interpolation.shape[0], spectrogram_interpolation.shape[1], num_GM))
        spectrogram_WM = np.zeros( shape=(spectrogram_interpolation.shape[0], spectrogram_interpolation.shape[1], num_WM))
        

        count_GM = 0
        count_WM = 0
        for e in range(len(columns)):
            electrode_name = columns[e]
            if (len(electrode_name) == 3): electrode_name = "{0}{1}{2}".format(electrode_name[0:2], 0, electrode_name[2])
            if any(np.array(electrode_localization["electrode_name"]) == electrode_name):#if any analyzed signals are not in the electrode localization file, dont' compute.
                loc = np.where(np.array(electrode_localization["electrode_name"]) == electrode_name)[0][0]
                distance = electrode_localization["distances_label_2"][loc]
                region_number =  electrode_localization["region_number"][loc] 
                if all([distance <= distance_considered_GM, region_number >= 2 ]):
                    spectrogram_GM[:, :, count_GM] = spectrogram_interpolation[:, :, e]
                    count_GM = count_GM + 1
                if all([distance > distance_considered_WM, region_number >= 2]):
                    spectrogram_WM[:, :, count_WM] = spectrogram_interpolation[:, :, e]
                    count_WM = count_WM + 1

        #some electrodes have locations, but are not recorded on iEEG.org, so need to delete the part of the arrays
        spectrogram_GM = np.delete(spectrogram_GM, range(count_GM, num_GM), axis=2)
        spectrogram_WM = np.delete(spectrogram_WM, range(count_WM, num_WM), axis=2)
        spectrogram_GM_mean = np.nanmean(spectrogram_GM, axis=2)
        spectrogram_WM_mean = np.nanmean(spectrogram_WM, axis=2)
        print("saving file {0}\n\n".format(ofname_spectrodram_interpolated_avg_sub_ID))
        with open(ofname_spectrodram_interpolated_avg_sub_ID, 'wb') as f: pickle.dump([spectrogram_GM_mean, spectrogram_WM_mean, freqs, xnew, columns], f)




np.sum(total_num_electrodes_GM)
np.sum(total_num_electrodes_WM)

#%%
#averaging by patient across each peri-ictal time
ofname_spectrogram_interpolated_avg_all = ospj(ofpath_spectrogram_interpolated_avg_all,"ALL_patient_spectrograms_averaged.pickle")

ALL_data_GM = np.zeros(shape=(129, 200, 4 ))
ALL_data_WM = np.zeros(shape=(129, 200, 4 ))

descriptors = ["interictal","preictal","ictal","postictal"]
for per in range(len(descriptors)):
    print("{0}\n\n".format(per))
    count_sub = 0
    spectrogram_ALL_GM = np.zeros(shape=(129, 200, len(sub_IDs_unique) ))
    spectrogram_ALL_WM = np.zeros(shape=(129, 200, len(sub_IDs_unique)))
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
            ofpath_spectrogram_interpolated_avg_sub_ID = ospj(ofpath_spectrogram_interpolated_avg, "sub-{0}".format(sub_ID))
            ofname_spectrodram_interpolated_avg_sub_ID =  ospj(ofpath_spectrogram_interpolated_avg_sub_ID, "sub-{0}_{1}_{2}_{3}_spectrogram_filtered_interpolated_avg.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))

            with open(ofname_spectrodram_interpolated_avg_sub_ID, 'rb') as f: spectrogram_GM_mean, spectrogram_WM_mean, freqs, xnew, columns = pickle.load(f)
            spectrogram_ALL_GM[:, :, count_sub] = np.log10(spectrogram_GM_mean)
            spectrogram_ALL_WM[:, :, count_sub] = np.log10(spectrogram_WM_mean)
            count_sub = count_sub + 1
            

           
    spectrogram_ALL_GM_mean = np.nanmean(spectrogram_ALL_GM, axis=2)
    spectrogram_ALL_WM_mean = np.nanmean(spectrogram_ALL_WM, axis=2)
    ALL_data_GM[:,:,per] = spectrogram_ALL_GM_mean
    ALL_data_WM[:,:,per] = spectrogram_ALL_WM_mean

print("saving file {0}\n\n".format(ofname_spectrogram_interpolated_avg_all))
with open(ofname_spectrogram_interpolated_avg_all, 'wb') as f: pickle.dump([ALL_data_GM, ALL_data_WM, freqs, xnew], f)

np.sum(total_num_electrodes_GM)
np.sum(total_num_electrodes_WM)
