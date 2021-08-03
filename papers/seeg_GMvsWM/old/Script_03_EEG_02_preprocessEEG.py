"""
2020.08.01
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
path = "/media/arevell/sharedSSD/linux/papers/paper005" 
import sys
import os
from os.path import join as ospj
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import multiprocessing
import pickle
import json, codecs
import copy

#import custom
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
sys.path.append(ospj(path, "seeg_GMvsWM", "code" ,"tools", "ieegpy"))
from preprocessEEG import filter_eeg_data
from imagingToolsRevell import printProgressBar, show_eeg, plot_adj, plot_adj_allbands
import echobase

#%

fname_EEG_times = ospj(path,"data/data_raw/iEEG_times/EEG_times.xlsx")
fpath_EEG = ospj(path,"data/data_raw/EEG")
fpath_electrodeLoc = ospj(path,"data/data_processed/electrode_localization")
fpath_electrodesRemoved = ospj(path,"data/data_processed/EEG/electrodesRemoved/")
fpath_filtered_eeg = ospj(path,"data/data_processed/EEG/filtered/")
fpath_video = ospj(path,"videos/adjacency")
fpath_video = ospj(path,"videos")
fpath_connectivity = ospj(path,"data/data_processed/connectivity_matrices/function")

if not (os.path.isdir(fpath_filtered_eeg)): os.makedirs(fpath_filtered_eeg, exist_ok=True)
if not (os.path.isdir(fpath_electrodesRemoved)): os.makedirs(fpath_electrodesRemoved, exist_ok=True)
if not (os.path.isdir(fpath_video)): os.makedirs(fpath_video, exist_ok=True)

#% Load Study Meta Data
eegTimes = pd.read_excel(fname_EEG_times)    


#%%Remove electrodes outside brain

for i in range(len(eegTimes)):
    #parsing data DataFrame to get iEEG information
    sub_ID = eegTimes.iloc[i].RID
    iEEG_filename = eegTimes.iloc[i].file
    ignore_electrodes = eegTimes.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(eegTimes.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(eegTimes.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = eegTimes.iloc[i].descriptor
    print(f"{sub_ID}; {descriptor}")
    #input filename EEG
    fpath_EEG_subID = ospj(fpath_EEG, f"sub-{sub_ID}")
    fname_EEG_subID = ospj(fpath_EEG_subID, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEG.csv")
    fname_EEG_subID_MD = ospj(fpath_EEG_subID, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEG_metadata.csv")
    #input electrode localization
    fname_electrodeLoc = ospj(fpath_electrodeLoc, f"sub-{sub_ID}", f"sub-{sub_ID}_electrode_localization.csv")
    #Output electrodes removed EEG
    fpath_EEG_subID_removed = ospj(fpath_electrodesRemoved, f"sub-{sub_ID}")
    fname_EEG_subID_removed = ospj(fpath_EEG_subID_removed, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEGremoved.csv")
    
    if not all([os.path.exists(fname_EEG_subID_removed), True]):
        #read files
        eeg = pd.read_csv(fname_EEG_subID)
        elecLoc = pd.read_csv(fname_electrodeLoc)
        eeg_metadata = pd.read_csv(fname_EEG_subID_MD)
        fs = int(eeg_metadata["fs"][0])
        
        #remove eeg electrodes not in localization file
        name_loc = np.array(elecLoc["electrode_name"])
        name_eeg = np.array(eeg.columns) 
        
        name_intersect = np.intersect1d(name_loc, name_eeg, return_indices = True )
        eeg_ordered = eeg.iloc[:, name_intersect[2] ]
        name_eegOrdered = np.array(eeg_ordered.columns) 
    
        #remove eeg electrodes outside the brain
        ind_outside = np.where(np.array(elecLoc["Tissue_segmentation_region_number"]) == 0)
        name_outside = name_loc[ind_outside]
        
        name_outside_intersect = np.intersect1d(name_outside, name_eegOrdered, return_indices = True )
        if len(name_outside_intersect[2]) > 0:
            eeg_ordered_dropOutside = eeg_ordered.drop(eeg_ordered.columns[name_outside_intersect[2]], axis=1)
        else:
            eeg_ordered_dropOutside = eeg_ordered
        #fname_EEG_subID_MD = ospj(fpath_EEG_subID, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEGremoved_metadata.csv")
        if not (os.path.isdir(fpath_EEG_subID_removed)): os.mkdir(fpath_EEG_subID_removed)#if the path doesn't exists, then make the directory
        pd.DataFrame.to_csv(eeg_ordered_dropOutside, fname_EEG_subID_removed, header=True, index=False)
        #pd.DataFrame.to_csv(eeg_metadata, fname_EEG_subID_MD, header=True, index=False)
        
        data = np.array(eeg_ordered_dropOutside)
        #calculate parameters
        n_samp, n_chan = data.shape
        
        metadata ={"sub-ID": sub_ID, "iEEG_filename": iEEG_filename, 
               "start_time_usec": start_time_usec, "stop_time_usec":stop_time_usec,
               "descriptor":descriptor,
               "fs": fs, "n_samples": n_samp, 
               "n_channels": n_chan, 
               "channels": np.array(eeg_ordered_dropOutside.columns).tolist(), "channels_ignore": ignore_electrodes, 
               "channels_outside": name_outside.tolist()}   
        fname_EEG_subID_MD = ospj(fpath_EEG_subID_removed, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEGremoved.json")
        with open(fname_EEG_subID_MD, 'w', encoding='utf-8') as f: json.dump(metadata, f, ensure_ascii=False, indent=4)

#%%
for i in range(0,4): #range(len(eegTimes)):
    #parsing data DataFrame to get iEEG information
    sub_ID = eegTimes.iloc[i].RID
    iEEG_filename = eegTimes.iloc[i].file
    ignore_electrodes = eegTimes.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(eegTimes.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(eegTimes.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = eegTimes.iloc[i].descriptor
    print(f"\n{sub_ID}; {descriptor}")
    #input filename EEG
    fpath_EEG_subID = ospj(fpath_electrodesRemoved, f"sub-{sub_ID}")
    fname_EEG_subID = ospj(fpath_EEG_subID, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEGremoved.csv")
    fname_EEG_subID_MD = ospj(fpath_EEG_subID, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_EEGremoved.json")
    
    #Output files
    fpath_FC = ospj(fpath_connectivity,f"sub-{sub_ID}")
    if not (os.path.isdir(fpath_FC)): os.mkdir(fpath_FC)
    ref = "CAR" 
    fpath_FC_filt = ospj(fpath_FC, ref)
    if not (os.path.isdir(fpath_FC_filt)): os.mkdir(fpath_FC_filt)
    fname_xcorr = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_crossCorrelation.pickle")
    fname_pearson = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_pearson.pickle")
    fname_pearsonPval = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_pearsonPval.pickle")
    fname_spearman = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_spearman.pickle")
    fname_spearmanPval = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_spearmanPval.pickle")
    fname_coherence = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_coherence.pickle")
    fname_mi = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_mutualInformation.pickle")
    
    #read files
    eeg = pd.read_csv(fname_EEG_subID)
    eeg_metadata = json.loads(codecs.open(fname_EEG_subID_MD, 'r', encoding='utf-8').read())
    fs = int(eeg_metadata["fs"])
    data = np.array(eeg)
    
    #calculate parameters
    n_samp, n_chan = data.shape
    mw = 1 #Moving Window; in seconds
    ws = 2 #Window Size; in seconds
    times = np.floor(np.arange(0, n_samp, fs * mw))
    times = times[np.flatnonzero( n_samp - times > fs * ws)].astype(int) #only consider times where can fit into window size
    t = len(times)

    #metadata
    metadata_names = ["Broadband", "Delta", "Theta", "Alpha", "Beta", "gammaLow", "gammaMid", "gammaHigh"]
    metadata ={"sub-ID": sub_ID, "iEEG_filename": iEEG_filename, 
               "start_time_usec": start_time_usec, "stop_time_usec":stop_time_usec,
               "descriptor":descriptor, "reference": ref,
               "frequencies": metadata_names, "fs": fs, "n_samples": n_samp, 
               "n_channels": n_chan, "moving_window_seconds": mw,
               "window_size_seconds": ws, "window_num": t, "channels": np.array(eeg.columns).tolist(), 
               "channels_ignore": ignore_electrodes, "channels_outside": eeg_metadata["channels_outside"]}   
    fname_metadata = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_metadata.json")
    with open(fname_metadata, 'w', encoding='utf-8') as f: json.dump(metadata, f, ensure_ascii=False, indent=4)

    #calculate FC measures
    adj_xcorr = np.zeros(shape = (8, n_chan, n_chan , t))
    adj_pearson = np.zeros(shape = (8, n_chan, n_chan , t))
    adj_pearson_pval = np.zeros(shape = (8, n_chan, n_chan , t))
    adj_spearman = np.zeros(shape = (8, n_chan, n_chan , t))
    adj_spearman_pval = np.zeros(shape = (8, n_chan, n_chan , t))
    adj_coherence = np.zeros(shape = (8, n_chan, n_chan , t))
    adj_mi = np.zeros(shape = (8, n_chan, n_chan , t))
    
    
    
    if not (os.path.exists(fname_xcorr)):
        for w in times:
            t0 = time.time()
            start = w
            stop = w + fs * ws
            count = np.where(times == w)[0][0]
            data_win = data[start:stop,:]
            adj_xcorr[0, :,:,count], adj_xcorr[1, :,:,count], adj_xcorr[2, :,:,count], adj_xcorr[3, :,:,count], adj_xcorr[4, :,:,count], adj_xcorr[5, :,:,count], adj_xcorr[6, :,:,count], adj_xcorr[7, :,:,count] = echobase.crossCorrelation_wrapper(data_win, fs, avgref=True)
            t1= time.time(); td = t1-t0; tr = td*(t-count)/60; 
            printProgressBar(w, times[-1], prefix = "Progress:", suffix = f"{stop}/{n_samp}; {count+1}/{t}; Time: {np.round(tr,2)} min" )
            print(f"\n{sub_ID}; {descriptor}")
        with open(fname_xcorr, 'wb') as f: pickle.dump(adj_xcorr, f)
    else: print(f"file exists: {fname_xcorr}")
        
    if not (os.path.exists(fname_pearson)):
        for w in times:
            t0 = time.time()
            start = w
            stop = w + fs * ws
            count = np.where(times == w)[0][0]
            data_win = data[start:stop,:]
            [adj_pearson[0, :,:,count], adj_pearson[1, :,:,count], adj_pearson[2, :,:,count], adj_pearson[3, :,:,count], adj_pearson[4, :,:,count], adj_pearson[5, :,:,count], adj_pearson[6, :,:,count], adj_pearson[7, :,:,count]], [adj_pearson_pval[0, :,:,count], adj_pearson_pval[1, :,:,count], adj_pearson_pval[2, :,:,count], adj_pearson_pval[3, :,:,count], adj_pearson_pval[4, :,:,count], adj_pearson_pval[5, :,:,count], adj_pearson_pval[6, :,:,count], adj_pearson_pval[7, :,:,count]]  = echobase.pearson_wrapper(data_win, fs, avgref=True)
            t1= time.time(); td = t1-t0; tr = td*(t-count)/60; 
            printProgressBar(w, times[-1], prefix = "Progress:", suffix = f"{stop}/{n_samp}; {count+1}/{t}; Time: {np.round(tr,2)} min" )
            print(f"\n{sub_ID}; {descriptor}")
        with open(fname_pearson, 'wb') as f: pickle.dump(adj_pearson, f)
        with open(fname_pearsonPval, 'wb') as f: pickle.dump(adj_pearson_pval, f)
    else: print(f"file exists: {fname_pearson}")
        
    if not (os.path.exists(fname_spearman)):
        for w in times:
            t0 = time.time()
            start = w
            stop = w + fs * ws
            count = np.where(times == w)[0][0]
            data_win = data[start:stop,:]
            [adj_spearman[0, :,:,count], adj_spearman[1, :,:,count], adj_spearman[2, :,:,count], adj_spearman[3, :,:,count], adj_spearman[4, :,:,count], adj_spearman[5, :,:,count], adj_spearman[6, :,:,count], adj_spearman[7, :,:,count]], [adj_spearman_pval[0, :,:,count], adj_spearman_pval[1, :,:,count], adj_spearman_pval[2, :,:,count], adj_spearman_pval[3, :,:,count], adj_spearman_pval[4, :,:,count], adj_spearman_pval[5, :,:,count], adj_spearman_pval[6, :,:,count], adj_spearman_pval[7, :,:,count]]  = echobase.spearman_wrapper(data_win, fs, avgref=True)
            t1= time.time(); td = t1-t0; tr = td*(t-count)/60; 
            printProgressBar(w, times[-1], prefix = "Progress:", suffix = f"{stop}/{n_samp}; {count+1}/{t}; Time: {np.round(tr,2)} min" )
            print(f"\n{sub_ID}; {descriptor}")
        with open(fname_spearman, 'wb') as f: pickle.dump(adj_spearman, f)
        with open(fname_spearmanPval, 'wb') as f: pickle.dump(adj_spearman_pval, f)
    else: print(f"file exists: {fname_spearman}")
            
        
    if not (os.path.exists(fname_coherence)):
        for w in times:
            t0 = time.time()
            start = w
            stop = w + fs * ws
            count = np.where(times == w)[0][0]
            data_win = data[start:stop,:]
            adj_coherence[0, :,:,count], adj_coherence[1, :,:,count], adj_coherence[2, :,:,count], adj_coherence[3, :,:,count], adj_coherence[4, :,:,count], adj_coherence[5, :,:,count], adj_coherence[6, :,:,count], adj_coherence[7, :,:,count] = echobase.coherence_wrapper(data_win, fs, avgref=True)
            t1= time.time(); td = t1-t0; tr = td*(t-count)/60; 
            printProgressBar(w, times[-1], prefix = "Progress:", suffix = f"{stop}/{n_samp}; {count+1}/{t}; Time: {np.round(tr,2)} min" )
            print(f"\n{sub_ID}; {descriptor}")
        with open(fname_coherence, 'wb') as f: pickle.dump(adj_coherence, f)
    else: print(f"file exists: {fname_coherence}")

    
    if not (os.path.exists(fname_mi)):
        for w in times:
            t0 = time.time()
            start = w
            stop = w + fs * ws
            count = np.where(times == w)[0][0]
            data_win = data[start:stop,:]
            adj_mi[0, :,:,count], adj_mi[1, :,:,count], adj_mi[2, :,:,count], adj_mi[3, :,:,count], adj_mi[4, :,:,count], adj_mi[5, :,:,count], adj_mi[6, :,:,count], adj_mi[7, :,:,count] = echobase.mutualInformation_wrapper(data_win, fs, avgref=True)
            t1= time.time(); td = t1-t0; tr = td*(t-count)/60; 
            printProgressBar(w, times[-1], prefix = "Progress:", suffix = f"{stop}/{n_samp}; {count+1}/{t}; Time: {np.round(tr,2)} min" )
        with open(fname_mi, 'wb') as f: pickle.dump(adj_mi, f)
    else: print(f"file exists: {fname_mi}")
    
        
        
            

    
    
    
    
    
    
    
    

   
    
    
    
    
#%%    
    

def get_triu(adj, k = 1):
    adj_hat = adj[np.triu_indices( len(adj), k = k)]    
    return adj_hat
    
    
def get_tissue_ind(metadata, elecLoc, WM_definition = 0):
    ch_eeg = np.asarray(metadata["channels"])
    ch_eloc = np.asarray(elecLoc["electrode_name"].to_list())
    eloc_ind_gm = np.where(np.array(elecLoc["tissue_segmentation_distance_from_label_2"]) <= WM_definition)
    eloc_ind_wm = np.where(np.array(elecLoc["tissue_segmentation_distance_from_label_2"]) > WM_definition)
    ch_eloc_gm = ch_eloc[eloc_ind_gm]
    ch_eloc_wm = ch_eloc[eloc_ind_wm]
    ch_intersect_gm = np.intersect1d(ch_eeg, ch_eloc_gm, return_indices = True )
    ch_intersect_wm = np.intersect1d(ch_eeg, ch_eloc_wm, return_indices = True )
    return ch_intersect_gm[1], ch_intersect_wm[1]

    
def get_adj_tissue_ind(adj, ind_gm, ind_wm):
    adj_gm = adj[:, ind_gm[:,None], ind_gm[None,:], :]
    adj_wm = adj[:, ind_wm[:,None], ind_wm[None,:], :]
    adj_gmwm = adj[:, ind_gm[:,None], ind_wm[None,:], :]
    return adj_gm, adj_wm, adj_gmwm
    
def get_adj_reorder(adj, metadata, elecLoc):
    ch_eeg = np.asarray(metadata["channels"])
    ch_eloc = np.asarray(elecLoc["electrode_name"].to_list())
    ch_intersect = np.intersect1d(ch_eeg, ch_eloc, return_indices = True )
    
    eloc_sort = np.argsort(np.array(elecLoc["tissue_segmentation_distance_from_label_2"])[ch_intersect[2]]  )
    adj_sort = adj[:, eloc_sort[:,None], eloc_sort[None,:], :]

    return adj_sort

def movingaverage(x, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(x, window, 'valid')
    
    
def plot_adj_time(tt , all_adj, all_adj_gm, all_adj_wm, all_adj_gmwm, bars = [], prefix = "img", fpath =  ospj(fpath_video,"distributions"), vmin = -1, vmax = 1):
    n_freq, n_chan, n_chan, t_total = all_adj.shape
    colors = ["#000000", "#836953", "#78afe0", "#787be0" ]
    #title names
    titles = ["Broadband", "Delta", "Theta", "Alpha", "Beta", "Gamma - Low", "Gamma - Mid", "Gamma - High"]

    #initialization of means
    adj_mean = [ [] for i in range(n_freq)]
    adj_gm_mean = [ [] for i in range(n_freq)]
    adj_wm_mean = [ [] for i in range(n_freq)]
    adj_gmwm_mean = [ [] for i in range(n_freq)]

    t0 = time.time()
    for t in range(t_total):
        #setting up plot layout
        for f in range(n_freq):
            adj_ft = all_adj[f,:,:,t]
            adj_gm_ft = all_adj_gm[f,:,:,t]
            adj_wm_ft = all_adj_wm[f,:,:,t]
            adj_gmwm_ft = all_adj_gmwm[f,:,:,t]
            
            #plotting histograms
            x = get_triu(adj_ft, k = 1).flatten()
            adj_mean[f].append(np.mean(x))
            
            x = get_triu(adj_gm_ft, k = 1).flatten()
            adj_gm_mean[f].append(np.mean(x))    
            
            x = get_triu(adj_wm_ft, k = 1).flatten()
            adj_wm_mean[f].append(np.mean(x))       
            
            x = adj_gmwm_ft.flatten()
            adj_gmwm_mean[f].append(np.mean(x))    
            
    fig = plt.figure(constrained_layout=False, figsize=(24,15) , dpi=100)
    left1 = 0.03; right1 = 0.48; left2 = 0.52; right2 = 0.97
    bottom1 = 0.77; bottom2 = 0.52; bottom3 = 0.27; bottom4 = 0.02
    top1 = 0.97; top2 = 0.72; top3 = 0.47; top4 = 0.22
    gs1 = fig.add_gridspec(nrows = 2, ncols = 6, left = left1, right = right1, bottom = bottom1, top = top1)
    gs2 = fig.add_gridspec(nrows = 2, ncols = 6, left = left2, right = right2, bottom = bottom1, top = top1)
    gs3 = fig.add_gridspec(nrows = 2, ncols = 6, left = left1, right = right1, bottom = bottom2, top = top2)
    gs4 = fig.add_gridspec(nrows = 2, ncols = 6, left = left2, right = right2, bottom = bottom2, top = top2)
    gs5 = fig.add_gridspec(nrows = 2, ncols = 6, left = left1, right = right1, bottom = bottom3, top = top3)
    gs6 = fig.add_gridspec(nrows = 2, ncols = 6, left = left2, right = right2, bottom = bottom3, top = top3)
    gs7 = fig.add_gridspec(nrows = 2, ncols = 6, left = left1, right = right1, bottom = bottom4, top = top4)
    gs8 = fig.add_gridspec(nrows = 2, ncols = 6, left = left2, right = right2, bottom = bottom4, top = top4)
    gs = [gs1, gs2, gs3, gs4, gs5, gs6, gs7, gs8]
    ax = [None] * n_freq
    for freq in range(n_freq):
        ax[freq] = [fig.add_subplot(gs[freq][:2, :2]), fig.add_subplot(gs[freq][0, 2]), fig.add_subplot(gs[freq][0, 3]),  fig.add_subplot(gs[freq][1, 2]), fig.add_subplot(gs[freq][1, 3]),
                    fig.add_subplot(gs[freq][0, 4]),fig.add_subplot(gs[freq][0, 5]), fig.add_subplot(gs[freq][1, 4]) , fig.add_subplot(gs[freq][1, 5])]
    for f in range(n_freq):  
        t = tt
        print(f"t: {t+1}; f: {f+1}")
        adj_ft = all_adj[f,:,:,t]
        adj_gm_ft = all_adj_gm[f,:,:,t]
        adj_wm_ft = all_adj_wm[f,:,:,t]
        adj_gmwm_ft = all_adj_gmwm[f,:,:,t]
        
        #plotting histograms
        x = get_triu(adj_ft, k = 1).flatten()
        sns.histplot(x, ax = ax[f][5], kde = True, color = colors[0])   
        x = get_triu(adj_gm_ft, k = 1).flatten()
        sns.histplot(x, ax = ax[f][6], kde = True, color = colors[1])  
        x = get_triu(adj_wm_ft, k = 1).flatten()
        sns.histplot(x, ax = ax[f][7], kde = True, color = colors[2])    
        x = adj_gmwm_ft.flatten()
        sns.histplot(x, ax = ax[f][8], kde = True, color = colors[3])    
        for l in range(5,9):   
            ax[f][l].set_xlim([vmin, vmax])
            ax[f][l].set_yticks([])
            ax[f][l].set_ylabel("")
            
        #plotting heatmaps
        sns.heatmap(adj_ft, square=True, ax = ax[f][1], vmin = vmin, vmax = vmax, cbar=False, xticklabels = False, yticklabels = False)
        sns.heatmap(adj_gm_ft, square=True, ax = ax[f][2], vmin = vmin, vmax = vmax, cbar=False, xticklabels = False, yticklabels = False)
        sns.heatmap(adj_wm_ft, square=True, ax = ax[f][3], vmin = vmin, vmax = vmax, cbar=False, xticklabels = False, yticklabels = False)
        sns.heatmap(adj_gmwm_ft, square=True, ax = ax[f][4], vmin = vmin, vmax = vmax, cbar=False, xticklabels = False, yticklabels = False)
        
        #plotting means
        size = 8
        sns.scatterplot(x = range(1, t+2), y =  np.array(adj_mean[f])[range(0,t+1)] , ax = ax[f][0], color = colors[0] , s = size)
        sns.scatterplot(x = range(1, t+2), y =  np.array(adj_gm_mean[f])[range(0,t+1)] , ax = ax[f][0], color = colors[1], s = size)
        sns.scatterplot(x = range(1, t+2), y =  np.array(adj_wm_mean[f])[range(0,t+1)] , ax = ax[f][0], color = colors[2], s = size)
        sns.scatterplot(x = range(1, t+2), y =  np.array(adj_gmwm_mean[f])[range(0,t+1)] , ax = ax[f][0], color = colors[3] , s = size )
        
        plt.setp([ax[f][0]], title=titles[f])
        #plotting vertical lines for preictal, ictal, etc.
        if len(bars)>0:
            xcoords = np.array(bars)
            for b in range(1, len(xcoords)):
                xcoords[b] = xcoords[b-1] + bars[b]
            xcoords = xcoords + 1
            xhor = xcoords[np.flatnonzero(xcoords <=t+ 1)]
            if len(xhor) > 0:
                for xc in xhor:
                    ax[f][0].axvline(x=xc, linestyle='--', color="#000000")
     
        
        #plotting moving averages
        w=10 #window to average over
        if t >= w:
            x = movingaverage(np.array(adj_mean[f])[range(0,t+1)], w)
            sns.lineplot(x = range(w,t+2), y =  np.array(x) , ax = ax[f][0], color = colors[0] )
            x = movingaverage(np.array(adj_gm_mean[f])[range(0,t+1)], w)
            sns.lineplot(x = range(w,t+2), y =  np.array(x) , ax = ax[f][0], color = colors[1] )
            x = movingaverage(np.array(adj_wm_mean[f])[range(0,t+1)], w)
            sns.lineplot(x = range(w,t+2), y =  np.array(x) , ax = ax[f][0], color = colors[2] )
            x = movingaverage(np.array(adj_gmwm_mean[f])[range(0,t+1)], w)
            sns.lineplot(x = range(w,t+2), y =  np.array(x) , ax = ax[f][0], color = colors[3] )
                
    t1 = time.time(); td = t1-t0; tr = td*(t_total-tt)/60
    print(f"time remaining: {np.round(tr, 2)} min")
    
    plt.savefig(  ospj(fpath, f"{prefix}_{t:03d}.png")   )
    plt.show()
        
#%%        
    

FC_names = ["xcorr", "pearson", "spearman", "coherence", "mutualInformation"]
FC_t = 0
for FC_t in range(len(FC_names)):
    for i in range(0,4): #range(len(eegTimes)):
    #for i in np.concatenate([range(12,16), range(20,24)]): #range(len(eegTimes)):
        #parsing data DataFrame to get iEEG information
        sub_ID = eegTimes.iloc[i].RID
        iEEG_filename = eegTimes.iloc[i].file
        ignore_electrodes = eegTimes.iloc[i].ignore_electrodes.split(",")
        start_time_usec = int(eegTimes.iloc[i].connectivity_start_time_seconds*1e6)
        stop_time_usec = int(eegTimes.iloc[i].connectivity_end_time_seconds*1e6)
        descriptor = eegTimes.iloc[i].descriptor
        print(f"{sub_ID}; {descriptor}")
        #input filename EEG
    
        fpath_FC = ospj(fpath_connectivity,f"sub-{sub_ID}")
        ref = "CAR" 
        fpath_FC_filt = ospj(fpath_FC, ref)
        
        fname_electrodeLoc = ospj(fpath_electrodeLoc, f"sub-{sub_ID}", f"sub-{sub_ID}_electrode_localization.csv")
        
        #Output files
        
        #read files
        fname_metadata = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_metadata.json")
        metadata = json.loads(codecs.open(fname_metadata, 'r', encoding='utf-8').read())
        fs = int(metadata["fs"])
        elecLoc = pd.read_csv(fname_electrodeLoc)
        
     
        fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_crossCorrelation.pickle")
        with open(fname, 'rb') as f: adj_xcorr = pickle.load(f)
   
        fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_pearson.pickle")
        with open(fname, 'rb') as f: adj_pearson = pickle.load(f)    
    
        #fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_pearsonPval.pickle")
        #with open(fname, 'rb') as f: adj_pearson_pval = pickle.load(f)    
        

        fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_spearman.pickle")
        with open(fname, 'rb') as f: adj_spearman = pickle.load(f)    
        
        #fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_spearmanPval.pickle")
        #with open(fname, 'rb') as f: adj_spearman_pval = pickle.load(f)    
        

        fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_coherence.pickle")
        with open(fname, 'rb') as f: adj_coherence = pickle.load(f)    
        
        fname = ospj(fpath_FC_filt, f"sub-{sub_ID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_mutualInformation.pickle")
        with open(fname, 'rb') as f: adj_mi = pickle.load(f)    
        
        FC = [adj_xcorr, adj_pearson, adj_spearman, adj_coherence, adj_mi]
        
        adj = copy.deepcopy(FC[FC_t])
    
        ind_gm, ind_wm = get_tissue_ind(metadata, elecLoc, WM_definition = 0)
        adj_gm, adj_wm, adj_gmwm = get_adj_tissue_ind(adj, ind_gm, ind_wm)
        adj_sort = get_adj_reorder(adj, metadata, elecLoc)
        
        #st = "state"
        if descriptor == "interictal": st = 0;
        if descriptor == "preictal": st = 1; 
        if descriptor == "ictal": st = 2
        if descriptor == "postictal": st = 3
        
        
        plt.interactive(False)
        if st == 0:
            all_adj = adj_sort[:,:,:,range( adj.shape[3]-60, adj.shape[3])]; all_adj_gm = adj_gm[:,:,:,range( adj.shape[3]-60, adj.shape[3])]; all_adj_wm = adj_wm[:,:,:,range( adj.shape[3]-60, adj.shape[3])]; all_adj_gmwm = adj_gmwm[:,:,:,range( adj.shape[3]-60, adj.shape[3])]
            seizure_bars= [all_adj.shape[3]]
        if st >0: 
            all_adj = np.concatenate([all_adj, adj_sort], axis = 3); all_adj_gm = np.concatenate([all_adj_gm, adj_gm], axis = 3); all_adj_wm = np.concatenate([all_adj_wm, adj_wm], axis = 3); all_adj_gmwm = np.concatenate([all_adj_gmwm, adj_gmwm], axis = 3)
            seizure_bars.append(adj.shape[3])
        if st == 3:
            prefix = FC_names[FC_t]
            print(prefix)
            if prefix == "coherence" or prefix == "mutualInformation": vmin = 0
            else: vmin = -1
            if False:
                fpath = ospj(fpath_video, "distributions", f"sub-{sub_ID}", f"{prefix}")
                if not (os.path.isdir(fpath)): os.makedirs(fpath, exist_ok=True)
                for tt in range(  all_adj.shape[3]-1 ) :
                    plot_adj_time(tt , all_adj, all_adj_gm, all_adj_wm, all_adj_gmwm, bars = seizure_bars, prefix = prefix, fpath =  fpath, vmin = vmin, vmax = 1)
            else:  
                fpath = ospj(fpath_video, "timeseries", f"sub-{sub_ID}")
                if not (os.path.isdir(fpath)): os.makedirs(fpath, exist_ok=True)
                for tt in  [all_adj.shape[3]-1] :
                    plot_adj_time(tt , all_adj, all_adj_gm, all_adj_wm, all_adj_gmwm, bars = seizure_bars, prefix = prefix, fpath =  fpath, vmin = vmin, vmax = 1)
      
#%%
for i in range(len(eegTimes)):
    #parsing data DataFrame to get iEEG information
    sub_ID = eegTimes.iloc[i].RID
    iEEG_filename = eegTimes.iloc[i].file
    ignore_electrodes = eegTimes.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(eegTimes.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(eegTimes.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = eegTimes.iloc[i].descriptor
    #input filename EEG
    fpath_EEG_sub_ID = ospj(fpath_EEG, "sub-{0}".format(sub_ID))
    #Output filtered EEG
    fpath_EEG_sub_ID = ospj(fpath_filtered_eeg, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(fpath_EEG_sub_ID)): os.mkdir(fpath_EEG_sub_ID)#if the path doesn't exists, then make the directory
    fname_EEG = "{0}/sub-{1}_{2}_{3}_{4}_EEG.pickle".format(fpath_EEG_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    fname_EEG_filtered = "{0}/sub-{1}_{2}_{3}_{4}_EEG_filtered.pickle.pickle".format(fpath_EEG_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    print("\n\n\nID: {0}\nDescriptor: {1}".format(sub_ID, descriptor))
    if (os.path.exists(fname_EEG_filtered)):
        print("File already exists: {0}".format(fname_EEG_filtered))
    else:#if file already exists, don't run below
        filter_eeg_data(fname_EEG, fname_EEG_filtered)


    

