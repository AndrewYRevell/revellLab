# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:21:54 2020

@author: andyr
"""

import pickle
import numpy as np
import os
import sys
from os.path import join as ospj
path = ospj("/media","arevell","sharedSSD","linux","papers","paper005") #Parent directory of project
#path = ospj("E:\\","linux","pastates","paper005") #Parent directory of project
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
import pandas as pd
import copy
import json, codecs
from dataclasses import dataclass

#%% Input/Output Paths and File names
fname_EEG_times = ospj( path, "data","data_raw","iEEG_times","EEG_times.xlsx")
fpath_electrode_localization = ospj( path, "data","data_processed","electrode_localization")
fname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
fpath_FC = ospj(path, "data","data_processed","connectivity_matrices","function")
fpath_SC = ospj(path, "data","data_processed","connectivity_matrices","structure")


fpath_data =  ospj(path, "data","data_processed", "aggregated_data")
fpath_figure = ospj(path, "seeg_GMvsWM","figures","sfc")


if not (os.path.isdir(fpath_figure)): os.makedirs(fpath_figure, exist_ok=True)

#% Load Study Meta Data
data = pd.read_excel(fname_EEG_times)    
atlases = pd.read_csv(fname_atlases_csv)    
#% Processing Meta Data: extracting sub-IDs

subIDs_unique =  np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]


#%%

#Need python 3.7+
@dataclass
class DATA:
    subID: str
    electrode_localization: None
    streamlines: None
    function_EEG:  None
    function_rsfMRI: None


@dataclass
class streamlines_data_raw:
    atlas_name: str = "unknown"
    data: None = np.nan
    
    def clean_structural_data(self):
        SC = self.data.drop([0,1], axis=1)
        SC = SC.drop([0], axis=0)
        SC = SC.iloc[:, :-1]
        SC = np.array(SC.iloc[1:, :]).astype('float64')  #finally turn into numpy array
        return SC
    
    def get_region_names(self, basename):
        SC = self.data
        SC_regionNames = np.array([e[len(basename)+1:] for e in np.array(SC.iloc[1][2:-1])])
        return SC_regionNames
    
    
#%%
descriptors = ["interictal","preictal","ictal","postictal"]
references = ["CAR"]
for i in range(0,len(data)):
    #parsing data DataFrame to get iEEG information
    subID = data.iloc[i].RID
    subRID = "sub-{0}".format(subID)
    iEEG_filename = data.iloc[i].file
    ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    print( "\n\n{0}: {1}".format(subID,descriptor) )
    if descriptor == descriptors[0]: state = 0
    if descriptor == descriptors[1]: state = 1
    if descriptor == descriptors[2]: state = 2
    if descriptor == descriptors[3]: state = 3
    
    if state == 2:
        start_time_usec_ictal = start_time_usec
        stop_time_usec_ictal =  stop_time_usec
    
    #Inputs and Outputs
    #input filename EEG
    fpath_FC_subID = ospj(fpath_FC, subRID)
    fpath_electrode_localization_subID =  ospj(fpath_electrode_localization, subRID)
    fpath_SC_subID = ospj(fpath_SC, subRID)
    fname_electrode_localization = ospj(fpath_electrode_localization_subID, "sub-{0}_electrode_localization.csv".format(subID))
    electrode_localization = pd.read_csv(fname_electrode_localization)
    
    
    #initialize
    if state == 0:
        function_eeg_state = dict(interictal = np.nan, preictal = np.nan, ictal = np.nan, postictal = np.nan)
        function_eeg_all = dict(CAR = copy.deepcopy(function_eeg_state), bipolar = copy.deepcopy(function_eeg_state), laplacian =  copy.deepcopy(function_eeg_state))
    
    for ref in range(1):
        fname_FC_metadata = ospj(fpath_FC_subID,references[ref],  "sub-{0}_{1}_{2}_{3}_metadata.json".format(subID, iEEG_filename, start_time_usec, stop_time_usec))
        #GET DATA
        metadata = json.loads(codecs.open(fname_FC_metadata, 'r', encoding='utf-8').read())
        #function EEG
        with open( ospj(fpath_FC_subID,references[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_crossCorrelation.pickle"), 'rb') as f: FC_xcorr = pickle.load(f)
        with open( ospj(fpath_FC_subID,references[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_pearson.pickle"), 'rb') as f: FC_pearson = pickle.load(f)
        with open( ospj(fpath_FC_subID,references[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_pearsonPval.pickle"), 'rb') as f: FC_pearsonPval = pickle.load(f)
        with open( ospj(fpath_FC_subID,references[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_spearman.pickle"), 'rb') as f: FC_spearman = pickle.load(f)
        with open( ospj(fpath_FC_subID,references[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_spearmanPval.pickle"), 'rb') as f: FC_spearmanPval = pickle.load(f)
        with open( ospj(fpath_FC_subID,references[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_coherence.pickle"), 'rb') as f: FC_coherence = pickle.load(f)
        #with open( ospj(fpath_FC_subID,reference[ref],  f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_mutualInformation.pickle"), 'rb') as f: FC_mutualInformation = pickle.load(f)
        
        func = dict(metadata = metadata, 
                    xcorr = FC_xcorr, 
                    pearson = FC_pearson, 
                    pearsonPval = FC_pearsonPval, 
                    spearman = FC_spearman, 
                    spearmanPval = FC_spearmanPval, 
                    coherence = FC_coherence )
        
        function_eeg_all[ references[ref]  ][   descriptors[state]   ] = func
        
        function_eeg_all["bipolar"]
            
    if state ==3:

            
        #streamlines
        streamlines_list_data = [None] * len(atlases)
        if (os.path.exists(fpath_SC_subID)): #If structure exists
            for a in range(len(atlases)):
                atlas_fname = os.path.splitext(os.path.splitext(  atlases.iloc[a]["atlas_filename"] )[0])[0]
                atlas_name = os.path.splitext(os.path.splitext(  atlases.iloc[a]["atlas_name"] )[0])[0]
                print(f"{subID}; {atlas_name}")
                basename = f"sub-{subID}_preop3T_{atlas_fname}"
                fname_SC =  ospj( fpath_SC_subID, f"sub-{subID}.{basename}.count.pass.connectogram.txt") 
                
                
                SC = pd.read_table(fname_SC, header=None, dtype=object)
                
                st = streamlines_data_raw(atlas_fname, SC)
                streamlines_list_data[a] = dict(atlas = st.atlas_name, data = st.clean_structural_data(), region_names = st.get_region_names(basename) )
    
        all_data = dict(subID = subID, electrode_localization = electrode_localization, streamlines = streamlines_list_data, function_eeg = function_eeg_all)
    
        fname = ospj(fpath_data,   f"sub-{subID}_{iEEG_filename}_{start_time_usec_ictal}_{stop_time_usec_ictal}_data.pickle" )
        if not (os.path.exists(fname)):
            with open(fname, 'wb') as f: pickle.dump(all_data, f)



#with open(fname, 'rb') as f: all_data_upload = pickle.load(f)

#all_data_upload.keys()

#%%

    
    
    