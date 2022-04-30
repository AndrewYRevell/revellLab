#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:21:07 2021

@author: arevell
"""

import pickle
import numpy as np
import os
import sys
from os.path import join as ospj
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from scipy import signal
import copy
from scipy import interpolate
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename
import glob
from revellLab.packages.utilities import utils

@dataclass
class diffusionModels:
    SC: str = "unknown"
    SC_regions: None = None
    spread: None = None
    spread_regions: None = None
    electrodeLocalization: None = None
    
    
    def preprocess_SC(self, SC, threshold = 0.4, log_normalize = False):
        
        #log normalizing
        if log_normalize:
            SC[np.where(SC == 0)] = 1
            SC = np.log10(SC)
        SC = SC/np.max(SC)
        
        
        threshold = 0.4 #bottom X percent of edge weights are eliminated
        C_thresh = copy.deepcopy(SC)

        number_positive_edges = len(np.where(C_thresh > 0)[0])
        cutoff = int(np.round(number_positive_edges*threshold))

        positive_edges = C_thresh[np.where(C_thresh > 0)]
        cutoff_threshold = np.sort(positive_edges)[cutoff]
        C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
        return C_thresh
    
    
    def LTM(self, SC, seed = 0, time_steps = 50, threshold = 0.2):
        
        N = len(SC) #number of nodes
        node_state = np.zeros(shape = (time_steps, N))
        node_state[0, seed] = 1 #make seed active
    
        #neighbor_sum_distribution = np.zeros(shape = (time_steps, N))
        for t in range(1, time_steps):
            #print(t)
            for i in range(N): #loop thru all nodes
                #find neighbors of node i
                previous_state = node_state[t-1,i]
                neighbors = np.where(SC[i,:] > 0)
                neighbors_weight = SC[i,neighbors] 
                neighbors_state = node_state[t-1, neighbors]
                neighbors_sum = np.sum(neighbors_weight * neighbors_state)
                strength = np.sum(neighbors_weight)
                if neighbors_sum >= threshold*strength: #if sum is greater than threshold, make that node active
                    node_state[t, i] = 1
                if neighbors_sum < threshold*strength:
                    node_state[t, i] = 0
                if previous_state == 1:
                    node_state[t, i] = 1
                #neighbor_sum_distribution[t,i] = neighbors_sum/strength 
        return node_state
    
    #SC=  SC_hat
    def gradient_diffusion(self, SC, seed = 0, time_steps = 50, gradient = 0.1):
        
        N = len(SC) #number of nodes
        node_state = np.zeros(shape = (time_steps, N))
        node_state[0, seed] = 1 #make seed active
    

        for t in range(1, time_steps):
            #print(t)
            for i in range(N): #loop thru all nodes
                #find neighbors of node i
                previous_state = node_state[t-1,i]
                neighbors = np.where(SC[i,:] > 0)
                neighbors_weight = SC[i,neighbors] 
                neighbors_state = node_state[t-1, neighbors]
                neighbors_sum = np.sum(neighbors_weight * neighbors_state * gradient)
                
                # Add the cumulative sum from the neighbors to the node's current (i.e. previous) state. If greater than 1, make it 1
                # 
                
                node_state[t, i] = previous_state + neighbors_sum
                if node_state[t, i] > 1: node_state[t, i] = 1

                
                #fig, axes = utils.plot_make()
                #sns.heatmap(node_state)
        return node_state
    
    def gradient_LTM(self, SC, seed = 0, time_steps = 50, gradient = 0.1, threshold = 0.2):
        
        N = len(SC) #number of nodes
        node_state = np.zeros(shape = (time_steps, N))
        node_state[0, seed] = 1 #make seed active
    

        for t in range(1, time_steps):
            #print(t)
            for i in range(N): #loop thru all nodes
                #find neighbors of node i
                previous_state = node_state[t-1,i]
                neighbors = np.where(SC[i,:] > 0)
                neighbors_weight = SC[i,neighbors] 
                neighbors_state = node_state[t-1, neighbors]
                neighbors_sum = np.sum(neighbors_weight * neighbors_state * gradient)
                
                # Add the cumulative sum from the neighbors to the node's current (i.e. previous) state. If greater than 1, make it 1
                # 
                
                node_state[t, i] = previous_state + neighbors_sum
                if node_state[t, i] > 1: 
                    node_state[t, i] = 1
                if node_state[t, i] >= threshold: #if sum is greater than threshold, make that node active
                    node_state[t, i] = 1
                
                
                #fig, axes = utils.plot_make()
                #sns.heatmap(node_state)
        return node_state
    
    


    def equalize_sf(self, atlas = "AAL2", reference = "CAR", state = "ictal", functional_connectivity_measure = "xcorr", frequency = 0):
        SC, region_names, FC, FC_channels, el = self.get_structure_and_function( atlas = atlas, reference = reference, state = state, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        el_names = np.array(el["electrode_name"])
        
        #normalize SC
        SC[np.where(SC == 0)] = 1
        SC = np.log10(SC)
        SC = SC/SC.max()
        
        #remove any electrodes in FC with no localization data
        ind = np.intersect1d(FC_channels, el_names, return_indices=True)
        FC_mod = FC[  ind[1][:,None],  ind[1][None, :],:  ]
        el_mod = el.iloc[ind[2],:]
        el_mod_regions = np.array(el_mod[atlas + "_region_number"]).astype('str')

        #remove structure with no electrodes
        ind2 = np.intersect1d(region_names, el_mod_regions, return_indices=True)
        SC_mod = SC[ ind2[1][:,None],  ind2[1][None]]
        
        #select a random electrode from a region with multiple electrodes 
        SC_mod_region_names = ind2[0]
        ind3 = []
        np.random.seed(41)
        for s in range(len(SC_mod_region_names)):
            ind3.append(np.random.choice(np.where(SC_mod_region_names[s] == el_mod_regions)[0],1)[0])
        ind3 = np.array(ind3)
        FC_mod_SC = FC_mod[  ind3[:,None],  ind3[None, :],:  ]    
        el_mod_sc = el_mod.iloc[ind3,:]
        electrode_names = np.array(el_mod_sc["electrode_name"])
        return SC_mod, FC_mod_SC, electrode_names
    
   
        
    
    def resample(self, atlas = "AAL2", reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0, N= 100):
        ii, pi, ic, po = self.get_sfc_states( atlas = atlas, reference = reference, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        ii = signal.resample(ii, N)
        pi = signal.resample(pi, N)
        ic = signal.resample(ic, N)
        po = signal.resample(po, N)
        return ii, pi, ic, po
    
   
            
    def movingaverage(self, x, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(x, window, 'valid')

#%%


def get_diffusion_model_correlations(SC, SC_regions, spread, channels_spread, electrodeLocalization, atlas_name, atlas_label, dm_data, diffusion_model_type = 0, threshold=None, time_steps = None, gradient = None, visualize = False, r_to_visualize = None):
    
    if diffusion_model_type == 0:
        if time_steps == None:
            time_steps = 8
        if threshold == None:
            threshold = 0.1
    if diffusion_model_type == 1: 
        if gradient == None:
            gradient = 0.1
        if time_steps == None:
            time_steps = 25
            
    if visualize == True and r_to_visualize == None:
        r_to_visualize = SC_regions[0]
    if type(r_to_visualize) == int:
        r_to_visualize = str(r_to_visualize)
    
    print(f"\nModel:  {diffusion_model_type} (LTM=0;gDM=1;gLTM=2)\n\
              time_steps: {time_steps}\n\
              threshold: {threshold}\n\
              gradient: {gradient}\n\n\n")
    
    corrs = np.zeros(shape = (len(SC)))
    for r in range(len(corrs)):
        
        if diffusion_model_type == 0: #LTM
            node_state = dm_data.LTM(SC, seed=r, threshold=threshold, time_steps = time_steps)
        elif diffusion_model_type == 1: #Diffusion gradient
            node_state = dm_data.gradient_diffusion(SC, seed = r, time_steps = time_steps, gradient = gradient)
            #sns.heatmap(node_state.T, cbar=False)
        elif diffusion_model_type == 2: #gradientLTM
            node_state = dm_data.gradient_LTM(SC, seed = r, time_steps = time_steps, gradient = gradient,  threshold = threshold)
            #sns.heatmap(node_state.T, cbar=False)
        else:
            print("no diffusion model type given. Doing LTM")
            node_state = dm_data.LTM(SC, seed=r, threshold=threshold, time_steps = time_steps)
        #equalizing node state and spread
        el = copy.deepcopy(electrodeLocalization)
        #remove any electrodes in spread with no localization data
        ind = np.intersect1d(channels_spread, el["channel"], return_indices=True)
        spreadMod = spread[:,ind[1]]
        #sns.heatmap(spread.T, cbar=False)  
        #sns.heatmap(spreadMod.T, cbar=False)  
        el_mod = el.iloc[ind[2],:]
        el_mod_regions = np.array(el_mod[atlas_name + "_region_number"]).astype('str')
        #remove structure with no electrodes
        ind2 = np.intersect1d(SC_regions, el_mod_regions, return_indices=True)
        node_state_mod = node_state[:,ind2[1]]
        #select a random electrode from a region with multiple electrodes 
        SC_mod_region_names = ind2[0]
        ind3 = []
        #np.random.seed(41)
        for s in range(len(SC_mod_region_names)):
            ind3.append(np.random.choice(np.where(SC_mod_region_names[s] == el_mod_regions)[0],1)[0])
        ind3 = np.array(ind3)
        spreadMod_state = spreadMod[:, ind3]   
        #el_mod_sc = el_mod.iloc[ind3,:]
        #electrode_names = np.array(el_mod_sc["electrode_name"])
        #interpolate node state
        xp = np.tile(np.linspace(0, node_state_mod.shape[0]-1, spreadMod_state.shape[0]), (spreadMod_state.shape[1],1)).T
        interpolation =   interpolate.interp1d(np.arange(0, node_state_mod.shape[0], 1), node_state_mod, axis = 0)
        node_state_mod_resample = interpolation(xp)[:,0,:]
        #sns.heatmap(node_state_mod, cbar=False)   
        #sns.heatmap(node_state_mod_resample, cbar=False)   
        #sns.heatmap(spreadMod_state, cbar=False)   
        if (node_state_mod_resample == 0).all():#if entire prediction is zero, make the correlation zero
            corrs[r] = 0
        else:
            corrs[r] = spearmanr(node_state_mod_resample.flatten(), spreadMod_state.flatten())[0]
        top = SC_regions[np.where(corrs == np.max(corrs))[0][0]]
        
        top_region = get_region_name(atlas_label, top)
        print(f"\r({np.round( np.max(corrs),2)}) {top}: {top_region}; {np.round( np.max(corrs),2)}; {np.round((r+1)/len(corrs)*100,2)}%                                       ", end = "\r")
        
        SC_regions[np.where(corrs == np.max(corrs))[0][0]]
        
        #visualize:
        if visualize and SC_regions[r]== r_to_visualize:
            fig, axes = utils.plot_make(r = 2, c = 2)
            axes = axes.flatten()
            sns.heatmap(node_state.T, cbar=False, ax = axes[0])
            sns.heatmap(spread.T, cbar=False, ax = axes[1]) 
            sns.heatmap(node_state_mod_resample.T, cbar=False, ax = axes[2]) 
            sns.heatmap(spreadMod_state.T, cbar=False, ax = axes[3])   
            axes[0].set_title(f"Model: {diffusion_model_type}, region = {SC_regions[r]}\n{get_region_name(atlas_label, SC_regions[r])}", size = 10)
            axes[1].set_title(f"Spread Pattern", size = 10)
            axes[2].set_title(f"Equalized Model, corr = {np.round(corrs[r],2)}", size = 10)
            axes[3].set_title(f"Equalized Spread", size = 10)

        
    print("\n\n")
    return corrs


def get_region_name(atlas_label, region_number):
    return atlas_label.iloc[np.where(atlas_label["region"] == int(region_number) )[0][0]]["label"]

#%

def compute_diffusion_model_correlations_for_atlas(sub, paths, atlas_index, secondsBefore, skipWindow, probability_arr_movingAvg, channels, SESSION_RESEARCH3T, session, diffusion_model_type, threshold=None, time_steps = None, gradient = None, spread_before = 20, spread_after = 30, SC_threshold = 0.4, log_normalize = False, visualize = False, r_to_visualize = None):
    
    # Get SC data
    sub =sub
    ses = basename(glob.glob( join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{SESSION_RESEARCH3T}"))[0])[4:]
    sc_paths = glob.glob(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{ses}", "matrices", "*connectogram.txt"))
    
    
    a=atlas_index
    sc_atlas = sc_paths[a]
    atlas_name = basename(sc_atlas).split(".")[1]
    print(f"\n\n\n{atlas_name}\n\n\n")
    
    SC = utils.read_DSI_studio_Txt_files_SC(sc_atlas)
    SC_regions = utils.read_DSI_studio_Txt_files_SC_return_regions(sc_atlas, atlas_name)
    
    # Get electrode localization data
    paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION
    el_path = glob.glob( join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{sub}",  f"ses-{session}", "*atlasLocalization.csv"))[0]
    eloc = pd.read_csv(el_path)
    
    eloc["channel"] = utils.channel2std(np.array(eloc["channel"]) )
    electrodeLocalization = eloc[["channel", f"{atlas_name}_region_number", f"{atlas_name}_label"]]
    
    # Get atlas label data
    atlas_label_path = glob.glob( join(paths.ATLAS_LABELS, f"{atlas_name}.csv"))
    if len(atlas_label_path) > 0:
        atlas_label = atlas_label_path[0]
        atlas_label = pd.read_csv(atlas_label, header=1)
    else:
        atlas_label = pd.DataFrame(  dict( region= SC_regions.astype(int), label =  SC_regions))
    
    st =int((secondsBefore-spread_before)/skipWindow)
    stp = int((secondsBefore +spread_after)/skipWindow)
    spread = copy.deepcopy(probability_arr_movingAvg[st:stp,:])
    channels_spread = copy.deepcopy(channels)  
    #%
    dm_data = diffusionModels(SC, SC_regions, spread, channels_spread, electrodeLocalization)    
    SC_hat = dm_data.preprocess_SC(SC, SC_threshold, log_normalize = log_normalize)    
        
    
    #%
    
    corrs = get_diffusion_model_correlations(SC_hat, SC_regions, spread, channels_spread, electrodeLocalization, atlas_name, atlas_label, dm_data, diffusion_model_type = diffusion_model_type, threshold=threshold, time_steps = time_steps, gradient = gradient, visualize = visualize, r_to_visualize = r_to_visualize)
    
    ind = np.argsort(corrs)[::-1]
    corrs_order = corrs[ind]
    """
    if visualize:
        fig, ax = plt.subplots(1,1, figsize = (5,5), dpi = 300)    
        sns.lineplot(x = range(len(corrs_order)), y = corrs_order, ax = ax)
        #sns.lineplot(x="variable", y="value", data=corrs_order_null_df,ax = ax, color = "red")
        ax.set_xlabel("Brain Region Number")
        ax.set_ylabel("Spearman Rank Correlation")
        ax.set_title("LTM: Brain regions most correlated to measured seizure pattern")
    """
    if len(SC_regions) >=10:
        for i in range(10):
            region_number = SC_regions[ind][i]
            region_label = get_region_name(atlas_label, region_number)
            correlation = corrs_order[i]
            print(f" ({np.round(correlation,2)}) {region_number}: {region_label}")
            
        
    corrs_atlas =  np.vstack([SC_regions[ind],corrs_order ]).T
    return corrs_atlas, atlas_name

#aaaa = np.vstack([SC_regions[ind],corrs_order ]).T


#% Create nifti brains

def create_nifti_corrs_brain(corrs_atlas, sub, paths, SESSION_RESEARCH3T, atlas_name):
    ses = basename(glob.glob( join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{SESSION_RESEARCH3T}"))[0])[4:]
    save_loc = join(paths.SEIZURE_SPREAD_ATLASES_PROBABILITIES, f"sub-{sub}")
    utils.checkPathAndMake(save_loc, save_loc, printBOOL=False)
    
    
    atlas_path = glob.glob(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{ses}", "atlas_registration", f"*{atlas_name}.nii.gz"))[0]
    
    img, img_data = utils.load_img_and_data(atlas_path)
    img_data_cors = copy.deepcopy(img_data)
    img_data_cors[np.where(img_data_cors == 0 )] = -1
    #loop thru regions and replace regions with correlation data
    
    for r in range(len(corrs_atlas)):
        utils.printProgressBar(r+1, len(corrs_atlas))
        ind = np.where(img_data == int(corrs_atlas[r,0]) )
        img_data_cors[ind] =  float(corrs_atlas[r,1])
    
    
    save_loc_atlas = join(save_loc, f"{atlas_name}.nii.gz")
    utils.save_nib_img_data(img_data_cors, img, save_loc_atlas)
    

    T1_path = glob.glob(join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}",  f"ses-{ses}", "atlas_registration", "*desc-preproc_T1w_std.nii.gz"))[0]
    if utils.checkIfFileDoesNotExist(f"{join(save_loc, basename(T1_path))}"):
        utils.executeCommand(cmd = f"cp {T1_path} {join(save_loc, basename(T1_path))}")
"""

import os
from scipy.io import loadmat
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

#%%
path = "/media/arevell/sharedSSD/linux/papers/paper003" 
ifbase = "sub-RID0278_ses-preop3T_dwi-eddyMotionB0Corrected.aal_res-1x1x1.count.pass.connectivity.mat"
ifpath = os.path.join (path, "data", "data_processed","connectivity_matrices","structure", "sub-RID0278", "aal_res-1x1x1", )
ifname = os.path.join(ifpath, ifbase)

ifname_centroids = os.path.join (path, "data", "data_processed","atlas_centroids","AAL", "AAL_centroid.csv")
#%%

C = loadmat(ifname)["connectivity"]

centroids = pd.read_csv( ifname_centroids )
#%% Preprocess connectivity matrices
#log normalizing
#C[np.where(C == 0)] = 1
#C = np.log10(C)
C = C/np.max(C)

#%%
#Thresholding
threshold = 0.4 #bottom X percent of edge weights are eliminated
C_thresh = copy.deepcopy(C)

number_positive_edges = len(np.where(C > 0)[0])
cutoff = int(np.round(number_positive_edges*threshold))

positive_edges = C[np.where(C > 0)]
cutoff_threshold = np.sort(positive_edges)[cutoff]
C_thresh[np.where(C_thresh < cutoff_threshold)] = 0

len(np.where(C_thresh > 0)[0])
#%%

#parameters
seed = 32 #ROI number where activity starts
time_steps = 50 #number of time steps before termination of simulation

N = len(C_thresh) #number of nodes
node_state = np.zeros(shape = (time_steps, N))

node_state[0, seed] = 1 #make seed active


#%%
threshold_LTM = 0.2 #Threshold in linear threshold model - the cutoff where a node can become in active state
#%%
#LTM

neighbor_sum_distribution = np.zeros(shape = (time_steps, N))
for t in range(1, time_steps):
    #print(t)
    for i in range(N): #loop thru all nodes
        #find neighbors of node i
        previous_state = node_state[t-1,i]
        neighbors = np.where(C_thresh[i,:] > 0)
        neighbors_weight = C_thresh[i,neighbors] 
        neighbors_state = node_state[t-1, neighbors]
        neighbors_sum = np.sum(neighbors_weight * neighbors_state)
        strength = np.sum(neighbors_weight)
        if neighbors_sum >= threshold_LTM*strength: #if sum is greater than threshold, make that node active
            node_state[t, i] = 1
        if neighbors_sum < threshold_LTM*strength:
            node_state[t, i] = 0
        if previous_state == 1:
            node_state[t, i] = 1
        neighbor_sum_distribution[t,i] = neighbors_sum/strength 
        

tt = 1
sns.displot(neighbor_sum_distribution[1,:] ,binwidth=0.05); plt.xlim(0, 1)
sns.displot(neighbor_sum_distribution[2,:] ,binwidth=0.05); plt.xlim(0, 1)
sns.displot(neighbor_sum_distribution[9,:] ,binwidth=0.05); plt.xlim(0, 1)
print(node_state[1,:])
print(node_state[3,:])
print(node_state[6,:])

#%%
#Cascading






#parameters
seed = 36 #ROI number where activity starts
time_steps = 25 #number of time steps before termination of simulation

N = len(C_thresh) #number of nodes
node_state = np.zeros(shape = (time_steps, N))

node_state[0, seed] = 1 #make seed active

#%%

cascading_denominator = np.max(C)/0.3 # converts edge weights into probabilities

#%%
activation_probability_threshold_distribution = np.zeros(shape = (time_steps, N))
for t in range(1, time_steps):
    print(t)
    for i in range(N): #loop thru all nodes
        #find neighbors of node i
        previous_state = node_state[t-1,i]
        activation_probability = random.betavariate(1.2, 1.2) #random.uniform(0, 1) #probability of being activated
        neighbors = np.where(C_thresh[i,:] > 0)
        neighbors_weight = C_thresh[i,neighbors] 
        neighbors_state = node_state[t-1, neighbors]
        neighbors_probability = neighbors_weight/cascading_denominator #convets edge weights into probabilites
        activation_probability_threshold = 1-np.prod((1-(neighbors_probability * neighbors_state))) #the proabbility that any nieghbors activates the node
        
        if activation_probability <= activation_probability_threshold: #if sum is greater than threshold, make that node active
            node_state[t, i] = 1
        if activation_probability > activation_probability_threshold:
            node_state[t, i] = 0
        if previous_state == 1:
            node_state[t, i] = 1
        activation_probability_threshold_distribution[t,i] = activation_probability_threshold
   

   
node_state_df = pd.DataFrame(node_state, columns = centroids['label'])

node_state_df.to_csv(os.path.join (path, "blender", "diffusion_models_simulation", "sub-RID0278_AAL_seed36.csv"), index=False)



Nnumbers = 10000
numbers = np.zeros(shape = Nnumbers)
for k in range(Nnumbers):
    numbers[k] = random.betavariate(2,2)
np.mean(numbers)
sns.histplot(numbers)












"""

