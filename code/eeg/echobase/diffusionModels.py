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

@dataclass
class diffusionModels:
    SC: str = "unknown"
    SC_regions: None = None
    spread: None = None
    spread_regions: None = None
    electrodeLocalization: None = None
    
    
    def preprocess_SC(self, SC, threshold = 0.4):
        
        #log normalizing
        #C[np.where(C == 0)] = 1
        #C = np.log10(C)
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
    
        neighbor_sum_distribution = np.zeros(shape = (time_steps, N))
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
                neighbor_sum_distribution[t,i] = neighbors_sum/strength 
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