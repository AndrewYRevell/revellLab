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

@dataclass
class sfc:
    subID: str = "unknown"
    electrode_localization: None = None
    streamlines: None = None
    function_eeg:  None = None
    function_rsfMRI: None = None

    def get_atlas_names(self)   : 
        all_atlases = np.array([a["atlas"] for a in self.streamlines])
        return all_atlases
    def get_structure_and_function(self, atlas = "AAL2", reference = "CAR", state = "ictal", functional_connectivity_measure = "xcorr", frequency = 0):
        #get all atlases:
        all_atlases = np.array([a["atlas"] for a in self.streamlines]) #
        if not any(all_atlases == atlas):
            raise Exception(f"Valid atlases are:\n {all_atlases} \n\n Input atlas is not contained within the dataset. Please check spelling.")
            
        #find the inputted atlases data
        ind = np.where(all_atlases == atlas)[0][0]
        
        SC = self.streamlines[ind]["data"]
        region_names = self.streamlines[ind]["region_names"]
        FC = self.function_eeg[reference][state][functional_connectivity_measure][frequency]
        FC_channels = np.array(self.function_eeg[reference][state]["metadata"]["channels"])
        el = self.electrode_localization[ ["electrode_name",  atlas + "_region_number"]]
        return SC, region_names, FC, FC_channels, el

    def get_structure(self, atlas = "AAL2"):
        all_atlases = np.array([a["atlas"] for a in self.streamlines]) #
        if not any(all_atlases == atlas):
            raise Exception(f"Valid atlases are:\n {all_atlases} \n\n Input atlas is not contained within the dataset. Please check spelling.")
            
        #find the inputted atlases data
        ind = np.where(all_atlases == atlas)[0][0]
        
        SC = self.streamlines[ind]["data"]
        region_names = self.streamlines[ind]["region_names"]
        return SC, region_names
    
    def get_electrodeLocalization(self, atlas = "AAL2"):
        el = self.electrode_localization[ ["electrode_name",  atlas + "_region_number"]]
        return el

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
    
    def get_sfc(self, atlas = "AAL2", reference = "CAR", state = "ictal", functional_connectivity_measure = "xcorr", frequency = 0 ):
        SC_mod, FC_mod_SC, electrode_names = self.equalize_sf( atlas = atlas, reference = reference, state = state, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        
        SC_triu = SC_mod[np.triu_indices( len(SC_mod), k=1)]
        FC_triu = FC_mod_SC[np.triu_indices( len(FC_mod_SC), k=1)]
        
        t_len =  FC_triu.shape[1]
        sfc_timeseries = np.zeros(shape = (t_len,))
        if len(SC_triu) > 1: #there must be at least three electrodes in three different regions to compute correlations. If not, then set all correlations to zero
            for t in range( t_len ):
                sfc_timeseries[t] = spearmanr(  SC_triu ,  np.abs(FC_triu[:,t]) )[0]

        return sfc_timeseries
    
    def get_sfc_states(self, atlas = "AAL2", reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0):
        ii = self.get_sfc( atlas = atlas, reference = reference, state = "interictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        pi = self.get_sfc( atlas = atlas, reference = reference, state = "preictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        ic = self.get_sfc( atlas = atlas, reference = reference, state = "ictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        po = self.get_sfc( atlas = atlas, reference = reference, state = "postictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        
        return ii, pi, ic, po
        
    
    def resample(self, atlas = "AAL2", reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0, N= 100):
        ii, pi, ic, po = self.get_sfc_states( atlas = atlas, reference = reference, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        ii = signal.resample(ii, N)
        pi = signal.resample(pi, N)
        ic = signal.resample(ic, N)
        po = signal.resample(po, N)
        return ii, pi, ic, po
    
    def get_all_atlases_resample(self, reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0, N = 100):
        all_atlases = np.array([a["atlas"] for a in self.streamlines])
        
        #initialize
        array = np.zeros(shape = (N,4, len(all_atlases)))
        for a in range(len(all_atlases)):
            array[:,0, a], array[:,1, a], array[:,2, a], array[:,3, a] = self.resample(atlas = all_atlases[a], reference = reference, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, N= N)
            print(all_atlases[a])
            
        return array, all_atlases
    
    def plot_sfc(self, atlas = "AAL2", reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0, ylim = [-0.1, 0.5] ):
        ii, pi, ic, po = self.get_sfc_states( atlas = atlas, reference = reference, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        timeseries = np.concatenate([ii, pi, ic, po])
        
        fig,ax = plt.subplots(1,1,figsize=(8,4), dpi = 300)
        sns.scatterplot( x = range(len(timeseries)), y = timeseries, s = 2 ,ax=ax)
        ax.axvline(x=len(ii), ls = "--")
        ax.axvline(x=len(ii) + len(pi), ls = "--")
        ax.axvline(x=len(ii) + len(pi)+ len(ic), ls = "--")
        w=20
        y = self.movingaverage(timeseries, w)
        sns.lineplot(x = range(w-1,len(timeseries)), y =  y , ax = ax )
        ax.set_ylim(ylim)
        
    def plot_sfc_resample(self, atlas = "AAL2", reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0, ylim = [-0.1, 0.5] ):
        ii, pi, ic, po = self.resample( atlas = atlas, reference = reference, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
        timeseries = np.concatenate([ii, pi, ic, po])
        
        fig,ax = plt.subplots(1,1,figsize=(8,4), dpi = 300)
        sns.scatterplot( x = range(len(timeseries)), y = timeseries, s = 2 ,ax=ax)
        ax.axvline(x=len(ii), ls = "--")
        ax.axvline(x=len(ii) + len(pi), ls = "--")
        ax.axvline(x=len(ii) + len(pi)+ len(ic), ls = "--")
        w=15
        y = self.movingaverage(timeseries, w)
        sns.lineplot(x = range(w-1,len(timeseries)), y =  y , ax = ax )
        ax.set_ylim(ylim)
        
    def plot_struc_vs_func(self, atlas = "AAL2", reference = "CAR", state = "ictal", functional_connectivity_measure = "xcorr", frequency = 0, t = 0, s = 50, lw = 5 ):
        SC_mod, FC_mod_SC, electrode_names = self.equalize_sf( atlas = atlas, reference = reference, state = state, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)

        SC_triu = SC_mod[np.triu_indices( len(SC_mod), k=1)]
        FC_triu = FC_mod_SC[np.triu_indices( len(FC_mod_SC), k=1)]
        fig,ax = plt.subplots(1,1,figsize=(4,4), dpi = 300)
        sns.regplot(x = SC_triu,  y = np.abs(FC_triu[:,t]), ax = ax, scatter_kws = dict(s = s, edgecolors='none'), line_kws= dict(color="#902d30", lw = lw),  color="#00000077")
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([-0.05,1.05])
        pson = pearsonr(  SC_triu ,  np.abs(FC_triu[:,t]) )
        sman = spearmanr(  SC_triu ,  np.abs(FC_triu[:,t]) )
        print(   f"pearson = {pearsonr(  SC_triu ,  np.abs(FC_triu[:,t]) )[0]}; p = {pearsonr(  SC_triu ,  np.abs(FC_triu[:,t]) )[1]} " )
        print(   f"spearman = {spearmanr(  SC_triu ,  np.abs(FC_triu[:,t]) )[0]}; p = {spearmanr(  SC_triu ,  np.abs(FC_triu[:,t]) )[1]} " )
        return pson, sman

    def plot_all_atlases(self, reference = "CAR", functional_connectivity_measure = "xcorr", frequency = 0, N = 100, atlas_names = []):
        if len(atlas_names) == 1:
            atlas_names = np.array([a["atlas"] for a in self.streamlines])
 
        array, all_atlases = self.get_all_atlases_resample(reference = reference, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, N = N)   
        self.plot_all_atlases_func(array, atlas_names)     
            
    def plot_all_atlases_func(self, array, atlas_names = []):
        if len(atlas_names) == 0:
            atlas_names = np.array([a["atlas"] for a in self.streamlines])
        all_atlases = self.get_atlas_names()
        nrow, ncol = 10, 12
        fig  = plt.figure(figsize=(40,50), dpi = 150)
        ax = [None] * (nrow*ncol)
        ylim = [-0.2, 0.8]
        for a in range(1, len(all_atlases)+1):
            ax[a] = fig.add_subplot(nrow, ncol, a)
            timeseries = array[:,:, a-1]
            ii = timeseries[:, 0]
            pi = timeseries[:, 1]
            ic = timeseries[:, 2]
            po = timeseries[:, 3]
            timeseries = np.concatenate([ii, pi, ic, po])
            sns.scatterplot( x = range(len(timeseries)), y = timeseries, s = 0.5 ,ax=ax[a])
            
            ax[a].axvline(x=len(ii), ls = "--")
            ax[a].axvline(x=len(ii) + len(pi), ls = "--")
            ax[a].axvline(x=len(ii) + len(pi)+ len(ic), ls = "--")
            w=15
            y = self.movingaverage(timeseries, w)
            sns.lineplot(x = range(w-1,len(timeseries)), y =  y , ax = ax[a] )
            
            ax[a].set_yticklabels([])
            ax[a].set_xticklabels([])
            ax[a].set_ylim(ylim)
            ax[a].set_title(atlas_names[a-1], fontsize=10 )
            
    def movingaverage(self, x, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(x, window, 'valid')


