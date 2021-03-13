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
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import mannwhitneyu
import numpy as np
import copy
import json, codecs
import statsmodels.api as sm
import bct
import networkx as nx
import matplotlib.colors
from colormap import rgb2hex #  pip install colormap;  pip install easydev
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.mplot3d import Axes3D
np.seterr(divide = 'ignore')
from sklearn.linear_model import LinearRegression
from statsmodels.gam.api import GLMGam, BSplines
import pygam #  pip install pygam
from scipy import stats
from sklearn.linear_model import TweedieRegressor
from pygam import LinearGAM
from imagingToolsRevell import printProgressBar
import matplotlib.ticker as ticker
from dataclasses import dataclass
from matplotlib.offsetbox import AnchoredText
from statsmodels.stats.power import TTestIndPower

#% Input/Output Paths and File names
fname_EEG_times = ospj( path, "data","data_raw","iEEG_times","EEG_times.xlsx")
fname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")


fpath_data =  ospj(path, "data","data_processed", "aggregated_data")
fpath_figure = ospj(path, "seeg_GMvsWM","figures")


#if not (os.path.isdir(fpath_figure)): os.makedirs(fpath_figure, exist_ok=True)

#% Load Study Meta Data
data = pd.read_excel(fname_EEG_times)    
atlases = pd.read_csv(fname_atlases_csv)    
#% Processing Meta Data: extracting sub-IDs

subIDs_unique =  np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]

descriptors = ["interictal","preictal","ictal","postictal"]
references = ["CAR"]
data_ictal = data[data["descriptor"] == "ictal"]
freq_label = ["Broadband", "Delta", "Theta", "Alpha", "Beta", "Gamma - Low", "Gamma - Mid", "Gamma - High"]
N=100
#%%Debugging region
"""
sfc_data = sfc( **all_data  )

#SC_mod, FC_mod_SC, electrode_names = sfc_data.get_sfc(atlas = "HarvardOxford-sub-ONLY_maxprob-thr25-1mm")


all_atlases = np.array([a["atlas"] for a in sfc_data.streamlines])

#initialize
array_all_atlases = []
array = np.zeros(shape = (100,4))
for a in range(len(all_atlases)):
    array[:,0], array[:,1], array[:,2], array[:,3] = sfc_data.resample(atlas = all_atlases[a])
    array_all_atlases.append(copy.deepcopy(array))
    print(all_atlases[a])
"""
#%%
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





#%%Figure 4 data
fpath_figure_structure_vs_function = ospj(path, "seeg_GMvsWM","figures", "structure_vs_function")
if not (os.path.isdir(fpath_figure_structure_vs_function)): os.makedirs(fpath_figure_structure_vs_function, exist_ok=True)


i=1
N=100
all_arrays = np.zeros( shape = (N, 4 , len(atlases), len(data_ictal)) )

#parsing data DataFrame to get iEEG information
subID = data_ictal.iloc[i].RID
subRID = "sub-{0}".format(subID)
iEEG_filename = data_ictal.iloc[i].file
ignore_electrodes = data_ictal.iloc[i].ignore_electrodes.split(",")
start_time_usec = int(data_ictal.iloc[i].connectivity_start_time_seconds*1e6)
stop_time_usec = int(data_ictal.iloc[i].connectivity_end_time_seconds*1e6)
descriptor = data_ictal.iloc[i].descriptor
print( "\n\n{0}: {1}".format(subID,descriptor) )
fname = ospj(fpath_data,   f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )
if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = sfc( **all_data  ) #adding dictionary of data to dataclass sfc


atlas = "MMP_in_MNI_resliced"
functional_connectivity_measure = "pearson"
frequency = 0
ylim = [-0.2, 0.5]
sfc_data.plot_sfc_resample(atlas = atlas,functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, ylim = ylim)


sfc_data.get_atlas_names()

#sfc_data.plot_sfc(atlas = atlas,functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, ylim = ylim)


#%%Figure 4 Structure vs function plots

atlas = "AAL2"
functional_connectivity_measure = "xcorr"
frequency = 0

#%
state = "ictal"
#SC_mod, FC_mod_SC, electrode_names = sfc_data.equalize_sf( atlas = atlas, state = state, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)
#sfc_data.plot_sfc( atlas = atlas, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)

size = 60
lw = 5
pson, sman = sfc_data.plot_struc_vs_func(atlas = atlas, state = "interictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, t = 90,  s = size, lw = lw)
fname = ospj(fpath_figure_structure_vs_function , f"interictal_90_{np.round(pson[0], 2)}.pdf")
plt.savefig(fname)


pson, sman = sfc_data.plot_struc_vs_func(atlas = atlas, state = "preictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, t = 90,  s = size, lw = lw)
fname = ospj(fpath_figure_structure_vs_function , f"preictal_90_{np.round(pson[0], 2)}.pdf")
plt.savefig(fname)


pson, sman = sfc_data.plot_struc_vs_func(atlas = atlas, state = "ictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, t = 40,  s = size, lw = lw)
fname = ospj(fpath_figure_structure_vs_function , f"ictal_40_{np.round(pson[0], 2)}.pdf")
plt.savefig(fname)


pson, sman = sfc_data.plot_struc_vs_func(atlas = atlas, state = "ictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, t = 87,  s = size, lw = lw)
fname = ospj(fpath_figure_structure_vs_function , f"ictal_89_{np.round(pson[0], 2)}.pdf")
plt.savefig(fname)



pson, sman = sfc_data.plot_struc_vs_func(atlas = atlas, state = "postictal", functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, t = 177, s = size, lw = lw)
fname = ospj(fpath_figure_structure_vs_function , f"postictal_180_{np.round(pson[0], 2)}.pdf")
plt.savefig(fname)


#%%Figure 4 SFC

fpath_figure_4D = ospj(path, "seeg_GMvsWM","figures", "figure4D")
if not (os.path.isdir(fpath_figure_4D)): os.makedirs(fpath_figure_4D, exist_ok=True)



functional_connectivity_measure = "xcorr"
frequency = 0


#sfc_data.plot_all_atlases(functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, atlas_names=atlases["atlas_shortname"])

#sfc_data.plot_sfc_resample(atlas = atlases_to_plot[3],functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, ylim = ylim)


sfc_data.get_atlas_names()
atlases_to_plot =  ["AAL2",  "cc400_roi_atlas", "mni_icbm152_CerebrA_tal_nlin_sym_09c", "Hammersmith_atlas_n30r83_SPM5"]
atlases_to_plot_legend =  ["AAL2",  "Craddock400", "CerebrA", "Hammersmith"]

sns.set_style(style='white') 
fig4 = plt.figure(constrained_layout=False, dpi=300, figsize=(14, 4))
gs1 = fig4.add_gridspec(nrows=2, ncols=1, left=0.06, right=0.87, bottom=0.1, top=0.98, wspace=0.02, hspace = 0.1) #standard
axes = []
for r in range(2): #standard
    axes.append(fig4.add_subplot(gs1[r, 0]))

ylim1 = [-0.1, 0.40]
ylim2 = [-0.1, 0.40]
colors = ["#408be5", "#e59b40", "#e54840", "#238c28"]
gap = 2
label_x = 380
lw = 5
ls = ["-", "dashdot", "-", "--"]
for a in range(len(atlases_to_plot)):
    atlas_data = sfc_data.get_sfc_states( atlas = atlases_to_plot[a], functional_connectivity_measure=functional_connectivity_measure, frequency =frequency)
    timeseries_sz = np.concatenate( [  signal.resample( atlas_data[1], 100)  , signal.resample( atlas_data[2], 100),  signal.resample( atlas_data[3], 100)  ]    )
    N = 30
    timeseries_sz_avg = np.convolve(timeseries_sz, np.ones((N,))/N, mode='valid')
    timeseries_ii_avg = np.convolve( signal.resample( atlas_data[0], 100), np.ones((N,))/N, mode='valid')
    fill_array = np.empty( (1,gap+N-1)  )[0]
    fill_array[:] = np.NaN
    y =  np.concatenate([  timeseries_ii_avg, fill_array   ,  timeseries_sz_avg ,fill_array]) 
    sns.lineplot( x = range(len(y) ), y = y , legend = False, ci = 95, n_boot = 100, color= colors[a], ax = axes[0], linewidth=lw, **dict(ls = ls[a])   )
    
    if a == 0: y_pos = -0.022
    if a == 1: y_pos = 0.03
    if a == 2: y_pos = -0.07
    if a == 3: y_pos = 0.2
    #y_pos = timeseries_sz_avg[-1] * mult
    axes[0].text(x = label_x, y = y_pos, s = atlases_to_plot_legend[a], fontsize = 11, color = colors[a], fontweight = "bold" )

axes[0].set_ylim(ylim1);axes[1].set_ylim(ylim2)


sfc_data.get_atlas_names()
atlases_to_plot =  ["RandomAtlas0000030",  "RandomAtlas0000100" ,  "RandomAtlas0001000",  "RandomAtlas0010000"]
#atlases_to_plot =  ["RandomAtlas0000010",  "RandomAtlas0000100" ]
atlases_to_plot_legend =  ["30",  "100", "1,000",  "10,000"]

colors = ["#000000", "#2864ff", "#ff6961", "#999999"]
ls = ["-", "dashdot", "-", "--"]
for a in range(len(atlases_to_plot)):
    atlas_data_tmp = []
    for v in range(5): 
        atlas_data_random = sfc_data.get_sfc_states( atlas = atlases_to_plot[a]+ f"_v000{v+1}", functional_connectivity_measure=functional_connectivity_measure, frequency =frequency)
        atlas_data_tmp.append(atlas_data_random  )
    atlas_data_mean = []
    for s in range(4): 
        tmp = np.zeros(shape = (  len(atlas_data_tmp[v][s]),  5     )   )  
        for v in range(5):
            tmp[:,v] = atlas_data_tmp[v][s]
        atlas_data_mean.append(  np.nanmean( tmp, axis = 1   )  )
        
        
    timeseries_sz = np.concatenate( [  signal.resample( atlas_data_mean[1], 100)  , signal.resample( atlas_data_mean[2], 100),  signal.resample( atlas_data_mean[3], 100)  ]    )
    timeseries_sz_avg = np.convolve(timeseries_sz, np.ones((N,))/N, mode='valid')
    timeseries_ii_avg = np.convolve(signal.resample(atlas_data_mean[0],100), np.ones((N,))/N, mode='valid')
    fill_array = np.empty( (1,gap+N-1)  )[0]
    fill_array[:] = np.NaN
    y =  np.concatenate([  timeseries_ii_avg, fill_array   ,  timeseries_sz_avg ,fill_array]) 
    sns.lineplot( x = range(len(y) ), y = y , legend = False, ci = 95, n_boot = 100, color= colors[a], ax = axes[1], linewidth=lw, **dict(ls = ls[a]) )
    
    mult = 1
    if a == 0: mult = 0.6
    if a == 1: mult = 0.8
    if a == 2: mult = 0.84
    if a == 3: mult = 1.02
    y_pos = timeseries_sz_avg[-1] * mult
    axes[1].text(x = label_x, y = y_pos, s = atlases_to_plot_legend[a], fontsize = 11, fontweight = "bold")




axes[0].set_ylabel('');axes[1].set_ylabel('')  
axes[0].set_xlabel('');axes[1].set_xlabel('')




fontsize = 12
s = 0

for s in range(2):
    xlim = axes[s].get_xlim()
    
    axes[s].set(xticklabels=[], xticks=[])
    axes[s].set(yticklabels=[], yticks=[])
    
    #setting ticks

    ylim = ylim2
    axes[s].set(yticklabels=[], yticks=[0, 0.2, 0.4 ])    
    axes[s].tick_params(axis='both', which='both', length=2.75, direction = "in")
    axes[s].set(yticklabels=["0.0", 0.2, 0.4], yticks=[0, 0.2,  0.4 ]) 
    axes[s].tick_params(axis='y', which='both', length=2.75, labelcolor = "black", labelsize = fontsize)
    
    if s ==0:
        mult_factor_x = 17
        mult_factor_y =0.98
        #axes[s].text(x = (xlim[0] + 100-10 )/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "interictal", va='baseline', ha = "center", fontsize = fontsize )
        axes[s].text(x = (2*len(timeseries_ii_avg) +  2*gap + 100+int(N) )/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "Preictal", va='top', ha = "center", fontsize = fontsize)
        axes[s].text(x = (2*len(timeseries_ii_avg) + 2*gap + 200 + 100+int(N))/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "Ictal", va='top', ha = "center", fontsize = fontsize)
        #axes[s].text(x = ( 300 + xlim[1])/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "postictal", va='baseline', ha = "center", fontsize = fontsize)  
        #axes[s].text(x = (xlim[0] + 100-10 )/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "Standard Atlases", va='top', ha = "center", fontsize = fontsize )
        axes[s].text(x = label_x, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "Standard\nAtlases", va='top', fontsize = 11, fontweight = "bold" )

    
    xticks=[25, len(timeseries_ii_avg) + gap+int(N/2),  len(timeseries_ii_avg) +  gap + 100+int(N/2),  len(timeseries_ii_avg) +  gap + 200+int(N/2), len(y) - len(fill_array) ]
    axes[s].xaxis.set_major_locator(ticker.FixedLocator(xticks))
    axes[s].xaxis.set_ticks(xticks)
    if s ==1:
        axes[s].set(xticklabels=["~6 hrs before", "-90", "0", "90", "180"], xticks=xticks) 
        axes[s].tick_params(axis='x', which='both', length=0, labelcolor = "black", labelsize = fontsize)
        
     
        axes[s].text(x = label_x, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "Random\nAtlases", va='top', fontsize = 11 , fontweight = "bold")
    
    vertical_line = [len(timeseries_ii_avg) + gap +int(N/2), len(timeseries_ii_avg) +  gap + 100 +int(N/2), len(timeseries_ii_avg) +  gap + 200 +int(N/2) ]
    for x in vertical_line:
        axes[s].axvline(x, color='k', linestyle='--', lw = 0.5)
    
    
    ylim = axes[s].get_ylim()
    at = matplotlib.patches.FancyBboxPatch( ( len(timeseries_ii_avg) + gap+int(N/2), ylim[0]  ) , width= 100, height= (ylim[1] - ylim[0]) , color = "#33339933"  )
    axes[s].add_artist(at)
    at = matplotlib.patches.FancyBboxPatch( ( len(timeseries_ii_avg) +  gap + 100+int(N/2), ylim[0]  ) , width= 100, height= (ylim[1] - ylim[0]) , color = "#99333333"  )
    axes[s].add_artist(at)
        
sns.despine()


fig4.text(0.5, 0.01, 'Time (s) ', ha='center', fontsize = fontsize)
fig4.text(0.02, 0.5, 'Spearman Rank Correlation', ha='left',  va = "center", rotation = 'vertical', fontsize = fontsize)



fname = ospj(fpath_figure_4D , f"{sfc_data.subID}_figure4D.pdf")
plt.savefig(fname)





#%%Figure 5 data



all_arrays = np.zeros( shape = (N, 4 , len(atlases), len(data_ictal)) )
for i in range(0, len(data_ictal)):
    #parsing data DataFrame to get iEEG information
    subID = data_ictal.iloc[i].RID
    subRID = "sub-{0}".format(subID)
    iEEG_filename = data_ictal.iloc[i].file
    ignore_electrodes = data_ictal.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data_ictal.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data_ictal.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data_ictal.iloc[i].descriptor
    print( "\n\n{0}: {1}".format(subID,descriptor) )
    fname = ospj(fpath_data,   f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )
    if (os.path.exists(fname)):
        with open(fname, 'rb') as f: all_data = pickle.load(f)

    sfc_data = sfc( **all_data  ) #adding dictionary of data to dataclass sfc

    functional_connectivity_measure = "xcorr"
    frequency = 0
    all_arrays[:,:,:,i], all_atlases = sfc_data.get_all_atlases_resample(functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)



    
#%%turn to data frame

gap = 20
for i in range(len(subIDs_unique)):
    for a in range(len(all_atlases)):
        ind_choose = np.concatenate( [ np.arange(N) ,  np.arange(len(np.arange(N)) + gap, len(np.arange(N)) + gap+ N), np.arange(len(np.arange(N)) + gap+ N, len(np.arange(N)) + gap+ N + N) ,  np.arange(len(np.arange(N)) + gap+ N + N, len(np.arange(N)) + gap+ N + N + N)  ])
        pd.DataFrame(  np.reshape(all_arrays[range(100),0,a,i], (1,100)),  columns=ind_choose[range(0,100)])
       
        df1 = pd.concat([pd.Series( [subIDs_unique[i]  ], name = "subID"), pd.Series([all_atlases[a]  ], name = "atlas"), pd.Series([0], name = "state"), pd.DataFrame(  np.reshape(all_arrays[range(100),0,a,i], (1,100)),  columns=ind_choose[range(0,100)]) ] , axis = 1 )
        df1 = pd.melt(df1,id_vars=["subID", "atlas", "state"], var_name='index', value_name='SFC')
        df2 = pd.concat([pd.Series( [subIDs_unique[i]  ], name = "subID"), pd.Series([all_atlases[a]  ], name = "atlas"), pd.Series([1], name = "state"), pd.DataFrame(  np.reshape(all_arrays[range(100),1,a,i], (1,100)),  columns=ind_choose[range(100,200)]) ] , axis = 1 )
        df2 = pd.melt(df2,id_vars=["subID", "atlas", "state"], var_name='index', value_name='SFC')
        df3 = pd.concat([pd.Series( [subIDs_unique[i]  ], name = "subID"), pd.Series([all_atlases[a]  ], name = "atlas"), pd.Series([2], name = "state"), pd.DataFrame(  np.reshape(all_arrays[range(100),2,a,i], (1,100)),  columns=ind_choose[range(200,300)]) ] , axis = 1 )
        df3 = pd.melt(df3,id_vars=["subID", "atlas", "state"], var_name='index', value_name='SFC')
        df4 = pd.concat([pd.Series( [subIDs_unique[i]  ], name = "subID"), pd.Series([all_atlases[a]  ], name = "atlas"), pd.Series([3], name = "state"), pd.DataFrame(  np.reshape(all_arrays[range(100),3,a,i], (1,100)),  columns=ind_choose[range(300,400)]) ] , axis = 1 )
        df4 = pd.melt(df4,id_vars=["subID", "atlas", "state"], var_name='index', value_name='SFC')
        if i ==0 and a ==0:
            df_data = pd.concat([df1, df2,df3,df4] )
        else:
            df_tmp = pd.concat([df1, df2,df3,df4] )
            df_data = df_data.append( df_tmp)
        print(f"{subIDs_unique[i]}; {all_atlases[a]} ")


fname = ospj( path, f"data/data_processed/dataframes/data_{functional_connectivity_measure}_freq{frequency}.csv")

    
with open(fname, 'wb') as f: pickle.dump(df_data, f)
#%%Figure 5

atlases_to_plot =  ["AAL2", 
                    "HarvardOxford-combined",
                    "AAL600", 
                    "BN_Atlas_246_1mm", 
                    "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm", 
                    "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_resliced",
                    "Hammersmith_atlas_n30r83_SPM5",
                    "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2", 
                    "MMP_in_MNI_resliced",
                    "AAL_JHU_combined", 
                    "RandomAtlas0000030_v0005",
                    "RandomAtlas0000050_v0001",
                    "RandomAtlas0000100_v0001",
                    "RandomAtlas0001000_v0001",
                    "RandomAtlas0010000_v0001"]
color = sns.color_palette("coolwarm", 4)
tmp = color[2]
color[2] = color[3]
color[3] = tmp
tmp = color[0]
color[0] = color[1]
color[1] = tmp

sns.set_style("dark", {"ytick.left": True, "xtick.bottom": True })
sns.set_context("paper", font_scale=0.6, rc={"lines.linewidth": 0.75})


fig5 = plt.figure(constrained_layout=False, dpi=300, figsize=(6.5, 3.5))
gs1 = fig5.add_gridspec(nrows=3, ncols=5, left=0.05, right=0.99, bottom=0.07, top=0.95, wspace=0.02, hspace = 0.1) #standard
axes = []
for r in range(3): #standard
    for c in range(5):
        axes.append(fig5.add_subplot(gs1[r, c]))

fontsize = 6.5
ylim = [-0.1, 0.45]
t=0


for s in range(len(axes)):
 
        
    
    df_plot =   df_data[df_data.atlas.eq(atlases_to_plot[s])]
    sns.lineplot(data =df_plot,  x = "index", y = "SFC", hue= "state" ,  ax = axes[s], legend = False, ci = 75, n_boot = 50, palette= color)
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes[s].set_ylim(ylim)
    axes[s].set_ylabel('')    
    axes[s].set_xlabel('')
    
    #writing interictal, pretictal, ictal, postictal labels
    xlim = axes[s].get_xlim()
    if s == 0:
        mult_factor_x = 17
        mult_factor_y = 0.1
        axes[s].text(x = (xlim[0] + 100 + gap)/2,        y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "inter", va='baseline', ha = "center", fontsize = fontsize)
        axes[s].text(x = (-xlim[0] + gap + 100 + 200)/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "pre", va='baseline', ha = "center", fontsize = fontsize)
        axes[s].text(x = (-xlim[0] + gap + 200 + 300)/2, y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "ictal", va='baseline', ha = "center", fontsize = fontsize)
        axes[s].text(x = (gap + 300 + xlim[1])/2,        y = ylim[0]+ (ylim[1] - ylim[0])*mult_factor_y, s = "post", va='baseline', ha = "center", fontsize = fontsize)  
        
    vertical_line = [100+ gap, 200 + gap, 300 + gap]
    
    for x in vertical_line:
        axes[s].axvline(x, color='k', linestyle='--', lw = 0.5)
    
    axes[s].set(xticklabels=[], xticks=[])
    axes[s].set(yticklabels=[], yticks=[])
    axes[s].set(yticklabels=[], yticks=[0, 0.1 ,0.2, 0.3, 0.4 ])    
    axes[s].tick_params(axis='both', which='both', length=2.75, direction = "in")
    #setting ticks
    if s ==0:
        axes[s].set(yticklabels=["0.0", 0.1, 0.2, 0.3, 0.4], yticks=[0, 0.1 ,0.2, 0.3, 0.4 ]) 
        axes[s].tick_params(axis='y', which='both', length=2.75, labelcolor = "black", labelsize = fontsize)
    
    if s == 12:
        xticks=[100 +gap, 200+gap, 300 + gap ]
        axes[s].xaxis.set_major_locator(ticker.FixedLocator(xticks))
        axes[s].xaxis.set_ticks(xticks)
        axes[s].set(xticklabels=["-100", "0", "100"], xticks=xticks) 
        axes[s].tick_params(axis='x', which='both', length=0, labelcolor = "black", labelsize = fontsize) 
    #Setting plot titles
    #axes[s].text(x = (xlim[0] + xlim[1])/2, y = ylim[1] + (ylim[1] - ylim[0])*0.15 , fontsize = 7, s = atlas_labels[t], va='top', ha = "center")
    
    #Writing rsSFC
    if s ==0:
        axes[s].text((xlim[0] + 100 + gap)/2, y = ylim[0]+ (ylim[1] - ylim[0])*0.95 , fontsize = fontsize, s = "rsSFC", va='top', ha = "center")
        axes[s].text( (-xlim[0] + gap + 100 + 200)/2, y = ylim[0]+ (ylim[1] - ylim[0])*0.95 , fontsize = fontsize, s = r'$\Delta$SFC', va='top', ha = "center")
        arrow_properties = dict(color="black",arrowstyle= '-|>', lw=0.5)
        x1 = (xlim[0] + 100 + gap)/2
        y1 = ylim[0]+ (ylim[1] - ylim[0])*0.85
        x2 = (xlim[0] + 100 + gap)/2
        y2 = ylim[0]+ (ylim[1] - ylim[0])*0.48
        axes[s].annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_properties)
    
        x1 = (-xlim[0] + gap + 100 + 200+55)/2;    y1 = ylim[0]+ (ylim[1] - ylim[0])*0.52
        x2 = (-xlim[0] + gap + 100 + 200+80)/2;    y2 = ylim[0]+ (ylim[1] - ylim[0])*0.52
        axes[s].plot([x1, x2], [y1, y2], '-', lw=0.5, color="black")
       
        x1 = (-xlim[0] + gap + 100 + 200+55)/2;    y1 = ylim[0]+ (ylim[1] - ylim[0])*0.70
        x2 = (-xlim[0] + gap + 100 + 200+80)/2;    y2 = ylim[0]+ (ylim[1] - ylim[0])*0.70
        axes[s].plot([x1, x2], [y1, y2], '-', lw=0.5, color="black")
       
        x1 = (-xlim[0] + gap + 100 + 200+55)/2;    y1 = ylim[0]+ (ylim[1] - ylim[0])*0.52
        x2 = (-xlim[0] + gap + 100 + 200+55)/2;    y2 = ylim[0]+ (ylim[1] - ylim[0])*0.70
        axes[s].plot([x1, x2], [y1, y2], '-', lw=0.5, color="black")
        
        x1 = (-xlim[0] + gap + 100 + 200+10)/2;    y1 = ylim[0]+ (ylim[1] - ylim[0])*(0.52+0.70)/2
        x2 = (-xlim[0] + gap + 100 + 200+55)/2;    y2 = ylim[0]+ (ylim[1] - ylim[0])*(0.52+0.70)/2
        axes[s].plot([x1, x2], [y1, y2], '-', lw=0.5, color="black")
        
        x1 = (-xlim[0] + gap + 100 + 200+10)/2;    y1 = ylim[0]+ (ylim[1] - ylim[0])*(0.52+0.70)/2
        x2 = (-xlim[0] + gap + 100 + 200+10)/2;    y2 = ylim[0]+ (ylim[1] - ylim[0])*0.83
        axes[s].plot([x1, x2], [y1, y2], '-', lw=0.5, color="black")
        
        
        
    kwargs = {'lw':0.5}
    at = AnchoredText(s+1,loc='upper right', prop=dict(size=7), frameon=True)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.5" )
    at.patch.set_linewidth(0.5)
    axes[s].add_artist(at)
        
    t = t + 1
    print(s)




fig5.text(0.52, 0.01, 'Time: Normalized to 100; Seizure start = 0', ha='center', fontsize = fontsize)
fig5.text(0.02, 0.45, 'Spearkman Rank Correlation', ha='left',  va = "center", rotation = 'vertical', fontsize = fontsize)

    
fig5.text(0.5, 0.99, f'Average All Seizures {functional_connectivity_measure} ({freq_label[frequency]})', ha='center', va = "top", fontsize = 8)

fpath_figure_5 = ospj(path, "seeg_GMvsWM","figures", "figure5")
fname = ospj(fpath_figure_5 , f"figure5_{functional_connectivity_measure}_{freq_label[frequency]}.pdf")
plt.savefig(fname)


#%%Figure 6




#Grouping/averaging each period's 100 points together
Data_mean = df_data.groupby(['subID','atlas', "state"])["SFC"].mean().reset_index()

#Getting volumes and sphericity data
volumes_and_sphericity_means    =  pd.read_csv( ospj(path, "data/data_processed/atlas_morphology/volumes_and_sphericity.csv"), sep=",")    
volumes_and_sphericity_means.columns = ["atlas_name", volumes_and_sphericity_means.columns[1], volumes_and_sphericity_means.columns[2]]
volumes_and_sphericity_means.volume_voxels = np.log10(volumes_and_sphericity_means.volume_voxels)
volumes_and_sphericity_means = pd.merge(volumes_and_sphericity_means,atlases, on="atlas_name"  )
volumes_and_sphericity_means["atlas_filename"] = [os.path.splitext(x)[0] for x in volumes_and_sphericity_means["atlas_filename"]]
volumes_and_sphericity_means["atlas_filename"] = [os.path.splitext(x)[0] for x in volumes_and_sphericity_means["atlas_filename"]]
volumes_and_sphericity_means = volumes_and_sphericity_means.drop(["atlas_name", "atlas_shortname"], axis = 1)
volumes_and_sphericity_means.columns = [volumes_and_sphericity_means.columns[0], volumes_and_sphericity_means.columns[1], "atlas"]

#adding volume and sphericity data to the mean data
Data_mean = pd.merge(Data_mean,volumes_and_sphericity_means, on="atlas"  )



#Getting just interictal data for figure 7A (baseline SFC)
Data_mean_interictal = Data_mean[Data_mean.state.eq(0)]


#Getting random atlas data
Data_mean_interictal_RandomAtlas = Data_mean_interictal.iloc[np.where(Data_mean_interictal["atlas"].str.contains("RandomAtlas"))[0],:]
Data_mean_interictal_RandomAtlas = Data_mean_interictal_RandomAtlas.iloc[np.where(~Data_mean_interictal_RandomAtlas["atlas"].str.contains("RandomAtlas0000010"))[0],:].reset_index(drop = True)


#Getting standard atlas data
Data_mean_interictal_StandardAtlas = Data_mean_interictal.iloc[np.where(~Data_mean_interictal["atlas"].str.contains("RandomAtlas"))[0],:]
#remove JHU
#Data_mean_interictal_broadband_StandardAtlas = Data_mean_interictal_broadband_StandardAtlas.iloc[np.where(~Data_mean_interictal_broadband_StandardAtlas["atlas"].str.contains("JHU_res-1x1x1"))[0],:]


#Calculating Differences between ictal and preictal SFC

#getting preictal and ictal data
Data_mean_preictal = Data_mean[Data_mean.state.eq(1)]
Data_mean_ictal = Data_mean[Data_mean.state.eq(2)]

#Subtracting and making new dataframe with the difference
Difference = np.array(Data_mean_ictal.SFC) -  np.array(Data_mean_preictal.SFC)
Data_mean_difference = copy.deepcopy(Data_mean_ictal)
Data_mean_difference.SFC = Difference


#Getting random atlas data
Data_mean_difference_RandomAtlas = Data_mean_difference.iloc[np.where(Data_mean_difference["atlas"].str.contains("RandomAtlas"))[0],:]
Data_mean_difference_RandomAtlas = Data_mean_difference_RandomAtlas.iloc[np.where(~Data_mean_difference_RandomAtlas["atlas"].str.contains("RandomAtlas0000010"))[0],:].reset_index(drop = True)

#Getting standard atlas data
Data_mean_difference_StandardAtlas = Data_mean_difference.iloc[np.where(~Data_mean_difference["atlas"].str.contains("RandomAtlas"))[0],:]
#remove JHU
#Data_mean_difference_StandardAtlas = Data_mean_difference_broadband_StandardAtlas.iloc[np.where(~Data_mean_difference_broadband_StandardAtlas["atlas"].str.contains("JHU_res-1x1x1"))[0],:]

all_atlases


#%%
atlases_to_plot =  [
                    "HarvardOxford-combined",
                    "HarvardOxford-cort-NONSYMMETRIC-maxprob-thr25-1mm", 
                    "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm",
                    "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2", 
                    "AAL2", 
                    "AAL600", 
                    "BN_Atlas_246_1mm", 
                    "MMP_in_MNI_resliced",
                    "cc400_roi_atlas",
                    "Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm",
                    "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_resliced",
                    "Hammersmith_atlas_n30r83_SPM5",
                    ]


standard_atlas_names_labels = [
                    "HO-combined",
                    "HO Cort Only",
                    "Schaefer 100",
                    "DKT31 OASIS", 
                    "AAL2", 
                    "AAL600", 
                    "BNA", 
                    "MMP",
                    "Craddock 400   ",
                    "Schaefer 1000",
                    "Yeo 17",
                    "Hammersmith",
                    ]


atlas_labels = [os.path.splitext(x)[0] for x in atlases["atlas_filename"]]
atlas_labels = [os.path.splitext(x)[0] for x in atlas_labels]





colors_pre = sns.hls_palette(len(atlases_to_plot), l=.5, s=0.4)
colors_ic = sns.hls_palette(len(atlases_to_plot), l=.8, s=0.6)

colors_pre = ((colors_pre[0],) * len(atlases_to_plot)  )
colors_ic = ((colors_ic[0],) * len(atlases_to_plot)  )
random_color = sns.color_palette("Blues_r", 10)[0]

 
    
markersize = 3
    
fontsize = 8
axes = []
sns.set_style("dark", {"ytick.left": True, "xtick.bottom": True })
sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 0.75})
 
fig6 = plt.figure(constrained_layout=False, dpi=300, figsize=(6.4, 4))
gs1 = fig6.add_gridspec(nrows=1, ncols=1, left=0.08, right=0.4, bottom=0.52, top=0.89, wspace=0.02, hspace = 0.2)
gs2 = fig6.add_gridspec(nrows=1, ncols=1, left=0.08, right=0.4, bottom=0.1, top=0.5, wspace=0.02, hspace = 0.05)
gs3 = fig6.add_gridspec(nrows=2, ncols=6, left=0.45, right=0.99, bottom=0.52, top=0.89, wspace=0.05, hspace=0.1)
gs4 = fig6.add_gridspec(nrows=1, ncols=6, left=0.45, right=0.99, bottom=0.31, top=0.50, wspace=0.05, hspace=0.1)
gs5 = fig6.add_gridspec(nrows=1, ncols=7, left=0.45, right=0.99, bottom=0.1, top=0.29, wspace=0.05, hspace=0.05)

axes.append(fig6.add_subplot(gs1[:, :]))
axes.append(fig6.add_subplot(gs2[:, :]))
for r in range(2): #standard
    for c in range(6):
        axes.append(fig6.add_subplot(gs3[r, c]))
for r in range(1): #random
    for c in range(6):
        axes.append(fig6.add_subplot(gs4[r, c]))
for r in range(1): #random
    for c in range(7):
        axes.append(fig6.add_subplot(gs5[r, c]))
#Fig 6A
df = Data_mean_interictal_RandomAtlas
sns.lineplot(data =df,  x = "volume_voxels", y = "SFC" , legend = False, color= random_color,  linewidth=1, alpha = 0.7, ci = 95, marker="o", ax = axes[0], markeredgewidth=0, ms = markersize)

for a in range(12):
    df = Data_mean_interictal_StandardAtlas[Data_mean_interictal_StandardAtlas.atlas.eq(atlases_to_plot[a]  )]
    sns.lineplot(data =df,  x = "volume_voxels", y = "SFC" ,color = np.array(colors_pre)[a]  ,  linewidth=2, alpha = 0.9, err_style="bars", marker="o", ax = axes[0], markeredgewidth=0, ms = markersize)

#Fig7B
df = Data_mean_difference_RandomAtlas
sns.lineplot(data =df,  x = "volume_voxels", y = "SFC" , legend = False, color= random_color,  linewidth=1, alpha = 0.7, ci = 95, marker="o", ax = axes[1], markeredgewidth=0, ms = markersize)

for a in range(12):
    df = Data_mean_difference_StandardAtlas[Data_mean_difference_StandardAtlas.atlas.eq(atlases_to_plot[a]  )]
    sns.lineplot(data =df,  x = "volume_voxels", y = "SFC" ,color = np.array(colors_pre)[a]  ,  linewidth=2, alpha = 0.9, err_style="bars", marker="o", ax = axes[1], markeredgewidth=0, ms = markersize)

alpha = 0.1
N_F = 14#* len(freq_label)

ylim1 = [-0.025, 0.35]
ylim2 = [-0.015, 0.175]
ylim3 = [-0.05, 0.6]
xlim = axes[0].get_xlim()
axes[0] .set_ylim(ylim1)
axes[0].set(xticklabels=[], xticks=[2.5,3,3.5,4,4.5])
axes[0].set(yticklabels=["0.00", "0.10" ,"0.20", "0.30"], yticks=[0, 0.1 ,0.2, 0.3 ])  
axes[0].set_ylabel('rsSFC',fontsize=fontsize)    
axes[0].set_xlabel('')
axes[0].text((xlim[0]+xlim[1])/2, ylim1[0] + (ylim1[1] - ylim1[0])*1.1, 'resting-state SFC', ha='center', va = "top", fontsize = 10)    
axes[0].text(xlim[0], ylim1[0] + (ylim1[1] - ylim1[0])*1.12, 'A.', ha='left', va = "top", fontsize = 15, fontweight = 'bold')    
axes[0].tick_params(axis='x', which='both', length=3, labelcolor = "black", labelsize = 8, direction = "out")
axes[0].tick_params(axis='y', which='both', length=3, labelcolor = "black", labelsize = 8, direction = "out")

xlim = axes[1].get_xlim()
axes[1] .set_ylim(ylim2)
axes[1].set(xticklabels=[2.5,3,3.5,4,4.5], xticks=[2.5,3,3.5,4,4.5])
axes[1].set(yticklabels=["0.00", 0.05, "0.10", 0.15], yticks=[0, 0.05, 0.1, 0.15]) 
axes[1].set_ylabel(r'$\Delta$SFC',fontsize=fontsize)    
axes[1].set_xlabel(r'Volume $(log_{10} mm^3)$',fontsize=fontsize)
axes[1].text((xlim[0]+xlim[1])/2,  ylim2[0] + (ylim2[1] - ylim2[0])*0.96, r'$\Delta$SFC', ha='center', va = "top", fontsize = 10)    
axes[1].text(xlim[0],  ylim2[0] + (ylim2[1] - ylim2[0])*0.98, 'C.', ha='left', va = "top", fontsize = 15, fontweight = 'bold')    
axes[1].tick_params(axis='x', which='both', length=3, labelcolor = "black", labelsize = 8, direction = "out")
axes[1].tick_params(axis='y', which='both', length=3, labelcolor = "black", labelsize = 8, direction = "out")
 

fs = 4
axes[0].text(2.97, 0.25, "Schaefer\n1000", ha='right', va = "bottom", fontsize = fs)
axes[0].text(4.55, 0.17, "Yeo 17", ha='left', va = "bottom", fontsize = fs)
axes[0].text(4.35, 0.25, "Hammersmith", ha='left', va = "bottom", fontsize = fs)
axes[0].text(4.15, 0.135, "AAL2", ha='left', va = "bottom", fontsize = fs)
axes[1].text(4.82, -0.008, "Yeo 17", ha='right', va = "bottom", fontsize = fs)
axes[1].text(4.24, -0.008, "Hammersmith", ha='right', va = "bottom", fontsize = fs)
axes[1].text(4.15, 0.105, "AAL2", ha='left', va = "bottom", fontsize = fs)
axes[1].text(2.99, 0.069, "Schaefer\n1000", ha='right', va = "bottom", fontsize = fs)
#%$
#Standard Atlas
s= 2
for a in range(len(atlases_to_plot) ):
    
    Data_mean_atlas = Data_mean[Data_mean.atlas.eq(atlases_to_plot[a])]

    df = copy.deepcopy(Data_mean_atlas)
    sns.violinplot(x="state", y="SFC", data=df, order = [ 1, 2], inner="quartile", ax = axes[s], palette=[colors_pre[a], colors_ic[a]])
    

    handles, labels = axes[s].get_legend_handles_labels()
    axes[s].legend(handles[:], labels[:])
    axes[s].legend([],[], frameon=False)
    axes[s] .set_ylim(ylim3) 
    axes[s].tick_params(axis='both', which='both', length=3, direction = "in")
    axes[s].set_ylabel('')    
    axes[s].set_xlabel('')
    
    xlim = axes[s].get_xlim()
    ylim = axes[s].get_ylim()
    if s == 2:
        axes[s].text(xlim[0]-1, ylim1[0] + (ylim[1] - ylim[0])*1.2, 'B.', ha='left', va = "top", fontsize = 15, fontweight = 'bold')    

    axes[s].set(xticklabels=[], xticks=[])
    axes[s].set(yticklabels=[], yticks=[0, 0.1 ,0.2, 0.3, 0.4, 0.5,0.6])  
    #title
    axes[s].text(xlim[0]+1, ylim[1]*1.1, standard_atlas_names_labels[a], ha='center', va = "top", fontsize = 6) 
    #np.array(atlases["atlas_shortname"].iloc[np.array(atlas_labels) == atlases_to_plot[a] ])[0]
    #stats
    group1 = df.where(df.state== 1).dropna()['SFC']
    group2 = df.where(df.state== 2).dropna()['SFC']
    difference =  (np.mean(group2)-np.mean(group1))
    effect_size_cohensD = (np.mean(group2)-np.mean(group1))   / ((   (  (np.std(group1)**2) + (np.std(group2)**2)     )/2)**0.5)  
  
    alpha_pwr = 0.05/(43*5)
    power = 0.8
 
    
 
    
    p_value = stats.wilcoxon(group1, group2)[1]
    analysis = TTestIndPower()
    sample_size  = analysis.solve_power(effect_size_cohensD, power=power, nobs1=None, ratio=1.0, alpha=alpha_pwr)
    #print(f"{atlases_to_plot[a]} p-value: {np.round(p_value,2)}      { p_value < alpha/N_F}   sample size: {np.round(sample_size,0)}       effect size: {effect_size_cohensD}    difference: {difference}")
    print(f"{atlases_to_plot[a]} p-value: {np.round(p_value,2)}          { p_value < alpha/N_F} ") 
    if p_value *N_F < alpha:
        axes[s].text(xlim[0]+1, ylim[0] + ( ylim[1]-ylim[0])*1.15, "\n*", ha='center', va = "top", fontsize = 15, fontweight = 'bold')    

    s = s +1



#Random Atlas

all_atlases
atlases_to_plot =  [
                    "RandomAtlas0000030_v0001",
                    "RandomAtlas0000075_v0001",
                    "RandomAtlas0000050_v0001", 
                    "RandomAtlas0000100_v0001", 
                    "RandomAtlas0000200_v0002", 
                    "RandomAtlas0000300_v0002",
                    "RandomAtlas0000400_v0002",
                    "RandomAtlas0000500_v0002", 
                    "RandomAtlas0000750_v0002", 
                    "RandomAtlas0001000_v0001",
                    "RandomAtlas0002000_v0001", 
                    "RandomAtlas0005000_v0001",
                    "RandomAtlas0010000_v0001"]

random_atlas_names_labels = [ "30", "50", "75", "100", "200", "300", "400", "500", "750", "1,000", "2,000", "5,000", "10,000"]


for a in range(len(atlases_to_plot) ):
    
    Data_mean_atlas = Data_mean[Data_mean.atlas.eq(atlases_to_plot[a])]
    df = copy.deepcopy(Data_mean_atlas)
    sns.violinplot(x="state", y="SFC", data=df, order = [ 1, 2], inner="quartile", ax = axes[s], palette="Blues_r")
    
#%
    handles, labels = axes[s].get_legend_handles_labels()
    axes[s].legend(handles[:], labels[:])
    axes[s].legend([],[], frameon=False)
    axes[s] .set_ylim(ylim3)
    axes[s].tick_params(axis='both', which='both', length=3, direction = "in")
    axes[s].set_ylabel('')    
    axes[s].set_xlabel('')

    xlim = axes[s].get_xlim()
    ylim = axes[s].get_ylim()
    if s == 23:
        axes[s].set(xticklabels=["pre  ", "  ictal"])
        #axes[s].text(xlim[0]+0.5, ylim[0], 'pre', ha='center', va = "bottom", fontsize = 7)    
        #axes[s].text(xlim[0]+1.5, ylim[0], 'ictal', ha='center', va = "bottom", fontsize = 7)    
        axes[s].tick_params(axis='x', which='both', length=3, labelcolor = "black", labelsize = 8, direction = "out")
        axes[s].set_xlabel(r'Preictal vs Ictal SFC ($\Delta$SFC)',fontsize=8)
        axes[s].set(yticklabels=[], yticks=[0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6])  
    elif s == 20:
        axes[s].set(yticklabels=["0.0", 0.1, 0.2, 0.3, 0.4, 0.5], yticks=[0, 0.1 ,0.2, 0.3,0.4, 0.5,0.6])  
        axes[s].tick_params(axis='y', which='both', length=3, labelcolor = "black", labelsize = 8, direction = "in")
        axes[s].set(xticklabels=[], xticks=[])
    else:
        axes[s].set(xticklabels=[], xticks=[])
        axes[s].set(yticklabels=[], yticks=[0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6])  
    #title
    axes[s].text(xlim[0]+1, ylim[1]*0.97, random_atlas_names_labels[a] , ha='center', va = "top", fontsize = 6)    
    

    
    #stats
    group1 = df.where(df.state== 1).dropna()['SFC']
    group2 = df.where(df.state== 2).dropna()['SFC']
    difference =  (np.mean(group2)-np.mean(group1))
    effect_size_cohensD = (np.mean(group2)-np.mean(group1))   / ((   (  (np.std(group1)**2) + (np.std(group2)**2)     )/2)**0.5)  
    alpha_pwr = 0.05
    power = 0.8
 
    
    p_value = stats.wilcoxon(group1, group2)[1]
    analysis = TTestIndPower()
    sample_size  = analysis.solve_power(effect_size_cohensD, power=power, nobs1=None, ratio=1.0, alpha=alpha_pwr)
    #if reaches statistical significance 
    print("{0} p-value: {1}      {2}   sample size:{3}       effect size: {4}    difference: {5}".format(random_atlas_names_labels[a], p_value,  p_value < alpha/N_F, sample_size, effect_size_cohensD, difference))
    if p_value *N_F < alpha:
        axes[s].text(xlim[0]+1, ( ylim[1]-ylim[0])*1.1, "\n*", ha='center', va = "top", fontsize = 15, fontweight = 'bold')    
    
    s = s +1
    
    
    
 
    
 
          
       
fig6.text(0.5, 1, f'rsSFC and $\Delta$SFC', ha='center', va = "top")
fpath_figure_6 = ospj(path, "seeg_GMvsWM","figures", "figure6")
fname = ospj(fpath_figure_6 , f"figure6_{functional_connectivity_measure}_{freq_label[frequency]}.pdf")
plt.savefig(fname)
        

    
    
    




#%%subsequent analysis
sfc_data = sfc( **all_data  ) #adding dictionary of data to dataclass sfc
all_atlases = np.array([a["atlas"] for a in sfc_data.streamlines]) 


atlas = "AAL2"
functional_connectivity_measure = "xcorr"
frequency = 0
ylim = [-0.2, 0.5]

ii, pi, ic, po = sfc_data.get_sfc_states( atlas = atlas, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency)


#sfc_data.plot_sfc(atlas = atlas,functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, ylim = ylim)
sfc_data.plot_sfc_resample(atlas = atlas,functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, ylim = ylim)

#%%
state = "ictal"
sfc_data.plot_struc_vs_func(atlas = atlas, state = state, functional_connectivity_measure = functional_connectivity_measure, frequency = frequency, t = 20)

#%%


























def movingaverage(x, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(x, window, 'valid')

nrow, ncol = 10, 12
fig  = plt.figure(figsize=(40,50), dpi = 300)
ax = [None] * (nrow*ncol)
ylim = [-0.2, 0.8]
for a in range(1, len(all_atlases)+1):
    ax[a] = fig.add_subplot(nrow, ncol, a)
    timeseries = array[:,:,a-1]
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
    y = sfc_data.movingaverage(timeseries, w)
    sns.lineplot(x = range(w-1,len(timeseries)), y =  y , ax = ax[a] )
    
    
    ax[a].set_yticklabels([])
    ax[a].set_xticklabels([])
    ax[a].set_ylim(ylim)
    ax[a].set_title(atlases["atlas_shortname"][a-1], fontsize=10 )


#%%
    
    
    #%%
descriptors = ["interictal","preictal","ictal","postictal"]
references = ["CAR"]
data_ictal = data[data["descriptor"] == "ictal"]
i=1
N=100
