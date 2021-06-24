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

import pickle
import numpy as np
import os
import sys
from os.path import join as ospj
path = ospj("/media","arevell","sharedSSD","linux","papers","paper005") #Parent directory of project
path = ospj("E:\\","linux","papers","paper005") #Parent directory of project
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
import pingouin as pg
import statsmodels.api as sm
import bct
import networkx as nx
import matplotlib.colors
from colormap import rgb2hex #pip install colormap;  pip install easydev
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.mplot3d import Axes3D
np.seterr(divide = 'ignore')
from sklearn.linear_model import LinearRegression
from statsmodels.gam.api import GLMGam, BSplines
import pygam #pip install pygam

#%%
#from NetworkX:

def circular_layout(G, scale=1, center=None, dim=2):
    """Position nodes on a circle."""
    import numpy as np
    if dim < 2:
        raise ValueError("cannot handle dimensions < 2")
    paddims = max(0, (dim - 2))
    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        theta = np.linspace(1, 2, len(G) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack(
            [-np.cos(theta), np.sin(theta), np.zeros((len(G), paddims))]
        )
        pos = dict(zip(G, pos))
    return pos




#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data","data_raw","iEEG_times","EEG_times.xlsx")
ifpath_electrode_localization = ospj( path, "data","data_processed","electrode_localization")

ifpath_FC = ospj(path, "data","data_processed","connectivity_matrices","function")
ifpath_SC = ospj(path, "data","data_processed","connectivity_matrices","structure")


ofpath_FC_analysis = ospj(path, "data","data_processed","FC_analysis","montage","referential","filtered","original")
ofpath_figure = ospj(path, "seeg_GMvsWM","figures","networks")
ofpath_figure_dataMining = ospj(path, "seeg_GMvsWM","figures","data_mining")

if not (os.path.isdir(ofpath_FC_analysis)): os.makedirs(ofpath_FC_analysis, exist_ok=True)
if not (os.path.isdir(ofpath_figure)): os.makedirs(ofpath_figure, exist_ok=True)
if not (os.path.isdir(ofpath_figure_dataMining)): os.makedirs(ofpath_figure_dataMining, exist_ok=True)
#% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique =  np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]
#%

diff_FC_GMWM = [np.zeros(1)] *4
net_netMeasures_subjects =  pd.DataFrame({"Subject": [0], "Period":  [0],"Network Type":  [0] ,"Nodes":  [0], "Edges":  [0], "Density":  [0] , "Transitivity":  [0], "Characteristic Path Length": [0]}) 
net_nodeMeasures_subjects =  pd.DataFrame({"Subject": [0], "Period":  [0],"Network Type":  [0] ,"Degree":  [0], "Strength":  [0], "Clustering Coefficient":  [0] , "Betweenness Centrality":  [0]}) 

net_nodeMeasures_subjects_GMWM =  pd.DataFrame({"Subject": [0], "Period":  [0],"Network Type":  [0] ,"Degree":  [0], "Strength":  [0], "Clustering Coefficient":  [0]}) 

           

FC_GMGM_all_ii = np.zeros(1)
FC_WMWM_all_ii = np.zeros(1)
FC_GMGM_all_pi = np.zeros(1)
FC_WMWM_all_pi = np.zeros(1)
FC_GMGM_all_ic = np.zeros(1)
FC_WMWM_all_ic = np.zeros(1)
FC_GMGM_all_po = np.zeros(1)
FC_WMWM_all_po = np.zeros(1)

calculate_distances_boolean = True #calculating distances matrices to see if GM-GM distances are different from WM-WM distances. takes a while to compute, so only need to do it once
if (calculate_distances_boolean):
    GM_distances = np.zeros(1)
    WM_distances = np.zeros(1)
descriptors = ["interictal","preictal","ictal","postictal"]
#%%
#%
DATA = [None] * len(sub_IDs_unique)
for pt in range(len(sub_IDs_unique)):
    DATA[pt] = [None] * 5
    for per in range(5):
         DATA[pt][per] = [None] * 4

count_patient = 0
for freq in range(4):
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
        if descriptor == descriptors[0]: per = 0
        if descriptor == descriptors[1]: per = 1
        if descriptor == descriptors[2]: per = 2
        if descriptor == descriptors[3]: per = 3
        
        #Inputs and OUtputs
        #input filename EEG
        ifpath_FC_sub_ID = ospj(ifpath_FC, sub_RID)
        ifpath_electrode_localization_sub_ID =  ospj(ifpath_electrode_localization, sub_RID)
        ifname_FC_filtered = ospj(ifpath_FC_sub_ID, "sub-{0}_{1}_{2}_{3}_functionalConnectivity.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
        ifpath_SC_sub_ID = ospj(ifpath_SC, sub_RID)
    
        ifname_electrode_localization = ospj(ifpath_electrode_localization_sub_ID, "sub-{0}_electrode_localization.csv".format(sub_ID))
        ofpath_FC_sub_ID = ospj(ofpath_FC_analysis, sub_RID)
        ofname_FC_sub_ID = ospj(ofpath_FC_sub_ID, "sub-{0}_{1}_{2}_{3}_PSDvsDistance_filtered.pickle".format(sub_ID, iEEG_filename, start_time_usec, stop_time_usec))
        if not (os.path.isdir(ofpath_FC_sub_ID)): os.mkdir(ofpath_FC_sub_ID)
        
        
        #GET DATA
        #get localization and FC files
        with open(ifname_FC_filtered, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
        electrode_localization = pd.read_csv(ifname_electrode_localization)
        FC = [broadband, alphatheta, beta, lowgamma, highgamma]
        
        #remname to standard 4 character electrode name
        for e in range(len(electrode_row_and_column_names)):
            electrode_name = electrode_row_and_column_names[e]
            if (len(electrode_name) == 3): electrode_name = "{0}{1}{2}".format(electrode_name[0:2], 0, electrode_name[2])
            electrode_row_and_column_names[e] = electrode_name
        #get electrode names in FC and localization files
        electrode_names_FC = electrode_row_and_column_names
        electrode_names_localization = np.array(electrode_localization["electrode_name"])
        
        tmp2 = np.nanmean(lowgamma, axis=2)  
        
        #Preprocessing files
        #find electrodes in both files (i.e there are electrodes in localization files but not in FC files, and there are electrodes in FC files but not localized)
        electrode_names_intersect, electrode_names_FC_ind, electrode_names_localization_ind =  np.intersect1d(electrode_names_FC, electrode_names_localization, return_indices = True)
    
        #Equalizing discrepancies in localization files and FC files
        #removing electrodes in localization files not in FC
        electrode_localization_intersect = electrode_localization.iloc[electrode_names_localization_ind]
        electrode_names_localization_intersect = electrode_names_localization[electrode_names_localization_ind]
        
        #removing electrodes in FC not in localization files
        FC_intersect = copy.deepcopy(FC)
        for f in range(len(FC)):
            FC_intersect[f] = FC_intersect[f][electrode_names_FC_ind[:,None], electrode_names_FC_ind[None,:], :] 
        electrode_names_FC_intersect = electrode_names_FC[electrode_names_FC_ind]
        print(np.equal(electrode_names_localization_intersect, electrode_names_FC_intersect))#check electrode localization order aligns with FC
        
            
        #Only GM and WM tissues considered (not CSF=1, or outside brain = 0)
        labels = np.array(electrode_localization_intersect["region_number"])
        labels_gm_ind = np.where(labels >= 2)[0] #Only GM and WM tissues considered
        #removing electrode localization electrodes not in GM or WM
        electrode_localization_intersect_GMWM = electrode_localization_intersect.iloc[labels_gm_ind]
        #removing FC electrodes not in GM or WM
        FC_intersect_GMWM = copy.deepcopy(FC_intersect)
        for f in range(len(FC_intersect_GMWM)):
            FC_intersect_GMWM[f] = FC_intersect_GMWM[f][labels_gm_ind[:,None], labels_gm_ind[None,:], :] 
        
        
        
        #averaging FC
        FC_intersect_GMWM_mean = [None] * len(FC_intersect_GMWM)
        for f in range(len(FC_intersect_GMWM_mean)):
            FC_intersect_GMWM_mean[f] = np.nanmean(FC_intersect_GMWM[f], axis=2)    
        np.allclose(FC_intersect_GMWM_mean[f], FC_intersect_GMWM_mean[f].T, rtol=1e-05, atol=1e-08)#check if symmetric
        
    
        #%
        #Get distances
        distances = np.array(electrode_localization_intersect_GMWM["distances_label_2"])
        distances_order_ind = np.argsort(distances)
        distances[distances_order_ind]
        electrode_localization_intersect_GMWM_distances_order = electrode_localization_intersect_GMWM.iloc[distances_order_ind]
        tmp = electrode_localization_intersect_GMWM_distances_order.reset_index (drop=True)
        #Ordering FC nodes based on distances
        FC_intersect_GMWM_mean_distances_order = copy.deepcopy(FC_intersect_GMWM_mean)
        for f in range(len(FC)):
            FC_intersect_GMWM_mean_distances_order[f] = FC_intersect_GMWM_mean_distances_order[f][distances_order_ind[:,None], distances_order_ind[None,:]] 
        
        f
        freq = freq
    
        #Subset GM and WM FC
        distances = np.array(electrode_localization_intersect_GMWM_distances_order["distances_label_2"])
        distances_GM_ind = np.where(distances <= 0)[0] #if change this, MUST CHANGE delta_order_electrodes_GM below around line 305
        distances_WM_ind = np.where(distances > 0)[0]
        distances_GM_ind_order_electrode  = np.where(np.array(electrode_localization_intersect_GMWM["distances_label_2"]) <= 0)[0]
        distances_WM_ind_order_electrode  = np.where(np.array(electrode_localization_intersect_GMWM["distances_label_2"]) > 0)[0]
        #Subset GM
        FC_intersect_GMWM_mean_distances_order_GM = copy.deepcopy(FC_intersect_GMWM_mean_distances_order)
        for f in range(len(FC)):
            FC_intersect_GMWM_mean_distances_order_GM[f] = FC_intersect_GMWM_mean_distances_order_GM[f][distances_GM_ind[:,None], distances_GM_ind[None,:]] 
        #Subset WM
        FC_intersect_GMWM_mean_distances_order_WM = copy.deepcopy(FC_intersect_GMWM_mean_distances_order)
        for f in range(len(FC)):
            FC_intersect_GMWM_mean_distances_order_WM[f] = FC_intersect_GMWM_mean_distances_order_WM[f][distances_WM_ind[:,None], distances_WM_ind[None,:]] 
        
        
        #subset GM-WM connections
        FC_intersect_GMWM_mean_distances_order_GMWM = copy.deepcopy(FC_intersect_GMWM_mean_distances_order)
        for f in range(len(FC_intersect_GMWM_mean_distances_order_GMWM)):
            FC_intersect_GMWM_mean_distances_order_GMWM[f] = FC_intersect_GMWM_mean_distances_order_GMWM[f][distances_GM_ind[:,None], distances_WM_ind[None,:]] 
   
        FC_intersect_GMWM_mean_distances_order_GMWM_plot = copy.deepcopy(FC_intersect_GMWM_mean_distances_order)
        for f in range(len(FC_intersect_GMWM_mean_distances_order)):
            for r in range(len(distances)):
                for c in range(len(distances)):
                    electrode_1 = distances[r]
                    electrode_2 = distances[c]
                    if (electrode_1 == 0 and electrode_2 ==0): 
                        FC_intersect_GMWM_mean_distances_order_GMWM_plot[f][r,c] =0
                    if (electrode_1 > 0 and electrode_2 > 0): 
                        FC_intersect_GMWM_mean_distances_order_GMWM_plot[f][r,c] =0
        FC_intersect_GMWM_mean_GMWM_plot = copy.deepcopy(FC_intersect_GMWM_mean)
        for f in range(len(FC_intersect_GMWM_mean)):
            for r in range(len(distances)):
                for c in range(len(np.array(electrode_localization_intersect_GMWM["distances_label_2"]))):
                    electrode_1 = np.array(electrode_localization_intersect_GMWM["distances_label_2"])[r]
                    electrode_2 = np.array(electrode_localization_intersect_GMWM["distances_label_2"])[c]
                    if (electrode_1 == 0 and electrode_2 ==0): 
                        FC_intersect_GMWM_mean_GMWM_plot[f][r,c] =0
                    if (electrode_1 > 0 and electrode_2 > 0): 
                        FC_intersect_GMWM_mean_GMWM_plot[f][r,c] =0
                        
             
        
        tmp =  FC_intersect_GMWM_mean_distances_order_GMWM[freq]
        tm2p =  FC_intersect_GMWM_mean_distances_order_GMWM_plot[freq]
        
        #plt.figure(figsize=(10, 10), dpi = 100); sns.heatmap(FC_intersect_GMWM_mean_distances_order_GM[freq], square=True, vmin = 0, vmax = 1); plt.title("Broadband GM-GM \n{1} \n{0}".format(sub_RID,descriptor ))
        #plt.figure(figsize=(10, 10), dpi = 100); sns.heatmap(FC_intersect_GMWM_mean_distances_order_WM[freq], square=True, vmin = 0, vmax = 1); plt.title("Broadband WM-WM \n{1} \n{0}".format(sub_RID,descriptor ))
        FC_GMGM = FC_intersect_GMWM_mean_distances_order_GM[freq][np.triu_indices( len(FC_intersect_GMWM_mean_distances_order_GM[freq]), k = 1) ] 
        FC_WMWM = FC_intersect_GMWM_mean_distances_order_WM[freq][np.triu_indices( len(FC_intersect_GMWM_mean_distances_order_WM[freq]), k = 1) ] 
    
    
        df_long_GMGM = pd.DataFrame({"Tissue": np.repeat("GM-GM", len(FC_GMGM)), "FC": FC_GMGM})
        df_long_WMWM = pd.DataFrame({"Tissue": np.repeat("WM-WM", len(FC_WMWM)), "FC": FC_WMWM})
        
        FC_df_long = pd.concat([df_long_GMGM, df_long_WMWM] )
    
        np.nanmean(FC_GMGM )
        np.nanmean(FC_WMWM  )
        np.nanmean(FC_WMWM  ) - np.nanmean(FC_GMGM ) #df_diff_FC_GMWM
        if descriptor == descriptors[0]: diff_FC_GMWM[0] = np.concatenate([diff_FC_GMWM[0] ,np.array([np.nanmedian(FC_WMWM  ) - np.nanmedian(FC_GMGM )] )  ])
        if descriptor == descriptors[1]: diff_FC_GMWM[1] = np.concatenate([diff_FC_GMWM[1] ,np.array([np.nanmedian(FC_WMWM  ) - np.nanmedian(FC_GMGM )] )  ])
        if descriptor == descriptors[2]: diff_FC_GMWM[2] = np.concatenate([diff_FC_GMWM[2] ,np.array([np.nanmedian(FC_WMWM  ) - np.nanmedian(FC_GMGM )] )  ])
        if descriptor == descriptors[3]: diff_FC_GMWM[3] = np.concatenate([diff_FC_GMWM[3] ,np.array([np.nanmedian(FC_WMWM  ) - np.nanmedian(FC_GMGM )] )  ])
    
        #plot = sns.displot( data =FC_df_long  , kind = "kde", x="FC" , hue="Tissue"); plot.axes[0,0].set_ylim(0,10); plot.axes[0,0].set_xlim(0,1); plt.title("Broadband \n{1} \n{0}".format(sub_RID,descriptor ))
    
        
        if descriptor == descriptors[0]:
            FC_GMGM_all_ii = np.concatenate([FC_GMGM_all_ii,FC_GMGM ])
            FC_WMWM_all_ii = np.concatenate([FC_WMWM_all_ii,FC_WMWM ])
            per = 0
        if descriptor == descriptors[1]:
            FC_GMGM_all_pi = np.concatenate([FC_GMGM_all_pi,FC_GMGM ])
            FC_WMWM_all_pi = np.concatenate([FC_WMWM_all_pi,FC_WMWM ])
            per = 1
            pi_order_electrodes = FC_intersect_GMWM_mean[freq]
            pi_order_distances = FC_intersect_GMWM_mean_distances_order[freq]
        if descriptor == descriptors[2]:
            FC_GMGM_all_ic = np.concatenate([FC_GMGM_all_ic,FC_GMGM ])
            FC_WMWM_all_ic = np.concatenate([FC_WMWM_all_ic,FC_WMWM ])
            per = 2
            ic_order_electrodes = FC_intersect_GMWM_mean[freq]
            ic_order_distances = FC_intersect_GMWM_mean_distances_order[freq]
            delta_order_electrodes = ic_order_electrodes - pi_order_electrodes
            delta_order_distances = ic_order_distances - pi_order_distances
            
            delta_order_electrodes_GM = delta_order_electrodes[ distances_GM_ind_order_electrode[:,None], distances_GM_ind_order_electrode[None,:]]
            delta_order_electrodes_WM = delta_order_electrodes[distances_WM_ind_order_electrode[:,None],distances_WM_ind_order_electrode[None,:]]
            
            delta_order_distances_GM = delta_order_distances[distances_GM_ind[:,None], distances_GM_ind[None,:]]
            delta_order_distances_WM = delta_order_distances[distances_WM_ind[:,None], distances_WM_ind[None,:]]
            
            #subset GM-WM connections
            delta_order_electrodes_GMWM_plot = copy.deepcopy(delta_order_electrodes)
    
            for r in range(len(distances)):
                for c in range(len(np.array(electrode_localization_intersect_GMWM["distances_label_2"]))):
                    electrode_1 = np.array(electrode_localization_intersect_GMWM["distances_label_2"])[r]
                    electrode_2 = np.array(electrode_localization_intersect_GMWM["distances_label_2"])[c]
                    if (electrode_1 == 0 and electrode_2 ==0): 
                        delta_order_electrodes_GMWM_plot[r,c] =0
                    if (electrode_1 > 0 and electrode_2 > 0): 
                        delta_order_electrodes_GMWM_plot[r,c] =0
            
        
        if descriptor == descriptors[3]:
            FC_GMGM_all_po = np.concatenate([FC_GMGM_all_po,FC_GMGM ])
            FC_WMWM_all_po = np.concatenate([FC_WMWM_all_po,FC_WMWM ])
            per = 3
            
        
        
        
        #calculating distances matrices to see if GM-GM distances are different from WM-WM distances
        if (calculate_distances_boolean): #takes a while to compute, so only need to do it once
            if descriptor == descriptors[0]:
                
                electrode_localization_intersect_GMWM_distances_order
                distance_matrix = np.zeros(shape = (len(electrode_localization_intersect_GMWM_distances_order),len(electrode_localization_intersect_GMWM_distances_order) ))
                for e1 in range(len(electrode_localization_intersect_GMWM_distances_order)):
                    for e2 in range(len(electrode_localization_intersect_GMWM_distances_order)):
                        p1 = [electrode_localization_intersect_GMWM_distances_order.iloc[e1]["x_coordinate"], electrode_localization_intersect_GMWM_distances_order.iloc[e1]["y_coordinate"], electrode_localization_intersect_GMWM_distances_order.iloc[e1]["z_coordinate"]]
                        p2 = [electrode_localization_intersect_GMWM_distances_order.iloc[e2]["x_coordinate"], electrode_localization_intersect_GMWM_distances_order.iloc[e2]["y_coordinate"], electrode_localization_intersect_GMWM_distances_order.iloc[e2]["z_coordinate"]]
                        distance_matrix[e1,e2] = np.sqrt(  np.sum((np.array(p1)-np.array(p2))**2, axis=0)    )
                #colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#131e2a","#4a4aa2", "#cbcae7"])
                #plt.figure(figsize=(5, 5), dpi = 600); sns.heatmap(distance_matrix, square=True, cmap = colormap, center = 30); plt.title("Distance Matrix \n{0} ".format(sub_RID ))
                #plt.savefig(ospj(ofpath_figure_adj, "distance_matrix_ordered.pdf"))
                
                #Subset Distances by GM-GM, WM-WM pairs
                distance_matrix_GM = distance_matrix[distances_GM_ind[:,None], distances_GM_ind[None,:]] 
                distance_matrix_WM = distance_matrix[distances_WM_ind[:,None], distances_WM_ind[None,:]] 
                #plt.figure(figsize=(10, 10), dpi = 100); sns.heatmap(distance_matrix_GM, square=True); plt.title("Distance Matrix GM \n{0} ".format(sub_RID ))
                #plt.figure(figsize=(10, 10), dpi = 100); sns.heatmap(distance_matrix_WM, square=True); plt.title("Distance Matrix WM \n{0} ".format(sub_RID ))
                distance_GM = distance_matrix_GM[np.triu_indices( len(distance_matrix_GM), k = 1) ] 
                distance_WM = distance_matrix_WM[np.triu_indices( len(distance_matrix_WM), k = 1) ] 
                
                np.nanmean(distance_GM )
                np.nanmean(distance_WM  )
                GM_distances = np.concatenate([GM_distances, distance_GM]   )
                WM_distances = np.concatenate([WM_distances, distance_WM]   )
                
                mannwhitneyu(distance_GM ,distance_WM  )[1]
        
        
        #############################################
        

        #%   

            
    
        #Network Analysis
        
        FC_intersect_GMWM_mean_distances_order
        FC_intersect_GMWM_mean_distances_order_GM
        FC_intersect_GMWM_mean_distances_order_WM
        FC_intersect_GMWM_mean_distances_order_GMWM
        
        network_full = copy.deepcopy(FC_intersect_GMWM_mean_distances_order[freq])
        network_GM = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_GM[freq])
        network_WM = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_WM[freq])
        network_GMWM = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_GMWM_plot[freq])
    
        #thresholding
        threshold = 0.3
        network_full[np.where(network_full < threshold)] = 0
        network_GM[np.where(network_GM < threshold)] = 0
        network_WM[np.where(network_WM < threshold)] = 0
        network_GMWM[np.where(network_GMWM < threshold)] = 0
    
        G_full = nx.from_numpy_array(network_full) 
        G_GM = nx.from_numpy_array(network_GM) 
        G_WM = nx.from_numpy_array(network_WM) 
        G_GMWM = nx.from_numpy_array(network_GMWM) 
    
        #shortest path lengths
        G = G_full
        max_shortest = np.zeros(1)
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            max_shortest = np.concatenate([max_shortest , np.array([nx.average_shortest_path_length(C)] )    ]    )
        SPL_full = np.max(max_shortest)
        G = G_GM
        max_shortest = np.zeros(1)
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            max_shortest = np.concatenate([max_shortest , np.array([nx.average_shortest_path_length(C)] )    ]    )
        SPL_GM = np.max(max_shortest)
        G = G_WM
        max_shortest = np.zeros(1)
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            max_shortest = np.concatenate([max_shortest , np.array([nx.average_shortest_path_length(C)] )    ]    )
        SPL_WM = np.max(max_shortest)
        #SPL_full, SPL_GM, SPL_WM
        
        #nx.betweenness_centrality
        
        #bct.degrees_und(network_GMWM)
        #bct.clustering_coef_wu(network_GMWM)
        
        #print(nx.info(G))
        nx.density(G_full), nx.density(G_GM), nx.density(G_WM)
        nx.shortest_path(G_full), nx.shortest_path(G_GM), nx.shortest_path(G_WM)
        nx.transitivity(G_full), nx.transitivity(G_GM), nx.transitivity(G_WM)
        nx.degree(G_full), nx.degree(G_GM), nx.degree(G_WM)
        nx.betweenness_centrality(G_full), nx.betweenness_centrality(G_GM), nx.betweenness_centrality(G_WM)
        nx.betweenness_centrality(G_full).values()
        np.array(list(nx.betweenness_centrality(G_full).values()))
    
        
    
        N_full = len(network_full); N_GM = len(network_GM); N_WM = len(network_WM)
        net_nodeMeasures = pd.concat( [pd.DataFrame({"Subject": np.repeat(sub_RID, N_full), "Period": np.repeat(descriptor, N_full),"Network Type": np.repeat("Full Network", N_full), "Degree": bct.degrees_und(network_full) ,"Strength": bct.strengths_und(network_full), "Clustering Coefficient": bct.clustering_coef_wu(network_full) ,"Betweenness Centrality": np.array(list(nx.betweenness_centrality(G_full).values()))               }),
                                 pd.DataFrame({"Subject": np.repeat(sub_RID, N_GM), "Period": np.repeat(descriptor, N_GM), "Network Type": np.repeat("GM-only", N_GM),        "Degree": bct.degrees_und(network_GM  ) ,"Strength": bct.strengths_und(network_GM),"Clustering Coefficient": bct.clustering_coef_wu(network_GM)  ,"Betweenness Centrality": np.array(list(nx.betweenness_centrality(G_GM).values()))            }),
                                 pd.DataFrame({"Subject": np.repeat(sub_RID, N_WM), "Period": np.repeat(descriptor, N_WM), "Network Type": np.repeat("WM-only", N_WM  ),      "Degree": bct.degrees_und(network_WM  ) ,"Strength": bct.strengths_und(network_WM), "Clustering Coefficient": bct.clustering_coef_wu(network_WM)  ,"Betweenness Centrality": np.array(list(nx.betweenness_centrality(G_WM).values()))                })  ], 
                                 axis=0,ignore_index=True )
        net_netMeasures = pd.concat( [pd.DataFrame({"Subject": np.repeat(sub_RID, 1), "Period": np.repeat(descriptor, 1),"Network Type": np.repeat("Full Network", 1) ,"Nodes": N_full, "Edges":  bct.density_und(network_full)[2], "Density": bct.density_und(network_full)[0] , "Transitivity": bct.transitivity_wu(network_full), "Characteristic Path Length":SPL_full                   }),
                                 pd.DataFrame({"Subject": np.repeat(sub_RID, 1), "Period": np.repeat(descriptor, 1),"Network Type": np.repeat("GM-only", 1),       "Nodes": N_GM , "Edges":  bct.density_und(network_GM)[2], "Density": bct.density_und(network_GM  )[0]  ,"Transitivity": bct.transitivity_wu(network_GM)    , "Characteristic Path Length": SPL_GM                   })      , 
                                 pd.DataFrame({"Subject": np.repeat(sub_RID, 1), "Period": np.repeat(descriptor, 1),"Network Type": np.repeat("WM-only", 1  ),    "Nodes": N_WM      ,"Edges":  bct.density_und(network_WM)[2],   "Density": bct.density_und(network_WM  )[0] , "Transitivity": bct.transitivity_wu(network_WM)   , "Characteristic Path Length":SPL_WM              } )  ], 
                                 axis=0,ignore_index=True )
        net_netMeasures_GMWM = pd.concat( [pd.DataFrame({"Subject": np.repeat(sub_RID, N_full), "Period": np.repeat(descriptor, N_full),"Network Type": np.repeat("GM-WM", N_full) , "Degree": bct.degrees_und(network_GMWM) ,"Strength": bct.strengths_und(network_GMWM), "Clustering Coefficient": bct.clustering_coef_wu(network_GMWM)           })], 
                                         axis=0,ignore_index=True )
        
        
        net_netMeasures_subjects = net_netMeasures_subjects.append(net_netMeasures)
        net_nodeMeasures_subjects = net_nodeMeasures_subjects.append(net_nodeMeasures)
        net_nodeMeasures_subjects_GMWM  = net_nodeMeasures_subjects_GMWM.append(net_netMeasures_GMWM)
        """
        maximum = np.max(net_nodeMeasures["Degree"])
        plot = sns.displot( data =net_nodeMeasures  , kind = "kde",x="Degree",  hue = "Network Type"); plot.axes[0,0].set_ylim(0,); plot.axes[0,0].set_xlim(0,40); plt.title("Degree \n{2} \n{1} \n{0}".format(sub_RID,descriptor, np.array(order_of_matrices_in_pickle_file)[0][freq]))
        plot = sns.displot( data =net_nodeMeasures  , kind = "kde",x="Strength",  hue = "Network Type"); plot.axes[0,0].set_ylim(0,); plot.axes[0,0].set_xlim(0,20); plt.title("Strength \n{2} \n{1} \n{0}".format(sub_RID,descriptor, np.array(order_of_matrices_in_pickle_file)[0][freq]))
        plot = sns.displot( data =net_nodeMeasures  , kind = "kde",x="Clustering Coefficient",  hue = "Network Type"); plot.axes[0,0].set_ylim(0,); plot.axes[0,0].set_xlim(0,1.5); plt.title("Clustering Coefficient (weighted) \n{2} \n{1} \n{0}".format(sub_RID,descriptor, np.array(order_of_matrices_in_pickle_file)[0][freq]))
        plot = sns.displot( data =net_nodeMeasures  , kind = "kde",x="Betweenness Centrality",  hue = "Network Type"); plot.axes[0,0].set_ylim(0,); plot.axes[0,0].set_xlim(0,350); plt.title("Betweenness Centrality (weighted) \n{2} \n{1} \n{0}".format(sub_RID,descriptor, np.array(order_of_matrices_in_pickle_file)[0][freq]))
    
        """
        
        electrode_localization_intersect_GMWM_distances_order_community = copy.deepcopy(electrode_localization_intersect_GMWM_distances_order)
        electrode_localization_intersect_GMWM_distances_order_community = electrode_localization_intersect_GMWM_distances_order_community.reset_index()
        electrode_localization_intersect_GMWM_distances_order_community = pd.concat([ electrode_localization_intersect_GMWM_distances_order_community,  pd.DataFrame({"Module": bct.community_louvain(network_full)[0] })  ], axis=1,ignore_index=True )
        bct.community_louvain(network_full)
        bct.community_louvain(network_GM)
        bct.community_louvain(network_WM)
        
        
        ######################################################
        ######################################################
        #SFC analysis
        if (os.path.exists(ifpath_SC_sub_ID)): #If structure exists
            ifname_SC_sub_ID = ospj(ifpath_SC_sub_ID, "whole_brain_ROIs_cg.txt")
            structure = pd.read_table(ifname_SC_sub_ID, header=None)
            #cleaning up structural data 
            structure = structure.drop([0,1], axis=1)
            structure = structure.drop([0], axis=0)
            structure = structure.iloc[:, :-1]
            structure_electrode_names = np.array([e[-4:] for e in  np.array(structure.iloc[0])])
            structure = np.array(structure.iloc[1:, :]).astype('float64')  #finally turn into numpy array
            
            #reorder structure
            #remove electrodes not in FC 
            electrode_names_FC_distance_order = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])
          
            #find electrodes in both files (i.e there are electrodes in localization files but not in FC files, and there are electrodes in FC files but not localized)
            electrode_names_intersect_SC, electrode_names_FC_ind_SC, electrode_names_localization_ind_SC =  np.intersect1d(electrode_names_FC_distance_order, structure_electrode_names, return_indices = True)
        
            #Equalizing discrepancies in SC files and FC files
            #removing electrodes in SC not in FC files
            SC = copy.deepcopy(structure)
            SC = SC[electrode_names_localization_ind_SC[:,None], electrode_names_localization_ind_SC[None,:]] 
            
            electrode_names_SC = structure_electrode_names[electrode_names_localization_ind_SC]
            
            #reorder to same as distance FC. Seems can't do this with np.intersect1d. Python is such a shitty programming language compared to R and Matlab for data science and data mining holy crap
            order = []
            for e in range(len(electrode_names_SC)):
                order.append(  np.where(electrode_names_FC_distance_order[e]  ==  electrode_names_SC  )[0][0]   )
            order = np.array(order)
            SC_distance_order = SC[order[:,None], order[None,:]] 
            electrode_names_SC_distance_order = electrode_names_SC[order]
            #print(np.equal(electrode_names_FC_distance_order, electrode_names_SC_distance_order))#check electrode localization order aligns with FC
            
            #normalizing
            SC_distance_order_log = copy.deepcopy(SC_distance_order)
            SC_distance_order_log[np.where(SC_distance_order_log == 0)] = 1
            SC_distance_order_log = np.log10(SC_distance_order_log)
        
        
            tmp = FC_intersect_GMWM_mean_distances_order[freq]

            sc_log = SC_distance_order_log[  np.triu_indices( len(SC_distance_order_log), k = 1)   ]
            sc = SC_distance_order[  np.triu_indices( len(SC_distance_order), k = 1)   ]
            fc1 = FC_intersect_GMWM_mean_distances_order[freq][  np.triu_indices( len(FC_intersect_GMWM_mean_distances_order[freq]), k = 1)   ]
            dis1 = distance_matrix[  np.triu_indices( len(distance_matrix), k = 1)   ]
            #tmp2 = np.log10(tmp2 * 10)
            
            df = pd.DataFrame( dict(   distance = dis1, FC = fc1, SC = sc, sc_log = sc_log  )  )
            
        DATA[count_patient][freq][per] = [SC,
                               SC_distance_order,
                               FC_intersect_GMWM_mean,
                               FC_intersect_GMWM_mean_distances_order,
                               FC_intersect_GMWM_mean_distances_order_GM,
                               FC_intersect_GMWM_mean_distances_order_WM,
                               FC_intersect_GMWM_mean_distances_order_GMWM,
                               FC_intersect_GMWM_mean_distances_order_GMWM_plot,
                               FC_intersect_GMWM_mean_distances_order_GMWM,
                               FC_intersect_GMWM_mean_GMWM_plot,
                               distance_matrix,
                               distance_matrix_GM,
                               distance_matrix_WM,
                               electrode_localization_intersect_GMWM,
                               electrode_localization_intersect_GMWM_distances_order
                               ]
        count_patient = count_patient + 1
        """
         
            
            fig, ax = plt.subplots(nrows = 2, ncols=2,figsize=(10, 10), dpi = 600)
            sns.regplot(x = "distance", y = "FC", data = df, ax= ax[0][0], scatter_kws={'s':2})
            sns.residplot(x = "distance", y = "FC", data = df, ax= ax[0][1], scatter_kws={'s':2})
            sns.regplot(x = "distance", y = "SC", data = df, ax= ax[1][0], scatter_kws={'s':2})
            sns.residplot(x = "distance", y = "SC", data = df, ax= ax[1][1], scatter_kws={'s':2})
            
        
            mod = sm.OLS(df['FC'],sm.add_constant(df['distance']))
            res = mod.fit()
            res.resid
            test = df[['distance']].drop_duplicates().sort_values('distance')
            predictions = res.get_prediction(sm.add_constant(test))
            predictions = pd.concat([test,predictions.summary_frame(alpha=0.05)],axis=1)
            predictions.head()
            fig,axes = plt.subplots(1,3,figsize=(15,5))
            sns.scatterplot(x='distance',y='FC',ax=axes[0],data=df)
            axes[0].plot(predictions['distance'], predictions['mean'], lw=2)
            axes[0].fill_between(x=predictions['distance'],
                                 y1=predictions['mean_ci_lower'],y2=predictions['mean_ci_upper'],
                                 facecolor='blue', alpha=0.2)
            sns.scatterplot(x='distance',y='FC',ax=axes[1],
                            data=pd.DataFrame({'distance':df['distance'],
                                              'FC':res.resid})
                            )
            axes[1].axhline(0, ls='--',color="k")
            sns.distplot(res.resid,ax=axes[2],bins=20)
           
            
           
            
            mod = sm.OLS(df['SC'],sm.add_constant(df['distance']))
            res = mod.fit()
            res.resid
            test = df[['distance']].drop_duplicates().sort_values('distance')
            predictions = res.get_prediction(sm.add_constant(test))
            predictions = pd.concat([test,predictions.summary_frame(alpha=0.05)],axis=1)
            predictions.head()
            fig,axes = plt.subplots(1,3,figsize=(15,5))
            sns.scatterplot(x='distance',y='SC',ax=axes[0],data=df)
            axes[0].plot(predictions['distance'], predictions['mean'], lw=2)
            axes[0].fill_between(x=predictions['distance'],
                                 y1=predictions['mean_ci_lower'],y2=predictions['mean_ci_upper'],
                                 facecolor='blue', alpha=0.2)
            sns.scatterplot(x='distance',y='SC',ax=axes[1],
                            data=pd.DataFrame({'distance':df['distance'],
                                              'SC':res.resid})
                            )
            axes[1].axhline(0, ls='--',color="k")
            sns.distplot(res.resid,ax=axes[2],bins=20)
                            
            
            
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression().fit(  np.array(df['distance']).reshape(-1,1),     np.array(df['SC']).reshape(-1,1)  )   
            preds = model.predict(   np.array(df['distance']).reshape(-1,1)   )
            res_structure = np.array(df['SC']).reshape(-1,1)    - preds
              
            model = LinearRegression().fit(  np.array(df['distance']).reshape(-1,1),     np.array(df['FC']).reshape(-1,1)  )   
            preds = model.predict(   np.array(df['distance']).reshape(-1,1)   )
            res_function = np.array(df['FC']).reshape(-1,1)    - preds
            
            df = pd.concat([df, pd.DataFrame(  dict(SC_residuals = res_structure.flatten(), FC_residuals = res_function.flatten())   )  ],axis=1)
              
            size = 1
            fig, ax = plt.subplots(nrows = 3, ncols=4,figsize=(22, 14), dpi = 600)
            sns.scatterplot(x = "distance", y= "SC", data= df ,s=size, ax = ax[0][0])
            sns.scatterplot(x = "distance", y= "FC", data= df ,s=size, ax = ax[0][1])
            sns.scatterplot(x = "distance", y= "SC_residuals", data= df ,s=size, ax = ax[0][2])
            sns.scatterplot(x = "distance", y= "FC_residuals", data= df ,s=size, ax = ax[0][3])
            sns.scatterplot(x = "SC", y= "FC", data= df ,s=size, ax = ax[1][0])
            sns.scatterplot(x = "SC_residuals", y= "FC_residuals", data= df ,s=size, ax = ax[1][1])
            sns.scatterplot(x = "SC_residuals", y= "FC", data= df ,s=size, ax = ax[1][2])
            sns.scatterplot(x ="SC", y= "FC_residuals", data= df , s=size, ax = ax[1][3])
            sns.histplot(x ="SC_residuals", data= df  ,ax=ax[2][0],bins=20, kde = True)
            sns.histplot(x ="FC_residuals", data= df  ,ax=ax[2][1],bins=20, kde = True)
            ax[0][2].axhline(0, ls='--',color="k")
            ax[0][3].axhline(0, ls='--',color="k")
            
            
        
            GLM_model = sm.GLM(fc1, dis1, family=sm.families.Binomial()).fit()
            predicted = GLM_model.predict()
            #FC_residuals = GLM_model.resid # residual values
            fig, ax = plt.subplots(nrows = 1, ncols=2,figsize=(10, 5), dpi = 600)
            sns.scatterplot(x = dis1, y= fc1, s=10, ax = ax[0])
            sns.scatterplot(x = dis1, y= predicted, s=10, ax = ax[0])
            #sns.scatterplot(x = dis1, y= FC_residuals, s=10, ax = ax[1])
            

            print(GLM_model.summary())
            
            
        
            
            pearsonr(sc_log, fc1)
     
            spearmanr(df["SC"], df["FC"])
            spearmanr(df["SC_residuals"], df["FC_residuals"])
            spearmanr(df["SC"], df["FC_residuals"])
            spearmanr(df["SC_residuals"], df["FC"])

            distance_matrix
            

            ###########################################3
            ###########################################3
            ###########################################3
            ###########################################3
            ###########################################3
            ###########################################3
            ###########################################3
            ###########################################3

            #DATA = [SC_distance_order, FC_intersect_GMWM_mean_distances_order[freq], distance_matrix, SC_distance_order_log , electrode_localization_intersect_GMWM_distances_order]
            DATA = [SC_distance_order, FC_intersect_GMWM_mean_distances_order[freq], distance_matrix, SC_distance_order_log , 
                    electrode_localization_intersect_GMWM_distances_order.reset_index(drop=True), delta_order_distances]
            
            DATA
            DATA_FCi = 1
            DATA_SCi = 0
            threshold_SFC = 0.0
            threshold_ind = np.where(DATA[DATA_FCi] > threshold_SFC)
            
            p = 2
            p_list = []
            init = np.zeros(shape = (1,2))
            for p in range(len(threshold_ind[0])):
                r = threshold_ind[0][p]
                c = threshold_ind[1][p]
                fun = DATA[DATA_FCi][r,c]
                np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[r]
                np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[c]
                dist_i = DATA[2][r,c]
                
                tmp_dist = np.zeros(shape = (len(DATA[1] ),3 ))
                tmp_dist2 = np.zeros(shape = (len(DATA[1] ),3 ))
                for e in range(len(DATA[1] )):
                    tmp_dist[e,0] = DATA[2][r,e] 
                    tmp_dist[e,1] = DATA[2][c,e]
                    heron = (tmp_dist[e,1] + tmp_dist[e,0] + dist_i)/2
                    heronE = (3*dist_i)/2
                    A = np.sqrt(heron * (heron - tmp_dist[e,0] )* (heron - tmp_dist[e,1] )* (heron - dist_i ))
                    tmp_dist[e,2] = np.abs(A - np.sqrt(heronE * (heronE - dist_i )* (heronE - dist_i )* (heronE - dist_i )))#finding which ones are closes to an equilateral triangle
                    tmp_dist2[e,0] = np.abs( tmp_dist[e,1]  - tmp_dist[e,0])#finding which ones are closes to an equilateral triangle
         
                    tmp_dist2[e,1] = np.abs( tmp_dist[e,1]  -dist_i)#finding which ones are closes to an equilateral triangle
         
                    tmp_dist2[e,2] = np.abs( tmp_dist[e,2]  -dist_i)#finding which ones are closes to an equilateral triangle
         
            
                th1 = dist_i *0.8
                th2 = dist_i /0.8
                np.where(tmp_dist[:,0] >  th1)
                np.where(tmp_dist[:,1] >  th1)
                np.where(tmp_dist[:,0] < th2)
                np.where(tmp_dist[:,1] <  th2)
                #print( "{0}: {1}".format(p,np.any(  (tmp_dist[:,0] >  th1) & (tmp_dist[:,1] > th1) & (tmp_dist[:,0] < th2)  & (tmp_dist[:,1] <  th2) )  ))
                
                p_list.append(np.any(  (tmp_dist[:,0] >  th1) & (tmp_dist[:,1] > th1) & (tmp_dist[:,0] < th2)  & (tmp_dist[:,1] <  th2) ))
                if dist_i>0:
                    if (np.any(  (tmp_dist[:,0] >  th1) & (tmp_dist[:,1] > th1) & (tmp_dist[:,0] < th2)  & (tmp_dist[:,1] <  th2) ) == True):
                
                        c1 = np.where(  (tmp_dist[:,0] >  th1) & (tmp_dist[:,1] > th1) & (tmp_dist[:,0] < th2)  & (tmp_dist[:,1] <  th2)  )[0][0]
                        np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[c1]
                        
                        #tmp_dist = np.delete(tmp_dist,  [r, c], axis=0)
                        #tmp_dist_names = np.delete(np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"]), [r, c])
                        
                        #finding electrode most equidistant to the other two electrodes
                        #c1 = np.where(tmp_dist_names[np.where(np.min(tmp_dist[:,2])== tmp_dist[:,2])[0][0]] == np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"]))[0][0]
                        
                        #tmp_dist = np.abs(np.delete(DATA[2][r,:]- dist_i,c))
                        #c1 = np.where(  np.abs(DATA[2][r,:] - dist_i) == np.min( np.abs(tmp_dist))     )[0][0]
                        fun1 = DATA[DATA_FCi][r,c1]
                        dist_i1 = DATA[2][r,c1]
                        fun2 = DATA[DATA_FCi][c,c1]
                        dist_i2 = DATA[2][c,c1]
                        
                        struc = DATA[DATA_SCi][r,c]
                        struc1 = DATA[DATA_SCi][r,c1]
                        struc2 = DATA[DATA_SCi][c,c1]
                        
                        print("\n{2}-{3}: {0}. {1}mm. {9}\n{2}-{4}: {5}. {6}mm. {10}\n{3}-{4}: {7}. {8}mm. {11}".format(np.round(fun,2), np.round(dist_i,2), DATA[4].iloc[r]["electrode_name"], DATA[4].iloc[c]["electrode_name"] ,
                                                               DATA[4].iloc[c1]["electrode_name"], np.round(fun1,2), np.round(dist_i1,2),
                                                               np.round(fun2,2), np.round(dist_i2,2),
                                                               np.round(struc,2), np.round(struc1,2), np.round(struc2,2))
                              )
                        
                        #print (  "{0}, {1}".format(np.round(fun-fun1,2), np.round(struc-struc1,2) ))
                    
                        
                        init = np.vstack(  (init ,  np.array([fun-fun1, struc-struc1]).reshape(1,2)) )
                        
   
            init = np.delete(init, 0, axis = 0)  
                
            fig, ax = plt.subplots(nrows = 2, ncols=2,figsize=(10, 10), dpi = 600)
            sns.regplot(x = init[:,0], y = init[:,1], ax= ax[0][0], scatter_kws={'s':2})
                
                
            p_list = np.array(p_list)
            p_ind = np.where(p_list)
    
        #############################################
        
        #%
        #PLOT FOR PAPER
        ##############################%%
        #heatmaps
        len_gm = len(np.where(distances <= 0)[0])
    
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff", "#d26767", "#e4c4a4","#e4e4a4","#e4e4a4","#2d972d", "#a4c4e4", "#a4a4e4", "#c4a4e4", "#e4a4e4"])
    
        plt.figure(figsize=(5, 5), dpi = 600); 
        len(FC_intersect_GMWM_mean[freq])
        sns.heatmap(FC_intersect_GMWM_mean[freq], square=True, vmin = 0, vmax = 1, cmap = "mako", center=0.45); 
        plt.title("{0}_{1}_{2}_unordered".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        ofpath_figure_adj = ospj(ofpath_figure, "adjacency_matrices")
        if not (os.path.isdir(ofpath_figure_adj)): os.makedirs(ofpath_figure_adj, exist_ok=True)
        plt.savefig(ospj(ofpath_figure_adj, "{0}_{2}_{1}_{3}_unordered_NGM_{4}.pdf".format(sub_RID,descriptor, per, order_of_matrices_in_pickle_file.iloc[freq][0],len_gm  )))
        
        #%
        ###Unordered by distance - ordered by electrodes
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff", "#d26767", "#e4c4a4","#e4e4a4","#e4e4a4","#2d972d", "#a4c4e4", "#a4a4e4", "#c4a4e4", "#e4a4e4"])
        plt.figure(figsize=(5, 5), dpi = 600); 
        len(FC_intersect_GMWM_mean_distances_order[freq])
        sns.heatmap(FC_intersect_GMWM_mean_distances_order[freq], square=True, vmin = 0, vmax = 1, cmap = "mako", center=0.45); 
        plt.title("{0}_{1}_{2}_ordered".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        plt.savefig(ospj(ofpath_figure_adj, "{0}_{2}_{1}_{3}_ordered_NGM_{4}.pdf".format(sub_RID,descriptor, per, order_of_matrices_in_pickle_file.iloc[freq][0],len_gm  )))
        
        #GM-WM connections only - FULL MATRIX
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff", "#d26767", "#e4c4a4","#e4e4a4","#e4e4a4","#2d972d", "#a4c4e4", "#a4a4e4", "#c4a4e4", "#e4a4e4"])
        plt.figure(figsize=(5, 5), dpi = 600); 
        len(FC_intersect_GMWM_mean_distances_order_GMWM[freq])
        sns.heatmap(FC_intersect_GMWM_mean_distances_order_GMWM_plot[freq], square=True, vmin = 0, vmax = 1, cmap = "mako", center=0.45); 
        plt.title("{0}_{1}_{2}_ordered".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        plt.savefig(ospj(ofpath_figure_adj, "{0}_{2}_{1}_{3}_ordered_GM-WM_FULL_{4}.pdf".format(sub_RID,descriptor, per, order_of_matrices_in_pickle_file.iloc[freq][0],len_gm  )))
        
        #GM-WM connections only - GM-WM only
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff", "#d26767", "#e4c4a4","#e4e4a4","#e4e4a4","#2d972d", "#a4c4e4", "#a4a4e4", "#c4a4e4", "#e4a4e4"])
        plt.figure(figsize=(5, 5), dpi = 600); 
        len(FC_intersect_GMWM_mean_distances_order_GMWM[freq])
        sns.heatmap(FC_intersect_GMWM_mean_distances_order_GMWM[freq], square=True, vmin = 0, vmax = 1, cmap = "mako", center=0.45); 
        plt.title("{0}_{1}_{2}_ordered".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        plt.savefig(ospj(ofpath_figure_adj, "{0}_{2}_{1}_{3}_ordered_GM-WM_ONLY_{4}.pdf".format(sub_RID,descriptor, per, order_of_matrices_in_pickle_file.iloc[freq][0],len_gm  )))
        
        np.min(delta)
        np.max(delta)
        #Delta
        plt.figure(figsize=(5, 5), dpi = 600); 
        len(delta)
        sns.heatmap(delta, square=True, vmin = -0.16, vmax = 0.39, cmap = "mako", center=0.25); 
        plt.title("{0}_{1}_{2}_DELTA_ordered".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        ofpath_figure_adj = ospj(ofpath_figure, "adjacency_matrices")
        if not (os.path.isdir(ofpath_figure_adj)): os.makedirs(ofpath_figure_adj, exist_ok=True)
        plt.savefig(ospj(ofpath_figure_adj, "{0}_{2}_{1}_{3}_DELTA_ORDERED.pdf".format(sub_RID,descriptor, per, order_of_matrices_in_pickle_file.iloc[freq][0]  )))
        
        
        
        
        #%
        ##############################%%
        #Colorbars
        #GM
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#c6b4a5", "#c6b4a5"])
        electrode_localization_intersect_GMWM_distances_order
        distances_plot = np.zeros(shape=(1, len(electrode_localization_intersect_GMWM_distances_order)))
        distances_plot[0] = np.array(electrode_localization_intersect_GMWM_distances_order["distances_label_2"])
        plt.figure(figsize=(5, 1), dpi = 600); 
        sns.heatmap(distances_plot, square=True, cmap = colormap, center=0.6, cbar= False, xticklabels= False, yticklabels= False)
        plt.savefig(ospj(ofpath_figure_adj, "colorbar_GM.pdf"))
        
        #WM
        distances_plot = distances_plot[np.where(distances_plot > 0)]
        distances_plot = np.reshape(distances_plot, (1,len(distances_plot)))
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b6d4ee", "#184268"])
        plt.figure(figsize=(5, 1), dpi = 600); 
        sns.heatmap(distances_plot, square=True, cmap = colormap, cbar= False, xticklabels= False, yticklabels= False)
        plt.savefig(ospj(ofpath_figure_adj, "colorbar_WM.pdf"))
    
        #Unordered
        from colormap import rgb2hex
        #rgb2hex(int(colormap(0)[0]*255), int(colormap(0)[1]*255), int(colormap(0)[2]*255))
        #"#{0:02x}{1:02}{2:02}".format( int(colormap(0)[0]*255), int(colormap(0)[1]*255), int(colormap(0)[2]*255))
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b6d4ee", "#184268"])
        colors = np.ndarray.tolist(np.repeat("#c6b4a5" ,255) )
        for c in range(255):
            colors.append( rgb2hex(int(colormap(c)[0]*255), int(colormap(c)[1]*255), int(colormap(c)[2]*255))  )
      
        distances_plot = np.zeros(shape=(1, len(electrode_localization_intersect_GMWM)))
        distances_plot[0] = np.array(electrode_localization_intersect_GMWM["distances_label_2"])
        plt.figure(figsize=(5, 1), dpi = 600); 
        sns.heatmap(distances_plot, square=True, cmap =colors, center=0.5, cbar= False, xticklabels= False, yticklabels= False)
        plt.savefig(ospj(ofpath_figure_adj, "colorbar_unordered.pdf"))
        #############################   
        
        #plot distributions of FC
        plot, axes = plt.subplots(figsize=(5, 5), dpi = 600)
        sns.histplot( FC_intersect_GMWM_mean[freq][  np.triu_indices( len(FC_intersect_GMWM_mean[freq]), k = 1)   ], kde = True, color = (0.2, 0.2, 0.2, 0) , ax = axes); 
        axes.set_xlim([0.05, 1]); axes.set_ylim([0, 400])
        axes.title.set_text("{0}_{1}_{2}_FullNetwork".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_ylabel('Count')
        axes.set_xlabel(r'Coherence ($\gamma$)' )
        plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_ALL.pdf"))
        
        plot1, axes1 = plt.subplots(figsize=(5, 5), dpi = 600)
        sns.histplot( FC_intersect_GMWM_mean_distances_order_WM[freq][  np.triu_indices( len(FC_intersect_GMWM_mean_distances_order_WM[freq]), k = 1)   ], kde = True, color = (	0.463, 0.686, 0.875, 0) , ax = axes1); 
        sns.histplot( FC_intersect_GMWM_mean_distances_order_GM[freq][  np.triu_indices( len(FC_intersect_GMWM_mean_distances_order_GM[freq]), k = 1)   ], kde = True, color = (0.659, 0.553, 0.463, 0) , ax = axes1); 
        axes1.set_xlim([0.05, 1]); #axes1.set_ylim([0, 100])
        axes1.title.set_text("{0}_{1}_{2}_GMWM".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        axes1.spines['top'].set_visible(False)
        axes1.spines['right'].set_visible(False)
        axes1.set_ylabel('Count')
        axes1.set_xlabel(r'Coherence ($\gamma$)' )
        plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_GM_WM.pdf"))
        
        
        #GM-WM connections
        #plot distributions of FC
        plot10, axes10 = plt.subplots(figsize=(5, 5), dpi = 600)
        sns.histplot( np.ndarray.flatten(FC_intersect_GMWM_mean_distances_order_GMWM[freq]), kde = True, color = (0.506, 0.49, 0.773, 0) , ax = axes10); 
        axes10.set_xlim([0.05, 1]); axes10.set_ylim([0,  238.35])
        axes10.title.set_text("{0}_{1}_{2}_GM_to_WM_connections".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        axes10.spines['top'].set_visible(False)
        axes10.spines['right'].set_visible(False)
        axes10.set_ylabel('Count')
        axes10.set_xlabel(r'Coherence ($\gamma$)' )
        plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_GM-WM.pdf"))
        
        
    
        #Plot distribution ECDF differences
        FC_GMWM_for_plot_df = pd.concat([pd.DataFrame({"Tissue": np.repeat("GM-GM", len(FC_GMGM)), "FC": FC_GMGM}), pd.DataFrame({"Tissue": np.repeat("WM-WM", len(FC_WMWM)), "FC": FC_WMWM})] )
        plt.figure(figsize=(5, 5), dpi = 600); 
        plot2, axes2 = plt.subplots(figsize=(5, 5), dpi = 600)
        sns.ecdfplot( data =FC_GMWM_for_plot_df, x="FC" , hue="Tissue", ax = axes2)
        axes2.set_xlim([0.05, 1]); axes2.set_ylim([0, 1])
        axes2.title.set_text("{0}_{1}_{2}_ECDFplot".format(sub_RID,descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        axes2.spines['top'].set_visible(False)
        axes2.spines['right'].set_visible(False)
        axes2.set_ylabel('Count')
        axes2.set_xlabel(r'Coherence ($\gamma$)' )
        plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_ECDFplot.pdf"))
        print(mannwhitneyu(FC_GMGM ,FC_WMWM  )[1])
    
        #################################
        #Plot color bar for electrodes
        electrode_localization_intersect_GMWM_distances_order
        electrode_localization_intersect_GMWM
        distances_order_ind
        electrode_colors = ["#E7463D", "#6370E7",  "#E7CE60", "#ABE798", "#DD8EE7", "#76D2E7", "#E77C39", "#50934E", "#8858AB", "#7C1D22", "#7D481E", "#101E59", "#B9B9B9"] #ordered from LA to RG electrodes for sub-RID0278
        electrode_colors_name = ["LA", "LB","LC", "LD", "LE", "LF", "LG", "RA", "RB", "RC", "RE", "RF", "RG"]
        electrode_colors_bar_ordered_distance = []
        for e in range(len(electrode_localization_intersect_GMWM_distances_order["electrode_name"])):
            electrode_name = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[e]
            electrode_colors_bar_ordered_distance.append(   electrode_colors[np.where(   np.array(electrode_name[0:2] )  == electrode_colors_name)[0][0]] )
     
        electrode_colors_bar_ordered_electrodes = []
        for e in range(len(electrode_localization_intersect_GMWM["electrode_name"])):
            electrode_name = np.array(electrode_localization_intersect_GMWM["electrode_name"])[e]
            electrode_colors_bar_ordered_electrodes.append(   electrode_colors[np.where(   np.array(electrode_name[0:2] )  == electrode_colors_name)[0][0]] )
     
        plt.figure(figsize=(5, 1), dpi = 600); 
        electrodes_plot = np.zeros(shape=(1, len(electrode_localization_intersect_GMWM)))
        electrodes_plot[0] = np.array(range(len(electrode_localization_intersect_GMWM["electrode_name"])))
        sns.heatmap(electrodes_plot, square=True, cmap =electrode_colors_bar_ordered_electrodes, cbar= False, xticklabels= False, yticklabels= False)
        plt.savefig(ospj(ofpath_figure_adj, "colorbar_electrodes_ordered_byElectrodes.pdf"))
        
        plt.figure(figsize=(5, 1), dpi = 600); 
        sns.heatmap(electrodes_plot, square=True, cmap =electrode_colors_bar_ordered_distance, cbar= False, xticklabels= False, yticklabels= False)
        plt.savefig(ospj(ofpath_figure_adj, "colorbar_electrodes_ordered_distances.pdf"))
        
        ###############################################
        
        
        """
        
       
        """
        #plot Network
        tmp = copy.deepcopy(FC_intersect_GMWM_mean_distances_order[freq])
        tmp[np.where(tmp <threshold ) ] = 0
        G = nx.from_numpy_array(tmp) 
        G.edges(data=True)
        pos = circular_layout(G)
        #plt.figure(figsize=(10, 10), dpi = 100); nx.draw_circular(G, with_labels=True, font_weight='bold')
    
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
            
    
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            
        plt.figure(figsize=(5, 5), dpi = 300)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors_node)
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=1, edge_color = colors_edge)
        #nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif", labels = labels)
        plt.title("{0}_{1}_{2}_{3}".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
    
        
        ofpath_figure_circles = ospj(ofpath_figure, "circles")
        if not (os.path.isdir(ofpath_figure_circles)): os.makedirs(ofpath_figure_circles, exist_ok=True)
        #plt.savefig(ospj(ofpath_figure_circles, "{0}_{1}_{2}_{3}.pdf".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  )))
        
        
        
        #plot GM-WM
        tmp = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_GMWM_plot[freq])
        tmp[np.where(tmp <threshold ) ] = 0
        G = nx.from_numpy_array(tmp) 
        pos = circular_layout(G)
        
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
            plt.figure(figsize=(5, 5), dpi = 300)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors_node)
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=1, edge_color = ["#817dc5"])
        nx.draw_networkx_labels(G, pos, font_size=4, font_family="sans-serif", labels = labels)
        plt.title("{0}_{1}_{2}_{3}".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        """
        ##############################################
        #%
        """
        #plots for data mining
        for d in range(2):#plot one with annotations, plot one without
            if (d == 0): annot = False
            if (d == 1): annot = True
            print( "{0}   {1}    {2}".format(sub_RID, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
            font_size_annot = 0.01
            font_size_elec = 2.1
    
            ofpath_figure_dataMining_subID = ospj(ofpath_figure_dataMining, sub_RID, order_of_matrices_in_pickle_file.iloc[freq][0])
            if not (os.path.isdir(ofpath_figure_dataMining_subID)): os.makedirs(ofpath_figure_dataMining_subID, exist_ok=True)
            
            len_gm = len(np.where(distances <= 0)[0])
            len(FC_intersect_GMWM_mean[freq])
            ########################
            ########################
            ########################
            ##### HEATMAP
     
            fig, axs = plt.subplots(nrows = 3, ncols=6,figsize=(20, 10), dpi = 600, gridspec_kw=dict(width_ratios=[0.2,4,0.2,0.2,4,0.2], height_ratios=[0.2,4,0.2]))
            axs[0][0].axis('off'); axs[0][2].axis('off');axs[0][3].axis('off');axs[0][5].axis('off');axs[2][0].axis('off');axs[2][2].axis('off');axs[2][3].axis('off');axs[2][5].axis('off');
            fig.subplots_adjust(wspace=0.05)
            fig.subplots_adjust(hspace=0.05)
            fig.suptitle( "{0}   {1}    {2}".format(sub_RID, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ), y=0.93)
            #ordered by Electrodes
            sns.heatmap(FC_intersect_GMWM_mean[freq], vmin = 0, vmax = 1, cmap = "mako", center=0.4, annot=annot, cbar=False, ax=axs[1][1], annot_kws=dict(size=font_size_annot, rotation= 0)); 
            cbar = fig.colorbar(axs[1][1].collections[0], cax=axs[1][2]); cbar.outline.set_visible(False)
            #ordered by Distances from GM
            sns.heatmap(FC_intersect_GMWM_mean_distances_order[freq], vmin = 0, vmax = 1, cmap = "mako", center=0.4, annot=annot, cbar=False, ax=axs[1][4], annot_kws=dict(fontsize=font_size_annot, rotation= 0));  
            cbar = fig.colorbar(axs[1][4].collections[0], cax=axs[1][5]);  cbar.outline.set_visible(False)
         
           
            axs[1][2].yaxis.set_ticks_position('left'); axs[1][2].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
            axs[1][5].yaxis.set_ticks_position('left'); axs[1][5].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
            axs[1][1].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
            axs[1][4].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
    
            axs[1][1].set_xticklabels( axs[1][1].get_xmajorticklabels(), fontsize =font_size_elec)
            axs[1][1].set_yticklabels( axs[1][1].get_ymajorticklabels(), fontsize =font_size_elec)
            axs[1][4].set_xticklabels( axs[1][4].get_xmajorticklabels(), fontsize =font_size_elec)
            axs[1][4].set_yticklabels( axs[1][4].get_ymajorticklabels(), fontsize =font_size_elec)
            axs[1][1].tick_params(axis='both', which='major', pad=1.8)
            axs[1][4].tick_params(axis='both', which='major', pad=1.8)
            axs[1][2].tick_params(axis='both', which='major', pad=1)
            axs[1][5].tick_params(axis='both', which='major', pad=1)
            #Colorbars
            #Colorbar Distances
            distances_colorbar_orderElectrodes = np.zeros(shape=(1, len(electrode_localization_intersect_GMWM)))
            distances_colorbar_orderElectrodes[0] = np.array(electrode_localization_intersect_GMWM["distances_label_2"])
            distances_colorbar_orderDistances = np.zeros(shape=(1, len(electrode_localization_intersect_GMWM_distances_order)))
            distances_colorbar_orderDistances[0] = np.array(electrode_localization_intersect_GMWM_distances_order["distances_label_2"])
            colormap_distances = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b6d4ee", "#184268"])
            colors_distances = np.ndarray.tolist(np.repeat("#c6b4a5" ,255) )
            for c in range(255):
                colors_distances.append( rgb2hex(int(colormap_distances(c)[0]*255), int(colormap_distances(c)[1]*255), int(colormap_distances(c)[2]*255))  )
            #Colorbar for Distances  - Order Electrodes
            sns.heatmap(distances_colorbar_orderElectrodes, square=False, cmap =colors_distances, center=0.5, vmin = 0, vmax = 9, cbar= False, xticklabels= False, yticklabels= False, annot=True, ax=axs[2][1],annot_kws=dict(fontsize=font_size_elec, rotation= 90) )
            #Colorbarfor  Distances - Order Distances
            sns.heatmap(distances_colorbar_orderDistances, square=False, cmap =colors_distances, center=0.5, vmin = 0, vmax = 9, cbar= False, xticklabels= False, yticklabels= False, annot=True, ax=axs[2][4],annot_kws=dict(fontsize=font_size_elec, rotation= 90) )
            
            #Color Bar for electrodes
            electrode_localization_intersect_GMWM_distances_order
            electrode_localization_intersect_GMWM
            distances_order_ind
            #Colorbar - Get number of electrodes sticks:
            electrode_stick_names = np.unique(np.array([e[:2] for e in  np.array(electrode_localization_intersect_GMWM["electrode_name"])]))
            colormap_electrodes = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#d67777","#d6a677","#d6d677","#a7d677","#77a7d6","#7778d6", "#d677a7"]) #rainbow ROYGBIV
            electrode_stick_N = len(electrode_stick_names)
            #Colorbar - Make one unique color for each electrode stick base on the rainbow above
            colors_electrodes = []
            
            
            for c in np.linspace(start=0, stop=255, num=electrode_stick_N).astype(int):
                colors_electrodes.append( rgb2hex(int(colormap_electrodes(c)[0]*255), int(colormap_electrodes(c)[1]*255), int(colormap_electrodes(c)[2]*255))  )
            electrode_contact_names = np.array(electrode_localization_intersect_GMWM["electrode_name"])
            electrode_contact_names_orderDistances = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])
            electrode_colors_bar_ordered_electrodes = []
            for e in range(len(electrode_localization_intersect_GMWM["electrode_name"])):
                electrode_name = np.array(electrode_localization_intersect_GMWM["electrode_name"])[e]
                electrode_colors_bar_ordered_electrodes.append(   colors_electrodes[np.where(   np.array(electrode_name[0:2] )  == electrode_stick_names)[0][0]] )
        
            electrode_colors_bar_ordered_distance = []
            for e in range(len(electrode_localization_intersect_GMWM_distances_order["electrode_name"])):
                electrode_name = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[e]
                electrode_colors_bar_ordered_distance.append(   colors_electrodes[np.where(   np.array(electrode_name[0:2] )  == electrode_stick_names)[0][0]] )
            #Colorbars - PLOT
            electrodes_color_bar_plot = np.zeros(shape=(1, len(electrode_localization_intersect_GMWM)))
            electrodes_color_bar_plot[0] = np.array(range(len(electrode_localization_intersect_GMWM["electrode_name"])))
            sns.heatmap(electrodes_color_bar_plot, square=False, cmap =electrode_colors_bar_ordered_electrodes, cbar= False, xticklabels= False, yticklabels= False, annot=electrode_contact_names.reshape(electrodes_color_bar_plot.shape), fmt = '', ax=axs[0][1], annot_kws=dict(fontsize=font_size_elec, rotation= 90))
            sns.heatmap(np.transpose(electrodes_color_bar_plot), square=False, cmap =electrode_colors_bar_ordered_electrodes, cbar= False, xticklabels= False, yticklabels= False, annot=electrode_contact_names.reshape(np.transpose(electrodes_color_bar_plot).shape), fmt = '', ax=axs[1][0], annot_kws=dict(fontsize=font_size_elec))
            sns.heatmap(electrodes_color_bar_plot, square=False, cmap =electrode_colors_bar_ordered_distance, cbar= False, xticklabels= False,   yticklabels= False, annot=electrode_contact_names_orderDistances.reshape(electrodes_color_bar_plot.shape), fmt = '', ax=axs[0][4], annot_kws=dict(fontsize=font_size_elec, rotation= 90))
            sns.heatmap(np.transpose(electrodes_color_bar_plot), square=False, cmap =electrode_colors_bar_ordered_distance, cbar= False, xticklabels= False,   yticklabels= False, annot=electrode_contact_names_orderDistances.reshape(np.transpose(electrodes_color_bar_plot).shape), fmt = '', ax=axs[1][3], annot_kws=dict(fontsize=font_size_elec))
          
            if (d == 0): ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "{0}_{1}_{2}_{3}.pdf".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
            if (d == 1): ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "{0}_{1}_{2}_{3}_annotation.pdf".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
            plt.savefig(ofname_figure_adj)
            
            ########################
            ########################
            ########################
            ########################
            ########################
            ########################
            #Difference btweeen Ictal and Preictal  - Heatmap
            if descriptor == descriptors[2]:
                print(  "Ictal-Prictal Difference:{0}   {1}    {2}".format(sub_RID, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
                delta_order_electrodes
                delta_order_distances
                fig1, axs1 = plt.subplots(nrows = 3, ncols=6,figsize=(20, 10), dpi = 600, gridspec_kw=dict(width_ratios=[0.2,4,0.2,0.2,4,0.2], height_ratios=[0.2,4,0.2]))
                axs1[0][0].axis('off'); axs1[0][2].axis('off');axs1[0][3].axis('off');axs1[0][5].axis('off');axs1[2][0].axis('off');axs1[2][2].axis('off');axs1[2][3].axis('off');axs1[2][5].axis('off');
                fig1.subplots_adjust(wspace=0.05)
                fig1.subplots_adjust(hspace=0.05)
                fig1.suptitle( "Ictal-Prictal Difference:\n{0}   {1}    {2}".format(sub_RID, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ), y=0.93)
                vmin = np.min(delta_order_electrodes); vmax = np.max(delta_order_electrodes)
                #ordered by Electrodes
                sns.heatmap(delta_order_electrodes, square=False, vmin = vmin, vmax = vmax, cmap = "mako", annot=annot, cbar=False, ax=axs1[1][1], annot_kws=dict(fontsize=font_size_annot)); 
                #ordered by Distances from GM
                sns.heatmap(delta_order_distances, square=False, vmin = vmin, vmax = vmax, cmap = "mako", annot=annot, cbar=False, ax=axs1[1][4], annot_kws=dict(fontsize=font_size_annot));  
                cbar = fig1.colorbar(axs1[1][1].collections[0], cax=axs1[1][2]);  cbar.outline.set_visible(False) 
                cbar = fig1.colorbar(axs1[1][4].collections[0], cax=axs1[1][5]);  cbar.outline.set_visible(False)
                axs1[1][2].yaxis.set_ticks_position('left'); axs1[1][2].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
                axs1[1][5].yaxis.set_ticks_position('left'); axs1[1][5].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
                axs1[1][1].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
                axs1[1][4].tick_params(direction='in', length=0, width=0, labelsize=font_size_elec)
        
                axs1[1][1].set_xticklabels( axs1[1][1].get_xmajorticklabels(), fontsize =font_size_elec)
                axs1[1][1].set_yticklabels( axs1[1][1].get_ymajorticklabels(), fontsize =font_size_elec)
                axs1[1][4].set_xticklabels( axs1[1][4].get_xmajorticklabels(), fontsize =font_size_elec)
                axs1[1][4].set_yticklabels( axs1[1][4].get_ymajorticklabels(), fontsize =font_size_elec)
                axs1[1][1].tick_params(axis='both', which='major', pad=1.8)
                axs1[1][4].tick_params(axis='both', which='major', pad=1.8)
                axs1[1][2].tick_params(axis='both', which='major', pad=1)
                axs1[1][5].tick_params(axis='both', which='major', pad=1)
                        
                #colorbars
                sns.heatmap(distances_colorbar_orderElectrodes, square=False, cmap =colors_distances, center=0.5, vmin = 0, vmax = 9, cbar= False, xticklabels= False, yticklabels= False, annot=True, ax=axs1[2][1], annot_kws=dict(fontsize=font_size_annot) )
        
                sns.heatmap(distances_colorbar_orderDistances, square=False, cmap =colors_distances, center=0.5, vmin = 0, vmax = 9, cbar= False, xticklabels= False, yticklabels= False, annot=True, ax=axs1[2][4], annot_kws=dict(fontsize=font_size_annot) )
                sns.heatmap(electrodes_color_bar_plot, square=False, cmap =electrode_colors_bar_ordered_electrodes, cbar= False, xticklabels= False, yticklabels= False, annot=electrode_contact_names.reshape(electrodes_color_bar_plot.shape), fmt = '', ax=axs1[0][1], annot_kws=dict(fontsize=font_size_elec, rotation= 90))
                sns.heatmap(np.transpose(electrodes_color_bar_plot), square=False, cmap =electrode_colors_bar_ordered_electrodes, cbar= False, xticklabels= False, yticklabels= False, annot=electrode_contact_names.reshape(np.transpose(electrodes_color_bar_plot).shape), fmt = '', ax=axs1[1][0], annot_kws=dict(fontsize=font_size_elec))
                sns.heatmap(electrodes_color_bar_plot, square=False, cmap =electrode_colors_bar_ordered_distance, cbar= False, xticklabels= False,   yticklabels= False, annot=electrode_contact_names_orderDistances.reshape(electrodes_color_bar_plot.shape), fmt = '', ax=axs1[0][4], annot_kws=dict(fontsize=font_size_elec, rotation= 90))
                sns.heatmap(np.transpose(electrodes_color_bar_plot), square=False, cmap =electrode_colors_bar_ordered_distance, cbar= False, xticklabels= False,   yticklabels= False, annot=electrode_contact_names_orderDistances.reshape(np.transpose(electrodes_color_bar_plot).shape), fmt = '', ax=axs1[1][3], annot_kws=dict(fontsize=font_size_elec))
                if (d == 0): ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "{0}_DIFFERENCE_ictal-preictal_{1}.pdf".format(sub_RID, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
                if (d == 1): ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "{0}_DIFFERENCE_ictal-preictal_{1}_annotation.pdf".format(sub_RID, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
                plt.savefig(ofname_figure_adj)
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ######################## 
        #%
        #network Connectivity
        #plot Network
        print(  "{0}   {1}    {2}".format(sub_RID, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        #sns.set_style("whitegrid", {'axes.grid' : False})
        node_size = 12
        label_size = 1.5
        edge_size = 0.5
        
        fig2, axs2 = plt.subplots(nrows = 2, ncols=4,figsize=(10, 5), dpi = 600, gridspec_kw=dict(width_ratios=[1,1,1,1], height_ratios=[1,1]))
        fig2.subplots_adjust(hspace=-0.05, wspace=-0.05)
        fig2.suptitle( "{0}   {1}    {2}".format(sub_RID, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ), y=0.93)
        #FULL NETWORK - order: electrodes
        network = copy.deepcopy(FC_intersect_GMWM_mean[freq])
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        pos = circular_layout(G)
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"])[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM["region_number"])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
            
    
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[0,0])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs2[0,0])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[0,0])
        axs2[0,0].axis("off")
        
        
        #FULL NETWORK - order: distance
        network = copy.deepcopy(FC_intersect_GMWM_mean_distances_order[freq])
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        pos = circular_layout(G)
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
            
    
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[0,2])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs2[0,2])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[0,2])
        axs2[0,2].axis("off")
        
        
        
        
        #GM Network - order: electrodes
        network = copy.deepcopy(FC_intersect_GMWM_mean[freq][distances_GM_ind_order_electrode[:,None], distances_GM_ind_order_electrode[None,:]]  )
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        G.edges(data=True)
        pos = circular_layout(G)
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"][electrode_localization_intersect_GMWM["region_number"] == 2 ]  )[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[1,0])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs2[1,0])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[1,0])
        axs2[1,0].axis("off")
        
        
        #GM Network - order: distance
        network = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_GM[freq])
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        G.edges(data=True)
        pos = circular_layout(G)
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ]  )[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[1,2])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs2[1,2])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[1,2])
        axs2[1,2].axis("off")
        
        
        
        #WM Network - order: electrodes
        network = copy.deepcopy(FC_intersect_GMWM_mean[freq][distances_WM_ind_order_electrode[:,None], distances_WM_ind_order_electrode[None,:]] )
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        G.edges(data=True)
        pos = circular_layout(G)
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"][electrode_localization_intersect_GMWM["region_number"] == 3 ]  )[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[1,1])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs2[1,1])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[1,1])
        axs2[1,1].axis("off")
        
        #WM Network - order: distance
        network = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_WM[freq])
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        G.edges(data=True)
        pos = circular_layout(G)
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ]  )[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[0]] 
            region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[1]] 
            if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
            if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
            if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
            if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
            colors_edge.append(clr)
        edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
        colors_edge = [] 
        for idx, edge in enumerate(G.edges()): 
            if edges[idx][2]["weight"] > threshold:
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[1,3])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs2[1,3])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[1,3])
        axs2[1,3].axis("off")
    
    
        #plot GM-WM - order: electrodes
        network = copy.deepcopy(FC_intersect_GMWM_mean_GMWM_plot[freq])
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        pos = circular_layout(G)
        
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"])[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM["region_number"])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[0,1])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = ["#817dc5"], ax = axs2[0,1])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[0,1])
        axs2[0,1].axis("off")
        
        
        #plot GM-WM - order: distance
        network = copy.deepcopy(FC_intersect_GMWM_mean_distances_order_GMWM_plot[freq])
        network[np.where(network <threshold ) ] = 0
        G = nx.from_numpy_array(network) 
        pos = circular_layout(G)
        
        labels = {} 
        for idx, node in enumerate(G.nodes()): 
            labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[idx] 
        colors_node = [] 
        for idx, node in enumerate(G.nodes()): 
            region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[idx] 
            if region_id ==2: clr = "#c6b4a5"
            if region_id ==3: clr = "#b6d4ee"
            colors_node.append(clr)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs2[0,3])
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = ["#817dc5"], ax = axs2[0,3])
        nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs2[0,3])
        axs2[0,3].axis("off")
        ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "NETWORK_{0}_{1}_{2}_{3}.pdf".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        plt.savefig(ofname_figure_adj)
       
        ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "NETWORK_{0}_{1}_{2}_{3}.pdf".format(sub_RID, per, descriptor, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
        plt.savefig(ofname_figure_adj)
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ########################
        ######################## 
        #%
        if descriptor == descriptors[2]:
            print(  "Ictal-Prictal Difference:\n{0}  {1}".format(sub_RID, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
            #Network Connectivity Differences between ictal and preictal
            #plot Network
            sns.set_style("whitegrid", {'axes.grid' : False})
        
            threshold_delta_positive = 0.15
            fig3, axs3 = plt.subplots(nrows = 2, ncols=4,figsize=(10, 5), dpi = 600, gridspec_kw=dict(width_ratios=[1,1,1,1], height_ratios=[1,1]))
            fig3.subplots_adjust(hspace=-0.05, wspace=-0.05)
            fig3.suptitle( "Ictal-Prictal Difference:\n{0}  {1}".format(sub_RID, order_of_matrices_in_pickle_file.iloc[freq][0]  ), y=0.93)
            #FULL NETWORK - order: electrodes
            network = copy.deepcopy(delta_order_electrodes)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            G.edges(data=True)
            pos = circular_layout(G)
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"])[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM["region_number"])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
                
        
            edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                if edges[idx][2]["weight"] > threshold_delta_positive:
                    region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[0]] 
                    region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"])[edge[1]] 
                    if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                    if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                    if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                    if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                    colors_edge.append(clr)
                
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[0,0])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs3[0,0])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[0,0])
            axs3[0,0].axis("off")
            
            
            #FULL NETWORK - order: distance
            network = copy.deepcopy(delta_order_distances)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            G.edges(data=True)
            pos = circular_layout(G)
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
                
        
            edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                if edges[idx][2]["weight"] > threshold_delta_positive:
                    region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[0]] 
                    region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[edge[1]] 
                    if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                    if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                    if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                    if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                    colors_edge.append(clr)
                
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[0,2])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs3[0,2])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[0,2])
            axs3[0,2].axis("off")
            
            
            #GM Network - order: electrodes
            network = copy.deepcopy(delta_order_electrodes_GM)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            G.edges(data=True)
            pos = circular_layout(G)
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"][electrode_localization_intersect_GMWM["region_number"] == 2 ]  )[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                if edges[idx][2]["weight"] > threshold_delta_positive:
                    region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[0]] 
                    region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 2 ])[edge[1]] 
                    if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                    if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                    if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                    if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                    colors_edge.append(clr)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[1,0])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs3[1,0])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[1,0])
            axs3[1,0].axis("off")
            
            
            
            
            #GM Network - order: distance
            network = copy.deepcopy(delta_order_distances_GM)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            G.edges(data=True)
            pos = circular_layout(G)
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ]  )[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                if edges[idx][2]["weight"] > threshold_delta_positive:
                    region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[0]] 
                    region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 2 ])[edge[1]] 
                    if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                    if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                    if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                    if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                    colors_edge.append(clr)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[1,2])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs3[1,2])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[1,2])
            axs3[1,2].axis("off")
            
            
            
            #WM Network - order: electrodes
            network = copy.deepcopy(delta_order_electrodes_WM)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            G.edges(data=True)
            pos = circular_layout(G)
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"][electrode_localization_intersect_GMWM["region_number"] == 3 ]  )[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                if edges[idx][2]["weight"] > threshold_delta_positive:
                    region_id_1 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[0]] 
                    region_id_2 = np.array(electrode_localization_intersect_GMWM["region_number"][electrode_localization_intersect_GMWM["region_number"] == 3 ])[edge[1]] 
                    if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                    if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                    if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                    if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                    colors_edge.append(clr)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[1,1])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs3[1,1])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[1,1])
            axs3[1,1].axis("off")
            
        
            
            
            #WM Network - order: distance
            network = copy.deepcopy(delta_order_distances_WM)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            G.edges(data=True)
            pos = circular_layout(G)
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ]  )[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[0]] 
                region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[1]] 
                if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                colors_edge.append(clr)
            edges = [(u,v,d) for (u,v,d) in G.edges(data=True) if G.has_edge(v,u)]
            colors_edge = [] 
            for idx, edge in enumerate(G.edges()): 
                if edges[idx][2]["weight"] > threshold_delta_positive:
                    region_id_1 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[0]] 
                    region_id_2 = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"][electrode_localization_intersect_GMWM_distances_order["region_number"] == 3 ])[edge[1]] 
                    if (region_id_1 ==2 and region_id_2 == 2 ): clr = "#90735c"
                    if (region_id_1 ==3 and region_id_2 == 3 ): clr = "#8dbce4"
                    if (region_id_1 ==2 and region_id_2 == 3 ): clr = "#33333333"
                    if (region_id_1 ==3 and region_id_2 == 2 ): clr = "#33333333"
                    colors_edge.append(clr)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[1,3])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = colors_edge, ax = axs3[1,3])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[1,3])
            axs3[1,3].axis("off")
            
        
            delta_order_electrodes_GMWM_plot
            #plot GM-WM - order: electrodes
            network = copy.deepcopy(delta_order_electrodes_GMWM_plot)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            pos = circular_layout(G)
            
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM["electrode_name"])[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM["region_number"])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[0,1])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = ["#817dc5"], ax = axs3[0,1])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[0,1])
            axs3[0,1].axis("off")    
        
            
            #plot GM-WM - order: distance
            network = copy.deepcopy(delta_order_distances)
            network[np.where(network <threshold_delta_positive ) ] = 0
            G = nx.from_numpy_array(network) 
            pos = circular_layout(G)
            
            labels = {} 
            for idx, node in enumerate(G.nodes()): 
                labels[node] = np.array(electrode_localization_intersect_GMWM_distances_order["electrode_name"])[idx] 
            colors_node = [] 
            for idx, node in enumerate(G.nodes()): 
                region_id = np.array(electrode_localization_intersect_GMWM_distances_order["region_number"])[idx] 
                if region_id ==2: clr = "#c6b4a5"
                if region_id ==3: clr = "#b6d4ee"
                colors_node.append(clr)
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors_node, ax = axs3[0,3])
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold_delta_positive]
            nx.draw_networkx_edges(G, pos, edgelist= elarge, width=edge_size, edge_color = ["#817dc5"], ax = axs3[0,3])
            nx.draw_networkx_labels(G, pos, font_size=label_size, font_family="sans-serif", labels = labels, ax = axs3[0,3])
            axs3[0,3].axis("off")    
            ofname_figure_adj = ospj(ofpath_figure_dataMining_subID, "NETWORKDIFF_{0}_{1}.pdf".format(sub_RID, order_of_matrices_in_pickle_file.iloc[freq][0]  ))
            plt.savefig(ofname_figure_adj)
  """        
#%%
    


for l in range(len(diff_FC_GMWM)):
    diff_FC_GMWM[l] = np.delete(diff_FC_GMWM[l], 0)
FC_GMGM_all_ii = np.delete(FC_GMGM_all_ii, 0); FC_WMWM_all_ii = np.delete(FC_WMWM_all_ii, 0) #removing the initialized zero
FC_GMGM_all_pi = np.delete(FC_GMGM_all_pi, 0); FC_WMWM_all_pi = np.delete(FC_WMWM_all_pi, 0) #removing the initialized zero
FC_GMGM_all_ic = np.delete(FC_GMGM_all_ic, 0); FC_WMWM_all_ic = np.delete(FC_WMWM_all_ic, 0) #removing the initialized zero
FC_GMGM_all_po = np.delete(FC_GMGM_all_po, 0); FC_WMWM_all_po = np.delete(FC_WMWM_all_po, 0) #removing the initialized zero


net_netMeasures_subjects_copy = copy.deepcopy(net_netMeasures_subjects)
net_netMeasures_subjects = net_netMeasures_subjects.iloc[1:]
net_nodeMeasures_subjects = net_nodeMeasures_subjects.iloc[1:]
#%%
colors = ["#c6b4a5", "#b6d4ee"]
sns.set_palette(sns.color_palette(colors))


df_long_GMGM_all_ii = pd.DataFrame({"Tissue": np.repeat("GM-GM", len(FC_GMGM_all_ii)), "FC": FC_GMGM_all_ii})
df_long_GMGM_all_pi = pd.DataFrame({"Tissue": np.repeat("GM-GM", len(FC_GMGM_all_pi)), "FC": FC_GMGM_all_pi})
df_long_GMGM_all_ic = pd.DataFrame({"Tissue": np.repeat("GM-GM", len(FC_GMGM_all_ic)), "FC": FC_GMGM_all_ic})
df_long_GMGM_all_po = pd.DataFrame({"Tissue": np.repeat("GM-GM", len(FC_GMGM_all_po)), "FC": FC_GMGM_all_po})

df_long_WMWM_all_ii = pd.DataFrame({"Tissue": np.repeat("WM-WM", len(FC_WMWM_all_ii)), "FC": FC_WMWM_all_ii})
df_long_WMWM_all_pi = pd.DataFrame({"Tissue": np.repeat("WM-WM", len(FC_WMWM_all_pi)), "FC": FC_WMWM_all_pi})
df_long_WMWM_all_ic = pd.DataFrame({"Tissue": np.repeat("WM-WM", len(FC_WMWM_all_ic)), "FC": FC_WMWM_all_ic})
df_long_WMWM_all_po = pd.DataFrame({"Tissue": np.repeat("WM-WM", len(FC_WMWM_all_po)), "FC": FC_WMWM_all_po})


FC_df_long_all_ii = pd.concat([df_long_GMGM_all_ii, df_long_WMWM_all_ii] )
FC_df_long_all_pi = pd.concat([df_long_GMGM_all_pi, df_long_WMWM_all_pi] )
FC_df_long_all_ic = pd.concat([df_long_GMGM_all_ic, df_long_WMWM_all_ic] )
FC_df_long_all_po = pd.concat([df_long_GMGM_all_po, df_long_WMWM_all_po] )
plot = sns.displot( data =FC_df_long_all_ii  , kind = "ecdf", x="FC" , hue="Tissue"); plot.axes[0,0].set_ylim(0,1); plot.axes[0,0].set_xlim(0,1); plt.title("{0} \nInterictal".format(order_of_matrices_in_pickle_file.iloc[freq][0] ))
plot = sns.displot( data =FC_df_long_all_pi  , kind = "ecdf", x="FC" , hue="Tissue"); plot.axes[0,0].set_ylim(0,1); plot.axes[0,0].set_xlim(0,1); plt.title("{0} \nPreictal".format(order_of_matrices_in_pickle_file.iloc[freq][0]  ))
plot = sns.displot( data =FC_df_long_all_ic  , kind = "ecdf", x="FC" , hue="Tissue"); plot.axes[0,0].set_ylim(0,1); plot.axes[0,0].set_xlim(0,1); plt.title("{0} \nIctal".format(order_of_matrices_in_pickle_file.iloc[freq][0] ))
plot = sns.displot( data =FC_df_long_all_po  , kind = "ecdf", x="FC" , hue="Tissue"); plot.axes[0,0].set_ylim(0,1); plot.axes[0,0].set_xlim(0,1); plt.title("{0} \nPostictal".format(order_of_matrices_in_pickle_file.iloc[freq][0]  ))


FC_df_long_all = [FC_df_long_all_ii, FC_df_long_all_pi, FC_df_long_all_ic, FC_df_long_all_po]


per = 0
for per in range(4):
    #Plot distribution ECDF differences
    plt.figure(figsize=(5, 5), dpi = 600); 
    plot3, axes3 = plt.subplots(figsize=(5, 5), dpi = 600)
    sns.ecdfplot( data =FC_df_long_all[per], x="FC" , hue="Tissue", ax = axes3)
    
    axes3.set_xlim([0.05, 1]); axes3.set_ylim([0, 1])
    axes3.title.set_text( "FC_distribution_all_{0}_{1}_{2}_ECDFplot.pdf".format(per, descriptors[per], order_of_matrices_in_pickle_file.iloc[freq][0]  ))
    axes3.spines['top'].set_visible(False)
    axes3.spines['right'].set_visible(False)
    axes3.set_ylabel('Proportion')
    axes3.set_xlabel(r'Coherence ($\gamma$)' )
    ofpath_figure_adj = ospj(ofpath_figure, "adjacency_matrices")
    
    plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_all_{0}_{1}_{2}_ECDFplot.pdf".format(per, descriptors[per], order_of_matrices_in_pickle_file.iloc[freq][0]  )))




from scipy.stats import kruskal

np.nanmean(FC_GMGM_all_ii )
np.nanmean(FC_WMWM_all_ii  )
print(np.nanmedian(FC_WMWM_all_ii ) - np.nanmedian(FC_GMGM_all_ii  ))

np.nanmean(FC_GMGM_all_pi )
np.nanmean(FC_WMWM_all_pi  )
print(np.nanmedian(FC_WMWM_all_pi ) - np.nanmedian(FC_GMGM_all_pi  ))

np.nanmean(FC_GMGM_all_ic )
np.nanmean(FC_WMWM_all_ic  )
print(np.nanmedian(FC_WMWM_all_ic ) - np.nanmedian(FC_GMGM_all_ic  ))

np.nanmean(FC_GMGM_all_po )
np.nanmean(FC_WMWM_all_po  )
print(np.nanmedian(FC_WMWM_all_po ) - np.nanmedian(FC_GMGM_all_po  ))


print(kruskal(FC_GMGM_all_ii, FC_WMWM_all_ii)[1])
print(kruskal(FC_GMGM_all_pi, FC_WMWM_all_pi)[1])
print(kruskal(FC_GMGM_all_ic, FC_WMWM_all_ic)[1])
print(kruskal(FC_GMGM_all_po, FC_WMWM_all_po)[1])
kruskal(FC_GMGM_all_ic, FC_WMWM_all_ic)[1]



#%%

#calculating distances matrices to see if GM-GM distances are different from WM-WM distances
if (calculate_distances_boolean): 
    GM_distances = np.delete(GM_distances, 0); WM_distances = np.delete(WM_distances, 0) #removing the initialized zero
    df_long_distances_GM = pd.DataFrame({"Tissue": np.repeat("GM-GM", len(GM_distances)), "distances": GM_distances})
    df_long_distances_WM = pd.DataFrame({"Tissue": np.repeat("WM-WM", len(WM_distances)), "distances": WM_distances})
    df_long_distances = pd.concat([df_long_distances_GM, df_long_distances_WM] )
    np.nanmean(GM_distances )
    np.nanmean(WM_distances  )
    print(np.nanmean(GM_distances ) - np.nanmean(WM_distances  ))
    plot = sns.displot( data =df_long_distances  , kind = "hist", x="distances" , hue="Tissue", kde=True); plot.axes[0,0].set_ylim(0,); plot.axes[0,0].set_xlim(0,); plt.title("{0} \nPostictal".format(order_of_matrices_in_pickle_file.iloc[freq][0]  ))
    
    print(mannwhitneyu(GM_distances ,WM_distances  )[1])






#%%
#Regressing out distances

per = range(4)
fig,axes = plt.subplots(4,4,figsize=(20,5))
for p in range(len(per)):
    FC_vs_distance = pd.concat([FC_df_long_all[p], df_long_distances.drop(["Tissue"], axis=1)   ], axis=1)
    plt.figure(dpi = 100)
    #plot =  sns.regplot(data =FC_vs_distance, x = "distances", y = "FC", marker='o', scatter_kws={'s':0.5});  plt.title("Distances vs FC {0}\n{1}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p] ))
    
    
    
    linear_regression = pg.linear_regression(FC_vs_distance['distances'], FC_vs_distance['FC'])
    
    
    
    all(FC_df_long_all_ii["Tissue"] == df_long_distances["Tissue"])
    
    
    OLS_model = sm.OLS(FC_vs_distance['FC'] , sm.add_constant ( FC_vs_distance['distances'])).fit()  # training the model
    predicted_values = OLS_model.predict()  # predicted values
    OLS_model.resid
    residual_values = OLS_model.resid # residual values
    residual_values = residual_values.rename("residuals")
    
    FC_vs_distance_residuals = pd.concat([FC_vs_distance, residual_values],  axis=1)
    
    
    #plot = sns.displot( data =FC_vs_distance_residuals  , kind = "ecdf", x="residuals" , hue="Tissue"); plot.axes[p][0,0].set_ylim(0,); plt.title("{0} \n{1}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p]  ))
    FC_vs_distance_residuals["Tissue"]
    

    print("Significance: {0}, {1}: {2}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p] ,    mannwhitneyu(FC_vs_distance_residuals.loc[FC_vs_distance_residuals['Tissue'] == "GM-GM"]["residuals"] ,FC_vs_distance_residuals.loc[FC_vs_distance_residuals['Tissue'] == "WM-WM"]["residuals"]  )[1]))
    
    
    mod = sm.OLS(FC_vs_distance['FC'],sm.add_constant(FC_vs_distance['distances']))
    res = mod.fit()
    res.resid
    test = FC_vs_distance[['distances']].drop_duplicates().sort_values('distances')
    predictions = res.get_prediction(sm.add_constant(test))
    predictions = pd.concat([test,predictions.summary_frame(alpha=0.05)],axis=1)
    predictions.head()
    sns.scatterplot(x='distances',y='FC',ax=axes[p][0],data=FC_vs_distance, s= 1)
    axes[p][0].plot(predictions['distances'], predictions['mean'], lw=2)
    axes[p][0].fill_between(x=predictions['distances'],
                         y1=predictions['mean_ci_lower'],y2=predictions['mean_ci_upper'],
                         facecolor='blue', alpha=0.2)
    sns.scatterplot(x='distances',y='FC',ax=axes[p][1], s = 1,
                    data=pd.DataFrame({'distances':FC_vs_distance['distances'],
                                      'FC':res.resid})
                    )
    axes[p][1].axhline(0, ls='--',color="k")
    sns.histplot(res.resid,ax=axes[p][2],bins=20, kde=True)
    
    FC_vs_distance_residuals = pd.concat([FC_vs_distance, pd.DataFrame( dict(residuals=res.resid))],  axis=1)
    sns.ecdfplot( data =FC_vs_distance_residuals  , x="residuals" , hue="Tissue", ax=axes[p][3]);# ax=axes[p][3].set_ylim(0,); #plt.title("{0} \n{1}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p]  ))
 
    
    
    
    model = LinearRegression().fit(  np.array(FC_vs_distance['distances']).reshape(-1,1),     np.array(FC_vs_distance['FC']).reshape(-1,1)  )   
    preds = model.predict(   np.array(FC_vs_distance['distances']).reshape(-1,1)   )
    res_function = np.array(FC_vs_distance['FC']).reshape(-1,1)    - preds
  

from pygam import LinearGAM, s, f
X0 = FC_vs_distance[['distances']].to_numpy()
y0 = FC_vs_distance['FC'].to_numpy()
gam = LinearGAM(s(0) ).fit(X0, y0)

r0 = gam.deviance_residuals(X0, y0)


y1 = gam.predict(X0)
y0 - y1

fig,axes = plt.subplots(1,4,figsize=(15,5), dpi = 300)
sns.scatterplot(x=X0.flatten() ,y=y0 ,ax=axes[0], s = 1)
sns.scatterplot(x=X0.flatten() ,y=y1 ,ax=axes[0], s = 1)

FC_vs_distance_residuals = pd.concat([FC_vs_distance.reset_index(drop=True), pd.DataFrame( dict(residuals=r0))],  axis=1)
sns.ecdfplot( data =FC_vs_distance_residuals  , x="residuals" , hue="Tissue");# ax=axes[p][3].set_ylim(0,); #plt.title("{0} \n{1}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p]  ))
 
    
#%%
# Set your custom color palette
colors = ["#c6b4a5", "#b6d4ee"]
sns.set_palette(sns.color_palette(colors))

diff_FC_GMWM



{"sub_ID": sub_IDs_unique, "interictal":diff_FC_GMWM[0] , "preictal":diff_FC_GMWM[1], "ictal":diff_FC_GMWM[2], "postictal":diff_FC_GMWM[3] }
df_diff_FC_GMWM = pd.DataFrame({"sub_ID": sub_IDs_unique, "interictal":diff_FC_GMWM[0] , "preictal":diff_FC_GMWM[1], "ictal":diff_FC_GMWM[2], "postictal":diff_FC_GMWM[3] })
#df_diff_FC_GMWM = df_diff_FC_GMWM.drop([4,6,7])
df_diff_FC_GMWM_long = pd.melt(df_diff_FC_GMWM, id_vars=['sub_ID'], value_vars=['interictal', 'preictal', 'ictal', 'postictal'])




plt.figure(dpi = 300); 
flierprops = dict(marker='o', markerfacecolor='#00000088', markersize=0.0, linestyle='none', markeredgecolor='#00000000')
ax = sns.boxplot(x="variable", y="value", data=df_diff_FC_GMWM_long,  linewidth=0.9, whis = 1, width=0.8, flierprops = flierprops, showfliers = False, palette = ["#d6cdf1", "#b19fe4", "#8c71d7", "#b19fe4"])
ax = sns.swarmplot(x="variable", y="value", data=df_diff_FC_GMWM_long, color=".25")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([-0.019, 0.14])
ax.title.set_text( "FC_distribution_delta_plot_{0}.pdf".format(order_of_matrices_in_pickle_file.iloc[freq][0]  ))
ax.set_xlabel('')
ax.set_ylabel(r'Mean $\Delta$' )
ofpath_figure_adj = ospj(ofpath_figure, "adjacency_matrices")

plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_delta_plot_{0}.pdf".format(order_of_matrices_in_pickle_file.iloc[freq][0]  )))





#ax.set_ylim([-0.015,0.2])
print(df_diff_FC_GMWM)
#%%

from scipy import stats

# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
effect = cohen_d(df_diff_FC_GMWM[ "ictal"], df_diff_FC_GMWM["preictal"])
alpha = 0.05/20
power = 0.8
# perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)
effect = cohen_d(df_diff_FC_GMWM[ "ictal"], df_diff_FC_GMWM["interictal"])
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)


print(   stats.ttest_rel(diff_FC_GMWM[2],diff_FC_GMWM[0])[1]   )



np.mean(diff_FC_GMWM[0])
np.mean(diff_FC_GMWM[1])
np.mean(diff_FC_GMWM[2])
np.mean(diff_FC_GMWM[3])


print(mannwhitneyu(diff_FC_GMWM[0] ,diff_FC_GMWM[2]  )[1])

df_diff_FC_GMWM.mean(axis=1)


plot = sns.displot( data =df_diff_FC_GMWM_long  , kind = "kde", x="value" , hue="sub_ID"); plot.axes[0,0].set_ylim(0,); plt.title("{0} \n{1}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p]  ))
#plot = sns.displot( data =df_diff_FC_GMWM.mean(axis=1)  , kind = "kde"); plot.axes[0,0].set_ylim(0,); plt.title("{0} \n{1}".format(order_of_matrices_in_pickle_file.iloc[freq][0], descriptors[p]  ))

stats.ttest_1samp(diff_FC_GMWM[2]  , 0)[1]
stats.ttest_1samp(  np.concatenate([diff_FC_GMWM[0], diff_FC_GMWM[1],diff_FC_GMWM[2],diff_FC_GMWM[3] ])  , 0)[1]
stats.ttest_1samp( np.array(df_diff_FC_GMWM.mean(axis=1))  , 0)[1]
effect = np.mean(np.array(df_diff_FC_GMWM.mean(axis=1))) /np.std(np.array(df_diff_FC_GMWM.mean(axis=1)) )
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
import statsmodels
statsmodels.stats.power.tt_solve_power(effect_size=effect, nobs=None, alpha=0.05, power=0.8, alternative='two-sided')


FC_df_long_all_ii 
FC_df_long_all_pi 
FC_df_long_all_ic 
FC_df_long_all_po 

#%%

df_diff_FC_GMWM_by_elec = pd.DataFrame({"Tissue": FC_df_long_all_ii["Tissue"], "interictal":FC_df_long_all_ii["FC"] , "preictal":FC_df_long_all_pi["FC"] , "ictal":FC_df_long_all_ic["FC"] , "postictal":FC_df_long_all_po["FC"]  })

df_diff_FC_GMWM_by_elec_long = pd.melt(df_diff_FC_GMWM_by_elec, id_vars=['Tissue'], value_vars=['interictal', 'preictal', 'ictal', 'postictal'])




#sns.violinplot(x="variable", y="value", hue="Tissue", data=df_diff_FC_GMWM_by_elec_long, palette="muted")


# Set your custom color palette
colors = ["#c6b4a5", "#b6d4ee"]
sns.set_palette(sns.color_palette(colors))

plt.figure(dpi = 300); 
flierprops = dict(marker='o', markerfacecolor='#00000000', markersize=0.0, linestyle='none', markeredgecolor='#00000033')
ax = sns.boxplot(x="variable", y="value", hue="Tissue", data=df_diff_FC_GMWM_by_elec_long,  linewidth=0.9, whis = 1, width=0.8, flierprops = flierprops)



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.set_ylim([0.1,0.65])





#%%

NETWORK_MEASURE = "Nodes"

colors = ["#c6b4a5", "#b6d4ee"]



net_netMeasures_subjects_long = pd.melt(net_netMeasures_subjects, id_vars=['Subject','Period', 'Network Type'], value_vars=['Nodes', 'Edges', 'Density', 'Transitivity' ,'Characteristic Path Length'])

DENSITY = net_netMeasures_subjects_long.loc[net_netMeasures_subjects_long['variable'] == NETWORK_MEASURE]

per = 2
DENSITY_per = DENSITY.loc[DENSITY['Period'] == descriptors[per]]
#print(stats.ttest_rel(np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "WM-only"]["value"]),np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "GM-only"]["value"]))[1]) 

print(  mannwhitneyu(  np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "WM-only"]["value"]),      np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "GM-only"]["value"])  )  [1] )

colors = ["#c6b4a5", "#b6d4ee"]
sns.set_palette(sns.color_palette(colors))

plt.figure(dpi = 300); 
flierprops = dict(marker='o', markerfacecolor='#00000000', markersize=0.0, linestyle='none', markeredgecolor='#00000033')
ax = sns.boxplot(x="Period", y=NETWORK_MEASURE, hue="Network Type", data=net_netMeasures_subjects,  linewidth=0.9, whis = 1, width=0.8, showfliers=False, palette = ["#c6b4a5", "#666666", "#b6d4ee"], hue_order =["GM-only", "Full Network", "WM-only"])
ax = sns.swarmplot(x="Period", y=NETWORK_MEASURE, hue="Network Type", data=net_netMeasures_subjects, color=".25", dodge=True, size = 3,hue_order =["GM-only", "Full Network", "WM-only"])

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3],labels[0:3], frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.set_ylim([0, 2000])


ofpath_figure_adj = ospj(ofpath_figure, "network_measures")
plt.savefig(ospj(ofpath_figure_adj, "{1}_{0}.pdf".format(order_of_matrices_in_pickle_file.iloc[freq][0], NETWORK_MEASURE  )))



#%%
NETWORK_MEASURE = "Betweenness Centrality"
net_nodeMeasures_subjects_long = pd.melt(net_nodeMeasures_subjects, id_vars=['Subject','Period', 'Network Type'], value_vars=['Degree', 'Strength', 'Clustering Coefficient' ,'Betweenness Centrality'])

DENSITY = net_nodeMeasures_subjects_long.loc[net_nodeMeasures_subjects_long['variable'] == NETWORK_MEASURE]

per = 1
DENSITY_per = DENSITY.loc[DENSITY['Period'] == descriptors[per]]
print(stats.ttest_ind(np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "WM-only"]["value"]),  np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "GM-only"]["value"]))[1]) 

print(kruskal(np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "WM-only"]["value"]),  np.array(DENSITY_per.loc[DENSITY_per['Network Type'] == "GM-only"]["value"]))[1])

"""
net_nodeMeasures_subjects

plt.figure(dpi = 300); 
flierprops = dict(marker='o', markerfacecolor='#00000000', markersize=0.0, linestyle='none', markeredgecolor='#00000033')
ax = sns.boxplot(x="Period", y=NETWORK_MEASURE, hue="Network Type", data=net_nodeMeasures_subjects,  linewidth=0.9, whis = 1, width=0.8, showfliers=False, palette = ["#c6b4a5", "#666666", "#b6d4ee"], hue_order =["GM-only", "Full Network", "WM-only"])
#ax = sns.swarmplot(x="Period", y="Betweenness Centrality", hue="Network Type", data=net_nodeMeasures_subjects, color=".25", dodge=True, size=0.2)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3],labels[0:3], frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

"""


per = 0
for per in range(4):
    DENSITY_per = DENSITY.loc[DENSITY['Period'] == descriptors[per]]
    #Plot distribution ECDF differences
    plt.figure(figsize=(5, 5), dpi = 600); 
    plot4, axes4 = plt.subplots(figsize=(5, 5), dpi = 600)
    sns.kdeplot( data =DENSITY_per, x="value" , hue="Network Type", ax = axes4, palette = ["#c6b4a5", "#666666", "#b6d4ee"], hue_order =["GM-only", "Full Network", "WM-only"])
    
    axes4.set_xlim([-50, 200]); axes4.set_ylim([0, 0.044]) #Degree
    #axes4.set_xlim([-0.2, 1]); axes4.set_ylim([0, 3.2]) #CC
    #axes4.set_xlim([-0.03, 0.05]); axes4.set_ylim([0, 52]) #Betweenness Centrality
    axes4.title.set_text( "{3} {0} {1} {2}".format(per, descriptors[per], order_of_matrices_in_pickle_file.iloc[freq][0], NETWORK_MEASURE  ))
    axes4.spines['top'].set_visible(False)
    axes4.spines['right'].set_visible(False)
    axes4.set_ylabel('Kernel Density')
    axes4.set_xlabel(NETWORK_MEASURE )
    
    ofpath_figure_adj = ospj(ofpath_figure, "network_measures/nodal_measures")
    plt.savefig(ospj(ofpath_figure_adj, "{3}_{2}_{0}_{1}.pdf".format(per, descriptors[per], order_of_matrices_in_pickle_file.iloc[freq][0], NETWORK_MEASURE  )))

    
    #plt.savefig(ospj(ofpath_figure_adj, "FC_distribution_all_{0}_{1}_{2}_ECDFplot.pdf".format(per, descriptors[per], order_of_matrices_in_pickle_file.iloc[freq][0]  )))












































