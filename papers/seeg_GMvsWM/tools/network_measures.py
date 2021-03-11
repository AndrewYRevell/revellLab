"""
2020.01.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://sites.google.com/site/bctnet/Home/functions

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    1. infname_connectivity: path and filename of the adjacency matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    1. charPath- characteristic path length
    2. clust- clustering coefficient
    3. degree- mean degree 
    4. smallWorld-normalized small world measure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:
    python3.6 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
import sys
import pandas as pd
import numpy as np
#from scipy.io import loadmat #fromn scipy to read mat files (structural connectivity)
import bct

#%% Paths and File names

#ifname_connectivity = "data_processed/connectivity_matrices/structure/sub-RID0194/aal_res-1x1x1/sub-RID0194_ses-preop3T_dwi-eddyMotionB0Corrected.aal_res-1x1x1.count.pass.connectivity.mat"
#ifname_connectivity = "data_processed/connectivity_matrices/structure/sub-RID0278/RandomAtlas0500/sub-RID0278_ses-preop3T_dwi-eddyMotionB0Corrected.RandomAtlas0500_v0001.count.pass.connectivity.mat"
#%% Paramters


#%%


def get_network_measures(fname_connectivity):
    
    C = pd.read_table(fname_connectivity, header=None, dtype=object)
    #cleaning up structural data 
    C = C.drop([0,1], axis=1)
    C = C.drop([0], axis=0)
    C = C.iloc[:, :-1]
    #C_electrode_names = np.array([e[-4:] for e in  np.array(C.iloc[0])])
    C = np.array(C.iloc[1:, :]).astype('float64')  #finally turn into numpy array

    #binarize connectivity matrix
    C_binarize = bct.weight_conversion(C, "binarize")

    #Calculate Network Measures:

        
    # 1. Density
    density = bct.density_und(C_binarize)[0]
    
    # 2. Degree
    degree_mean = np.mean(bct.degrees_und(C_binarize))
        
    # 3. Clustering Coefficient
    clustering_coefficient = bct.clustering_coef_bu(C_binarize)
    clustering_coefficient_mean = np.mean(clustering_coefficient)    

    
    # 4. characteristic path length (i.e. average shortest path length)
    #Get distance 
    C_dist  = bct.distance_bin(C_binarize)
    #If there are any disjointed nodes set them equal to the largest non-Inf length
    C_dist_max = np.nanmax(C_dist[C_dist != np.inf]) #find the max length (that's not infinity)
    C_dist[np.where(C_dist == np.inf)] = C_dist_max #find the inifnities, and replace with max
    characteristic_path_length = bct.charpath(C_dist)[0]
    
    

    # 5. Small Worldness
    Cr = degree_mean/len(C_binarize)
    Lr = np.log10(len(C_binarize))/np.log10(degree_mean)
    
    gamma = clustering_coefficient_mean/Cr
    lamb = characteristic_path_length/Lr
    
    sigma = gamma/lamb
    small_worldness = sigma

    network_measures = np.zeros(shape=(1,5))
    network_measures[0,:] =  [density, degree_mean, clustering_coefficient_mean, characteristic_path_length, small_worldness]
    colLabels = ["Density", "degree_mean", "clustering_coefficient_mean", "characteristic_path_length", "small_worldness"]
    network_measures_df = pd.DataFrame(network_measures, columns= colLabels) 
    return network_measures_df
   
    
   
    
   
    #%%
   
    
"""
Old code

def clustering_coef_bu(C_binarize):
    n= len(C_binarize)
    clustering_coefficient = np.zeros(shape= (n, 1))
    for u in  range(n):
        
        V = np.where(C_binarize[u, :] > 0)[0]
    
        k=len(V)
        if k >= 2:#degree must be at least 2
            S = 0
            for x in range(k):
                for y in range(k):
                    S = S + C_binarize[ V[x], V[y]]
            clustering_coefficient[u, 0] = S/(np.power(k,2)-k)
    return clustering_coefficient


    # 5. Small Worldness
    
    R_binarize = bct.randmio_und(C_binarize, 4)[0]
    R_clustering_coefficient_mean = np.mean(bct.clustering_coef_bu(R_binarize))    
    R_dist = bct.distance_bin(R_binarize)
    R_dist_max = np.nanmax(R_dist[R_dist != np.inf]) #find the max length (that's not infinity)
    R_dist[np.where(R_dist == np.inf)] = R_dist_max #find the inifnities, and replace with max
    R_characteristic_path_length = bct.charpath(R_dist)[0]
    
    small_worldness = (clustering_coefficient_mean/R_clustering_coefficient_mean)/(characteristic_path_length/R_characteristic_path_length)
    
    L_binarize = bct.latmio_und(C_binarize, 4)
    L_binarize_0 = L_binarize[0]
    L_clustering_coefficient_mean = np.mean(bct.clustering_coef_bu(L_binarize_0))  
    
    ((R_characteristic_path_length/characteristic_path_length)   - (clustering_coefficient_mean/L_clustering_coefficient_mean))*2

"""
    