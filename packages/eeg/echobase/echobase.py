"""
Purpose:
Function pipelines for filtering time-varying data
Logic of code:
    1. Default parameters for filtering 
    2. Calculating connectivity for cross-correlation, pearson, spearman, coherence, and mutal inforomation for 
       broadband, delta, theta, alpha, beta, gamma-high, gamma-mid, and gamma-low frequency bands
    3. Supporting code to enact filtering, type checking, etc.
Table of Contents:
A. Set up
    Parameters
B. Main
    Wrappers:
    1. crossCorrelation_wrapper
    2. pearson_wrapper
    3. spearman_wrapper
    4. coherence_wrapper
    5. mutualInformation_wrapper
    
    Calculating connectivity: 
    6. pearson_connectivity
    7. spearman_connectivity
    8. crossCorrelation_connectivity
    9. mutualInformation_connectivity
    10. coherence_connectivity
    
C. Supporting Code:
    11. common_avg_ref
    12. automatic_bipolar_ref
    13. manual_bipolar_ref
    14. laplacian_bipolar_ref
    15. ar_one
    16. elliptic
    17. elliptic_bandFilter 
    18. butterworth_filt 
    
D. Utilities
    19. check_path
    20. getNextCol
    21. getIndexes
    22. make_path
    23. check_path_overwrite
    24. check_has_key
    25. check_dims
    26. check_type
    27. check_function 
    28. printProgressBar
    29. show_eeg_compare
    30. plot_adj
    31. plot_adj_allbands
See individual function comments for inputs and outputs
Change Log
----------
2021 March 9: Formalized by Andy Revell. Documentation by Lena Armstrong
"""

import os
import math
import time
import copy
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate 
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
#from mtspec import mt_coherence #needed if uncommented out for mtspec
#from __future__ import division

#%%

"""
A. Set up
"""
# Parameter set - the following are the default parameters for filtering 

#Bands
param_band = {}
param_band['Broadband'] = [1., 127.]
param_band['delta'] = [1., 4.]
param_band['theta'] = [4., 8.]
param_band['alpha'] = [8., 13.]
param_band['beta'] = [13., 30.]
param_band['gammaLow'] = [30., 40.]
param_band['gammaMid'] = [70., 100.]
param_band['gammaHigh'] = [100., 127.]

#Filters
param = {}
g = 1.0
gpass = 1.0 #changed from 0.1 to 1 because seems like 0.1 does not work for notch filter. Do not use 2.0 --> does not work for delta band. 1.0 seems good
gstop = 60.0
param['Notch_60Hz'] = {'wp': [58.0, 62.0],
                       'ws': [59.0, 61.0],
                       'gpass': gpass,
                       'gstop': gstop}
param['Broadband'] = {'wp': param_band['Broadband'],
                    'ws': [ param_band['Broadband'][0]-0.5 , param_band['Broadband'][1]+0.5  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['delta'] = {'wp': param_band['delta'],
                    'ws': [ param_band['delta'][0]-0.5 , param_band['delta'][1]+g  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['theta'] = {'wp': param_band['theta'],
                    'ws': [ param_band['theta'][0]-g , param_band['theta'][1]+g  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['alpha'] = {'wp': param_band['alpha'],
                    'ws': [ param_band['alpha'][0]-g , param_band['alpha'][1]+g  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['beta'] = {'wp': param_band['beta'],
                    'ws': [ param_band['beta'][0]-g , param_band['beta'][1]+g  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['gammaLow'] = {'wp': param_band['gammaLow'],
                    'ws': [ param_band['gammaLow'][0]-g , param_band['gammaLow'][1]+g  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['gammaMid'] = {'wp': param_band['gammaMid'],
                    'ws': [ param_band['gammaMid'][0]-g , param_band['gammaMid'][1]+g  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['gammaHigh'] = {'wp': param_band['gammaHigh'],
                    'ws': [ param_band['gammaHigh'][0]-g , param_band['gammaHigh'][1]+0.5  ],
                    'gpass': gpass,
                    'gstop': gstop}
param['XCorr'] = {'tau': 0.5}



"""
n_samp, n_chan = data.shape
start = 0
stop = 1
adj_crossCorrelation_all = crossCorrelation_wrapper(data[range(start, fs*stop), :], fs, param, avgref=True)
plot_adj_allbands(adj_crossCorrelation_all, vmin = -0.7, vmax = 0.9)
adj_pearson_all, adj_pearson_all_pval = pearson_wrapper(data[range(start, fs*stop), :], fs, param, avgref=True)
plot_adj_allbands(adj_pearson_all, vmin = -0.7, vmax = 0.9)
plot_adj_allbands(adj_pearson_all_pval, vmin = 0, vmax = 0.05/((n_chan*n_chan-n_chan)/2)    )
adj_spearman_all, adj_spearman_all_pval = spearman_wrapper(data[range(start, fs*stop), :], fs, param, avgref=True)
plot_adj_allbands(adj_spearman_all, vmin = -0.7, vmax = 0.9)
plot_adj_allbands(adj_spearman_all_pval, vmin = 0, vmax = 0.05/((n_chan*n_chan-n_chan)/2)   )
adj_coherence_all = coherence_wrapper(data[range(start, fs*stop), :], fs, param, avgref=True)
plot_adj_allbands(adj_coherence_all, vmin = 0, vmax = 1 )
adj_mi_all = mutualInformation_wrapper(data[range(start, fs*stop), :], fs, param, avgref=True)
plot_adj_allbands(adj_mi_all, vmin = 0, vmax = 1 )
"""


"""
B. Main
"""
#%%
# Wrapper scripts
def crossCorrelation_wrapper(data, fs, param = param, avgref=True):
    """
    Pipeline function using cross-correlation for computing a band-specific functional network from ECoG.
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
            
        param: set to default 
        avgref: True/False
            Re-reference data to the common average (default: True)
    Returns
    -------
        adj_xcorr_bb: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation broadband 
            
        adj_xcorr_d: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation delta 
            
        adj_xcorr_t: ndarray, shape ( N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation theta 
            
        adj_xcorr_a: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation alpha 
            
        adj_xcorr_b: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation beta 
            
        adj_xcorr_gl: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation gamma-low 
            
        adj_xcorr_gm: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation gamma-mid 
            
        adj_xcorr_gh: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for cross-correlation gamma-high 
    """

    # Standard param checks
    check_type(data,np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Build pipeline
    """
    if avgref:
        data_ref = common_avg_ref(data)
    else:
        data_ref = data.copy()
    data_ref_ar = ar_one(data_ref)
    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data_ref_ar, fs, param)
    """

    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data, fs, param)
    
    print("Cross Correlation Broadband")
    adj_xcorr_bb = crossCorrelation_connectivity(data_bb, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Delta")
    adj_xcorr_d = crossCorrelation_connectivity(data_d, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Theta")
    adj_xcorr_t = crossCorrelation_connectivity(data_t, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Alpha")
    adj_xcorr_a = crossCorrelation_connectivity(data_a, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Beta")
    adj_xcorr_b = crossCorrelation_connectivity(data_b, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Gamma - Low")
    adj_xcorr_gl = crossCorrelation_connectivity(data_gl, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Gamma - Mid")
    adj_xcorr_gm = crossCorrelation_connectivity(data_gm, fs, **param['XCorr'], absolute=False)
    print("Cross Correlation Gamma - High")
    adj_xcorr_gh = crossCorrelation_connectivity(data_gh, fs, **param['XCorr'], absolute=False)

    return adj_xcorr_bb, adj_xcorr_d, adj_xcorr_t, adj_xcorr_a, adj_xcorr_b, adj_xcorr_gl, adj_xcorr_gm, adj_xcorr_gh
      

def pearson_wrapper(data, fs, param = param, avgref=True):
    """
    Pipeline function using pearson correlation for computing a band-specific functional network from ECoG.
    
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
        
        param: set to default
        avgref: True/False
            Re-reference data to the common average (default: True)
    Returns
    -------
        adj_pearson_bb, adj_pearson_bb_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation broadband and corresponding matrix of p-values
            
        adj_pearson_d, adj_pearson_d_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation delta and corresponding matrix of p-values
 
        adj_pearson_t, adj_pearson_t_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation theta and corresponding matrix of p-values           
       
       adj_pearson_a, adj_pearson_a_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation alpha and corresponding matrix of p-values 
            
       adj_pearson_b, adj_pearson_b_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation beta and corresponding matrix of p-values 
       
       adj_pearson_gl, adj_pearson_gl_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation gamma-low and corresponding matrix of p-values           
       
       adj_pearson_gm, adj_pearson_gm_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation gamma-mid and corresponding matrix of p-values 
            
       adj_pearson_gh, adj_pearson_gh_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for pearson correlation gamma-high and corresponding matrix of p-values 
    """

    # Standard param checks
    check_type(data,np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Build pipeline
    """
    if avgref:
        data_ref = common_avg_ref(data)
    else:
        data_ref = data.copy()
    data_ref_ar = ar_one(data_ref)

    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data_ref_ar, fs, param)
    """
    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data, fs, param)
    print("Pearson Correlation Broadband")
    adj_pearson_bb, adj_pearson_bb_pval  = pearson_connectivity(data_bb, fs)
    print("Pearson Correlation Delta")
    adj_pearson_d, adj_pearson_d_pval = pearson_connectivity(data_d, fs)
    print("Pearson Correlation Theta")
    adj_pearson_t, adj_pearson_t_pval = pearson_connectivity(data_t, fs)
    print("Pearson Correlation Alpha")
    adj_pearson_a, adj_pearson_a_pval = pearson_connectivity(data_a, fs)
    print("Pearson Correlation Beta")
    adj_pearson_b, adj_pearson_b_pval = pearson_connectivity(data_b, fs)
    print("Pearson Correlation Gamma - Low")
    adj_pearson_gl, adj_pearson_gl_pval = pearson_connectivity(data_gl, fs)
    print("Pearson Correlation Gamma - Mid")
    adj_pearson_gm, adj_pearson_gm_pval = pearson_connectivity(data_gm, fs)
    print("Pearson Correlation Gamma - High")
    adj_pearson_gh, adj_pearson_gh_pval = pearson_connectivity(data_gh, fs)

    return [adj_pearson_bb, adj_pearson_d, adj_pearson_t, adj_pearson_a, adj_pearson_b, adj_pearson_gl, adj_pearson_gm, adj_pearson_gh], [adj_pearson_bb_pval, adj_pearson_d_pval, adj_pearson_t_pval, adj_pearson_a_pval, adj_pearson_b_pval, adj_pearson_gl_pval, adj_pearson_gm_pval, adj_pearson_gh_pval]

      
def spearman_wrapper(data, fs, param = param, avgref=True):
    """
    Pipeline function using separman correlation for computing a band-specific functional network from ECoG.
    
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
            
        param: set to default
        avgref: True/False
            Re-reference data to the common average (default: True)
    Returns
    -------
        adj_spearman_bb, adj_spearman_bb_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation broadband and corresponding matrix of p-values
            
        adj_spearman_d, adj_spearman_d_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation delta and corresponding matrix of p-values
 
        adj_spearman_t, adj_spearman_t_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation theta and corresponding matrix of p-values           
       
       adj_spearman_a, adj_spearman_a_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation alpha and corresponding matrix of p-values 
            
       adj_spearman_b, adj_spearmann_b_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation beta and corresponding matrix of p-values 
       
       adj_spearman_gl, adj_spearman_gl_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation gamma-low and corresponding matrix of p-values           
       
       adj_spearman_gm, adj_spearman_gm_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation gamma-mid and corresponding matrix of p-values 
            
       adj_spearman_gh, adj_spearman_gh_pvalue: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for spearman correlation gamma-high and corresponding matrix of p-values 
    """

    # Standard param checks
    check_type(data,np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Build pipeline
    """
    if avgref:
        data_ref = common_avg_ref(data)
    else:
        data_ref = data.copy()
    data_ref_ar = ar_one(data_ref)

    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data_ref_ar, fs, param)
    """
    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data, fs, param)
    print("spearman Correlation Broadband")
    adj_spearman_bb, adj_spearman_bb_pval  = spearman_connectivity(data_bb, fs)
    print("spearman Correlation Delta")
    adj_spearman_d, adj_spearman_d_pval = spearman_connectivity(data_d, fs)
    print("spearman Correlation Theta")
    adj_spearman_t, adj_spearman_t_pval = spearman_connectivity(data_t, fs)
    print("spearman Correlation Alpha")
    adj_spearman_a, adj_spearman_a_pval = spearman_connectivity(data_a, fs)
    print("spearman Correlation Beta")
    adj_spearman_b, adj_spearman_b_pval = spearman_connectivity(data_b, fs)
    print("spearman Correlation Gamma - Low")
    adj_spearman_gl, adj_spearman_gl_pval = spearman_connectivity(data_gl, fs)
    print("spearman Correlation Gamma - Mid")
    adj_spearman_gm, adj_spearman_gm_pval = spearman_connectivity(data_gm, fs)
    print("spearman Correlation Gamma - High")
    adj_spearman_gh, adj_spearman_gh_pval = spearman_connectivity(data_gh, fs)

    return [adj_spearman_bb, adj_spearman_d, adj_spearman_t, adj_spearman_a, adj_spearman_b, adj_spearman_gl, adj_spearman_gm, adj_spearman_gh], [adj_spearman_bb_pval, adj_spearman_d_pval, adj_spearman_t_pval, adj_spearman_a_pval, adj_spearman_b_pval, adj_spearman_gl_pval, adj_spearman_gm_pval, adj_spearman_gh_pval]  
    
 
def coherence_wrapper(data, fs, param = param, avgref=True):
    """
    Pipeline function using coherence for computing a band-specific functional network from ECoG.
    See: Khambhati, A. N. et al. (2016).
    Virtual Cortical Resection Reveals Push-Pull Network Control
    Preceding Seizure Evolution. Neuron, 91(5).
    Data --> CAR Filter --> Multi-taper Coherence
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
            
        param: set to default
        avgref: True/False
            Re-reference data to the common average (default: True)
    Returns
    -------
        adj_coherence_bb: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence broadband 
            
        adj_coherence_d: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence delta 
            
        adj_coherence_t: ndarray, shape ( N, N, T)
            Adjacency matrix for N variates over time T for coherence theta 
            
        adj_coherence_a: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence alpha 
            
        adj_coherence_b: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence beta 
            
        adj_coherence_gl: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence gamma-low 
            
        adj_coherence_gm: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence gamma-mid 
            
        adj_coherence_gh: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for coherence gamma-high 
    """

    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Build pipeline
    """
    if avgref:
        data_ref = common_avg_ref(data)
    else:
        data_ref = data.copy()
    data_ref_60 = elliptic(data_ref, fs, **param['Notch_60Hz'])
    """
    data_ref_60 = elliptic(data, fs, **param['Notch_60Hz'])

    print("Coherence Broadband")
    adj_coherence_bb = coherence_connectivity(data_ref_60, fs, param["Broadband"]['wp'])
    print("Coherence Delta")
    adj_coherence_d = coherence_connectivity(data_ref_60, fs, param["delta"]['wp'])
    print("Coherence Theta")
    adj_coherence_t = coherence_connectivity(data_ref_60, fs, param["theta"]['wp'])
    print("Coherence Alpha")
    adj_coherence_a = coherence_connectivity(data_ref_60, fs, param["alpha"]['wp'])
    print("Coherence Beta")
    adj_coherence_b = coherence_connectivity(data_ref_60, fs, param["beta"]['wp'])
    print("Coherence Gamma - Low")
    adj_coherence_gl = coherence_connectivity(data_ref_60, fs, param["gammaLow"]['wp'])
    print("Coherence Gamma - Mid")
    adj_coherence_gm = coherence_connectivity(data_ref_60, fs, param["gammaMid"]['wp'])
    print("Coherence Gamma - High")
    adj_coherence_gh = coherence_connectivity(data_ref_60, fs, param["gammaHigh"]['wp'])

    return adj_coherence_bb, adj_coherence_d, adj_coherence_t, adj_coherence_a, adj_coherence_b, adj_coherence_gl, adj_coherence_gm, adj_coherence_gh
    
    
def mutualInformation_wrapper(data, fs, param = param, avgref=True):
    """
    Pipeline function using mutual information for computing a broadband functional network from ECoG.
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
            
        param: set to default
        avgref: True/False
            Re-reference data to the common average (default: True)
    Returns
    -------
       adj_mi_bb: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information broadband 
            
        adj_mi_d: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information delta 
            
        adj_mi_t: ndarray, shape ( N, N, T)
            Adjacency matrix for N variates over time T for mutual information theta 
            
        adj_mi_a: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information alpha 
            
        adj_mi_b: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information beta 
            
        adj_mi_gl: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information gamma-low 
            
        adj_mi_gm: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information gamma-mid 
            
        adj_mi_gh: ndarray, shape (N, N, T)
            Adjacency matrix for N variates over time T for mutual information gamma-high 
    """

    # Standard param checks
    check_type(data,np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Build pipeline
    """
    if avgref:
        data_ref = common_avg_ref(data)
    else:
        data_ref = data.copy()
    data_ref_ar = ar_one(data_ref)

    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data_ref_ar, fs, param)
    """
    data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh = elliptic_bandFilter(data, fs, param)
    print("Mutual Information Broadband")
    adj_mi_bb = mutualInformation_connectivity(data_bb, fs)
    print("Mutual Information Delta")
    adj_mi_d = mutualInformation_connectivity(data_d, fs)
    print("Mutual Information Theta")
    adj_mi_t = mutualInformation_connectivity(data_t, fs)
    print("Mutual Information Alpha")
    adj_mi_a = mutualInformation_connectivity(data_a, fs)
    print("Mutual Information Beta")
    adj_mi_b = mutualInformation_connectivity(data_b, fs)
    print("Mutual Information Gamma - Low")
    adj_mi_gl = mutualInformation_connectivity(data_gl, fs)
    print("Mutual Information Gamma - Mid")
    adj_mi_gm = mutualInformation_connectivity(data_gm, fs)
    print("Mutual Information Gamma - High")
    adj_mi_gh = mutualInformation_connectivity(data_gh, fs)

    return adj_mi_bb, adj_mi_d, adj_mi_t, adj_mi_a, adj_mi_b, adj_mi_gl, adj_mi_gm, adj_mi_gh
          
    
#%%
# Calculating Connectivity 

   
def pearson_connectivity(data, fs):
    """
    Uses pearson correlation to compute a band-specific functional network from ECoG
    
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
    Returns
    -------
        adj, adj_pvalue: ndarray, shape (F, N, N, T)
            Adjacency matrix for N variates for F frequency-bands over time T for pearson correlation
            and corresponding matrix of p-values 
    """

    # Retrieve data attributes
    n_samp, n_chan = data.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    adj_pvalue = np.zeros((n_chan, n_chan))

    # Compute all coherences
    count = 0
    for n1, n2 in zip(triu_ix, triu_iy):
        t0 = time.time()
        adj[n1, n2] = pearsonr(data[:,n1], data[:,n2])[0]
        adj_pvalue[n1, n2] = pearsonr(data[:,n1], data[:,n2])[1]
        t1 = time.time(); 
        td = t1-t0; tr = td*(len(triu_ix)-count)/60; printProgressBar(count+1, len(triu_ix), prefix = '', suffix = f"{count}  {np.round(tr,2)} min", decimals = 1, length = 20, fill = "X", printEnd = "\r"); count += 1

    adj += adj.T
    adj_pvalue += adj_pvalue.T
    return adj, adj_pvalue


def spearman_connectivity(data, fs):
    """
    Uses spearman correlation to compute a band-specific functional network from ECoG
    
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
    Returns
    -------
        adj, adj_pvalue: ndarray, shape (F, N, N, T)
            Adjacency matrix for N variates for F frequency-bands over time T for spearman correlation
            and corresponding matrix of p-values 
    """
    
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    adj_pvalue = np.zeros((n_chan, n_chan))
    
    # Compute all coherences
    count = 0
    for n1, n2 in zip(triu_ix, triu_iy):
        t0 = time.time()
        adj[n1, n2] = spearmanr(data[:,n1], data[:,n2])[0]
        adj_pvalue[n1, n2] = spearmanr(data[:,n1], data[:,n2])[1]
        t1 = time.time(); td = t1-t0; tr = td*(len(triu_ix)-count)/60; printProgressBar(count+1, len(triu_ix), prefix = '', suffix = f"{count}  {np.round(tr,2)} min", decimals = 1, length = 20, fill = "X", printEnd = "\r"); count += 1
    adj += adj.T
    adj_pvalue += adj_pvalue.T
    return adj, adj_pvalue


def crossCorrelation_connectivity(data_hat, fs, tau, absolute=False):
    """
    Uses FFT-based cross-correlation (using convolution) to compute a band-specific functional network from ECoG
    Parameters
    ----------
        data_hat: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
        tau: float
            The max lag limits of cross-correlation in seconds
        
        absolute: True/False
            (default: False)
    Returns
    -------
        adj: ndarray, shape (F, N, N, T)
            Adjacency matrix for N variates for F frequency-bands over time T for cross-correlation
    """

    # Standard param checks
    check_type(data_hat, np.ndarray)
    check_dims(data_hat, 2)
    check_type(fs, int)
    check_type(tau, float)

    # Get data_hat attributes
    n_samp, n_chan = data_hat.shape
    tau_samp = int(tau*fs)
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Normalize the signal
    data_hat -= data_hat.mean(axis=0)
    data_hat /= data_hat.std(axis=0)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    lags = np.hstack((range(0, n_samp, 1),
                      range(-n_samp, 0, 1)))
    tau_ix = np.flatnonzero(np.abs(lags) <= tau_samp)

    # Use FFT to compute cross-correlation
    data_hat_fft = np.fft.rfft( np.vstack((data_hat, np.zeros_like(data_hat))), axis=0)

    # Iterate over all edges
    count = 0
    for n1, n2 in zip(triu_ix, triu_iy):
        t0 = time.time()
        xc = 1 / n_samp * np.fft.irfft( data_hat_fft[:, n1] * np.conj(data_hat_fft[:, n2]))
        if absolute:
            adj[n1, n2] = np.max( np.abs(xc[tau_ix])  )
        #taking the absolute max value, whether negative or positive, but preserving sign
        elif not absolute:
            if xc[tau_ix].max() > np.abs(xc[tau_ix].min()):
                adj[n1, n2] = xc[tau_ix].max()
            else:
                adj[n1, n2] = xc[tau_ix].min()
        t1 = time.time(); td = t1-t0; tr = td*(len(triu_ix)-count)/60; printProgressBar(count+1, len(triu_ix), prefix = '', suffix = f"{count}  {np.round(tr,2)} min", decimals = 1, length = 20, fill = "X", printEnd = "\r"); count += 1
    adj += adj.T

    return adj

   
def mutualInformation_connectivity(data_hat, fs):
    """
    Uses mutual information to compute a band-specific functional network from ECoG
     
    https://www.roelpeters.be/calculating-mutual-information-in-python/
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#id6
   
    Parameters
    ----------
        data_hat: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
    Returns
    -------
        adj: ndarray, shape (F, N, N, T)
            Adjacency matrix for N variates for F frequency-bands over time T for mutal information
    """
   
    # Retrieve data_hat attributes
    n_samp, n_chan = data_hat.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))

    # Compute all coherences
    count = 0
    for n1, n2 in zip(triu_ix, triu_iy):
        t0 = time.time()
        #METHOD 1 - wrong. Need to treat continuous variable (time series EEG) differently. MI is for classes
        #adj[n1, n2] = normalized_mutual_info_score(  data_hat[:,n1], data_hat[:,n2])
        
        #METHOD 2 - better
        #c_xy = np.histogram2d( data_hat[:,n1],  data_hat[:,n2], 1000)[0]  
        #adj[n1, n2] = mutual_info_score( None, None, contingency=c_xy)
        
        #METHOD 3 - best
        adj[n1, n2] = mutual_info_regression( data_hat[:,n1].reshape(-1,1), data_hat[:,n2], n_neighbors=3  ) #Note: Very slow
        t1 = time.time(); td = t1-t0; tr = td*(len(triu_ix)-count)/60; printProgressBar(count+1, len(triu_ix), prefix = '', suffix = f"{count}  {np.round(tr,2)} min", decimals = 1, length = 20, fill = "X", printEnd = "\r"); count += 1
    adj += adj.T
    return adj


def coherence_connectivity(data_hat, fs, cf):
    """
    Uses coherence to compute a band-specific functional network from ECoG
    Parameters
    ----------
        data_hat: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
        cf: list
            Frequency range over which to compute coherence [-NW+C, C+NW]
    Returns
    -------
        adj: ndarray, shape (F, N, N, T)
            Adjacency matrix for N variates for F frequency-bands over time T for coherence 
    """

    # Standard param checks
    check_type(data_hat, np.ndarray)
    check_dims(data_hat, 2)
    check_type(cf, list)

    if not len(cf) == 2:
        raise Exception('Must give a frequency range in list of length 2')

    # Get data_hat attributes
    n_samp, n_chan = data_hat.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))

    # Compute all coherences
    count = 0
    for n1, n2 in zip(triu_ix, triu_iy):
        t0 = time.time()
        if (data_hat[:, n1] == data_hat[:, n2]).all():
            adj[n1, n2] = np.nan
        else:
            out = signal.coherence(x= data_hat[:, n1],
                                   y = data_hat[:, n2],
                                   fs = fs,
                                   window= range(int(fs-fs/3)) #if n_samp = fs, the window has to be less than fs, or else you will get output as all ones. So I modified to be fs - fs/3, and not just fs
                                   )

            # Find closest frequency to the desired center frequency
            cf_idx = np.flatnonzero((out[0] >= cf[0]) &
                                    (out[0] <= cf[1]))

            # Store coherence in association matrix
            adj[n1, n2] = np.mean(out[1][cf_idx])
        t1 = time.time(); td = t1-t0; tr = td*(len(triu_ix)-count)/60; printProgressBar(count+1, len(triu_ix), prefix = '', suffix = f"{count}  {np.round(tr,2)} min", decimals = 1, length = 20); count += 1

    adj += adj.T

    return adj    
    

"""
def multitaper(data, fs, time_band, n_taper, cf):
    
    The multitaper function windows the signal using multiple Slepian taper
    functions and then computes coherence between windowed signals.
    Note 2020.05.06
    To install mtspec:
    See https://krischer.github.io/mtspec/ for more documentation
    1. Need to have gfortran installed on computer
    2. It is different for Linux and Mac
    Linux:
    #apt-get install gfortran
    #pip install mtspec
    #or
    # conda config --add channels conda-forge
    # conda install mtspec
    Mac OS:
    Need homebrew, then do:
    #brew install gcc
    #brew cask install gfortran
    #pip install mtspec
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
        time_band: float
            The time half bandwidth resolution of the estimate [-NW, NW];
            such that resolution is 2*NW
        n_taper: int
            Number of Slepian sequences to use (Usually < 2*NW-1)
        cf: list
            Frequency range over which to compute coherence [-NW+C, C+NW]
    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
            
            
    param_cohe['time_band'] = 5.
    param_cohe['n_taper'] = 9
    
    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(time_band, float)
    check_type(n_taper, int)
    check_type(cf, list)
    if n_taper >= 2*time_band:
        raise Exception('Number of tapers must be less than 2*time_band')
    if not len(cf) == 2:
        raise Exception('Must give a frequency range in list of length 2')
    # Get data attributes
    n_samp, n_chan = data.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)
    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    # Compute all coherences
    count = 0
    for n1, n2 in zip(triu_ix, triu_iy):
        t0 = time.time()
        if (data[:, n1] == data[:, n2]).all():
            adj[n1, n2] = np.nan
        else:
            out = mt_coherence(1.0/fs,
                               data[:, n1],
                               data[:, n2],
                               time_band,
                               n_taper,
                               int(n_samp/2.), 0.95,
                               iadapt=1,
                               cohe=True, freq=True)
            # Find closest frequency to the desired center frequency
            cf_idx = np.flatnonzero((out['freq'] >= cf[0]) &
                                    (out['freq'] <= cf[1]))
            # Store coherence in association matrix
            adj[n1, n2] = np.mean(out['cohe'][cf_idx])
        t1 = time.time(); td = t1-t0; tr = td*(len(triu_ix)-count)/60; printProgressBar(count+1, len(triu_ix), prefix = '', suffix = f"{count}  {np.round(tr,2)} min", decimals = 1, length = 20, fill = "X", printEnd = "\r"); count += 1
    adj += adj.T
    return adj    
"""



#%%
"""
C. Referencing and Filters
"""

def common_avg_ref(data):
    """
    The common_avg_ref function subtracts the common mode signal from the original
    signal. Suggested for removing correlated noise, broadly over a sensor array.
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
    Returns
    -------
        data_reref: ndarray, shape (T, N)
            Referenced signal with common mode removed
    """
    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    # Remove common mode signal
    data_reref = (data.T - data.mean(axis=1)).T
    return data_reref


def automaticBipolarMontageSEEG(data, data_columns):
    channels = np.array(data_columns)
    
  

    
    nchan = len(channels)
    #naming to standard 4 character channel Name: (Letter)(Letter)(Number)(Number)
    channels = channel2std(channels)
    count = 0
    for ch in range(nchan-1):
        ch1Ind = ch
        ch1 = channels[ch1Ind]    
        #find sequential index
        ch2 = ch1[0:2] + f"{(int(ch1[2:4]) + 1):02d}"
        
        ch2exists = np.where(channels == ch2)[0]
        if len(ch2exists) > 0:
            ch2Ind = ch2exists[0]
            bipolar = pd.Series((data[:,ch1Ind] - data[:,ch2Ind])).rename(ch1)
            if count == 0: #initialize
                dfBipolar = pd.DataFrame( bipolar)
                count = count + 1
            else:
                dfBipolar = pd.concat([dfBipolar, pd.DataFrame( bipolar)], axis=1)
    return np.array(dfBipolar), np.array(dfBipolar.columns)


def manual_bipolar_ref(data, data_col, csvcols):
    """
    The manual_bipolar_ref function calculates the bipolar montaged signal with
    a CSV file that informs the subtractions.
    
    Parameters
    ----------
    data: ndarray, shape (T, N)
        Input signal with T samples over N variates
    
    csvcols: the CSV file with 2 columns: the original electrode, and the bipolar reference
    
    data_col: column names of the data Pandas frame (list of electrode names)
    
    Returns
    -------
    data_reref: ndarray, shape (T, N)
        Bipolar Referenced signal
    """
    #csvcols = np.array(pd.read_csv('manual_bipolar_ref_columns.csv'))
    data_bp = pd.DataFrame()
    for i in range(0, csvcols.shape[0]):
        input_1 = data[:, list(data_col).index(csvcols[i][0])]
        input_2 = data[:, list(data_col).index(csvcols[i][1])]
        data_bp[data_col[i]] = input_1 - input_2
    data_reref = np.array(data_bp)
    return data_reref


def laplacian_ref(data, data_col, n, csvcoor):
    """
    The laplacian_ref function calculates the laplacian montaged signal with a
    CSV file that has the coordinates.
 
    Parameters
    ----------
    data: ndarray, shape (T, N)
        Input signal with T samples over N variates
   
    n: number of electrodes to average (Laplacian)
    
    csvcols: the CSV file with MNI coordinates
    
    data_col: column names of the data Pandas frame (list of electrode names)
 
    Returns
    -------
    data_reref: ndarray, shape (T, N)
        Bipolar Referenced signal
    """
    data_lp = pd.DataFrame()
    #csvcoor = pd.read_csv('sub-RID0278_electrode_localization.csv')
    csv_names = csvcoor['electrode_name']
    for j in range(0, csv_names.shape[0]):
        if (getIndexes(pd.DataFrame(data.columns), csv_names[j]) == []):
            csvcoor = csvcoor.drop(j)
    # note I need to get rid of the electrodes that are not in data but are in the csv file above
    coordata = csvcoor[['x_coordinate', 'y_coordinate', 'z_coordinate']]
    coorindex = list(csvcoor[['electrode_name']]['electrode_name'])
    # coorindex = [item for elem in coorindex for item in elem]
    csvcoor = pd.DataFrame(coordata.values, columns=['xcord', 'ycord', 'zcord'], index=coorindex)
    distcoor = pd.DataFrame(distance_matrix(csvcoor.values, csvcoor.values),
    index=csvcoor.index, columns=csvcoor.index)
    for i in range(0, data.shape[1] - 1):
        input_1 = data[:, i]
        # find the n least numbers
        min_elecs = distcoor.nsmallest(n, [data_col[i]]).index.tolist()
        for j in range(0, n):
            if j == 0:
                input_2 = data[:, list(data_col).index(min_elecs[j])]
            else:
                input_2 = input_2 + data[:, list(data_col).index(min_elecs[j])]
        input_2 = input_2 / n
        data_lp[data_col[i]] = input_1 - input_2
    data_reref = np.array(data_lp)
    return data_reref


def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """
    
    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])
    return data_white


def elliptic(data_hat, fs, wp, ws, gpass, gstop):
    """
    The elliptic function implements bandpass, lowpass, highpass filtering
    This implements zero-phase filtering to pre-process and analyze
    frequency-dependent network structure. Implements Elliptic IIR filter.
    Parameters
    ----------
        data_hat: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
        wp: tuple, shape: (1,) or (1,1)
            Pass band cutoff frequency (Hz)
        ws: tuple, shape: (1,) or (1,1)
            Stop band cutoff frequency (Hz)
        gpass: float
            Pass band maximum loss (dB)
        gstop: float
            Stop band minimum attenuation (dB)
    Returns
    -------
        data_hat_filt: ndarray, shape (T, N)
            Filtered signal with T samples over N variates
    """

    # Standard param checks
    check_type(data_hat, np.ndarray)
    check_dims(data_hat, 2)
    check_type(fs, int)
    check_type(wp, list)
    check_type(ws, list)
    check_type(gpass, float)
    check_type(gstop, float)
    if not len(wp) == len(ws):
        raise Exception('Frequency criteria mismatch for wp and ws')
    if not (len(wp) < 3):
        raise Exception('Must only be 1 or 2 frequency cutoffs in wp and ws')

    # Design filter
    nyq = fs / 2.0

    # new code. Works with scipy 1.4 (2020.05.06)
    wpass_nyq = [iter*0 for iter in range(len(wp))]
    for m in range(0, len(wp)):
        wpass_nyq[m] = wp[m] / nyq

    # new code. Works with scipy 1.4 (2020.05.06)
    wstop_nyq = [iter*0 for iter in range(len(ws))]
    for m in range(0, len(ws)):
        wstop_nyq[m] = ws[m] / nyq

    #wpass_nyq = map(lambda f: f/nyq, wp) #old code. Works with scipy 0.18
    #wstop_nyq = map(lambda f: f/nyq, wstop) #old code. Works with scipy 0.18
    b, a = signal.iirdesign(wp=wpass_nyq,
                                     ws=wstop_nyq,
                                     gpass=gpass,
                                     gstop=gstop,
                                     ftype='ellip')
    # Perform filtering and dump into signal_packet
    data_hat_filt = signal.filtfilt(b, a, data_hat, axis=0)
    return data_hat_filt


def elliptic_bandFilter(data, fs, param = param):
    """
    This function serves as a wrapper for the elliptic filter, in order to filter data into frequency bands 
    by first filtering out 60Hz and then filtering data for each frequency band. 
    
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
            
        param: set to default
    Returns
    -------
        data_bb: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for broadband 
            
        data_d: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for delta
            
        data_t: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for theta
        
        data_a: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for alpha
            
        data_b: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for beta
            
        data_gl: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for gamma-low
        
        data_gm: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for gamma-mid
            
        data_gh: ndarray, shape (T, N)
            Filtered signal with T samples over N variates for gamma-high
    """
    
    data_60 = elliptic(data, fs, **param['Notch_60Hz']) 
    band = "Broadband"
    data_bb = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_bb = elliptic(data_bb, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)

    band = "delta"
    data_d = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_d = elliptic(data_d, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)
    
    band = "theta"
    data_t = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_t = elliptic(data_t, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)
    
    band = "alpha"
    data_a = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_a = elliptic(data_a, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)

    band = "beta"
    data_b = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_b = elliptic(data_b, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)
    
    band = "gammaLow"
    data_gl = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_gl = elliptic(data_gl, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)
    
    band = "gammaMid"
    data_gm = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_gm = elliptic(data_gm, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)

    band = "gammaHigh"
    data_gh = elliptic(data_60, fs, [param[band]['wp'][1]], [param[band]['ws'][1]], gpass,  gstop)
    data_gh = elliptic(data_gh, fs, [param[band]['wp'][0]], [param[band]['ws'][0]], gpass,  gstop)

    return data_bb, data_d, data_t, data_a, data_b, data_gl, data_gm, data_gh


def butterworth_filt(data, fs):
    """
    This function filters data with butterworth filter.
    
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
    Returns
    -------
       notched: ndarray, shape (T, N)
            Filtered signal with T samples over N variates
    """
    
    filtered = np.zeros(data.shape)
    w = np.array([1, 120])  / np.array([(fs / 2), (fs / 2)])  # Normalize the frequency
    b, a = signal.butter(4, w, 'bandpass')
    filtered = signal.filtfilt(b, a, data)
    for i in range(data.shape[1]): filtered[:, i] = signal.filtfilt(b, a, data[:, i])
    filtered = filtered + (data[0] - filtered[0])  # correcting offset created by filtfilt
    b, a = signal.iirnotch(60, 30, fs)
    notched = np.zeros(data.shape)
    for i in range(data.shape[1]): notched[:, i] = signal.filtfilt(b, a, filtered[:, i])
    
    return notched


def preprocess(df, fs, fsds, montage = "bipolar", prewhiten = True):
    data = np.array(df)
    data_columns = np.array(df.columns)
    if montage == "car":
        data_ref = common_avg_ref(data) 
        channels = data_columns
        channels = channel2std(channels)
    if montage == "bipolar":
        data_ref, channels = automaticBipolarMontageSEEG(data, data_columns)
    if prewhiten == True: data_ar = ar_one(data_ref)    
    else: data_ar = data_ref   
    data_filt = elliptic_bandFilter(data_ar, int(fs))[0]
    return data, data_ref, data_ar, data_filt, channels

#%%
        
def movingaverage(x, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(x, window, 'valid')



def distanceBetweenPoints(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def lineLength(vector):
    l = len(vector)
    length = np.zeros(shape = l)
    x = np.array(range(l))
    for v in range(l-1):
        length[v] = distanceBetweenPoints( (x[v], vector[v]) , (x[v+1], vector[v+1])  )
    lineLen = np.sum(length)
    return lineLen

def lineLengthOfArray(data):
    windows, nsamp, nchan = data.shape
    lineLength_arr = np.zeros(shape = (windows, nchan))  
    for c in range(nchan):
        print(c)
        for win in range(windows):
            lineLength_arr[win, c] = lineLength(data[win, :, c])
            
    lineLengthNorm = copy.deepcopy(lineLength_arr)
    for c in range(nchan):
        lineLengthNorm[:,c] = lineLength_arr[:,c]/np.max(lineLength_arr[:,c])
    return lineLength_arr, lineLengthNorm


#%% signal analysis

    
def get_power(data, fs, avg = False):
    nchan = data.shape[1]
    if avg: #calculate over entire segment. else, calculate in one second windows
        power = np.zeros(shape = (129,nchan))
        for ch in range(nchan):
            _, power[:,ch] = signal.welch(data[:,ch], fs, nperseg=1 * fs)
    else:
        index = np.arange(0, len(data), step = fs *1 )
        power = np.zeros(shape = (129, len(index),nchan))
        for w in range(len(index)-1):
            st = index[w]
            sp = index[w+1]
            for ch in range(nchan):
                _, power[:,w,ch] = signal.welch(data[st:sp,ch], fs, nperseg=1 * fs)
    return power



def power_interpolate(data, dataInterictal, ictalStartIndex, ictalEndIndex, length = 200):
    nchan = data.shape[2]
    powerInterp = np.zeros(shape = (129, length*4, nchan))
    for ch in range(nchan):
        ii = dataInterictal[:,:,ch]
        pi =  data[:,0:ictalStartIndex,ch]
        ic =  data[:,ictalStartIndex:ictalEndIndex,ch]
        po =  data[:,ictalEndIndex:,ch]
        interpII = interpolate.interp1d(np.linspace(1,100, ii.shape[1]),  ii, axis=1 )
        interpPI = interpolate.interp1d(np.linspace(1,100, pi.shape[1]),  pi, axis=1 )
        interpIC = interpolate.interp1d(np.linspace(1,100, ic.shape[1]),  ic, axis=1 )
        interpPO = interpolate.interp1d(np.linspace(1,100, po.shape[1]),  po, axis=1 )
        powerInterp[:,0:length,ch]= interpII(np.linspace(1,100,length))
        powerInterp[:,length:length*2,ch]= interpPI(np.linspace(1,100,length))
        powerInterp[:,length*2:length*3,ch]= interpIC(np.linspace(1,100,length))
        powerInterp[:,length*3:length*4,ch]= interpPO(np.linspace(1,100,length))
    return powerInterp



#%%
"""
D. Utilities:
"""
def channel2std(channelsArr):
    nchan = len(channelsArr)
    for ch in range(nchan):
        if len(channelsArr[ch]) < 4:
            channelsArr[ch] = channelsArr[ch][0:2] + f"{int(channelsArr[ch][2:]):02d}"
    return channelsArr



def getIndexes(dfObj, value):
    """
    This getIndexes function is a helper function for the Laplacian montaging. It makes sure the CSV
    electrode files have the same electrodes as the voltage data
 
    Parameters
    ----------
        dfObj: ndarray of columns 
        value: index in dfObj
    Returns
    -------
        listOfPos: list of positive indices 
    """
    listOfPos = list()
    result = dfObj.isin([value])
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    return listOfPos


def check_path(path):
    '''
    Check if path exists
    Parameters
    ----------
        path: str
            Check if valid path
    '''
    if not os.path.exists(path):
        raise IOError('%s does not exists' % path)


def makepathWithError(path):
    '''
    Make new path if path does not exist
    Parameters
    ----------
        path: str
            Make the specified path
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise IOError('Path: %s, already exists' % path)

def makePath(path):
    '''
    Make new path if path does not exist
    Parameters
    ----------
        path: str
            Make the specified path
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def check_path_overwrite(path):
    '''
    Prevent overwriting existing path
    Parameters
    ----------
        path: str
            Check if path exists
    '''
    if os.path.exists(path):
        raise IOError('%s cannot be overwritten' % path)


def check_has_key(dictionary, key_ref):
    '''
    Check whether the dictionary has the specified key
    Parameters
    ----------
        dictionary: dict
            The dictionary to look through
        key_ref: str
            The key to look for
    '''
    if key_ref not in dictionary.keys():
        raise KeyError('%r should contain the %r key' % (dictionary, key_ref))


def check_dims(arr, nd):
    '''
    Check if numpy array has specific number of dimensions
    Parameters
    ----------
        arr: numpy.ndarray
            Input array for dimension checking
        nd: int
            Number of dimensions to check against
    '''
    if not arr.ndim == nd:
        raise Exception('%r has %r dimensions. Must have %r' % (arr, arr.ndim, nd))


def check_type(obj, typ):
    '''
    Check if obj is of correct type
    Parameters
    ----------
        obj: any
            Input object for type checking
        typ: type
            Reference object type (e.g. str, int)
    '''
    if not isinstance(obj, typ):
        raise TypeError('%r is %r. Must be %r' % (obj, type(obj), typ))


def check_function(obj):
    '''
    Check if obj is a function
    Parameters
    ----------
        obj: any
            Input object for type checking
    '''
    if not inspect.isfunction(obj):
        raise TypeError('%r must be a function.' % (obj))

        
# Progress bar function
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = "X", printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
        
        
def show_eeg_compare(data, data_hat, fs, channel = 0, start_sec = 0, stop_sec = 2):
    """
    Plots eeg for comparison 
    @params:
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
            
        data_hat: ndarray, shape (T, N)
            Input signal with T samples over N variates
        
        fs: int
            Sampling frequency
        
        channel: int
            (default = 0)
        
        start_sec: int
            Starting time (default = 0)
        
        stop_sec: int
            Stopping time (default = 2)
    """
    data_ch = data[:,channel]
    data_ch_hat = data_hat[:,channel]    
    fig,axes = plt.subplots(1,2,figsize=(8,4), dpi = 300)
    sns.lineplot(x =  np.array(range(fs*start_sec,fs*stop_sec))/1e6*fs, y = data_ch[range(fs*start_sec,fs*stop_sec)], ax = axes[0] , linewidth=0.5 )
    sns.lineplot(x =  np.array(range(fs*start_sec,fs*stop_sec))/1e6*fs, y = data_ch_hat[range(fs*start_sec,fs*stop_sec)] , ax = axes[1], linewidth=0.5 )
    plt.show()


def plot_adj(adj, vmin = -1, vmax = 1 ):
    """
    Plots adjacency matrix 
    @params:
        adj: ndarray, shape (N, N)
             Adjacency matrix for N variates
        
        vmin: int
            (default = -1)
        
        vmax: int
            (default = 1)
    """    
    fig,axes = plt.subplots(1,1,figsize=(4,4), dpi = 300)
    sns.heatmap(adj, square=True, ax = axes, vmin = vmin, vmax = vmax)

    
def plot_adj_allbands(adj_list, vmin = -1, vmax = 1, titles = ["Broadband", "Delta", "Theta", "Alpha", "Beta", "Gamma - Low", "Gamma - Mid", "Gamma - High"] ):
    """
    Plots adjacency matrix for all bands
    @params:
        adj: ndarray, shape (F, N, N)
             Adjacency matrix for N variates for F frequency bands 
        
        vmin: int
            (default = -1)
        
        vmax: int
            (default = 1)
        
        titles: frequency band names
            (default = "Broadband", "Delta", "Theta", "Alpha", "Beta", "Gamma - Low", "Gamma - Mid", "Gamma - High")
    """ 
    fig,axes = plt.subplots(2,4,figsize=(16,9), dpi = 300)
    count = 0
    for x in range(2):
        for y in range(4):
            sns.heatmap(adj_list[count], square=True, ax = axes[x][y], vmin = vmin, vmax = vmax)
            axes[x][y].set_title(titles[count], size=10)
            count = count+1
            

#%%
#visulaize
"""
vmin = -0.5; vmax = 0.9; title_size = 8
fig,axes = plt.subplots(4,2,figsize=(8,16), dpi = 300)
sns.heatmap(adj_xcorr, square=True, ax = axes[0][0], vmin = vmin, vmax = vmax); axes[0][0].set_title("X corr; tau: 0.25 ; elliptic", size=title_size)
sns.heatmap(np.abs(adj_xcorr), square=True, ax = axes[0][1], vmin = 0, vmax = vmax); axes[0][1].set_title("X corr Abs; tau: 0.25; elliptic", size=title_size)
   
sns.heatmap(adj_pear, square=True, ax = axes[1][0], vmin = vmin, vmax = vmax); axes[1][0].set_title("Pearson; elliptic", size=title_size)
sns.heatmap(adj_spear, square=True, ax = axes[1][1], vmin = vmin, vmax = vmax); axes[1][1].set_title("Spearman; elliptic", size=title_size)
sns.heatmap(adj_cohe_bb_m, square=True, ax = axes[2][0]); axes[2][0].set_title("Coherence: mt_spec; elliptic", size=title_size)
sns.heatmap(adj_cohe_bb, square=True, ax = axes[2][1]); axes[2][1].set_title("Coherence: Scipy; elliptic", size=title_size)
   
sns.heatmap(adj_MI, square=True, ax = axes[3][0]); axes[3][0].set_title("Mutual Information; elliptic", size=title_size)
fig,axes = plt.subplots(2,2,figsize=(8,8), dpi = 300)
sns.heatmap(adj_butter_xcorr, square=True, ax = axes[0][0], vmin = vmin, vmax = vmax)
sns.heatmap(np.abs(adj_butter_xcorr), square=True, ax = axes[0][1], vmin = vmin, vmax = vmax)
sns.heatmap(adj_butter_pear, square=True, ax = axes[1][0], vmin = vmin, vmax = vmax)
sns.heatmap(adj_butter_spear, square=True, ax = axes[1][1], vmin = vmin, vmax = vmax)
###########
###########
###########
###########
ch = 1
data_ch = data[:,ch]
data_ch_hat = data_hat[:,ch]
  
fig,axes = plt.subplots(1,2,figsize=(8,4), dpi = 300)
st = 0; sp = 15
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch[range(fs*st,fs*sp)], ax = axes[0] , linewidth=0.5 )
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch_hat[range(fs*st,fs*sp)] , ax = axes[1], linewidth=0.5 )
data_ch = data[:,ch]
data_ch_hat = data_butter[:,ch]
  
fig,axes = plt.subplots(1,2,figsize=(8,4), dpi = 300)
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch[range(fs*st,fs*sp)], ax = axes[0] , linewidth=0.5 )
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch_hat[range(fs*st,fs*sp)] , ax = axes[1], linewidth=0.5 )
###########
###########
###########
###########
fig,axes = plt.subplots(2,2,figsize=(8,8), dpi = 300)
sns.histplot(adj_xcorr_025[np.triu_indices( len(adj_xcorr_025), k = 1)], ax = axes[0][0])
sns.histplot(adj_pear[np.triu_indices( len(adj_pear), k = 1)], ax = axes[0][1])
sns.histplot(adj_spear[np.triu_indices( len(adj_spear), k = 1)], ax = axes[1][0])
fig,axes = plt.subplots(2,2,figsize=(8,8), dpi = 300)
sns.histplot(adj_butter_xcorr[np.triu_indices( len(adj_butter_xcorr), k = 1)], ax = axes[0][0])
sns.histplot(adj_butter_pear[np.triu_indices( len(adj_butter_pear), k = 1)], ax = axes[0][1])
sns.histplot(adj_butter_spear[np.triu_indices( len(adj_butter_spear), k = 1)], ax = axes[1][0])
###########
###########
###########
###########
n1=18; n2 = 37
d1= data_hat[:,n1]
d2= data_hat[:,n2]
print(f"\nx_xorr:   {np.round( adj_xcorr_025[n1,n2],2 )}"  ) 
print(f"Pearson:  {np.round( pearsonr(d1, d2)[0],2 )}; p-value: {np.round( pearsonr(d1, d2)[1],2 )}"  ) 
print(f"Spearman: {np.round( spearmanr(d1, d2)[0],2 )}; p-value: {np.round( spearmanr(d1, d2)[1],2 )}"  ) 
adj_xcorr_025[n1,n2]; adj_pear[n1,n2]; adj_spear[n1,n2]
fig,axes = plt.subplots(1,1,figsize=(8,4), dpi = 300)
sns.regplot(  x = data_hat[range(fs*st, fs*sp),  n1], y= data_hat[range(fs*st, fs*sp),n2], ax = axes , scatter_kws={"s":0.05})
d1= data_hat[:,n1]
d2= data_hat[:,n2]
print(f"\nx_xorr:   {np.round( adj_butter_xcorr[n1,n2],2 )}"  ) 
print(f"\nPearson:  {np.round( pearsonr(d1, d2)[0],2 )}; p-value: {np.round( pearsonr(d1, d2)[1],2 )}"  ) 
print(f"Spearman: {np.round( spearmanr(d1, d2)[0],2 )}; p-value: {np.round( spearmanr(d1, d2)[1],2 )}"  ) 
fig,axes = plt.subplots(1,1,figsize=(8,4), dpi = 300)
sns.regplot(  x = data_butter[range(fs*st, fs*sp),  n1], y= data_butter[range(fs*st, fs*sp),n2], ax = axes , scatter_kws={"s":0.1})
elecLoc["Tissue_segmentation_distance_from_label_2"]   
elecLoc["electrode_name"]   
    
    
eeg.columns    
    
    
tmp = np.intersect1d(elecLoc["electrode_name"]   , eeg.columns  , return_indices = True )    
    
    
elecLoc["Tissue_segmentation_distance_from_label_2"]   
tmp2 = np.array(elecLoc.iloc[tmp[1],:]["Tissue_segmentation_distance_from_label_2"]    )
    
adjjj = adj_xcorr
adjjj = adj_pear
adjjj = adj_spear
adjjj = adj_MI
ind_wm = np.where(tmp2 > 0)[0]
ind_gm = np.where(tmp2 <= 0)[0]
tmp_wm = adjjj[ind_wm[:,None], ind_wm[None,:]]
tmp_gm = adjjj[ind_gm[:,None], ind_gm[None,:]]
np.mean(tmp_gm)
np.mean(tmp_wm)
order = np.argsort(tmp2)
tmp2[order][63]
adj_xcorr_ord = adj_xcorr[order[:,None], order[None,:]]
adj_pear_ord = adj_pear[order[:,None], order[None,:]]
adj_spear_ord = adj_spear[order[:,None], order[None,:]]
adj_MI_ord = adj_MI[order[:,None], order[None,:]]
"""
