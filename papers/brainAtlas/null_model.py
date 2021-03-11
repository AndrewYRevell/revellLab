# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:51:22 2020

@author: asilv
"""
import numpy as np
import pandas as pd 
import pickle
import get_structure_function_correlation as g_sfc


def generate_fake_graph(A):
    # logical index for only the upper half     
    C = np.array((np.triu(np.ones(A.shape),1)),dtype=bool)
    # pull off non-diag elements 
    non_diag = A[C]
    # randomly shuffle the non-diag elements
    np.random.shuffle(non_diag)
    
    # read out matrix
    B = np.zeros(A.shape)
    
    count = 0
    for i in range(0,A.shape[0]):
        for j in range(0,i):
            B[i,j] = non_diag[count]
            count = count + 1
            
    C = B + np.transpose(B)
    return(C)

def generate_null_func_conn(function_file_path,n_perm):
    #Get functional connecitivty data in pickle file format
    FC_list_all = []
    with open(function_file_path, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
    FC_list = [broadband, alphatheta, beta, lowgamma,highgamma ]
    # loop over the permutations and append the shuffled graphs 
    for i in range(0,n_perm):
        # loop over the freq bands
        # create new copy of orig to start with
        #FC_list_temp = FC_list.copy()
        FC_list_tmp = []
        for freq in range(0,len(FC_list)):
            # loop over time points 
            freq_specifc = np.zeros((FC_list[freq].shape[1],FC_list[freq].shape[1],FC_list[freq].shape[2]))
            for t in range(0,FC_list[freq].shape[2]):
                A = FC_list[freq][:,:,t] 
                shuff = generate_fake_graph(A)
                freq_specifc[:,:,t] = shuff
            FC_list_tmp.append(freq_specifc)   
            
        FC_list_all.append(FC_list_tmp)
        
    return(FC_list_all,electrode_row_and_column_names)
            
  
    
def generate_null_model_SFC_for_atlas(FC_list_all,electrode_row_and_column_names,structure_file_path,electrode_localization_by_atlas_file_path, outputfile):
    perm_sfc_list = []
    for i in range(0,len(FC_list_all)):
        cur_sfc = g_sfc.SFC_for_null_model(structure_file_path,FC_list_all[i].copy(),electrode_row_and_column_names,electrode_localization_by_atlas_file_path)
        perm_sfc_list.append(cur_sfc)
    order_of_matrices_in_pickle_file = pd.DataFrame(["broadband", "alphatheta", "beta", "lowgamma" , "highgamma" ], columns=["Order of matrices in pickle file"])
    with open(outputfile, 'wb') as f: pickle.dump([perm_sfc_list,order_of_matrices_in_pickle_file], f)
    
    
        
    

            
            
            
            
            
            
            