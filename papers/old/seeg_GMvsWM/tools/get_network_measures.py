# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import bct as bct 
import numpy as np
import pandas as pd 
import pickle
from multiprocessing import Pool




def get_true_and_resampled_network(function_file_path,output_path,electrode_localization_by_classification_atlas_file_path):

    print('Starting ',function_file_path)
    
    with open(function_file_path, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
    FC_list = [broadband, alphatheta, beta, lowgamma,highgamma ]

    # set up the dataframe of electrodes to analyze 
    final_electrodes = pd.DataFrame(electrode_row_and_column_names,columns=['electrode_name'])
    final_electrodes = final_electrodes.reset_index()
    final_electrodes = final_electrodes.rename(columns={"index": "func_index"})
    
    # Get electrode localization by classification atlas
    electrode_localization_by_class_atlas = pd.read_csv(electrode_localization_by_classification_atlas_file_path)
    
    # now join in the classification region number
    final_electrodes = final_electrodes.merge(electrode_localization_by_class_atlas.iloc[:,[0,4]],on = 'electrode_name')
    final_electrodes = final_electrodes[final_electrodes.iloc[:,2]>=0]
    for i in range(len(FC_list)):
        FC_list[i] = FC_list[i][final_electrodes['func_index'],:,:]
        FC_list[i] = FC_list[i][:,final_electrodes['func_index'], :]  
        
    # reset the index after adjusting for the order 
    final_electrodes = final_electrodes.reset_index()
    final_electrodes = final_electrodes.drop(columns=['index','func_index'])
    final_electrodes = final_electrodes.reset_index()
    final_electrodes = final_electrodes.rename(columns={"index": "func_index"})
    
    # now average the functional network and, for now, just use broadband
    broad_avg = np.arctanh(FC_list[0])
    broad_avg = np.mean(broad_avg,axis=2)
    broad_avg = np.tanh(broad_avg)
    
    # the true network measures 
    true_network_stats = get_true_network_metrics(broad_avg)
    
    # the resampled measures 
    resampled_network_stats_grey_matter = resample_network_multi_core(broad_avg,1000,[0,.25,.50,.75,1],0,final_electrodes,22)
    
    resampled_network_stats_white_matter = resample_network_multi_core(broad_avg,1000,[0,.25,.50,.75,1],1,final_electrodes,22)
    
    order_of_dicts_in_pickle_file = pd.DataFrame(["true_metrics", "resampled_grey", "resampled_white"], columns=["Order of dicts in pickle file"])
    with open(output_path, 'wb') as f: pickle.dump([true_network_stats, resampled_network_stats_grey_matter, resampled_network_stats_white_matter, order_of_dicts_in_pickle_file], f)


    




def get_true_network_metrics(A):
    
    #control centrality 
    c_c = control_centrality(A)
    
    cc_fake = np.zeros((100,1))
    for i in range(0,100):
        cc_fake[i] = np.mean(control_centrality(generate_fake_graph(A)))

    m_cc_fake = np.mean(cc_fake)
    cc_norm = c_c/m_cc_fake
    
    # Get identity of node with lowest control centrality
    min_cc_true = np.where(c_c == np.amin(c_c))[0]
    
    # get synchronizability
    sync = synchronizability(A)
    
    # normalized sync
    sync_fake = np.zeros((100,1))
    for i in range(0,100):
        sync_fake[i] = synchronizability(generate_fake_graph(A))

    m_sync_fake = np.mean(sync_fake)
    sync_norm = sync/m_sync_fake
    
    # get betweeness centrality
    bc = betweenness_centrality(A)
    bc_fake = np.zeros((100,1))
    for i in range(0,100):
        bc_fake[i] = np.mean(betweenness_centrality(generate_fake_graph(A)))

    m_bc_fake = np.mean(bc_fake)
    bc_norm = bc/m_bc_fake
    
    # Get identity of node with max bc
    max_bc_true = np.where(bc == np.amax(bc))[0]
    
    
    # get eigenvector centrality
    ec = bct.eigenvector_centrality_und(A)
    ec_fake = np.zeros((100,1))
    for i in range(0,100):
        ec_fake[i] = np.mean(bct.eigenvector_centrality_und(generate_fake_graph(A)))

    m_ec_fake = np.mean(ec_fake)
    ec_norm = ec/m_ec_fake
    
    # Get identity of node with max ec
    max_ec_true = np.where(ec == np.amax(ec))[0]
    
    # get edge betweeness centrality
    edge_bc,ignore = bct.edge_betweenness_wei(A)
    edge_bc_fake = np.zeros((100,1))
    for i in range(0,100):
        edge_bc_fake[i] = np.mean(bct.edge_betweenness_wei(generate_fake_graph(A))[0])
    m_edge_bc_fake = np.mean(edge_bc_fake)
    edge_bc_norm = edge_bc/m_edge_bc_fake;
    
    
    # get clustering coeff
    clust = bct.clustering_coef_wu(A)
    clust_fake = np.zeros((100,1))
    for i in range(0,100):
        clust_fake[i] = np.mean(bct.clustering_coef_wu(generate_fake_graph(A)))

    m_clust_fake = np.mean(clust_fake)
    clust_norm = clust/m_clust_fake
    
    # Get identity of node with max clust
    max_clust_true = np.where(clust == np.amax(clust))[0]
    
    # get node strength
    ns = node_strength(A)
    ns_fake = np.zeros((100,1))
    for i in range(0,100):
        ns_fake[i] = np.mean(node_strength(generate_fake_graph(A)))

    m_ns_fake = np.mean(ns_fake)
    ns_norm = ns/m_ns_fake
    
    # Get identity of node with max clust
    max_ns_true = np.where(ns == np.amax(ns))[0]



    #Get true efficiency 
    Ci,ignore = bct.modularity_und(A)
    par = bct.participation_coef(A,Ci)   


    eff = bct.efficiency_wei(A, 0)
    eff_fake = np.zeros((100,1))
    for i in range(0,100):
        eff_fake[i] = (bct.efficiency_wei(generate_fake_graph(A)))

    m_eff_fake = np.mean(eff_fake)
    eff_norm = eff/m_eff_fake
    
    # Get true transistivity   
    trans = bct.transitivity_wu(A)
    trans_fake = np.zeros((100,1))
    for i in range(0,100):
        trans_fake[i] = (bct.transitivity_wu(generate_fake_graph(A)))

    m_trans_fake = np.mean(trans_fake)
    trans_norm = trans/m_trans_fake   
    
    # store output results in a dictionary 
    #nodal
    results = {}
    results['control_centrality'] = c_c
    results['control_centrality_norm'] = cc_norm
    results['min_cc_node'] = min_cc_true

    # global measure 
    results['sync'] = sync
    results['sync_norm'] = sync_norm
    
    # nodal 
    results['bc'] = bc
    results['bc_norm'] = bc_norm
    results['max_bc_node'] = max_bc_true
    
    # nodal
    results['ec'] = ec
    results['ec_norm'] = ec_norm
    results['max_ec_node'] = max_ec_true  
    
    # nodal 
    results['clust'] = clust
    results['clust_norm'] = clust_norm
    results['max_clust_node'] = max_clust_true
    
    # nodal 
    results['ns'] = ns
    results['ns_norm'] = ns_norm
    results['max_ns_node'] = max_ns_true

    # global
    results['eff'] = eff
    results['eff_norm'] = eff_norm

    # global 
    results['trans'] = trans
    results['trans_norm'] = trans_norm
    
    # nodal 
    results['par'] = par
    
    # edge
    results['edge_bc'] = edge_bc
    results['edge_bc_norm'] = edge_bc_norm
    
    return(results)


def resample_network_multi_core(A,n_perm,e_f,type_to_remove,final_electrodes,n_core):

    # type to remove : 1 for white matter, 0 for grey matter 
    
    
    # e_f : Following Erin + John convention. This is the fraction of nodes to 
    # keep in the network (ex.. ef=.2 means we remove 80% of the nodes)
    e_f = np.array(e_f)
    nch = A.shape[0]
    n_f = e_f.shape[0]
    
    
    
    # create sub dataframes for only the grey and white matter elec
    wm = final_electrodes[final_electrodes.iloc[:,2]>0]
    gm = final_electrodes[final_electrodes.iloc[:,2]==0]
    
    # numbers of each electrode type 
    numWhite = wm.shape[0]
    numGrey = gm.shape[0]
    
    # fraction to remove 
    if(type_to_remove==1):
        e_n = numWhite-np.ceil(e_f*numWhite)
    else: 
        e_n = numGrey-np.ceil(e_f*numGrey)
    
    # control centrality
    all_c_c = np.zeros((nch,n_f,n_perm))
    all_c_c[:] = np.nan
    cc_reg =  np.zeros((nch,n_f,n_perm))
    cc_reg[:] = np.nan

    all_cc_norm =  np.zeros((nch,n_f,n_perm))
    all_cc_norm[:] = np.nan
    
    #init node strengths
    all_ns =  np.zeros((nch,n_f,n_perm))
    all_ns[:] = np.nan

    all_ns_norm =  np.zeros((nch,n_f,n_perm))
    all_ns_norm[:] = np.nan
    
    # init betweenness centrality
    all_bc =  np.zeros((nch,n_f,n_perm))
    all_bc[:] = np.nan

    all_bc_norm =  np.zeros((nch,n_f,n_perm))
    all_bc_norm[:] = np.nan
    
    # synch 
    all_sync =  np.zeros((n_f,n_perm))
    all_sync[:] = np.nan

    all_sync_norm =  np.zeros((n_f,n_perm))
    all_sync_norm[:] = np.nan
   
    # efficiency
    all_eff =  np.zeros((n_f,n_perm))
    all_eff[:] = np.nan

    all_eff_norm =  np.zeros((n_f,n_perm))
    all_eff_norm[:] = np.nan
    
    # eigenvector centrality
    all_ec =  np.zeros((nch,n_f,n_perm))
    all_ec[:] = np.nan

    all_ec_norm =  np.zeros((nch,n_f,n_perm))
    all_ec_norm[:] = np.nan

    
    # clustering coeff
    all_clust =  np.zeros((nch,n_f,n_perm))
    all_clust[:] = np.nan

    all_clust_norm =  np.zeros((nch,n_f,n_perm))
    all_clust_norm[:] = np.nan

    # participation coeff
    all_par =  np.zeros((nch,n_f,n_perm))
    all_par[:] = np.nan

    
    # transistivity 
    all_trans =  np.zeros((n_f,n_perm))
    all_trans[:] = np.nan

    all_trans_norm =  np.zeros((n_f,n_perm))
    all_trans_norm[:] = np.nan
    
    # edge bc
    all_edge_bc =  []
    all_edge_bc_norm = []

    
    # get true particpation 
    Ci,ignore = bct.modularity_und(A)
    true_par = bct.participation_coef(A,Ci)  
    avg_par_removed =  np.zeros((n_f,n_perm))
    avg_par_removed[:] = np.nan

    
    # get the true bc
    true_bc = betweenness_centrality(A)
    avg_bc_removed = np.zeros((n_f,n_perm))
    avg_bc_removed[:] = np.nan
    

    
    # loop over all removal fractions and permutations 
    for f in range(0,n_f):
        all_edge_bc_cur_fraction = []
        all_edge_bc_norm_cur_fraction = []
        pool = Pool(n_core)
        all_As = []
        all_ch_inds = []
        print("Starting removal fraction {0}".format(f))
        for i_p in range(0,n_perm):
            
            
            # make a copy of the adjacency matrix (we will edit this each time) 
            A_tmp = A.copy()
            
            # picking the nodes to remove 
            if(type_to_remove==1):
                to_remove = wm.sample(int(e_n[f])).iloc[:,0]
            else:
                to_remove = gm.sample(int(e_n[f])).iloc[:,0]
            
            # take these electrodes out of the adjacency matrix 
            A_tmp = np.delete(A_tmp, to_remove, axis=0)
            A_tmp = np.delete(A_tmp, to_remove, axis=1)
            
            # create a new array to hold the identity of the channels 
            ch_ids = np.arange(0,nch)
            ch_ids = np.delete(ch_ids,to_remove)
            
            # append to list for multicore processing 
            all_As.append(A_tmp)
            all_ch_inds.append(ch_ids)
        
        return_list = pool.map(get_true_network_metrics, all_As)
        
        for p in range(0,len(return_list)):    
            r = return_list[p]
            ch_ids = all_ch_inds[p]
            # edge metric
            all_edge_bc_cur_fraction.append(r['edge_bc'])
            all_edge_bc_norm_cur_fraction.append(r['edge_bc_norm'])
            # populate the nodal measures 
            for i in range(0,ch_ids.shape[0]):
                all_c_c[ch_ids[i],f,i_p] = r['control_centrality'][i]
                all_ns[ch_ids[i],f,i_p] = r['ns'][i]
                all_bc[ch_ids[i],f,i_p] = r['bc'][i]
                all_par[ch_ids[i],f,i_p] = r['par'][i]
                all_ec[ch_ids[i],f,i_p] = r['ec'][i]
                all_clust[ch_ids[i],f,i_p] = r['clust'][i]
                
                all_cc_norm[ch_ids[i],f,i_p] = r['control_centrality_norm'][i]
                all_ns_norm[ch_ids[i],f,i_p] = r['ns_norm'][i]
                all_bc_norm[ch_ids[i],f,i_p] = r['bc_norm'][i]
                all_ec_norm[ch_ids[i],f,i_p] = r['ec_norm'][i]
                all_clust_norm[ch_ids[i],f,i_p] = r['clust_norm'][i]
    
            # populate the global measures
            all_sync[f,i_p] = r['sync']
            all_sync_norm[f,i_p] = r['sync_norm']
            
            all_eff[f,i_p] = r['eff']
            all_eff_norm[f,i_p] = r['eff_norm']            
                
            
            all_trans[f,i_p] = r['trans']
            all_trans_norm[f,i_p] = r['trans_norm']
        
        
    all_edge_bc.append(all_edge_bc_cur_fraction)
    all_edge_bc_norm.append(all_edge_bc_norm_cur_fraction)
    
    
    # construct the output dictionary from a resampling 

    #nodal
    results = {}
    results['control_centrality'] = all_c_c
    results['control_centrality_norm'] = all_cc_norm

    # global measure 
    results['sync'] = all_sync
    results['sync_norm'] = all_sync_norm
    
    # nodal 
    results['bc'] = all_bc
    results['bc_norm'] = all_bc_norm
    
    # nodal
    results['ec'] = all_ec
    results['ec_norm'] = all_ec_norm
    
    # nodal 
    results['clust'] = all_clust
    results['clust_norm'] = all_clust_norm
    
    # nodal 
    results['ns'] = all_ns
    results['ns_norm'] = all_ns_norm

    # global
    results['eff'] = all_eff
    results['eff_norm'] = all_eff_norm

    # global 
    results['trans'] = all_trans
    results['trans_norm'] = all_trans_norm
    
    # nodal 
    results['par'] = all_par
    
    #edge 
    results['edge_bc'] = all_edge_bc
    results['edge_bc_norm'] = all_edge_bc_norm
    
    return(results)
   




    

def resample_network(A,n_perm,e_f,type_to_remove,final_electrodes):
    
    # type to remove : 1 for white matter, 0 for grey matter 
    
    
    # e_f : Following Erin + John convention. This is the fraction of nodes to 
    # keep in the network (ex.. ef=.2 means we remove 80% of the nodes)
    e_f = np.array(e_f)
    nch = A.shape[0]
    n_f = e_f.shape[0]
    
    
    
    # create sub dataframes for only the grey and white matter elec
    wm = final_electrodes[final_electrodes.iloc[:,2]>0]
    gm = final_electrodes[final_electrodes.iloc[:,2]==0]
    
    # numbers of each electrode type 
    numWhite = wm.shape[0]
    numGrey = gm.shape[0]
    
    # fraction to remove 
    if(type_to_remove==1):
        e_n = numWhite-np.ceil(e_f*numWhite)
    else: 
        e_n = numGrey-np.ceil(e_f*numGrey)
    
    # control centrality
    all_c_c = np.zeros((nch,n_f,n_perm))
    all_c_c[:] = np.nan
    cc_reg =  np.zeros((nch,n_f,n_perm))
    cc_reg[:] = np.nan

    all_cc_norm =  np.zeros((nch,n_f,n_perm))
    all_cc_norm[:] = np.nan
    
    #init node strengths
    all_ns =  np.zeros((nch,n_f,n_perm))
    all_ns[:] = np.nan

    all_ns_norm =  np.zeros((nch,n_f,n_perm))
    all_ns_norm[:] = np.nan
    
    # init betweenness centrality
    all_bc =  np.zeros((nch,n_f,n_perm))
    all_bc[:] = np.nan

    all_bc_norm =  np.zeros((nch,n_f,n_perm))
    all_bc_norm[:] = np.nan
    
    # synch 
    all_sync =  np.zeros((n_f,n_perm))
    all_sync[:] = np.nan

    all_sync_norm =  np.zeros((n_f,n_perm))
    all_sync_norm[:] = np.nan
   
    # efficiency
    all_eff =  np.zeros((n_f,n_perm))
    all_eff[:] = np.nan

    all_eff_norm =  np.zeros((n_f,n_perm))
    all_eff_norm[:] = np.nan
    
    # eigenvector centrality
    all_ec =  np.zeros((nch,n_f,n_perm))
    all_ec[:] = np.nan

    all_ec_norm =  np.zeros((nch,n_f,n_perm))
    all_ec_norm[:] = np.nan

    
    # clustering coeff
    all_clust =  np.zeros((nch,n_f,n_perm))
    all_clust[:] = np.nan

    all_clust_norm =  np.zeros((nch,n_f,n_perm))
    all_clust_norm[:] = np.nan

    # participation coeff
    all_par =  np.zeros((nch,n_f,n_perm))
    all_par[:] = np.nan

    
    # transistivity 
    all_trans =  np.zeros((n_f,n_perm))
    all_trans[:] = np.nan

    all_trans_norm =  np.zeros((n_f,n_perm))
    all_trans_norm[:] = np.nan
    
    # edge bc
    all_edge_bc =  []
    all_edge_bc_norm = []

    
    # get true particpation 
    Ci,ignore = bct.modularity_und(A)
    true_par = bct.participation_coef(A,Ci)  
    avg_par_removed =  np.zeros((n_f,n_perm))
    avg_par_removed[:] = np.nan

    
    # get the true bc
    true_bc = betweenness_centrality(A)
    avg_bc_removed = np.zeros((n_f,n_perm))
    avg_bc_removed[:] = np.nan
    

    
    # loop over all removal fractions and permutations 
    for f in range(0,n_f):
        all_edge_bc_cur_fraction = []
        all_edge_bc_norm_cur_fraction = []
        for i_p in range(0,n_perm):
            
            if(i_p % 100 ==0):
                print("Doing permutation {0} for removal of fraction {1}".format(i_p,f))
            
            # make a copy of the adjacency matrix (we will edit this each time) 
            A_tmp = A.copy()
            
            # picking the nodes to remove 
            if(type_to_remove==1):
                to_remove = wm.sample(int(e_n[f])).iloc[:,0]
            else:
                to_remove = gm.sample(int(e_n[f])).iloc[:,0]
            
            # take these electrodes out of the adjacency matrix 
            A_tmp = np.delete(A_tmp, to_remove, axis=0)
            A_tmp = np.delete(A_tmp, to_remove, axis=1)
            
            # create a new array to hold the identity of the channels 
            ch_ids = np.arange(0,nch)
            ch_ids = np.delete(ch_ids,to_remove)
            
            
            
            # get the new metrics from A_tmp
            r = get_true_network_metrics(A_tmp)
            
            # edge metric
            all_edge_bc_cur_fraction.append(r['edge_bc'])
            all_edge_bc_norm_cur_fraction.append(r['edge_bc_norm'])
            # populate the nodal measures 
            for i in range(0,ch_ids.shape[0]):
                all_c_c[ch_ids[i],f,i_p] = r['control_centrality'][i]
                all_ns[ch_ids[i],f,i_p] = r['ns'][i]
                all_bc[ch_ids[i],f,i_p] = r['bc'][i]
                all_par[ch_ids[i],f,i_p] = r['par'][i]
                all_ec[ch_ids[i],f,i_p] = r['ec'][i]
                all_clust[ch_ids[i],f,i_p] = r['clust'][i]
                
                all_cc_norm[ch_ids[i],f,i_p] = r['control_centrality_norm'][i]
                all_ns_norm[ch_ids[i],f,i_p] = r['ns_norm'][i]
                all_bc_norm[ch_ids[i],f,i_p] = r['bc_norm'][i]
                all_ec_norm[ch_ids[i],f,i_p] = r['ec_norm'][i]
                all_clust_norm[ch_ids[i],f,i_p] = r['clust_norm'][i]

            # populate the global measures
            all_sync[f,i_p] = r['sync']
            all_sync_norm[f,i_p] = r['sync_norm']
            
            all_eff[f,i_p] = r['eff']
            all_eff_norm[f,i_p] = r['eff_norm']            
                
            
            all_trans[f,i_p] = r['trans']
            all_trans_norm[f,i_p] = r['trans_norm']
            
            
        all_edge_bc.append(all_edge_bc_cur_fraction)
        all_edge_bc_norm.append(all_edge_bc_norm_cur_fraction)
    
    
    # construct the output dictionary from a resampling 

    #nodal
    results = {}
    results['control_centrality'] = all_c_c
    results['control_centrality_norm'] = all_cc_norm

    # global measure 
    results['sync'] = all_sync
    results['sync_norm'] = all_sync_norm
    
    # nodal 
    results['bc'] = all_bc
    results['bc_norm'] = all_bc_norm
    
    # nodal
    results['ec'] = all_ec
    results['ec_norm'] = all_ec_norm
    
    # nodal 
    results['clust'] = all_clust
    results['clust_norm'] = all_clust_norm
    
    # nodal 
    results['ns'] = all_ns
    results['ns_norm'] = all_ns_norm

    # global
    results['eff'] = all_eff
    results['eff_norm'] = all_eff_norm

    # global 
    results['trans'] = all_trans
    results['trans_norm'] = all_trans_norm
    
    # nodal 
    results['par'] = all_par
    
    #edge 
    results['edge_bc'] = all_edge_bc
    results['edge_bc_norm'] = all_edge_bc_norm
    
    return(results)
   
def generate_resampled_statistics(A,n_perm,e_f,type_to_remove,final_electrodes):
    # first pull off the true metrics 
    true_results = get_true_network_metrics(A)
    # next pull off resampled metrics 
    re_results = resample_network(A,n_perm,e_f,type_to_remove,final_electrodes)
    
    # for global metrics pull off the relative difference
    sync_rel_diff = (re_results['sync']-true_results['sync'])/true_results['sync']
            
    eff_rel_diff = (re_results['eff']-true_results['eff'])/true_results['eff']
            
    trans_rel_diff = (re_results['trans']-true_results['trans'])/true_results['trans']

    sync_norm_rel_diff = (re_results['sync_norm']-true_results['sync_norm'])/true_results['sync_norm']
            
    eff_norm_rel_diff = (re_results['eff_norm']-true_results['eff_norm'])/true_results['eff_norm']
            
    trans_norm_rel_diff = (re_results['trans_norm']-true_results['trans_norm'])/true_results['trans_norm']    


    # for nodal measurs we can do the average relative difference 
    bc_rel_diff = nodal_avg_rel_diff('bc', re_results, true_results)
    bc_norm_rel_diff = nodal_avg_rel_diff('bc_norm', re_results, true_results)

    control_centrality_rel_diff = nodal_avg_rel_diff('control_centrality', re_results, true_results)
    control_centrality_norm_rel_diff = nodal_avg_rel_diff('control_centrality', re_results, true_results)

    ec_rel_diff = nodal_avg_rel_diff('ec', re_results, true_results)
    ec_norm_rel_diff = nodal_avg_rel_diff('ec_norm', re_results, true_results)

    clust_rel_diff = nodal_avg_rel_diff('clust', re_results, true_results)
    clust_norm_rel_diff = nodal_avg_rel_diff('clust_norm', re_results, true_results)

    ns_rel_diff = nodal_avg_rel_diff('ns', re_results, true_results)
    ns_norm_rel_diff = nodal_avg_rel_diff('ns_norm', re_results, true_results)
  
    par_rel_diff = nodal_avg_rel_diff('par', re_results, true_results)
      
    
    results = {}
    results['control_centrality'] = control_centrality_rel_diff
    results['control_centrality_norm'] = control_centrality_norm_rel_diff

    # global measure 
    results['sync'] = sync_rel_diff
    results['sync_norm'] = sync_norm_rel_diff
    
    # nodal 
    results['bc'] = bc_rel_diff
    results['bc_norm'] = bc_norm_rel_diff
    
    # nodal
    results['ec'] = ec_rel_diff
    results['ec_norm'] = ec_norm_rel_diff
    
    # nodal 
    results['clust'] = clust_rel_diff
    results['clust_norm'] = clust_norm_rel_diff
    
    # nodal 
    results['ns'] = ns_rel_diff
    results['ns_norm'] = ns_norm_rel_diff

    # global
    results['eff'] = eff_rel_diff
    results['eff_norm'] = eff_norm_rel_diff

    # global 
    results['trans'] = trans_rel_diff
    results['trans_norm'] = trans_norm_rel_diff
    
    # nodal 
    results['par'] = par_rel_diff
    
    return(results)
    # skip these metrics for now 
    # we can also do the spearman rank coorelation 
    
    # we can then do reliability 
    
    
    


def nodal_avg_rel_diff(metric,re_dict,true_dict):
    avg_nodal = np.nanmean(re_dict[metric],0)
    avg_true = np.nanmean(true_dict[metric])
    return((avg_nodal-avg_true)/avg_true)

    

def node_strength(A):
    return(np.sum(A,axis=1))

    
def betweenness_centrality(A):
    A[A==0] = np.amin(A[A>0])
    cost = 1/A
    bc = bct.betweenness_wei(cost)
    return(bc/2)



def synchronizability(A):
    # degree vector
    dvector = np.sum(A,axis=0)
    
    # convert to diag matrix
    D = np.diag(dvector)
    
    # Laplacian 
    L = D - A
    
    # compute eigenvalues of L
    w, v = np.linalg.eig(L)
    
    # sort 
    w = np.sort(w)
    
    # compute synch: ratio of second smallest eigenvalue to largest eigenvalue
    sync = abs(w[1]/w[-1])
    
    return(sync)
    
def control_centrality(A):
    n_ch = A.shape[0]
    
    # synch
    sync = synchronizability(A)
    
    c_c = np.zeros((n_ch,1))
    for ich in range(0,n_ch):
        
        # remove channel 
        A_temp = np.delete(A, ich, 0)
        A_temp = np.delete(A_temp,ich,1)
        
        # recompute synch
        sync_temp = synchronizability(A_temp)
        
        c_c[ich] = (sync_temp - sync)/sync
    
    return(c_c)


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
            
    
    