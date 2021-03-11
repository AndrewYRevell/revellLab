
"""
"""


import numpy as np
import pickle
import pandas as pd
from scipy.io import loadmat #fromn scipy to read mat files (structural connectivity)
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import re #regular expression
#%%
  
def SFC(ifname_SC, ifname_SCtxt, ifname_FC, ifname_electrode_localization, ofname_SFC):
    
    """
    :param ifname_SC:
    :param ifname_FC:
    :param ifname_electrode_localization:
    :return:
    """
    
    """
    from os.path import join as ospj
    #path = "/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001"
    path = "/mnt"
    #Example:
    ifname_SC = ospj(path, 'data_processed/connectivity_matrices/structure/sub-RID0278/aal_res-1x1x1/sub-RID0278_ses-preop3T_dwi-eddyMotionB0Corrected.aal_res-1x1x1.count.pass.connectivity.mat')
    ifname_electrode_localization = ospj(path, 'data_processed/electrode_localization_atlas_region/sub-RID0278/aal_res-1x1x1/sub-RID0278_electrode_coordinates_mni_aal_res-1x1x1.csv')
       
    
        
    #seizure 3
    
    #inter-ictal
    ifname_FC = ospj(path, 'data_processed/connectivity_matrices/function/sub-RID0278/sub-RID0278_HUP138_phaseII_394423190000_394603190000_functionalConnectivity.pickle')
    
    #pre-ictal 
    ifname_FC = ospj(path, 'data_processed/connectivity_matrices/function/sub-RID0278/sub-RID0278_HUP138_phaseII_415843190000_416023190000_functionalConnectivity.pickle')
    
    #ictal 
    ifname_FC = ospj(path, 'data_processed/connectivity_matrices/function/sub-RID0278/sub-RID0278_HUP138_phaseII_416023190000_416112890000_functionalConnectivity.pickle')
    
    #postictal 
    ifname_FC = ospj(path, 'data_processed/connectivity_matrices/function/sub-RID0278/sub-RID0278_HUP138_phaseII_416112890000_416292890000_functionalConnectivity.pickle')
    
     
    """
    #Get functional connecitivty data in pickle file format
    with open(ifname_FC, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
    FC_list = [broadband, alphatheta, beta, lowgamma,highgamma ]
    
    # set up the dataframe of electrodes to analyze 
    final_electrodes = pd.DataFrame(electrode_row_and_column_names,columns=['electrode_name'])
    final_electrodes = final_electrodes.reset_index()
    final_electrodes = final_electrodes.rename(columns={"index": "func_index"})
    
    #Get Structural Connectivity data in mat file format. Output from DSI studio
    structural_connectivity_array = np.array(pd.DataFrame(loadmat(ifname_SC)['connectivity']))
    
    structural_connectivity_names = pd.read_csv(ifname_SCtxt, sep="\t", header=1,index_col=1)
    structural_connectivity_names = structural_connectivity_names.columns[1:-1]
    SC_region_names = []
    for r in range(len(structural_connectivity_names)):
        SC_region_names.append(re.sub('region_', '', structural_connectivity_names[r])    ) 
    SC_region_names = np.array(SC_region_names)
    SC_region_names = SC_region_names.astype('int') 
    
    #Get electrode localization by atlas csv file data. From get_electrode_localization.py
    electrode_localization_by_atlas = pd.read_csv(ifname_electrode_localization)
    
    # normalizing and log-scaling the structural matrices
    structural_connectivity_array[structural_connectivity_array == 0] = 1;
    structural_connectivity_array = np.log10(structural_connectivity_array)  # log-scaling. Converting 0s to 1 to avoid taking log of zeros
    structural_connectivity_array = structural_connectivity_array / np.max(structural_connectivity_array)  # normalization
    
    
    #Only consider electrodes that are in both the localization and the functional connectivity file
    final_electrodes = final_electrodes.merge(electrode_localization_by_atlas.iloc[:,[0,4]],on='electrode_name')
    # Remove electrodes in the Functional Connectivity matrices that have a region of 0 
    final_electrodes = final_electrodes[final_electrodes['region_number']!=0]
    
    for i in range(len(FC_list)):
        FC_list[i] = FC_list[i][final_electrodes['func_index'],:,:]
        FC_list[i] = FC_list[i][:,final_electrodes['func_index'], :]
    
    #Fisher z-transform of functional connectivity data. This is to take means of correlations and do correlations to the structural connectivity
    #Fisher z transform is just arctanh
    for i in range(len(FC_list)):
        FC_list[i] = np.arctanh(FC_list[i])
    
    # Remove structural ROIs not in electrode_localization ROIs
    electrode_ROIs = np.unique(np.array(final_electrodes.iloc[:,2]))
    electrode_ROIs = electrode_ROIs[~(electrode_ROIs == 0)] #remove region 0
   
    structural_index = []
    for r in range(len(electrode_ROIs)):
        structural_index.append(  np.where(electrode_ROIs[r] ==SC_region_names  )[0][0]   )
    structural_index = np.array(structural_index)
    structural_connectivity_array = structural_connectivity_array[structural_index,:]
    structural_connectivity_array = structural_connectivity_array[:,structural_index]
    
    #taking average functional connectivity for those electrodes in same atlas regions
    for i in range(len(FC_list)):
        ROIs = np.array( final_electrodes.iloc[:,2])
        for r in range(len(electrode_ROIs)):
            index_logical = (ROIs == electrode_ROIs[r])
            index_first = np.where(index_logical)[0][0]
            index_second_to_end = np.where(index_logical)[0][1:]
            mean = np.nanmean(FC_list[i][index_logical,:,:], axis=0)
            # Fill in with mean.
            FC_list[i][index_first,:,:] = mean
            FC_list[i][:, index_first, :] = mean
            #delete the other rows and oclumns belonging to same region.
            FC_list[i] = np.delete(FC_list[i], index_second_to_end, axis=0)
            FC_list[i] = np.delete(FC_list[i], index_second_to_end, axis=1)
            #keeping track of which electrode labels correspond to which rows and columns
            ROIs = np.delete(ROIs, index_second_to_end, axis=0)
        #remove electrodes in the ROI labeld as zero
        index_logical = (ROIs == 0)
        index = np.where(index_logical)[0]
        FC_list[i] = np.delete(FC_list[i], index, axis=0)
        FC_list[i] = np.delete(FC_list[i], index, axis=1)
        ROIs = np.delete(ROIs, index, axis=0)
    
    
    #order FC matrices by ROIs
    order = np.argsort(ROIs)
    for i in range(len(FC_list)):
        FC_list[i] = FC_list[i][order,:,:]
        FC_list[i] = FC_list[i][:, order, :]
    
    #un-fisher ztranform
    for i in range(len(FC_list)):
        FC_list[i] = np.tanh(FC_list[i])
    
    
    #initialize correlation arrays
    Corrrelation_list = [None] * len(FC_list)
    for i in range(len(FC_list)):
        Corrrelation_list[i] = np.zeros(  [FC_list[0].shape[2]], dtype=float)
    
    correlation_type = 'spearman'
    #calculate Structure-Function Correlation.
    for i in range(len(FC_list)):
        for t in range(FC_list[i].shape[2]-1):
            #Spearman Rank Correlation: functional connectivity and structural connectivity are non-normally distributed. So we should use spearman
            if correlation_type == 'spearman':
                Corrrelation_list[i][t] = spearmanr(  np.ndarray.flatten(FC_list[i][:,:,t]) , np.ndarray.flatten(structural_connectivity_array)   ).correlation
                #print("spearman")
            # Pearson Correlation: This is calculated bc past studies use Pearson Correlation and we want to see if these results are comparable.
            if correlation_type == 'pearson':
                Corrrelation_list[i][t] = pearsonr(np.ndarray.flatten(FC_list[i][:, :, t]), np.ndarray.flatten(structural_connectivity_array))[0]
    
    
    

    order_of_matrices_in_pickle_file = pd.DataFrame(["broadband", "alphatheta", "beta", "lowgamma" , "highgamma" ], columns=["Order of matrices in pickle file"])
    with open(ofname_SFC, 'wb') as f: pickle.dump([Corrrelation_list[0], Corrrelation_list[1], Corrrelation_list[2], Corrrelation_list[3], Corrrelation_list[4], order_of_matrices_in_pickle_file], f)
    




#%%
"""
ofpath = ospj( path,  "brainAtlas", "figures", "correlation")

import seaborn as sns
i=0
tmp_plt_t = Corrrelation_list[i]
t=179
spearmanr(  np.ndarray.flatten(FC_list[i][:,:,t]) , np.ndarray.flatten(structural_connectivity_array)   ).correlation
tmp_array = FC_list[i][:,:,t]
tmp_FC = np.ndarray.flatten(FC_list[i][:,:,t])
tmp_SC =  np.ndarray.flatten(structural_connectivity_array)

tmp_SC_zeros_removed = np.delete(tmp_SC, np.where(tmp_SC == 0))
tmp_FC_zeros_removed = np.delete(tmp_FC, np.where(tmp_SC == 0))



fig = plt.figure(constrained_layout=False, dpi=300, figsize=(2.5, 2.5))

g = sns.regplot(x = tmp_SC_zeros_removed, y =tmp_FC_zeros_removed, color = "black", line_kws={'linewidth':6, 'color':"#993333cc"})
g.set(ylim=(0, 1))
g.set(xlim=(0, 1))

g.set(xticklabels=[])
g.set(xlabel=None)
g.set(yticklabels=[])
g.set(ylabel=None)

g.tick_params(axis='both', which='both', length=2.75, direction = "in")

#sns.despine()
fig.savefig(ospj(ofpath, "correlation_Craddock400_postictal_{0}_corr_{1}".format(t, int(np.round(tmp_plt_t[t],3)*1000)  ) ), format = "pdf")





tmp1 = Corrrelation_list[0]
tmp2 = Corrrelation_list[0]
tmp3 = Corrrelation_list[0]

x = range(len( tmp2))
y = tmp2

plt.scatter(x = x, y=y)
plt.ylim(-0.1, 0.5)




x = range(len( np.concatenate([ tmp1, tmp2])))
y = np.concatenate([ tmp1, tmp2])

plt.scatter(x = x, y=y)
plt.ylim(-0.1, 0.5)







x = range(len( np.concatenate([ tmp1, tmp2, tmp3])))
y = np.concatenate([ tmp1, tmp2, tmp3])

plt.scatter(x = x, y=y)
plt.ylim(-0.1, 0.5)

#plt.title(atlas)
"""


#%%


