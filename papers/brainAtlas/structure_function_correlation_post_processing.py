"""
2020.01.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:
    python3.6 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
from os.path import join as ospj
sys.path.append(ospj(path, "paper001/code/tools"))
import pickle
import os
import numpy as np
import copy
import scipy.interpolate

#%% Unit testing
#Paths and File names

#path = "/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001"
"""
sub_ID = "sub-RID0194"
atlas = "RandomAtlas0010000"
version = "v0001_"
atlas = "aal_res-1x1x1"
version = ""
ifpath_interictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0194_HUP134_phaseII_D02_157702933433_157882933433_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_preictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0194_HUP134_phaseII_D02_179122933433_179302933433_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_ictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0194_HUP134_phaseII_D02_179302933433_179381931054_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_postictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0194_HUP134_phaseII_D02_179381931054_179561931054_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))


sub_ID = "sub-RID0278"
atlas = "RandomAtlas0005000"
version = "v0001_"
ifpath_interictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0278_HUP138_phaseII_394423190000_394603190000_{1}_{2}SFC.pickle".format(sub_ID, atlas,version ))
ifpath_preictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0278_HUP138_phaseII_415843190000_416023190000_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_ictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0278_HUP138_phaseII_416023190000_416112890000_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_postictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0278_HUP138_phaseII_416112890000_416292890000_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))



sub_ID = "sub-RID0508"
atlas = "aal_res-1x1x1"
version = ""
ifpath_interictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0508_HUP184_phaseII_244435130000_244615130000_{1}_{2}SFC.pickle".format(sub_ID, atlas,version ))
ifpath_preictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0508_HUP184_phaseII_265855130000_266035130000_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_ictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0508_HUP184_phaseII_266035130000_267029570312_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_postictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0508_HUP184_phaseII_267029570312_267209570312_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))




sub_ID = "sub-RID0420"
atlas = "aal_res-1x1x1"
version = ""
ifpath_interictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0420_HUP186_phaseII_188682261173_188862261173_{1}_{2}SFC.pickle".format(sub_ID, atlas,version ))
ifpath_preictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0420_HUP186_phaseII_210102261173_210282261173_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_ictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0420_HUP186_phaseII_210282261173_210329885042_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_postictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0420_HUP186_phaseII_210329885042_210509885042_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))


 
sub_ID = "sub-RID0502"
atlas = "RandomAtlas0001000"
version = "v0001_"
atlas = "aal_res-1x1x1"
version = ""
ifpath_interictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0502_HUP182_phaseII_491691579507_491871579507_{1}_{2}SFC.pickle".format(sub_ID, atlas,version ))
ifpath_preictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0502_HUP182_phaseII_707511579507_707691579507_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_ictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0502_HUP182_phaseII_707691579507_707789139213_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))
ifpath_postictal = ospj(path, "data_processed/structure_function_correlation/{0}/{1}/sub-RID0502_HUP182_phaseII_707789139213_707969139213_{1}_{2}SFC.pickle".format(sub_ID, atlas,version))



ofname_SFC_aggregate = "sub-RID0420_HUP186_phaseII_seizure_21028226117_aal_res-1x1x1_SFC.pickle"
ofpath_SFC_aggregate = ospj(path, "data_processed/structure_function_correlation_processed/SFC_aggregated_by_seizure/sub-RID0420/aal_res-1x1x1", ofname_SFC_aggregate)


"""
#%%
#Aggregation
def aggregate_SFC(ifpath_interictal, ifpath_preictal, ifpath_ictal, ifpath_postictal, ofpath_SFC_aggregate):


    interictal = list([0, 0, 0, 0, 0])
    preictal = list([0, 0, 0, 0, 0])
    ictal = list([0, 0, 0, 0, 0])
    postictal = list([0, 0, 0, 0, 0])
    with open(ifpath_interictal, 'rb') as f: interictal[0], interictal[1], interictal[2], interictal[3], interictal[4], order_of_matrices_in_pickle_file = pickle.load(f)
    with open(ifpath_preictal, 'rb') as f: preictal[0],   preictal[1],   preictal[2],   preictal[3],   preictal[4],   order_of_matrices_in_pickle_file = pickle.load(f)
    with open(ifpath_ictal, 'rb') as f: ictal[0],      ictal[1],      ictal[2],      ictal[3],      ictal[4],      order_of_matrices_in_pickle_file = pickle.load(f)
    with open(ifpath_postictal, 'rb') as f: postictal[0],  postictal[1],  postictal[2],  postictal[3],  postictal[4],  order_of_matrices_in_pickle_file = pickle.load(f)
    
    oder_of_SFC_periods = ["interictal", "preictal", "ictal", "postictal"]
    SFC = [interictal, preictal, ictal, postictal]
    
    #replacing the last value (a zero) in the SFC vector with the second-to-last value because functional connectivity value is miss-calculated (= 0) due to 1 sec window roundind.
    for i in range(len(SFC)):
        for s in range(len(SFC[i])):
            SFC[i][s][-1] = SFC[i][s][-2] 
            
    
    with open(ofpath_SFC_aggregate, 'wb') as f: pickle.dump([SFC, order_of_matrices_in_pickle_file, oder_of_SFC_periods], f)



#%%
#Averaging

#ifname_SFC_aggregate_sub_ID_atlas =  ospj(path, "data_processed/structure_function_correlation_processed/SFC_aggregated_by_seizure/sub-RID0278/AAL600")
#ofpath_SFC_averages_by_atlas =  ospj(path, "data_processed/structure_function_correlation_processed/SFC_averages_by_atlas/averages/sub-RID0278/RandomAtlas0000010")
#ofpath_SFC_std_by_atlas =  ospj(path, "data_processed/structure_function_correlation_processed/SFC_averages_by_atlas/standard_deviations/sub-RID0278/RandomAtlas0000010")
def average_by_atlas(ifname_SFC_aggregate_sub_ID_atlas, ofpath_SFC_averages_by_atlas, ofpath_SFC_std_by_atlas):
    files = [f for f in sorted(os.listdir(ifname_SFC_aggregate_sub_ID_atlas))] 
    
    time = list()
    
    for f in range(len(files)):
        string = files[f]
        time.append( string[string.find("seizure_"):len(string)].split("_")[1]  )

    time_unique = np.unique(time)
    
    #tmp = []#unit testing
    #adding up file values
    for i in range(len(time_unique)):
        index = [t for t, sub in enumerate(files) if time_unique[i] in sub]
        numfiles = len(index)
        for sfc in range(len(index)):
            file = files[index[sfc]]
        
            file_path = ospj(ifname_SFC_aggregate_sub_ID_atlas, file)
            with open(file_path, 'rb') as f: SFC, order_of_matrices_in_pickle_file, oder_of_SFC_periods = pickle.load(f)
            if sfc == 0:
                SFC_avg = copy.deepcopy(SFC)
            else:
                for s in range(len(SFC)):
                    for freq in range(len(SFC[s])):
                        SFC_avg[s][freq] =  SFC_avg[s][freq] + SFC[s][freq]
            #tmp.append(SFC[0][0][0])#unit testing
            #print("{0} : {1}".format(   SFC_avg[0][0][0]    ,   SFC[0][0][0]     ) )#unit testing
            
            
        #taking mean
        for s in range(len(SFC)):
            for freq in range(len(SFC[s])):
                SFC_avg[s][freq] =  SFC_avg[s][freq]/numfiles
        
        #tmp = np.array(tmp) #unit testing
        #SFC_avg[0][0][0] == np.nanmean(tmp) #checking mean was calculated correctly for one case. They are about equal  
        #Standard Deviation
        
        #initialization of st dev list
        SFC_sd = copy.deepcopy(SFC_avg)
        for s in range(len(SFC)):
            for freq in range(len(SFC[s])):
                SFC_sd[s][freq] =  SFC_sd[s][freq]*0
            
        #this is the E(xi - mean)^2 part of standard deviation formula
        for sfc in range(len(index)):
            file = files[index[sfc]]
        
            file_path = ospj(ifname_SFC_aggregate_sub_ID_atlas, file)
            with open(file_path, 'rb') as f: SFC, order_of_matrices_in_pickle_file, oder_of_SFC_periods = pickle.load(f)
            for s in range(len(SFC)):
                for freq in range(len(SFC[s])):
                    SFC_sd[s][freq] =  SFC_sd[s][freq] + np.power((SFC[s][freq] -  SFC_avg[s][freq]  ),2)     #this is the E(xi - mean)^2 part of standard deviation formula
                    
        #this is the sqrt(divide by N) part of standard deviation formula
        for s in range(len(SFC)):
            for freq in range(len(SFC[s])):
                SFC_sd[s][freq] =  np.power(SFC_sd[s][freq]/numfiles, 0.5)            
                        
       # SFC_sd[0][0][0] == np.std(tmp)       #unit testing, checking standard deviation was calculated correctly for one case. They are about equal         
        
        if file.find("_v") <0:
            ofname = file[0:file.find("_SFC")]
        else:
            ofname = file[0:file.find("_v")]
        ofpath_average = ospj(ofpath_SFC_averages_by_atlas, "{0}_average_SFC.pickle".format(ofname) )
        ofpath_std = ospj(ofpath_SFC_std_by_atlas, "{0}_std_SFC.pickle".format(ofname) )
        with open(ofpath_average, 'wb') as f: pickle.dump([SFC_avg, order_of_matrices_in_pickle_file, oder_of_SFC_periods], f)
        with open(ofpath_std, 'wb') as f: pickle.dump([SFC_sd, order_of_matrices_in_pickle_file, oder_of_SFC_periods, numfiles], f)
    
   
     

#%%
#Interpolation

#ifpath_SFC_averages = ospj(path, "data_processed/structure_function_correlation_processed/SFC_averages_by_atlas/averages/sub-RID0278/RandomAtlas0000010/sub-RID0278_HUP138_phaseII_seizure_248525740000_RandomAtlas0000010_average_SFC.pickle")


def interpolation(ifpath_SFC_averages, ofname_SFC_interpolation):
    with open(ifpath_SFC_averages, 'rb') as f: SFC_avg, order_of_matrices_in_pickle_file, oder_of_SFC_periods = pickle.load(f)
    
    #initialization
    SFC_interpolation = [None]*len(SFC_avg)
    for s in range(len(SFC_avg)):
        SFC_interpolation[s] = [None]*len(SFC_avg[s])
    
    
    N=100 #interpolate to this many samples
    for s in range(len(SFC_avg)):
        for freq in range(len(SFC_avg[s])):
            data = SFC_avg[s][freq]
            func_interp = scipy.interpolate.interp1d(range(len(data)), data)
            xnew = np.linspace(0, len(data)-1, N, endpoint=False)
            SFC_interpolation[s][freq] =    func_interp(xnew)
            
    with open(ofname_SFC_interpolation, 'wb') as f: pickle.dump([SFC_interpolation, order_of_matrices_in_pickle_file, oder_of_SFC_periods], f)






#%%

"""
#Plotting for quick data mining purposes only

with open(file_path, 'rb') as f: SFC, order_of_matrices_in_pickle_file, oder_of_SFC_periods = pickle.load(f)

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


ylim = [-0.1, 0.5]
L = 180 #length of ii, pi, and po. generally between 1 and 180 seconds
sns.set(style="darkgrid")
sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 3})
#Extracting file names for plot title
ifname_ictal = os.path.split(ifpath_ictal)[1]
if ifname_ictal.find('phaseII_D') == -1: a = 3; b = "phaseII"
else: a = 4; b = ifname_ictal[ifname_ictal.find('phaseII_D'):(ifname_ictal.find('phaseII_D') + 11)]
seizure_start = ifname_ictal.split("_")[a]
seizure_end = ifname_ictal.split("_")[a+1]
iEEG_fname = "{0}_{1}".format(ifname_ictal.split("_")[1], b)
   


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=(4,11), sharex=True)
axes = [ax1, ax2, ax3, ax4, ax5]
fig.text(0.5, 0.1, 'time (s)', ha='center')
fig.text(0.01, 0.4, 'Spearkman Rank Correlation', ha='center', rotation = 'vertical')
fig.text(0.5, 0.89, "Subject: {0} {1}\ntime: {3}-{4}\nAtlas: {2}".format(sub_ID, iEEG_fname, atlas, seizure_start, seizure_end ), ha='center')
for s in range(len(SFC[0])):
    freq = np.array(order_of_matrices_in_pickle_file)[s,0]
    
    #extracting L seconds from each segment. ii = interictal, pi = preictal, ic = ictal, po = post ictal
    ii = SFC[0][s][range(L)]
    pi = np.flipud(np.flipud(SFC[1][s])[range(L)])#need to extract immediately before ictal
    ic  = SFC[2][s]
    po = SFC[3][s][range(L)]
    
    data =   [ii, pi, ic ,po ]
    x = range(len( np.concatenate(data )))
    y = np.concatenate(data)
    sns.scatterplot(x = x, y = y, ax = axes[s], hue = 0, legend = False,  linewidth=0, alpha = 0.6)
    axes[s].set_ylim(ylim)
    
    win = 20
    piicpo_roll = np.array(pd.DataFrame(np.concatenate([pi, ic, po])).rolling(win).mean()    ).flatten()
    ii_roll  = np.array(pd.DataFrame(np.concatenate([ii])).rolling(win).mean()    ).flatten()
    data = piicpo_roll
    x = range(len(ii_roll), len(ii_roll)+len( data ))
    y = data
    sns.lineplot(x = x, y = y, ax = axes[s], hue=1,  palette=['orange'], legend = False, sizes = [2])
    
    
    data = ii_roll
    x = range(len(ii_roll))
    y = data
    sns.lineplot(x = x, y = y, ax = axes[s], hue=1,  palette=['orange'], legend = False, sizes = [2])


    #writing interictal, pretictal, ictal, postictal labels
    if s == 0:
        mult_factor = 0.95
        axes[s].text(x = 0, y = ylim[1]*mult_factor, s = "inter", verticalalignment='top')
        axes[s].text(x = 1.1* len(ii), y = ylim[1]*mult_factor, s = "pre", verticalalignment='top')
        axes[s].text(x = 1.1* len(ii)+ len(pi), y = ylim[1]*mult_factor, s = "ictal", verticalalignment='top')
        axes[s].text(x = 1.1* len(ii) + len(pi) + len(ic), y = ylim[1]*mult_factor, s = "post", verticalalignment='top')
        
    vertical_line = [len(ii), len(ii)+ len(pi), len(ii) + len(pi) + len(ic)]
    for x in vertical_line:
        axes[s].axvline(x, color='k', linestyle='--')
      




"""
#%%









