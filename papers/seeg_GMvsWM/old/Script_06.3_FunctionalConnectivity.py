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



#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data","data_raw","iEEG_times","EEG_times.xlsx")
ifpath_electrode_localization = ospj( path, "data","data_processed","electrode_localization")

ifpath_FC = ospj(path, "data","data_processed","connectivity_matrices","function")
ifpath_SC = ospj(path, "data","data_processed","connectivity_matrices","structure")



ofpath_figure = ospj(path, "seeg_GMvsWM","figures","sfc")


if not (os.path.isdir(ofpath_figure)): os.makedirs(ofpath_figure, exist_ok=True)

#% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#% Processing Meta Data: extracting sub-IDs

sub_IDs_unique =  np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]
#%%
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

#%%

DATA = [None] * len(sub_IDs_unique)
for pt in range(len(sub_IDs_unique)):
    DATA[pt] = [None] 
    for atlas in range(len(DATA[pt])):
        DATA[pt][atlas] = [None] * 3
        DATA[pt][atlas][2] = [None] * 4

descriptors = ["interictal","preictal","ictal","postictal"]

count_patient = 0
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
    
    
    #Preprocessing files
    #find electrodes in both files (i.e there are electrodes in localization files but not in FC files, and there are electrodes in FC files but not localized)
    electrode_names_intersect, electrode_names_FC_ind, electrode_names_localization_ind =  np.intersect1d(electrode_names_FC, electrode_names_localization, return_indices = True)

    #Equalizing discrepancies in localization files and FC files
    #removing electrodes in localization files not in FC
    electrode_localization_intersect = electrode_localization.iloc[electrode_names_localization_ind].reset_index(drop=True)

    
    #removing electrodes in FC not in localization files
    FC_intersect = copy.deepcopy(FC)
    for f in range(len(FC)):
        FC_intersect[f] = FC_intersect[f][electrode_names_FC_ind[:,None], electrode_names_FC_ind[None,:], :] 
    electrode_names_FC_intersect = electrode_names_FC[electrode_names_FC_ind]

    
        
    #Only GM and WM tissues considered (not CSF=1, or outside brain = 0)
    labels = np.array(electrode_localization_intersect["region_number"])
    labels_gmwm_ind = np.where(labels >= 2)[0] #Only GM and WM tissues considered
    #removing electrode localization electrodes not in GM or WM
    electrode_localization_intersect_GMWM = electrode_localization_intersect.iloc[labels_gmwm_ind].reset_index(drop=True)
    #removing FC electrodes not in GM or WM
    FC_intersect_GMWM = copy.deepcopy(FC_intersect)
    for f in range(len(FC_intersect_GMWM)):
        FC_intersect_GMWM[f] = FC_intersect_GMWM[f][labels_gmwm_ind[:,None], labels_gmwm_ind[None,:], :] 
    electrode_names_FC_intersect_GMWM = np.array(electrode_localization_intersect_GMWM["electrode_name"])
    
    #averaging FC
    FC_intersect_GMWM_mean = [None] * len(FC_intersect_GMWM)
    for f in range(len(FC_intersect_GMWM_mean)):
        FC_intersect_GMWM_mean[f] = np.nanmean(FC_intersect_GMWM[f], axis=2)    
    np.allclose(FC_intersect_GMWM_mean[f], FC_intersect_GMWM_mean[f].T, rtol=1e-05, atol=1e-08)#check if symmetric
    



    #calculating distances matrices to see if GM-GM distances are different from WM-WM distances
    #takes a while to compute, so only need to do it once
    if descriptor == descriptors[0]:
        
        electrode_localization_intersect_GMWM
        distance_matrix = np.zeros(shape = (len(electrode_localization_intersect_GMWM),len(electrode_localization_intersect_GMWM) ))
        for e1 in range(len(electrode_localization_intersect_GMWM)):
            for e2 in range(len(electrode_localization_intersect_GMWM)):
                p1 = [electrode_localization_intersect_GMWM.iloc[e1]["x_coordinate"], electrode_localization_intersect_GMWM.iloc[e1]["y_coordinate"], electrode_localization_intersect_GMWM.iloc[e1]["z_coordinate"]]
                p2 = [electrode_localization_intersect_GMWM.iloc[e2]["x_coordinate"], electrode_localization_intersect_GMWM.iloc[e2]["y_coordinate"], electrode_localization_intersect_GMWM.iloc[e2]["z_coordinate"]]
                distance_matrix[e1,e2] = np.sqrt(  np.sum((np.array(p1)-np.array(p2))**2, axis=0)    )


    #Structure
    if (os.path.exists(ifpath_SC_sub_ID)): #If structure exists
        atlas= 0
        ifname_SC_sub_ID = ospj(ifpath_SC_sub_ID, "whole_brain_ROIs_cg.txt")#streamlines
        ifname_md_sub_ID = ospj(ifpath_SC_sub_ID, "whole_brain_ROIs_cg_md.txt")#streamlines
        ifname_fa_sub_ID = ospj(ifpath_SC_sub_ID, "whole_brain_ROIs_cg_fa.txt")#streamlines
        ifname_qa_sub_ID = ospj(ifpath_SC_sub_ID, "whole_brain_ROIs_cg_qa.txt")#streamlines
        ifname_ML_sub_ID = ospj(ifpath_SC_sub_ID, "whole_brain_ROIs_cg_ML.txt")#streamlines
        structure = pd.read_table(ifname_SC_sub_ID, header=None)
        #cleaning up structural data 
        structure = structure.drop([0,1], axis=1)
        structure = structure.drop([0], axis=0)
        structure = structure.iloc[:, :-1]
        structure_electrode_names = np.array([e[-4:] for e in  np.array(structure.iloc[0])])
        structure = np.array(structure.iloc[1:, :]).astype('float64')  #finally turn into numpy array
        
        #reorder structure
        #remove electrodes not in FC 

        #find electrodes in both files (i.e there are electrodes in localization files but not in FC files, and there are electrodes in FC files but not localized)
        electrode_names_intersect_SC, electrode_names_FC_ind_SC, electrode_names_localization_ind_SC =  np.intersect1d(electrode_names_FC_intersect_GMWM, structure_electrode_names, return_indices = True)
        #Equalizing discrepancies in SC files and FC files
        #removing electrodes in SC, but not in FC files
        SC = copy.deepcopy(structure)
        SC = SC[electrode_names_localization_ind_SC[:,None], electrode_names_localization_ind_SC[None,:]] 
        
        electrode_names_SC = structure_electrode_names[electrode_names_localization_ind_SC]
        
        print(electrode_names_SC == electrode_names_FC_intersect_GMWM)
        

 
    DATA[count_patient][atlas][0] = [electrode_localization_intersect_GMWM, distance_matrix ]
    DATA[count_patient][atlas][1] = [SC, distance_matrix   ] 
    DATA[count_patient][atlas][2][per] = [FC_intersect_GMWM, FC_intersect_GMWM_mean]

    if descriptor == descriptors[3]: count_patient = count_patient + 1




frequencies = np.hstack(np.asarray(order_of_matrices_in_pickle_file))

#%%
#regress out distances

freq=0
pt=1 
per=2
atlas = 0
count = 0
for freq in range(5):
    for pt in range(len(sub_IDs_unique)):
        if (pt == 0): all_FC = [None]* 4;  all_SC= [None]* 4; all_distance = [None]* 4
        for pp in range(4):
            if (pt == 0): all_FC[pp] = DATA[pt][atlas][2][pp][1][freq][np.triu_indices( len(DATA[pt][atlas][2][pp][1][freq]), k = 1) ] 
            if (pt == 0): all_SC[pp] = DATA[pt][atlas][1][0][np.triu_indices( len(DATA[pt][atlas][1][0]), k = 1) ] 
            if (pt == 0): all_distance[pp] = DATA[pt][atlas][1][1][np.triu_indices( len(DATA[pt][atlas][1][1]), k = 1) ]
            else:
                all_FC[pp] = np.concatenate([all_FC[pp],    DATA[pt][atlas][2][pp][1][freq][np.triu_indices( len(DATA[pt][atlas][2][pp][1][freq]), k = 1) ]] )
                all_SC[pp] = np.concatenate([all_SC[pp],DATA[pt][atlas][1][0][np.triu_indices( len(DATA[pt][atlas][1][0]), k = 1) ]]) #/ np.max(DATA[pt][atlas][pp][1][np.triu_indices( len(DATA[pt][atlas][pp][1]), k = 1) ] )
                all_distance[pp] = np.concatenate([all_distance[pp], DATA[pt][atlas][1][1][np.triu_indices( len(DATA[pt][atlas][1][1]), k = 1) ] ])
        
        
        
    all_FC = np.array([item for sublist in all_FC for item in sublist])
    all_SC = np.array([item for sublist in all_SC for item in sublist])
    all_distance = np.array([item for sublist in all_distance for item in sublist])
    
    """       
            
    X0_FC = all_distance.reshape(-1,1)
    y0_FC = all_FC
    gam_FC = LinearGAM( ).fit(X0_FC, y0_FC) ; 
    r0_FC = gam_FC.deviance_residuals(X0_FC, y0_FC)
    y1_FC = gam_FC.predict(X0_FC)
        
    X0_SC = all_distance.reshape(-1,1)
    y0_SC = all_SC
    gam_SC = LinearGAM().fit(X0_SC, y0_SC) ; 
    r0_SC = gam_SC.deviance_residuals(X0_SC, y0_SC)
    y1_SC = gam_SC.predict(X0_SC)
    y1_SC-y0_SC
    """
    
            
    X0_FC = all_distance.reshape(-1,1)
    y0_FC = copy.deepcopy(all_FC ) # np.mean(y0_FC)**4   np.var(y0_FC)  sns.histplot(y0_FC, kde = True)
    GLM_FC = TweedieRegressor(power=1, alpha=0.5, link='log')
    GLM_FC.fit(X0_FC,y0_FC)
    y1_FC = GLM_FC.predict(  X0_FC   )
    r0_FC = y0_FC - y1_FC   #   sns.histplot(y1_FC, kde = True);  import pylab; stats.probplot(r0_FC, dist="norm", plot=pylab )    ; 
    GLM_FC.score(X0_FC,y0_FC   )
    
    X0_SC = all_distance.reshape(-1,1)
    y0_SC = copy.deepcopy(all_SC)   # np.mean(y0_SC  )**2 -  np.var(y0_SC)  sns.histplot(y0_SC, kde = True);  y0_SC[np.where(y0_SC == 0)]=1; y0_SC = np.log10(y0_SC); y0_SC = y0_SC/np.max(y0_SC)
    GLM_SC = TweedieRegressor(power=1, alpha=0.5, link='log')
    GLM_SC.fit(X0_SC,y0_SC)
    y1_SC = GLM_SC.predict(  X0_SC   )
    r0_SC = y0_SC - y1_SC    #   sns.histplot(r0_SC, kde = True);  stats.probplot(r0_SC, dist="norm", plot=pylab )    ;
        
    
    
    #glm_binom  = sm.GLM(y0_FC, sm.add_constant ( X0_FC), family=sm.families.InverseGaussian())
    #res = glm_binom.fit()
    #sns.scatterplot(y=  res.mu, x= y0_FC, s = 0.1  )
    #fig,axes = plt.subplots(2,4,figsize=(8,4), dpi = 300)
    #sns.scatterplot(x=X0_FC.flatten() ,y=y0_FC ,ax=axes[0][0], s =0.1)
    #sns.scatterplot(x=X0_FC.flatten() ,y= res.mu ,ax=axes[0][0], s = 1)
    
    #glm_binom  = sm.GLM(y0_SC,  X0_SC, family=sm.families.Gamma())
    #res = glm_binom.fit()
    #sns.scatterplot(y=  res.mu, x= y0_SC, s = 0.1  )
    #fig,axes = plt.subplots(2,4,figsize=(8,4), dpi = 300)
    #sns.scatterplot(x=X0_FC.flatten() ,y=y0_SC ,ax=axes[0][0], s =0.1)
    #sns.scatterplot(x=X0_FC.flatten() ,y= res.mu ,ax=axes[0][0], s = 1)
    """
    size = 0.1/2
    fig,axes = plt.subplots(2,4,figsize=(8,4), dpi = 300)
    sns.scatterplot(x=X0_FC.flatten() ,y=y0_FC ,ax=axes[0][0], s =size)
    sns.scatterplot(x=X0_FC.flatten() ,y=y1_FC ,ax=axes[0][0], s = 1)
    sns.scatterplot(x=X0_SC.flatten() ,y=y0_SC ,ax=axes[0][1], s = size)
    sns.scatterplot(x=X0_SC.flatten() ,y=y1_SC ,ax=axes[0][1], s = 1)
    sns.scatterplot(x=X0_SC.flatten() ,y=r0_FC ,ax=axes[0][2], s =size)
    sns.scatterplot(x=X0_SC.flatten() ,y=r0_SC ,ax=axes[0][3], s = size)
    axes[0][2].axhline(0, ls='--',color="k")
    axes[0][3].axhline(0, ls='--',color="k")
    sns.regplot(x=y0_SC ,y=y0_FC ,ax=axes[1][0], scatter_kws={'s':size/10})  
    sns.regplot(x=r0_SC ,y=r0_FC ,ax=axes[1][1], scatter_kws={'s':size/10})  
    sns.regplot(x=y0_SC ,y=r0_FC ,ax=axes[1][2], scatter_kws={'s':size/10})  
    sns.regplot(x=r0_SC ,y=y0_FC ,ax=axes[1][3], scatter_kws={'s':size/10})  
    print(spearmanr(y0_SC, y0_FC)[0])
    print(spearmanr(y0_SC, r0_FC)[0])
    print(spearmanr(r0_SC, y0_FC)[0])
    print(spearmanr(r0_SC, r0_FC)[0])
    
    #model = LinearRegression().fit(  np.array(FC_vs_distance['distances']).reshape(-1,1),     np.array(FC_vs_distance['FC']).reshape(-1,1)  )   
    #preds = model.predict(   np.array(FC_vs_distance['distances']).reshape(-1,1)   )
    #res_function = np.array(FC_vs_distance['FC']).reshape(-1,1)    - preds
    
    print(   spearmanr(X0_FC.flatten(), r0_FC.flatten())[0]   )
    print(   spearmanr(X0_FC.flatten(), y0_FC.flatten())[0]   )
    print(spearmanr(X0_SC.flatten(), r0_SC.flatten())[0]    )
    print(spearmanr(X0_SC.flatten(), y0_SC.flatten())[0]    )
    
    """
#%%
count = 0
for freq in range(5):
    for pt in range(len(sub_IDs_unique)):
        for per in range(4):
            
            """
            for pat in range(len(sub_IDs_unique)):
                if (per == 0 and pat ==0): all_FC = [None]* 4;  all_SC= [None]* 4; all_distance = [None]* 4
                if (pat == 0): all_FC[per] = DATA[pat][per][3][freq][np.triu_indices( len(DATA[pat][per][3][freq]), k = 1) ] 
                if (pat == 0): all_SC[per] = DATA[pat][per][1][np.triu_indices( len(DATA[pat][per][1]), k = 1) ] 
                if (pat == 0): all_distance[per] = DATA[pat][per][10][np.triu_indices( len(DATA[pat][per][10]), k = 1) ]
                else:
                    all_FC[per] = np.concatenate([all_FC[per],    DATA[pat][per][3][freq][np.triu_indices( len(DATA[pat][per][3][freq]), k = 1) ]] )
                    all_SC[per] = np.concatenate([all_SC[per],DATA[pat][per][1][np.triu_indices( len(DATA[pat][per][1]), k = 1) ]]) #/ np.max(DATA[pat][per][1][np.triu_indices( len(DATA[pat][per][1]), k = 1) ] )
                    all_distance[per] = np.concatenate([all_distance[per], DATA[pat][per][10][np.triu_indices( len(DATA[pat][per][10]), k = 1) ] ])
            X0_FC = all_distance[per].reshape(-1,1)
            y0_FC = copy.deepcopy(all_FC[per] ) # np.mean(y0_FC)**4   np.var(y0_FC)  sns.histplot(y0_FC, kde = True)
            GLM_FC = TweedieRegressor(power=1, alpha=0.5, link='log')
            GLM_FC.fit(X0_FC,y0_FC)
            y1_FC = GLM_FC.predict(  X0_FC   )
            r0_FC = y0_FC - y1_FC   #   sns.histplot(y1_FC, kde = True);  import pylab; stats.probplot(r0_FC, dist="norm", plot=pylab )    ; 
            GLM_FC.score(X0_FC,y0_FC   )
            
            X0_SC = all_distance[per].reshape(-1,1)
            y0_SC = copy.deepcopy(all_SC[per])   # np.mean(y0_SC  )**2 -  np.var(y0_SC)  sns.histplot(y0_SC, kde = True);  y0_SC[np.where(y0_SC == 0)]=1; y0_SC = np.log10(y0_SC); y0_SC = y0_SC/np.max(y0_SC)
            GLM_SC = TweedieRegressor(power=1, alpha=0.5, link='log')
            GLM_SC.fit(X0_SC,y0_SC)
            y1_SC = GLM_SC.predict(  X0_SC   )
            r0_SC = y0_SC - y1_SC    # 
            """
            
            """
            all_FC = DATA[pt][per][3][freq][np.triu_indices( len(DATA[pt][per][3][freq]), k = 1) ] 
            all_SC = DATA[pt][per][1][np.triu_indices( len(DATA[pt][per][1]), k = 1) ] / np.max(DATA[pt][per][1][np.triu_indices( len(DATA[pt][per][1]), k = 1) ] )
            all_distance = DATA[pt][per][10][np.triu_indices( len(DATA[pt][per][10]), k = 1) ] 
            
            X0_FC = all_distance.reshape(-1,1)
            y0_FC = all_FC
            gam_FC = LinearGAM( ).fit(X0_FC, y0_FC) ; 
            r0_FC = gam_FC.deviance_residuals(X0_FC, y0_FC)  #  sns.histplot(r0_FC, kde = True);  stats.probplot(r0_FC, dist="norm", plot=pylab ) 
            y1_FC = gam_FC.predict(X0_FC)
            
            X0_SC = all_distance.reshape(-1,1)
            y0_SC = all_SC
            gam_SC = LinearGAM().fit(X0_SC, y0_SC) ; 
            r0_SC = gam_SC.deviance_residuals(X0_SC, y0_SC);  #  sns.histplot(r0_SC, kde = True);  stats.probplot(r0_SC, dist="norm", plot=pylab )    ;
            y1_SC = gam_SC.predict(X0_SC)
            y1_SC-y0_SC
            
            
            
            size = 0.1
            fig,axes = plt.subplots(2,4,figsize=(8,4), dpi = 300)
            sns.scatterplot(x=X0_FC.flatten() ,y=y0_FC ,ax=axes[0][0], s =size)
            sns.scatterplot(x=X0_FC.flatten() ,y=y1_FC ,ax=axes[0][0], s = 1)
            sns.scatterplot(x=X0_SC.flatten() ,y=y0_SC ,ax=axes[0][1], s = size)
            sns.scatterplot(x=X0_SC.flatten() ,y=y1_SC ,ax=axes[0][1], s = 1)
            sns.scatterplot(x=X0_SC.flatten() ,y=r0_FC ,ax=axes[0][2], s =size)
            sns.scatterplot(x=X0_SC.flatten() ,y=r0_SC ,ax=axes[0][3], s = size)
            axes[0][2].axhline(0, ls='--',color="k")
            axes[0][3].axhline(0, ls='--',color="k")
            sns.regplot(x=y0_SC ,y=y0_FC ,ax=axes[1][0], scatter_kws={'s':size/1})  
            sns.regplot(x=y0_SC ,y=r0_FC ,ax=axes[1][2], scatter_kws={'s':size/1})  
            sns.regplot(x=r0_SC ,y=y0_FC ,ax=axes[1][3], scatter_kws={'s':size/1})  
            sns.regplot(x=r0_FC ,y=r0_SC ,ax=axes[1][1], scatter_kws={'s':size/2})  
            #print(spearmanr(y0_SC, y0_FC))
            #print(spearmanr(y0_SC, r0_FC))
            #print(spearmanr(r0_SC, y0_FC))
            #print(spearmanr(r0_SC, r0_FC))
            fig.suptitle("{0}    {2}    {1}     {3}".format(sub_IDs_unique[pt], descriptors[per], frequencies[freq], np.round(spearmanr(r0_SC, r0_FC)[0], 2)    )) # or plt.suptitle('Main title')
            plt.show()
            
                
            fig,axes = plt.subplots(8,2,figsize=(4,28), dpi = 150)
            for pt in range(8):
            for per in np.array([ 1,2]):
            """
            
            
            
            for pp in range(4):
                if (pp == 0): all_FC = DATA[pt][atlas][2][pp][1][freq][np.triu_indices( len(DATA[pt][atlas][2][pp][1][freq]), k = 1) ] 
                if (pp == 0): all_SC = DATA[pt][atlas][1][0][np.triu_indices( len(DATA[pt][atlas][1][0]), k = 1) ] 
                if (pp == 0): all_distance = DATA[pt][atlas][1][1][np.triu_indices( len(DATA[pt][atlas][1][1]), k = 1) ]
                else:
                    all_FC = np.concatenate([all_FC,    DATA[pt][atlas][2][pp][1][freq][np.triu_indices( len(DATA[pt][atlas][2][pp][1][freq]), k = 1) ]] )
                    all_SC = np.concatenate([all_SC,DATA[pt][atlas][1][0][np.triu_indices( len(DATA[pt][atlas][1][0]), k = 1) ]]) #/ np.max(DATA[pt][atlas][pp][1][np.triu_indices( len(DATA[pt][atlas][pp][1]), k = 1) ] )
                    all_distance = np.concatenate([all_distance, DATA[pt][atlas][1][1][np.triu_indices( len(DATA[pt][atlas][1][1]), k = 1) ] ])
        
        
        
        
    

            
            X0_FC = all_distance.reshape(-1,1)
            y0_FC = copy.deepcopy(all_FC ) # np.mean(y0_FC)**4   np.var(y0_FC)  sns.histplot(y0_FC, kde = True)
            GLM_FC = TweedieRegressor(power=1, alpha=0.5, link='log')
            GLM_FC.fit(X0_FC,y0_FC)
            y1_FC = GLM_FC.predict(  X0_FC   )
            r0_FC = y0_FC - y1_FC   #   sns.histplot(y1_FC, kde = True);  import pylab; stats.probplot(r0_FC, dist="norm", plot=pylab )    ; 
            GLM_FC.score(X0_FC,y0_FC   )
            
            X0_SC = all_distance.reshape(-1,1)
            y0_SC = copy.deepcopy(all_SC)   # np.mean(y0_SC  )**2 -  np.var(y0_SC)  sns.histplot(y0_SC, kde = True);  y0_SC[np.where(y0_SC == 0)]=1; y0_SC = np.log10(y0_SC); y0_SC = y0_SC/np.max(y0_SC)
            GLM_SC = TweedieRegressor(power=1, alpha=0.5, link='log')
            GLM_SC.fit(X0_SC,y0_SC)
            y1_SC = GLM_SC.predict(  X0_SC   )
            r0_SC = y0_SC - y1_SC    #   sns.histplot(r0_SC, kde = True);  stats.probplot(r0_SC, dist="norm", plot=pylab )    ;
               
            
                    
            
            FC_residuals = np.zeros(shape = np.shape(DATA[pt][atlas][2][per][1][freq]))
            for r in range(len( (DATA[pt][atlas][2][per][1][freq]))):
                for c in range(len( (DATA[pt][atlas][2][per][1][freq]))):
                    FC_residuals[r,c] =     DATA[pt][atlas][2][per][1][freq][r,c]  - GLM_FC.predict(  DATA[pt][atlas][0][1][r,c].reshape(1, -1)   )[0]   
                    if r == c:
                        FC_residuals[r,c] = 0
                        
            SC_residuals = np.zeros(shape = np.shape( DATA[pt][atlas][1][0]))
            for r in range(len( (  DATA[pt][atlas][1][0]))):
                for c in range(len( ( DATA[pt][atlas][1][0]))):
                    SC_residuals[r,c] =    DATA[pt][atlas][1][0][r,c]   -  GLM_SC.predict(  DATA[pt][atlas][0][1][r,c].reshape(1, -1)   )[0]
                    if r == c:
                        SC_residuals[r,c] = 0
    
            #SC_residuals[np.where(SC_residuals == 0)] = 1
            #SC_residuals = np.log10(SC_residuals)
            #SC_residuals = SC_residuals/np.max(SC_residuals)
            
            elec = DATA[pt][atlas][0][0]
            WM_ind = np.where (  np.array( elec["distances_label_2"]) > 0 )[0]
            GM_ind = np.where (  np.array( elec["distances_label_2"]) <= 0 )[0]
            
            WM_FC = FC_residuals[WM_ind[:,None], WM_ind[None,:]]; 
            WM_FC =  WM_FC[np.triu_indices( len(WM_FC), k = 1) ] 
            GM_FC = FC_residuals[GM_ind[:,None], GM_ind[None,:]]; 
            GM_FC =  GM_FC[np.triu_indices( len(GM_FC), k = 1) ] 
            
            WM_SC = SC_residuals[WM_ind[:,None], WM_ind[None,:]]; 
            WM_SC =  WM_SC[np.triu_indices( len(WM_SC), k = 1) ] 
            GM_SC = SC_residuals[GM_ind[:,None], GM_ind[None,:]]; 
            GM_SC =  GM_SC[np.triu_indices( len(GM_SC), k = 1) ] 
            
            Full_SC = SC_residuals[np.triu_indices( len(SC_residuals), k = 1) ] 
            Full_FC = FC_residuals[np.triu_indices( len(FC_residuals), k = 1) ]
            
            """
            #analysis of WM FC with greatest WM-GM connections
            dist = DATA[pt][atlas][per][10]

            if per == 1: FC_residuals_pi = FC_residuals
            if per == 2: 
                FC_residuals_ic = FC_residuals
                
                delta = FC_residuals_ic - FC_residuals_pi
                delta_WM_FC =  delta[WM_ind[:,None], WM_ind[None,:]];
                delta_WM_FC = delta_WM_FC[np.triu_indices( len(delta_WM_FC), k = 1) ] 
                N = 0
                for N in range(   int(np.round(len(delta_WM_FC)*0.1,0))  ):
                    greatest_WM = np.sort(delta_WM_FC)[::-1]
                    node1_WM = np.where(delta == greatest_WM[N])[0][0]
                    node2_WM = np.where(delta == greatest_WM[N])[0][1]
                    
                    np.array( elec["electrode_name"])[node1_WM]
                    np.array( elec["electrode_name"])[node2_WM]
                    
                    greatest_GM_node1 = np.sort(delta[GM_ind[:,None], node1_WM].reshape(1,-1)[0]  )[::-1]
                    greatest_GM_node2 = np.sort(delta[GM_ind[:,None], node2_WM].reshape(1,-1)[0]  )[::-1]
                    
                    
                    node1_GM = np.where(delta == greatest_GM_node1[0])[0][0]
                    node2_GM = np.where(delta == greatest_GM_node2[0])[0][0]
                    
                    np.array( elec["electrode_name"])[node1_GM]
                    np.array( elec["electrode_name"])[node2_GM]
                    print("\n")
                    print( "{0}-{1}: {2}: {3}   {4}".format( np.array( elec["electrode_name"])[node1_WM], np.array( elec["electrode_name"])[node2_WM] , np.round(  SC_residuals[node1_WM, node2_WM],2  ), np.round( greatest_WM[N],2 ) , np.round( dist[node1_WM,node2_WM ],2 )       )               )
                    print( "{0}-{1}: {2}: {3}   {4}".format( np.array( elec["electrode_name"])[node1_WM], np.array( elec["electrode_name"])[node1_GM] ,   np.round(SC_residuals[node1_WM, node1_GM],2    ),np.round( greatest_GM_node1[0],2 ), np.round( dist[node1_WM,node1_GM ],2 )          ))
                    print( "{0}-{1}: {2}: {3}   {4}".format( np.array( elec["electrode_name"])[node2_WM], np.array( elec["electrode_name"])[node2_GM] , np.round ( SC_residuals[node2_WM, node2_GM],2),np.round( greatest_GM_node2[0],2 ) , np.round( dist[node2_WM,node2_GM ],2 )               ))
                    
                    print( "{0}-{1}: {2}: {3}   {4}".format( np.array( elec["electrode_name"])[node1_WM], np.array( elec["electrode_name"])[node2_GM] ,   np.round(SC_residuals[node1_WM, node2_GM],2    ),np.round( greatest_GM_node1[0],2 ), np.round( dist[node1_WM,node2_GM ],2 )          ))
                    print( "{0}-{1}: {2}: {3}   {4}".format( np.array( elec["electrode_name"])[node2_WM], np.array( elec["electrode_name"])[node2_GM] , np.round ( SC_residuals[node2_WM, node2_GM],2),np.round( greatest_GM_node2[0],2 ) , np.round( dist[node2_WM,node2_GM ],2 )               ))
                    
                    if N == 0: 
                        wmwm = np.array([ SC_residuals[node1_WM, node2_WM],  greatest_WM[N] ])
                        gmwm = np.array([ SC_residuals[node1_WM, node1_GM],  greatest_GM_node1[0] ])
                        gmwm = np.vstack(   (gmwm,  np.array([ SC_residuals[node2_WM, node1_GM],  greatest_GM_node1[0] ]) )   )
                    if N > 0: 
                        wmwm = np.vstack(   (wmwm,  np.array([ SC_residuals[node1_WM, node2_WM],  greatest_WM[N] ]) )   )
                        gmwm = np.vstack(   (gmwm,  np.array([ SC_residuals[node1_WM, node1_GM],  greatest_GM_node1[0] ]) )   )
                        gmwm = np.vstack(   (gmwm,  np.array([ SC_residuals[node2_WM, node1_GM],  greatest_GM_node1[0] ]) )   )
                        
                        
                
                sns.regplot( x = gmwm[:,0], y =  gmwm[:,1] , ax = axes[pt][0],scatter_kws={'s':1, "color": "#000000"}   )
                sns.regplot( x = wmwm[:,0], y =  wmwm[:,1]  , ax = axes[pt][1], scatter_kws={'s':1, "color": "#000000"}  )
                #plt.suptitle("{0}/{1}".format(sub_IDs_unique[pt], frequencies[freq]  ), y=1.1)
                axes[pt][0].title.set_text("{0}".format( np.round(spearmanr(gmwm[:,0],  gmwm[:,1]),2)     ))
                axes[pt][1].title.set_text("{0}".format( np.round(spearmanr(wmwm[:,0],  wmwm[:,1]),2)   ))
                #axes[pt][0].set_ylim(0, 0.7); axes[pt][1].set_ylim(0, 0.7)
                #axes[pt][0].set_xlim(-60000, 50000); axes[pt][1].set_xlim(-60000, 50000)
         
            dFC = delta[np.triu_indices( len(delta), k = 1) ] 
            sns.regplot( x =Full_SC, y = dFC ,scatter_kws={'s':1, "color": "#000000"}   )
            spearmanr(Full_SC, dFC)
            """
            """
            colors_p = ["#c6b4a5", "#b6d4ee"]
            fig,axes = plt.subplots(1,1,figsize=(8,8), dpi = 300)
            colormap_net = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#cccccc","#111111"]) 
            sns.heatmap(SC_residuals, square=True, ax = axes, annot = False,cbar = False, xticklabels=False, yticklabels=False, cmap = colormap_net )
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_SC_full.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            
            fig,axes = plt.subplots(1,1,figsize=(8,8), dpi = 300)
            colormap_net = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ddd2c9","#5a4839"]) 
            sns.heatmap(GM_SC, square=True, ax = axes, annot = False,cbar = False, xticklabels=False, yticklabels=False, cmap = colormap_net )
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_SC_GM.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            
            fig,axes = plt.subplots(1,1,figsize=(8,8), dpi = 300)
            colormap_net = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b6d4ee","#1f5686"]) 
            sns.heatmap(WM_SC, square=True, ax = axes, annot = False,cbar = False, xticklabels=False, yticklabels=False, cmap = colormap_net )
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_SC_WM.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            
            
            fig,axes = plt.subplots(1,1,figsize=(8,8), dpi = 300)
            colormap_net = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff","#000000"]) 
            sns.heatmap(FC_residuals, square=True, ax = axes, annot = False,cbar = False, xticklabels=False, yticklabels=False, cmap = colormap_net, center = 0.5 )
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_FC_full.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            
            fig,axes = plt.subplots(1,1,figsize=(8,8), dpi = 300)
            colormap_net = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ddd2c9","#5a4839"]) 
            sns.heatmap(GM_FC, square=True, ax = axes, annot = False,cbar = False, xticklabels=False, yticklabels=False, cmap = colormap_net, center = 0.3 )
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_FC_GM.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            
            fig,axes = plt.subplots(1,1,figsize=(8,8), dpi = 300)
            colormap_net = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b6d4ee","#1f5686"]) 
            sns.heatmap(WM_FC, square=True, ax = axes, annot = False,cbar = False, xticklabels=False, yticklabels=False, cmap = colormap_net )
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_FC_WM.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            """
            
            
            """
            size = 10
            fig,axes = plt.subplots(1,3,figsize=(10,2), dpi = 300)
            sns.regplot(x= Full_SC  , y=Full_FC  ,ax=axes[0], scatter_kws={'s':size/10, "color": "#999999"}, line_kws={"color": "#555555"}) 
            sns.regplot(x=GM_SC ,y=GM_FC ,ax=axes[1], scatter_kws={'s':size/10, "color": "#b7a08d"}, line_kws={"color": "#96785f"}) 
            sns.regplot(x=WM_SC  ,y=  WM_FC,ax=axes[2], scatter_kws={'s':size/10, "color": "#86b8e3"}, line_kws={"color": "#236196"}) 
            axes[0].title.set_text(   "{0}".format( np.round(spearmanr(Full_SC, Full_FC)[0],2)    ))
            axes[1].title.set_text(   "{0}".format(  np.round(spearmanr(GM_SC,    GM_FC)[0],2)  )   )
            axes[2].title.set_text(   "{0}".format(  np.round(spearmanr(WM_SC,    WM_FC)[0],2)  )   )
            plt.suptitle("{0}/{1}/{2}".format(sub_IDs_unique[pt], descriptors[per], frequencies[freq]  ), y=1.1)
            plt.savefig(ospj(ofpath_figure, "{0}_{1}_{2}_SFC.pdf".format(sub_IDs_unique[pt],  descriptors[per], frequencies[freq])))  
            plt.show()
            """
            
            df_fc = pd.concat( [pd.DataFrame(   {"Tissue": np.repeat("GM", len(GM_FC)), "FC": GM_FC}    ),  
                             pd.DataFrame(   {"Tissue": np.repeat("WM", len(WM_FC)), "FC":WM_FC}    ) 
                             ], axis=0)
            df_sc = pd.concat( [pd.DataFrame(   {"Tissue": np.repeat("GM", len(GM_SC)), "SC": GM_SC}    ),  
                             pd.DataFrame(   {"Tissue": np.repeat("WM", len(WM_SC)), "SC":WM_SC}    ) 
                             ], axis=0)
            
            df_cor = pd.concat( [  pd.DataFrame(   {"Patient": np.repeat("{0}".format(sub_IDs_unique[pt]), 3)}    )   ,
                                    pd.DataFrame(   {"Frequency": np.repeat("{0}".format(frequencies[freq]), 3)}    )  ,
                                    pd.DataFrame(   {"Period": np.repeat("{0}".format(descriptors[per]), 3)}    )      ,
                                    pd.DataFrame(   {"Network": np.array(["Full", "GM", "WM"]) }    )  ,
                                    pd.DataFrame(   { "SFC": np.array([spearmanr(Full_SC, Full_FC)[0], spearmanr(GM_SC, GM_FC)[0], spearmanr(WM_SC, WM_FC)[0]])           }    )
                                ], axis=1)
            

            
            #sns.histplot( data = df_sc, x="SC" , hue="Tissue", kde=True  )
            #sns.histplot( data = df_fc, x="FC" , hue="Tissue" , kde=True  )
            
            
            df_sfc =  pd.concat( [  pd.DataFrame(   {"Patient": np.repeat("{0}".format(sub_IDs_unique[pt]), len(df_fc))}    )   ,
                                    pd.DataFrame(   {"Frequency": np.repeat("{0}".format(frequencies[freq]), len(df_fc))}    )  ,
                                    pd.DataFrame(   {"Period": np.repeat("{0}".format(descriptors[per]), len(df_fc))}    )      ,
                                    df_fc.reset_index(drop=True),  
                                    pd.DataFrame( df_sc["SC"].reset_index(drop=True)) 
                                ], axis=1)
            if (pt == 0 and per ==0 and freq == 0):
                df_sfc_full = df_sfc
                df_cor_full = df_cor
            else:
                df_sfc_full =  pd.concat( [df_sfc_full, df_sfc], axis=0)
                df_cor_full =  pd.concat( [df_cor_full, df_cor], axis=0)
            printProgressBar(count+1, len(sub_IDs_unique)*4*5, suffix = "{0}/{1}".format(count+1, len(sub_IDs_unique)*4*5), decimals = 1, length = 25)
            count = count +1
            #print(df_cor_full)
#sns.lmplot( data = df_sfc, x="SC" , y = "FC", hue="Tissue",  scatter_kws={'s':3} )
#%%
colormap_subIDs = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#d67777","#d6a677","#d6d677","#a7d677","#77a7d6","#7778d6", "#d677a7"]) #rainbow ROYGBIV
sub_IDs_unique_N = len(sub_IDs_unique)
#Colorbar - Make one unique color for each electrode stick base on the rainbow above
colors_subIDs = []
for c in np.linspace(start=0, stop=255, num=sub_IDs_unique_N).astype(int):
    colors_subIDs.append( rgb2hex(int(colormap_subIDs(c)[0]*255), int(colormap_subIDs(c)[1]*255), int(colormap_subIDs(c)[2]*255))  )


#%%

for pt in range(8):
    freq = 0
    
    
    gm =  df_sfc_full.loc[df_sfc_full['Tissue'] == "GM"]
    wm = df_sfc_full.loc[df_sfc_full['Tissue'] == "WM"]
    
    pt_df =  df_sfc_full.loc[df_sfc_full['Patient'] == sub_IDs_unique[pt]]
    pt_gm =  gm.loc[gm['Patient'] == sub_IDs_unique[pt]]
    pt_wm =  wm.loc[wm['Patient'] == sub_IDs_unique[pt]]
    
    pt_df_freq =  pt_df.loc[pt_df['Frequency'] == frequencies[freq]]
    pt_gm_freq =  pt_gm.loc[pt_gm['Frequency'] == frequencies[freq]]
    pt_wm_freq =  pt_wm.loc[pt_wm['Frequency'] == frequencies[freq]]
    
    fig,axes = plt.subplots(4,3,figsize=(12,20), dpi = 300)
    for per in range(4):
        pt_df_freq_per =  pt_df_freq.loc[pt_df_freq['Period'] == descriptors[per]]
        pt_gm_freq_per =  pt_gm_freq.loc[pt_gm_freq['Period'] == descriptors[per]]
        pt_wm_freq_per =  pt_wm_freq.loc[pt_wm_freq['Period'] == descriptors[per]]
        
        sns.regplot( data = pt_df_freq_per, x="SC" , y = "FC", scatter=False, ax = axes[per][0],  line_kws = {"color": colors_subIDs[pt], "alpha": 1.0}  )
        sns.scatterplot( data = pt_df_freq_per, x="SC" , y = "FC",s= 4, color=colors_subIDs[pt], ax = axes[per][0] , legend = False)
        axes[per][0].title.set_text(   "{0}/{2}/{3}  -  {1}".format(sub_IDs_unique[pt], np.round(spearmanr(pt_df_freq_per["SC"], pt_df_freq_per["FC"])[0],2), descriptors[per], frequencies[freq]  )   )
        
        sns.regplot( data = pt_gm_freq_per, x="SC" , y = "FC", ax = axes[per][1],  line_kws = {"color": colors_subIDs[pt], "alpha": 1.0}, scatter=False  )
        sns.scatterplot( data = pt_gm_freq_per, x="SC" , y = "FC",s= 4, color=colors_subIDs[pt], ax = axes[per][1] , legend = False)
        axes[per][1].title.set_text(   "{0}".format(  np.round(spearmanr(pt_gm_freq_per["SC"],     pt_gm_freq_per["FC"])[0],2)  )   )
        
        sns.regplot( data = pt_wm_freq_per, x="SC" , y = "FC", ax = axes[per][2],  line_kws = {"color": colors_subIDs[pt], "alpha": 1.0}, scatter=False )
        sns.scatterplot( data = pt_wm_freq_per, x="SC" , y = "FC",s= 4, color=colors_subIDs[pt] , ax = axes[per][2] , legend = False)
        axes[per][2].title.set_text(   "{0}".format(np.round(spearmanr(pt_wm_freq_per["SC"],     pt_wm_freq_per["FC"])[0],2)  )   )
    
    
    


print(  np.round(spearmanr(df_sfc_full["SC"], df_sfc_full["FC"])[0],2)   )
print(  np.round(spearmanr(pt_gm_freq_per["FC"],     pt_gm_freq_per["SC"])[0],2))
print(  np.round(spearmanr(pt_wm_freq_per["SC"],    pt_wm_freq_per["FC"])[0],2) )   

          
print(  np.round(pearsonr(df_sfc_full["SC"], df_sfc_full["FC"])[0],2)   )
print(  np.round(pearsonr(pt_gm_freq_per["FC"],     pt_gm_freq_per["SC"])[0],2))
print(  np.round(pearsonr(pt_wm_freq_per["SC"],    pt_wm_freq_per["FC"])[0],2) )             





sns.scatterplot( data = pt_df_freq_per, x="SC" , y = "FC",s= 4, color=colors_subIDs[pt])
spearmanr(pt_df_freq_per["SC"], pt_df_freq_per["FC"])[0]

sns.scatterplot( data = pt_gm_freq_per, x="SC" , y = "FC",s= 4, color=colors_subIDs[pt])
spearmanr(pt_gm_freq_per["SC"], pt_gm_freq_per["FC"])[0]



fig,axes = plt.subplots(1,1,figsize=(4,4), dpi = 300)
g1 = sns.jointplot( data = gm, x="SC" , y = "FC", kind='reg')
g2 = sns.jointplot( data = wm, x="SC" , y = "FC", kind='reg')



df_sfc.loc[df_sfc['Tissue'] == "GM"]

print(spearmanr(X0_FC, r0_FC))




colors_p = ["#c6b4a5", "#b6d4ee"]
sns.set_palette(sns.color_palette(colors_p))





#%%




df_cor_full


fig, axes = plt.subplots(1,5,figsize=(20,4), dpi = 300)
for freq in range(0,5):
    
    df_cor_freq = df_cor_full.loc[df_cor_full['Frequency'] == frequencies[freq]]

    ax = sns.boxplot(x = "Period", y = "SFC", hue = "Network", data = df_cor_freq, ax = axes[freq], showfliers=False, palette = ["#c6b4a5", "#666666", "#b6d4ee"], hue_order =["GM", "Full", "WM"] )
    ax = sns.swarmplot(x = "Period", y = "SFC", hue = "Network", data=df_cor_freq, color=".25", dodge=True, ax =  axes[freq], size = 3, hue_order =["GM", "Full", "WM"])
    handles, labels = axes[freq].get_legend_handles_labels()
    
    axes[freq].legend(handles[0:3],labels[0:3], frameon=False)
    plt.legend(handles[0:3],labels[0:3], frameon=False)
    if freq <4: axes[freq].legend([],[])





    sfc_ictal = df_cor_freq.loc[df_cor_freq['Period'] == "ictal"]
    sfc_preictal = df_cor_freq.loc[df_cor_freq['Period'] == "preictal"]
    
    sfc_ictal_gm = sfc_ictal.loc[sfc_ictal['Network'] == "GM"]
    sfc_preictal_gm = sfc_preictal.loc[sfc_preictal['Network'] == "GM"]
    sfc_ictal_wm = sfc_ictal.loc[sfc_ictal['Network'] == "WM"]
    sfc_preictal_wm = sfc_preictal.loc[sfc_preictal['Network'] == "WM"]


    
    mannwhitneyu(sfc_ictal_gm["SFC"], sfc_preictal_gm["SFC"]   )[1]
    mannwhitneyu(sfc_ictal_wm["SFC"], sfc_preictal_wm["SFC"]   )[1]


    stats.ttest_rel(sfc_ictal_gm["SFC"], sfc_preictal_gm["SFC"])[1] 
    stats.ttest_rel(sfc_ictal_wm["SFC"], sfc_preictal_wm["SFC"])[1] 
    p_val = stats.wilcoxon(sfc_ictal_gm["SFC"], sfc_preictal_gm["SFC"])[1] 
    print(stats.wilcoxon(sfc_ictal_wm["SFC"], sfc_preictal_wm["SFC"])[1]) 
    
    
    axes[freq].title.set_text("{0}\n{1}".format(  frequencies[freq], np.round(p_val,2))   )

delta = sfc_ictal_gm["SFC"] - sfc_preictal_gm["SFC"]  


#sns.boxplot(x =delta , orient="x" )

stats.ttest_1samp(delta, 0)[1]
stats.wilcoxon(delta)[1]
    
    
    
    
    
    
    
#%%Plot for paper


freq = 0
df_cor_freq = df_cor_full.loc[df_cor_full['Frequency'] == frequencies[freq]]
    
    
    
fig, axes = plt.subplots(1,1,figsize=(10,10), dpi = 300)    
ax = sns.boxplot(x = "Period", y = "SFC", hue = "Network", data = df_cor_freq, ax = axes, showfliers=False, palette = ["#c6b4a5", "#666666", "#b6d4ee"], hue_order =["GM", "Full", "WM"] )
ax = sns.swarmplot(x = "Period", y = "SFC", hue = "Network", data=df_cor_freq, color=".25", dodge=True, ax =  axes, size = 6, hue_order =["GM", "Full", "WM"])
handles, labels = axes.get_legend_handles_labels()

axes.legend(handles[0:3],labels[0:3], frameon=False)
plt.legend(handles[0:3],labels[0:3], frameon=False)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
axes.set_ylim([0.0, 0.7])


sfc_ictal = df_cor_freq.loc[df_cor_freq['Period'] == "ictal"]
sfc_preictal = df_cor_freq.loc[df_cor_freq['Period'] == "preictal"]

sfc_ictal_gm = sfc_ictal.loc[sfc_ictal['Network'] == "GM"]
sfc_preictal_gm = sfc_preictal.loc[sfc_preictal['Network'] == "GM"]
sfc_ictal_wm = sfc_ictal.loc[sfc_ictal['Network'] == "WM"]
sfc_preictal_wm = sfc_preictal.loc[sfc_preictal['Network'] == "WM"]



stats.ttest_rel(sfc_ictal_gm["SFC"], sfc_preictal_gm["SFC"])[1] 
stats.ttest_rel(sfc_ictal_wm["SFC"], sfc_preictal_wm["SFC"])[1] 
p_val_gm = stats.wilcoxon(sfc_ictal_gm["SFC"], sfc_preictal_gm["SFC"])[1] 
stats.wilcoxon(sfc_ictal_wm["SFC"], sfc_preictal_wm["SFC"])[1] 


axes.title.set_text("{0}\n{1}".format(  frequencies[freq], np.round(p_val,2))   )

    
    
    
plt.savefig(ospj(ofpath_figure, "{0}_streamlines_Ysc_Nfc.pdf".format(frequencies[freq]  )))    
    
    
    
    
    
#%%
per=0
df_sc_plot = []

for pt in range(len(sub_IDs_unique)):
    SC_residuals = np.zeros(shape = np.shape( DATA[pt][per][1]))
    for r in range(len( ( DATA[pt][per][1]))):
        for c in range(len( ( DATA[pt][per][1]))):
            SC_residuals[r,c] =   DATA[pt][per][1][r,c]  #- GLM_SC.predict(  DATA[pt][per][10][r,c].reshape(1, -1)   )[0] 
            if r == c:
                SC_residuals[r,c] = 0

    #SC_residuals[np.where(SC_residuals == 0)] = 1
    #SC_residuals = np.log10(SC_residuals)
    #SC_residuals = SC_residuals/np.max(SC_residuals)
    
    elec = DATA[pt][per][14]
    WM_ind = np.where (  np.array( elec["distances_label_2"]) > 0 )[0]
    GM_ind = np.where (  np.array( elec["distances_label_2"]) <= 0 )[0]
    
 
    WM_SC = SC_residuals[WM_ind[:,None], WM_ind[None,:]]; WM_SC =  WM_SC[np.triu_indices( len(WM_SC), k = 1) ] 
    GM_SC = SC_residuals[GM_ind[:,None], GM_ind[None,:]]; GM_SC =  GM_SC[np.triu_indices( len(GM_SC), k = 1) ] 
    
    Full_SC = SC_residuals[np.triu_indices( len(SC_residuals), k = 1) ] 

    
    df_sc = pd.concat([   
        pd.DataFrame(   {"Tissue":  np.repeat("GM", len(GM_SC)) , "SC" :GM_SC   }           ),
        pd.DataFrame(   {"Tissue":  np.repeat("WM", len(WM_SC)) ,  "SC": WM_SC  }           ),
        pd.DataFrame(   {"Tissue":  np.repeat("Full", len(Full_SC)) ,  "SC": Full_SC  }           )
        
        
        ])
    
    df_sc = pd.concat([   
        pd.DataFrame(  { "Patient" : np.repeat("{0}".format(sub_IDs_unique[pt]), len(df_sc))       }   )  ,
        df_sc.reset_index(drop=True)
        
        ], axis = 1)
    
    
    if (pt == 0): df_sc_plot = df_sc

    else:
        df_sc_plot = pd.concat([  df_sc_plot, df_sc] )



    
    



log10 = np.array(df_sc_plot["SC"])
log10[np.where( log10== 0   )     ]   =1
log10 = np.log10(log10)

df_sc_plot_log10 = pd.concat([   df_sc_plot.reset_index(drop=True),
    pd.DataFrame(  { "log10" : log10      }   )  
    ], axis = 1)
    
df_sc_plot_log10_full = df_sc_plot_log10.loc[df_sc_plot_log10['Tissue'] =="Full"]
df_sc_plot_log10_GM_WM = df_sc_plot_log10.loc[(df_sc_plot_log10['Tissue'] =="GM")   | (df_sc_plot_log10['Tissue'] =="WM") ]
    
    
    
    

plot, axes = plt.subplots(figsize=(5, 5), dpi = 600)
sns.histplot( data =df_sc_plot_log10_full, x="log10" , ax = axes, kde = True, color = "#333333")
plt.savefig(ospj(ofpath_figure, "SC_full.pdf"))    
    
    
   
plot, axes = plt.subplots(figsize=(5, 5), dpi = 600)
sns.ecdfplot( data =df_sc_plot_log10_GM_WM, x="log10" , hue="Tissue" , ax = axes, palette=colors_p  )
plt.savefig(ospj(ofpath_figure, "SC_GMWM_ecdf.pdf"))    
  
plot, axes = plt.subplots(figsize=(5, 5), dpi = 600)
sns.histplot( data =df_sc_plot_log10_GM_WM, x="log10" , hue="Tissue" , ax = axes, kde = True, palette=colors_p)
plt.savefig(ospj(ofpath_figure, "SC_GMWM_hist.pdf"))  
  
    
    
    
    
#%%
#regress out distances

  

def regress_distance(structure, function, distance_matrix):
    
    for per in range(4):
            if (per == 0): all_FC = function[per][np.triu_indices( len(function[per]), k = 1) ] 
            else:
                all_FC = np.concatenate([all_FC,    function[per][np.triu_indices( len(function[per]), k = 1) ]] )

    all_distance = np.tile(distance_matrix[np.triu_indices( len(distance_matrix), k = 1) ]  , 4)
    all_SC = np.tile(structure[np.triu_indices( len(structure), k = 1) ],4)
        
    
    
    X0_FC = all_distance.reshape(-1,1)
    y0_FC = copy.deepcopy(all_FC ) # np.mean(y0_FC)**4   np.var(y0_FC)  sns.histplot(y0_FC, kde = True)
    GLM_FC = TweedieRegressor(power=1, alpha=0.5, link='log')
    GLM_FC.fit(X0_FC,y0_FC)
    y1_FC = GLM_FC.predict(  X0_FC   )
    r0_FC = y0_FC - y1_FC   #   sns.histplot(y1_FC, kde = True);  import pylab; stats.probplot(r0_FC, dist="norm", plot=pylab )    ; 
    GLM_FC.score(X0_FC,y0_FC   )
    
    X0_SC = all_distance.reshape(-1,1)
    y0_SC = copy.deepcopy(all_SC)   # np.mean(y0_SC  )**2 -  np.var(y0_SC)  sns.histplot(y0_SC, kde = True);  y0_SC[np.where(y0_SC == 0)]=1; y0_SC = np.log10(y0_SC); y0_SC = y0_SC/np.max(y0_SC)
    GLM_SC = TweedieRegressor(power=1, alpha=0.5, link='log')
    GLM_SC.fit(X0_SC,y0_SC)
    y1_SC = GLM_SC.predict(  X0_SC   )
    r0_SC = y0_SC - y1_SC    #   sns.histplot(r0_SC, kde = True);  stats.probplot(r0_SC, dist="norm", plot=pylab )    ;
        
    size = 0.2
    fig,axes = plt.subplots(2,4,figsize=(8,4), dpi = 300)
    sns.scatterplot(x=X0_FC.flatten() ,y=y0_FC ,ax=axes[0][0], s =size)
    sns.scatterplot(x=X0_FC.flatten() ,y=y1_FC ,ax=axes[0][0], s = 1)
    sns.scatterplot(x=X0_SC.flatten() ,y=y0_SC ,ax=axes[0][1], s = size)
    sns.scatterplot(x=X0_SC.flatten() ,y=y1_SC ,ax=axes[0][1], s = 1)
    sns.scatterplot(x=X0_SC.flatten() ,y=r0_FC ,ax=axes[0][2], s =size)
    sns.scatterplot(x=X0_SC.flatten() ,y=r0_SC ,ax=axes[0][3], s = size)
    axes[0][2].axhline(0, ls='--',color="k")
    axes[0][3].axhline(0, ls='--',color="k")
    sns.regplot(x=y0_SC ,y=y0_FC ,ax=axes[1][0], scatter_kws={'s':size/10})  
    sns.regplot(x=r0_SC ,y=y0_FC ,ax=axes[1][1], scatter_kws={'s':size/10})  
    sns.regplot(x=y0_SC ,y=r0_FC ,ax=axes[1][2], scatter_kws={'s':size/10})  
    sns.regplot(x=r0_SC ,y=r0_FC ,ax=axes[1][3], scatter_kws={'s':size/10})  
    xlim_dist = [0, 150]
    ylim_fc = [0.15,1]
    ylim_sc = [-1e4, 2.1e5]
    axes[0][0].set_xlim(xlim_dist); axes[0][1].set_xlim(xlim_dist); axes[0][2].set_xlim(xlim_dist); axes[0][3].set_xlim(xlim_dist)
    axes[1][0].set_xlim(ylim_sc); axes[1][1].set_xlim([np.min(r0_SC),np.max(r0_SC)]); axes[1][2].set_xlim([0,np.max(all_SC)]); axes[1][3].set_xlim([np.min(r0_SC),np.max(r0_SC)])
    axes[0][0].set_ylim(ylim_fc); axes[0][1].set_ylim(ylim_sc); axes[0][2].set_ylim([np.min(r0_FC),np.max(r0_FC)]); axes[0][3].set_ylim([np.min(r0_SC),np.max(r0_SC)])
    axes[1][0].set_ylim(ylim_fc); axes[1][1].set_ylim(ylim_fc); axes[1][2].set_ylim([np.min(r0_FC),np.max(r0_FC)]); axes[1][3].set_ylim([np.min(r0_FC),np.max(r0_FC)])


freq = 0

pt = 7
for pt in range(8):
    structure =  DATA[pt][atlas][1][0]
    per = 0
    function_ii = (DATA[pt][atlas][2][per][1][freq])
    per = 1
    function_pi = (DATA[pt][atlas][2][per][1][freq])
    per = 2
    function_ic = (DATA[pt][atlas][2][per][1][freq])
    per = 3
    function_po = (DATA[pt][atlas][2][per][1][freq])
    
    function = [function_ii,function_pi , function_ic, function_po]
    
    distance_matrix = DATA[pt][atlas][0][1]
    regress_distance(structure, function, distance_matrix)
        

    
#%%
#Plot time series SFC



def plot_time_series_SFC(structure, function ):

    function_residuals_per = [None] * 4
    for per in range(4):    
        FC_residuals = np.zeros(shape = np.shape(function[per]))
        for r in range(len(function[per])):
            print(r)
            for c in range(len( function[per])):
                for t in range(function[per].shape[2]):
                    FC_residuals[r,c,t] =     function[per][r,c,t] # - GLM_FC.predict(  DATA[pt][atlas][0][1][r,c].reshape(1, -1)   )[0]   
                    if r == c:
                        FC_residuals[r,c,t] = 0
        function_residuals_per[per] = FC_residuals
                    
    SC_residuals = np.zeros(shape = np.shape( structure))
    for r in range(len( (  structure))):
        for c in range(len( ( structure))):
            SC_residuals[r,c] =    structure[r,c]   #-  GLM_SC.predict(  DATA[pt][atlas][0][1][r,c].reshape(1, -1)   )[0]
            if r == c:
                SC_residuals[r,c] = 0
    

    sfc_time = [None] * 4
    
   
    for per in range(4):
        sfc_time[per] = np.zeros(shape = (1,  function_residuals_per[per].shape[2]   ))
        vec_sc =  SC_residuals[np.triu_indices( len(SC_residuals), k = 1) ] 
        for t in range( function_residuals_per[per].shape[2]):
            vec_fc =   function_residuals_per[per][:,:,t][np.triu_indices( len( function_residuals_per[per][:,:,t]), k = 1) ] 
            sfc_time[per][0,t] =  spearmanr(vec_sc, vec_fc)[0]
            

    for per in range(4):
         sfc_time[per] =  sfc_time[per][0]
    
    all_SFC = np.array([item for sublist in sfc_time for item in sublist])
    
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same')/w
        
        
    sns.scatterplot( x = range(len(all_SFC)),    y= all_SFC, s = 3)
    all_SFC_rolling_mean = moving_average(all_SFC, 10)
    sns.lineplot( x = range(len(all_SFC_rolling_mean)),    y= all_SFC_rolling_mean)
    plt.axvline(x= function_residuals_per[0].shape[2])
    plt.axvline(x= function_residuals_per[0].shape[2] + function_residuals_per[1].shape[2]              )
    plt.axvline(x= function_residuals_per[0].shape[2] + function_residuals_per[1].shape[2]     + function_residuals_per[2].shape[2]          )

    moving_average(all_SFC, 10)
    
    all_SFC.shape
    
    
pt = 0
structure =  DATA[pt][atlas][1][0]
per = 0
function_ii = (DATA[pt][atlas][2][per][0][freq])
per = 1
function_pi = (DATA[pt][atlas][2][per][0][freq])
per = 2
function_ic = (DATA[pt][atlas][2][per][0][freq])
per = 3
function_po = (DATA[pt][atlas][2][per][0][freq])

function = [function_ii,function_pi , function_ic, function_po]

    
plot_time_series_SFC(structure, function )



    
    
    
    
    
#%%    



def get_classes(electrode_localization):
    N= len(np.array(electrode_localization["electrode_name"]))
    classes_matrix = np.zeros(shape = ( N,N ))
    for i in range(N):
        for j in range(N):
            elec1 = np.array(electrode_localization["distances_label_2"])[i]
            elec2 = np.array(electrode_localization["distances_label_2"])[j]
            gm_thresh = 0
            if (elec1 <= gm_thresh and elec2 <= gm_thresh):
                class_pair = 0
            elif (elec1 <= gm_thresh and elec2 > gm_thresh):
                class_pair = 1
            elif (elec1 > gm_thresh and elec2 <= gm_thresh):
                class_pair = 1
            elif (elec1 > gm_thresh and elec2 > gm_thresh):
                class_pair = 2
            classes_matrix[i,j] = class_pair
    return classes_matrix
    

    
    
    
def plot_time_series_FC_vs_dist(function_time, classes_matrix, distance_matrix, ofpath_figure_FC_vs_distance ):    
    
 

    
    FC_time = [None] * 4
    for per in range(4):
        
        FC_t = function_time[per]
        
        length = len(FC_t[:,:,0][np.triu_indices( len( FC_t[:,:,0]), k = 1) ]) 

        FC_time[per] = np.zeros(shape = (length,  function_time[per].shape[2]   ))
        
        for t in range( FC_t.shape[2]):
            FC_time[per][:,t] =   FC_t[:,:,t][np.triu_indices( len( FC_t[:,:,t]), k = 1) ]
    
            

    all_distance = distance_matrix[np.triu_indices( len(distance_matrix), k = 1) ] 
    all_class = classes_matrix[np.triu_indices( len(classes_matrix), k = 1) ] 
        
    all_distance_gm = all_distance[ np.where(all_class == 0)  ]        
    all_distance_gmwm = all_distance[ np.where(all_class == 1)  ]        
    all_distance_wm = all_distance[ np.where(all_class == 2)  ]        

    X0_FC = all_distance.reshape(-1,1)
    X0_FC_gm = all_distance_gm.reshape(-1,1)
    X0_FC_gmwm = all_distance_gmwm.reshape(-1,1)
    X0_FC_wm = all_distance_wm.reshape(-1,1)
    colors_p = ["#a88d76","#8d76a8", "#3a8ad2"]
    colors_p_rgb = [[0.439, 0.353, 0.278], [0.451, 0.357, 0.561], [0.129, 0.373, 0.580]]
    for per in range(4):
        for t in range( FC_time[per].shape[1]):   
            
            
            y0_FC = copy.deepcopy(FC_time[per][:,t] ) 
            y0_FC_gm = copy.deepcopy(FC_time[per][:,t][ np.where(all_class == 0)  ]     ) 
            y0_FC_gmwm = copy.deepcopy(FC_time[per][:,t][ np.where(all_class == 1)  ]     ) 
            y0_FC_wm = copy.deepcopy(FC_time[per][:,t][ np.where(all_class == 2)  ]  ) 
            
            gam_FC = LinearGAM( ).fit(X0_FC, y0_FC) ; 
            gam_FC_gm = LinearGAM( ).fit(X0_FC_gm, y0_FC_gm) ; 
            gam_FC_gmwm = LinearGAM( ).fit(X0_FC_gmwm, y0_FC_gmwm) ; 
            gam_FC_wm = LinearGAM( ).fit(X0_FC_wm, y0_FC_wm) ; 
            

            y1_FC = gam_FC.predict(X0_FC)
            y1_FC_gm = gam_FC_gm.predict(X0_FC_gm)
            y1_FC_gmwm = gam_FC_gmwm.predict(X0_FC_gmwm)
            y1_FC_wm = gam_FC_wm.predict(X0_FC_wm)
            
            
            ind = np.argsort(X0_FC.flatten())
            ind_gm = np.argsort(X0_FC_gm.flatten())
            ind_gmwm = np.argsort(X0_FC_gmwm.flatten())
            ind_wm = np.argsort(X0_FC_wm.flatten())
            
            
            X0_FC_wm.flatten()[ind_wm]
            def moving_average(x, w):
                return np.convolve(x, np.ones(w), 'valid')/w
            w = 10
            y1_FC_avg = moving_average(y1_FC[ind], w)
            y1_FC_gm_avg = moving_average(y1_FC_gm[ind_gm], w)
            y1_FC_gmwm_avg = moving_average(y1_FC_gmwm[ind_gmwm], w)
            y1_FC_wm_avg = moving_average(y1_FC_wm[ind_wm], w)
            
            xlim = [0,150]
            ylim = [0.15, 1]
            size = 2
            fig,axes = plt.subplots(1,1,figsize=(5,5), dpi = 300)
            sns.scatterplot( x = X0_FC.flatten(), y = y0_FC, hue= all_class , s = size, ax = axes, palette = colors_p  )
            
            sns.lineplot( x =  X0_FC.flatten()[ind][w-1:], y =  y1_FC_avg, ax = axes, color=[0.2, 0.2, 0.2], linewidth=1.5 )
            sns.lineplot( x =  X0_FC_gm.flatten()[ind_gm][w-1:], y =  y1_FC_gm_avg, ax = axes, color=colors_p_rgb[0], linewidth=1.5  )
            sns.lineplot( x =  X0_FC_gmwm.flatten()[ind_gmwm][w-1:], y =  y1_FC_gmwm_avg, ax = axes , color=colors_p_rgb[1], linewidth=1.5 )
            sns.lineplot( x =  X0_FC_wm.flatten()[ind_wm][w-1:], y =  y1_FC_wm_avg, ax = axes , color=colors_p_rgb[2], linewidth=1.5 )
            axes.set_title("State: {0}\nTime: {1}".format(per, "{:03d}".format(t+1)))
            axes.set_xlabel('Pairwise Distance (mm)')
            axes.set_ylabel('Functional Connectivity')
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)
            axes.get_legend().remove()
            #plt.show()
            plt.savefig(os.path.join(ofpath_figure_FC_vs_distance, "state_{0}_time_{1}".format(per, "{:03d}".format(t+1)))) 

#%%
freq = 0
for pt in range(8):
    pt=6
    ofpath_figure_FC_vs_distance = ospj(ofpath_figure, "FC_vs_distance" ,"sub-{0}".format(sub_IDs_unique[pt])) 
    if not (os.path.isdir(ofpath_figure_FC_vs_distance)): os.makedirs(ofpath_figure_FC_vs_distance, exist_ok=True)
    
    electrode_localization =  DATA[pt][atlas][0][0]
    classes_matrix = get_classes(electrode_localization)
        
        
    per = 0
    function_ii = (DATA[pt][atlas][2][per][0][freq])
    per = 1
    function_pi = (DATA[pt][atlas][2][per][0][freq])
    per = 2
    function_ic = (DATA[pt][atlas][2][per][0][freq])
    per = 3
    function_po = (DATA[pt][atlas][2][per][0][freq])
    
    function_time = [function_ii,function_pi , function_ic, function_po]
    
    distance_matrix = DATA[pt][atlas][0][1]
    
    plot_time_series_FC_vs_dist(function_time, classes_matrix, distance_matrix, ofpath_figure_FC_vs_distance )    
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    