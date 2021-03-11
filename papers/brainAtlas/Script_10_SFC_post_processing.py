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
#path = "/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001"
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import structure_function_correlation_post_processing
import pandas as pd
import numpy as np
import pickle

#%% Paths and File names

ifname_EEG_times = ospj(path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_SFC = ospj( path, "data/data_processed/structure_function_correlation") 
ifpath_atlases_standard = ospj( path, "data/data_raw/atlases/standard_atlases")
ifpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ofpath_SFC_processed = ospj( path, "data/data_processed/structure_function_correlation_processed")
ofpath_SFC_aggregate = ospj( ofpath_SFC_processed, "SFC_aggregated_by_seizure")
ofpath_SFC_averages_by_atlas = ospj( ofpath_SFC_processed, "SFC_averages_by_atlas")
ofpath_SFC_interpolation = ospj( ofpath_SFC_processed, "SFC_interpolation")
ofpath_SFC_population_averages = ospj( ofpath_SFC_processed, "SFC_population_averages")



#%%Load Data
data = pd.read_excel(ifname_EEG_times)    
sub_ID_unique = np.unique(data.RID)


atlas_names_standard = [f for f in sorted(os.listdir(ifpath_atlases_standard))]
atlas_names_random = [f for f in sorted(os.listdir(ifpath_atlases_random))]


#%%

#IMPORTANT: assumes order listed meta data EEG times is (1) Interictal (2) Pre Ictal (3) Ictal (4) Postictal


#Aggregation
for i in range(0, len(data), 4):
    
    sub_ID = data.iloc[i].RID
    #Creating input and output names
    ifpath_SFC_sub_ID = ospj( ifpath_SFC, "sub-{0}".format(sub_ID)) 
    ofname_SFC_aggregate_sub_ID = ospj( ofpath_SFC_aggregate, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofname_SFC_aggregate_sub_ID)): os.mkdir(ofname_SFC_aggregate_sub_ID)
   
    
    #Random Atlases
    for a in range(len(atlas_names_random)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
        ifpath_SFC_sub_ID_atlas = ospj( ifpath_SFC_sub_ID, atlas_name) 
        ofname_SFC_aggregate_sub_ID_atlas = ospj( ofname_SFC_aggregate_sub_ID, atlas_name)
        
        if not (os.path.isdir(ofname_SFC_aggregate_sub_ID_atlas)): os.mkdir(ofname_SFC_aggregate_sub_ID_atlas)
       
        
        versions_length = len([f for f in sorted(os.listdir(ospj(ifpath_SFC_sub_ID, atlas_names_random[a])))]  )//4
        

        #getting seizure period SFC file names
        #ii = interictal pi = preictal ic = ictal po = postictal
        ii_start = int(data.iloc[i].connectivity_start_time_seconds  *1e6)
        ii_end   = int(data.iloc[i].connectivity_end_time_seconds    *1e6)
        pi_start = int(data.iloc[i+1].connectivity_start_time_seconds*1e6)
        pi_end   = int(data.iloc[i+1].connectivity_end_time_seconds  *1e6)
        ic_start = int(data.iloc[i+2].connectivity_start_time_seconds*1e6)
        ic_end   = int(data.iloc[i+2].connectivity_end_time_seconds  *1e6)
        po_start = int(data.iloc[i+3].connectivity_start_time_seconds*1e6)
        po_end   = int(data.iloc[i+3].connectivity_end_time_seconds  *1e6)
        
        iEEG_filename = data.iloc[i].file
        
        p = 0
        while p >=0:
            ifpath_interictal = "{0}/sub-{1}_{2}_{3}_{4}_{5}_v{6}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, ii_start, ii_end, atlas_name , '{:04}'.format(p+1))
            ifpath_preictal   = "{0}/sub-{1}_{2}_{3}_{4}_{5}_v{6}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, pi_start, pi_end, atlas_name , '{:04}'.format(p+1))
            ifpath_ictal      = "{0}/sub-{1}_{2}_{3}_{4}_{5}_v{6}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, ic_start, ic_end, atlas_name , '{:04}'.format(p+1))
            ifpath_postictal  = "{0}/sub-{1}_{2}_{3}_{4}_{5}_v{6}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, po_start, po_end, atlas_name , '{:04}'.format(p+1))
            
            
            ofname_SFC_aggregate = "sub-{0}_{1}_seizure_{2}_{3}_v{4}_SFC.pickle".format(sub_ID, iEEG_filename, ic_start, atlas_name, '{:04}'.format(p+1) )
            ofname_SFC_aggregate_fullpath = ospj(ofname_SFC_aggregate_sub_ID_atlas, ofname_SFC_aggregate)


            
            if os.path.exists(ifpath_interictal):
                if os.path.exists(ifpath_preictal):
                    if os.path.exists(ifpath_ictal):
                        if os.path.exists(ifpath_postictal):
                            print("Aggregating SFC: {0}".format(ofname_SFC_aggregate))
                            structure_function_correlation_post_processing.aggregate_SFC(ifpath_interictal, ifpath_preictal, ifpath_ictal, ifpath_postictal, ofname_SFC_aggregate_fullpath)
                            p = p + 1
                        else: p = -1; #print("File does not exist: {0}".format(ifpath_postictal))
                    else: p = -1; #print("File does not exist: {0}".format(ifpath_ictal))
                else: p = -1; #print("File does not exist: {0}".format(ifpath_preictal))
            else: p = -1; #print("File does not exist: {0}".format(ifpath_interictal))
             
    #Standard Atlases                   
    for a in range(len(atlas_names_standard)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
        ifpath_SFC_sub_ID_atlas = ospj( ifpath_SFC_sub_ID, atlas_name) 
        ofname_SFC_aggregate_sub_ID_atlas = ospj( ofname_SFC_aggregate_sub_ID, atlas_name)
        
        if not (os.path.isdir(ofname_SFC_aggregate_sub_ID_atlas)): os.mkdir(ofname_SFC_aggregate_sub_ID_atlas)
       

        #getting seizure period SFC file names
        #ii = interictal pi = preictal ic = ictal po = postictal
        ii_start = int(data.iloc[i].connectivity_start_time_seconds  *1e6)
        ii_end   = int(data.iloc[i].connectivity_end_time_seconds    *1e6)
        pi_start = int(data.iloc[i+1].connectivity_start_time_seconds*1e6)
        pi_end   = int(data.iloc[i+1].connectivity_end_time_seconds  *1e6)
        ic_start = int(data.iloc[i+2].connectivity_start_time_seconds*1e6)
        ic_end   = int(data.iloc[i+2].connectivity_end_time_seconds  *1e6)
        po_start = int(data.iloc[i+3].connectivity_start_time_seconds*1e6)
        po_end   = int(data.iloc[i+3].connectivity_end_time_seconds  *1e6)
        
        iEEG_filename = data.iloc[i].file
        
    
        ifpath_interictal = "{0}/sub-{1}_{2}_{3}_{4}_{5}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, ii_start, ii_end, atlas_name )
        ifpath_preictal   = "{0}/sub-{1}_{2}_{3}_{4}_{5}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, pi_start, pi_end, atlas_name )
        ifpath_ictal      = "{0}/sub-{1}_{2}_{3}_{4}_{5}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, ic_start, ic_end, atlas_name )
        ifpath_postictal  = "{0}/sub-{1}_{2}_{3}_{4}_{5}_SFC.pickle".format(ifpath_SFC_sub_ID_atlas, sub_ID, iEEG_filename, po_start, po_end, atlas_name )
        
        
        ofname_SFC_aggregate = "sub-{0}_{1}_seizure_{2}_{3}_SFC.pickle".format(sub_ID, iEEG_filename, ic_start, atlas_name )
        ofname_SFC_aggregate_fullpath = ospj(ofname_SFC_aggregate_sub_ID_atlas, ofname_SFC_aggregate)


        
        if os.path.exists(ifpath_interictal):
            if os.path.exists(ifpath_preictal):
                if os.path.exists(ifpath_ictal):
                    if os.path.exists(ifpath_postictal):
                        print("Aggregating SFC: {0}".format(ofname_SFC_aggregate))
                        structure_function_correlation_post_processing.aggregate_SFC(ifpath_interictal, ifpath_preictal, ifpath_ictal, ifpath_postictal, ofname_SFC_aggregate_fullpath)
                    else: print("File does not exist: {0}".format(ifpath_postictal))
                else: print("File does not exist: {0}".format(ifpath_ictal))
            else: print("File does not exist: {0}".format(ifpath_preictal))
        else: print("File does not exist: {0}".format(ifpath_interictal))
                            
                     

#%%
#averaging by atlas

for i in range(len(sub_ID_unique)):
    
    sub_ID = sub_ID_unique[i]
    #Creating input and output names
    ofname_SFC_aggregate_sub_ID = ospj( ofpath_SFC_aggregate, "sub-{0}".format(sub_ID))
    ofpath_SFC_averages_by_atlas_sub_ID = ospj( ofpath_SFC_averages_by_atlas,"averages", "sub-{0}".format(sub_ID))
    ofpath_SFC_std_by_atlas_sub_ID = ospj( ofpath_SFC_averages_by_atlas,"standard_deviations", "sub-{0}".format(sub_ID))
   
    if not (os.path.isdir(ofpath_SFC_averages_by_atlas_sub_ID)): os.mkdir(ofpath_SFC_averages_by_atlas_sub_ID)
    if not (os.path.isdir(ofpath_SFC_std_by_atlas_sub_ID)): os.mkdir(ofpath_SFC_std_by_atlas_sub_ID)
    
    #Random Atlases
    for a in range(len(atlas_names_random)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
        ofname_SFC_aggregate_sub_ID_atlas = ospj( ofname_SFC_aggregate_sub_ID, atlas_name)
        ofpath_SFC_averages_by_atlas_sub_ID_atlas = ospj( ofpath_SFC_averages_by_atlas_sub_ID, atlas_name)
        ofpath_SFC_std_by_atlas_sub_ID_atlas = ospj( ofpath_SFC_std_by_atlas_sub_ID, atlas_name)
        if not (os.path.isdir(ofpath_SFC_averages_by_atlas_sub_ID_atlas)): os.mkdir(ofpath_SFC_averages_by_atlas_sub_ID_atlas)
        if not (os.path.isdir(ofpath_SFC_std_by_atlas_sub_ID_atlas)): os.mkdir(ofpath_SFC_std_by_atlas_sub_ID_atlas)
        
        print("Averaging SFC in : {0}".format(ofname_SFC_aggregate_sub_ID_atlas))
        structure_function_correlation_post_processing.average_by_atlas(ofname_SFC_aggregate_sub_ID_atlas,ofpath_SFC_averages_by_atlas_sub_ID_atlas,ofpath_SFC_std_by_atlas_sub_ID_atlas)

    #Standard Atlases
    for a in range(len(atlas_names_standard)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
        ofname_SFC_aggregate_sub_ID_atlas = ospj( ofname_SFC_aggregate_sub_ID, atlas_name)
        ofpath_SFC_averages_by_atlas_sub_ID_atlas = ospj( ofpath_SFC_averages_by_atlas_sub_ID, atlas_name)
        ofpath_SFC_std_by_atlas_sub_ID_atlas = ospj( ofpath_SFC_std_by_atlas_sub_ID, atlas_name)
        if not (os.path.isdir(ofpath_SFC_averages_by_atlas_sub_ID_atlas)): os.mkdir(ofpath_SFC_averages_by_atlas_sub_ID_atlas)
        if not (os.path.isdir(ofpath_SFC_std_by_atlas_sub_ID_atlas)): os.mkdir(ofpath_SFC_std_by_atlas_sub_ID_atlas)
        
        print("Averaging SFC in : {0}".format(ofname_SFC_aggregate_sub_ID_atlas))
        structure_function_correlation_post_processing.average_by_atlas(ofname_SFC_aggregate_sub_ID_atlas,ofpath_SFC_averages_by_atlas_sub_ID_atlas,ofpath_SFC_std_by_atlas_sub_ID_atlas)




#%%
#interpolation


for i in range(len(sub_ID_unique)):
    
    sub_ID = sub_ID_unique[i]
    #Creating input and output names
    ofpath_SFC_averages_by_atlas_sub_ID = ospj( ofpath_SFC_averages_by_atlas,"averages", "sub-{0}".format(sub_ID))
    ofpath_SFC_interpolation_sub_ID =  ospj( ofpath_SFC_interpolation, "sub-{0}".format(sub_ID))
   
    if not (os.path.isdir(ofpath_SFC_interpolation_sub_ID)): os.mkdir(ofpath_SFC_interpolation_sub_ID)
    
    #Random Atlases
    for a in range(len(atlas_names_random)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
        ofpath_SFC_averages_by_atlas_sub_ID_atlas = ospj( ofpath_SFC_averages_by_atlas_sub_ID, atlas_name)
        ofpath_SFC_interpolation_sub_ID_atlas = ospj( ofpath_SFC_interpolation_sub_ID, atlas_name)
        if not (os.path.isdir(ofpath_SFC_interpolation_sub_ID_atlas)): os.mkdir(ofpath_SFC_interpolation_sub_ID_atlas)
        files = [f for f in sorted(os.listdir(ofpath_SFC_averages_by_atlas_sub_ID_atlas))]
        for s in range(len(files)):
            base_name = os.path.splitext(files[s])[0]
            ifpath_SFC_averages = ospj(ofpath_SFC_averages_by_atlas_sub_ID_atlas, files[s])
            os.path.exists(ifpath_SFC_averages)
            ofname = "{0}_interpolation_SFC.pickle".format(  base_name[  0:(len(base_name)-4)  ]     )
            ofname_SFC_interpolation = ospj(ofpath_SFC_interpolation_sub_ID_atlas, ofname )
            print("Interpolating SFC: {0}".format(ofname))
            structure_function_correlation_post_processing.interpolation(ifpath_SFC_averages,ofname_SFC_interpolation)

    #Standard Atlases
    for a in range(len(atlas_names_standard)):
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
        ofpath_SFC_averages_by_atlas_sub_ID_atlas = ospj( ofpath_SFC_averages_by_atlas_sub_ID, atlas_name)
        ofpath_SFC_interpolation_sub_ID_atlas = ospj( ofpath_SFC_interpolation_sub_ID, atlas_name)
        if not (os.path.isdir(ofpath_SFC_interpolation_sub_ID_atlas)): os.mkdir(ofpath_SFC_interpolation_sub_ID_atlas)
        files = [f for f in sorted(os.listdir(ofpath_SFC_averages_by_atlas_sub_ID_atlas))]
        for s in range(len(files)):
            base_name = os.path.splitext(files[s])[0]
            ifpath_SFC_averages = ospj(ofpath_SFC_averages_by_atlas_sub_ID_atlas, files[s])
            os.path.exists(ifpath_SFC_averages)
            ofname = "{0}_interpolation_SFC.pickle".format(  base_name[  0:(len(base_name)-4)  ]     )
            ofname_SFC_interpolation = ospj(ofpath_SFC_interpolation_sub_ID_atlas, ofname  )
            print("Interpolating SFC: {0}".format(ofname))
            structure_function_correlation_post_processing.interpolation(ifpath_SFC_averages,ofname_SFC_interpolation)






#%%
#Population Averaging
import time

print("\n\n\nAggregating all data to a long format dataframe in \nDirectory: {0}\n".format(ofpath_SFC_population_averages))
cols = ["subject", "atlas", "seizure", "period", "frequency", "period_index", "SFC"]
all_data = pd.DataFrame(columns=cols)
all_data = np.zeros(shape = (1, 7))
for i in range(len(sub_ID_unique)):
    
    sub_ID = sub_ID_unique[i]

    ofpath_SFC_interpolation_sub_ID =  ospj( ofpath_SFC_interpolation, "sub-{0}".format(sub_ID))
    atlases = [f for f in sorted(os.listdir(ofpath_SFC_interpolation_sub_ID))]
    
    for a in range(len(atlases)):
        ifname = ospj(ofpath_SFC_interpolation_sub_ID, atlases[a])
        files = [f for f in sorted(os.listdir(ifname))]
        for s in range(len(files)):
            with open(ospj(ifname, files[s]), 'rb') as f: SFC_interpolation, order_of_matrices_in_pickle_file, order_of_SFC_periods = pickle.load(f)
            
            order_of_matrices_in_pickle_file = np.array(order_of_matrices_in_pickle_file)
            string = files[s]
            seizure = string[string.find("seizure_"):len(string)].split("_")[1] 
            
            print(files[s])
            
            #startTime = time.time()
            
            for p in range(len(SFC_interpolation)):
                data = np.concatenate(  [  SFC_interpolation[p][0] , SFC_interpolation[p][1], SFC_interpolation[p][2], SFC_interpolation[p][3], SFC_interpolation[p][4] ]     ) 
                length = len(data )
                d = np.concatenate( [np.array(range(0,int(length/5)))]*5)
                
                freq = np.concatenate([np.array([order_of_matrices_in_pickle_file[0][0]] *int(length/5)), np.array([order_of_matrices_in_pickle_file[1][0]] *int(length/5)), np.array([order_of_matrices_in_pickle_file[2][0]] *int(length/5)),np.array([order_of_matrices_in_pickle_file[3][0]] *int(length/5)), np.array([order_of_matrices_in_pickle_file[4][0]] *int(length/5))])
                period = np.array([order_of_SFC_periods[p]] *length)
                seizure_ID = np.array([seizure] *length)
                atlas = np.array([atlases[a]] *length)
                subject =  np.array([sub_ID] *length)
                all_data_tmp = np.column_stack((  subject,atlas, seizure_ID, period , freq, d, data)  )
                
                all_data = np.concatenate(   [all_data, all_data_tmp] )
           # endTime = time.time()
            #workTime =  endTime - startTime
            #print(workTime)
            
            """
            startTime = time.time()
            for p in range(len(SFC_interpolation)):
                for freq in range(len(SFC_interpolation[p])):
                    data = SFC_interpolation[p][freq] 
                    for d in range(len(data)):
                        df = {"subject": [sub_ID], "atlas": [atlases[a]], "seizure": [seizure], "period": [order_of_SFC_periods[p]], "frequency": [order_of_matrices_in_pickle_file[freq][0]], "period_index": [d], "SFC": [data[d]] }
                        all_data = all_data.append(   pd.DataFrame( data= df , columns=cols))
            endTime = time.time()
            workTime =  endTime - startTime
            print(workTime)
            """
                    
           
    #pd.DataFrame.to_csv(all_data, ospj(ofpath_SFC_population_averages, "all_SFC_data.csv"), header=True, index=False)
    

import copy
all_data_copy = copy.deepcopy(all_data)

all_data = np.delete(all_data, 0, 0)
all_data_df = pd.DataFrame(all_data)
all_data_df.columns = ["subject", "atlas", "seizure","period", "frequency", "period_index", "SFC"]
pd.DataFrame.to_csv(all_data_df, ospj(ofpath_SFC_population_averages, "all_SFC_data.csv"), header=True, index=False)


"""
import seaborn as sns
i=7
sub_ID = sub_ID_unique[i]
ofpath_SFC_interpolation_sub_ID =  ospj( ofpath_SFC_interpolation, "sub-{0}".format(sub_ID))
ifname = ospj(ofpath_SFC_interpolation_sub_ID, "aal_res-1x1x1")

a = 29
ifname = ospj(ofpath_SFC_interpolation_sub_ID, atlases[a])
print(atlases[a])

files = [f for f in sorted(os.listdir(ifname))]
s=0
with open(ospj(ifname, files[s]), 'rb') as f: SFC_interpolation, order_of_matrices_in_pickle_file, order_of_SFC_periods = pickle.load(f)

tmp0 = SFC_interpolation[0][0] 
tmp1 = SFC_interpolation[1][0] 
tmp2 = SFC_interpolation[2][0] 
tmp3 = SFC_interpolation[3][0] 
tmp = np.concatenate((tmp0, tmp1, tmp2, tmp3))
N = 20
tmp_avg = np.convolve(tmp, np.ones((N,))/N, mode='same')
sns.lineplot(data = tmp)
sns.lineplot(data = tmp_avg)

"""
#%%




    