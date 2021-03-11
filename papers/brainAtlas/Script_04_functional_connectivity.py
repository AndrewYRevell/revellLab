"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 

    1. This is a wrapper script: Runs through meta-data to automatically calculate for all data
        Meta-data: data_raw/iEEG_times/EEG_times.xlsx
    2. Calculates functional connectivity
    3. Calls functional_connectivity.get_Functional_connectivity in paper001/code/tools
        See this function for more detail
        
        Note, this function can take a few hours for each iEEG snippet, depending on length.
        For a 180 second iEEG file sampled at 512 Hz, it may take around 3-6 hours.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Import appropriate tools 
    2. Get appropriate input and output paths and file names
    3. Setting appropriate parameters and preprocessing of data
    4. Calculates functional_connectivity
        1. Parses trough meta-data to get apprpriate:
            iEEG file names
            Start time (meta-data stores in seconds, but iEEG.org python API requires microseconds)
            Stop time
        2. Finds the downloaded iEEG file (from Script_01_download_iEEG_data.py) based on above information
        3. Calls functional_connectivity.get_Functional_connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    None. This is a wrapper scipt that automatically runs based on meta-data file
    Meta-data: data_raw/iEEG_times/EEG_times.xlsx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    Saves functional connectivity for each patinet's iEEG file 
    in appropriate directory.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python3.6 Script_04_functional_connectivity.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
from functional_connectivity import get_Functional_connectivity
import pandas as pd

#%% Paths and File names

inputfile_EEG_times = ospj(path, "data/data_raw/iEEG_times/EEG_times.xlsx")

inputpath_EEG = ospj(path, "data/data_raw/EEG")
outputpath_FC = ospj(path, "data/data_processed/connectivity_matrices/function")


                             
#%%Load Data
data = pd.read_excel(inputfile_EEG_times)    

#%%
for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    iEEG_filename = data.iloc[i].file
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    inputpath_EEG_sub_ID = ospj(inputpath_EEG, "sub-{0}".format(sub_ID))
    inputfile_EEG_sub_ID = "{0}/sub-{1}_{2}_{3}_{4}_EEG.pickle".format(inputpath_EEG_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    #check if EEG file exists
    if not (os.path.exists(inputfile_EEG_sub_ID)):
        print("EEG file does not exist: {0}".format(inputfile_EEG_sub_ID))
    else:
        #Output filename EEG
        outputpath_FC_sub_ID = ospj(outputpath_FC, "sub-{0}".format(sub_ID))
        if not (os.path.isdir(outputpath_FC_sub_ID)): os.mkdir(outputpath_FC_sub_ID)#if the path doesn't exists, then make the directory
        outputfile_FC = "{0}/sub-{1}_{2}_{3}_{4}_functionalConnectivity.pickle".format(outputpath_FC_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
        print("\nID: {0}\nDescriptor: {1}".format(sub_ID, descriptor))
        if (os.path.exists(outputfile_FC)):
            print("File already exists: {0}".format(outputfile_FC))
        else:#if file already exists, don't run below
            get_Functional_connectivity(inputfile_EEG_sub_ID, outputfile_FC)
   


   

#%%












