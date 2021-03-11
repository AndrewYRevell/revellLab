"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 
    1. This is a wrapper script: Runs through meta-data to automatically calculate for all data
        Meta-data: data_raw/iEEG_times/EEG_times.xlsx
    2. Get raw iEEG data: Downloads from iEEG.org the appropriate EEG time-series snippet for this study. 
        Removes artifact electrodes listed in meta-data file
    3. Calls download_iEEG_data.get_iEEG_data in paper001/code/tools
        See this function for more detail

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Import appropriate tools 
    2. Get appropriate input and output paths and file names
    3. Setting appropriate parameters and preprocessing of data
    4. Get iEEG data
        1. Parses trough meta-data to get apprpriate:
            iEEG file names
            Start time (meta-data stores in seconds, but iEEG.org python API requires microseconds)
            Stop time
            ignore electrode (these electrodes are artefacts or electrodes needed to be removed). 
                They are deleted before iEEG data are saved. See download_iEEG_data.get_iEEG_data for more info
        2. Calls download_iEEG_data.get_iEEG_data
            

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    username: first argument. Your iEEG.org username
    password: second argument. Your iEEG.org password

    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    
    Saves EEG timeseries in specified output directory in pickle file format
    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python3.6 Script_02_download_iEEG_data.py 'username' 'password'

    Note: may need to use "double quotes" for username and/or password if they have special characters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
sys.path.append(ospj(path, "brainAtlas/code/tools/ieegpy"))
from download_iEEG_data import get_iEEG_data
import pandas as pd

#%% Input/Output Paths and File names
ifname_EEG_times = ospj(path,"data/data_raw/iEEG_times/EEG_times.xlsx")
ofpath_EEG = ospj(path,"data/data_raw/EEG")

                              
#%% Load username and password input from command line arguments
username= sys.argv[1]
password= sys.argv[2]


#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% Get iEEG data
for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    iEEG_filename = data.iloc[i].file
    ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    #Output filename EEG
    ofpath_EEG_sub_ID = ospj(ofpath_EEG, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofpath_EEG_sub_ID)): os.mkdir(ofpath_EEG_sub_ID)#if the path doesn't exists, then make the directory
    outputfile_EEG = "{0}/sub-{1}_{2}_{3}_{4}_EEG.pickle".format(ofpath_EEG_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    print("\n\n\nID: {0}\nDescriptor: {1}".format(sub_ID, descriptor))
    if (os.path.exists(outputfile_EEG)):
        print("File already exists: {0}".format(outputfile_EEG))
    else:#if file already exists, don't run below
        get_iEEG_data(username,password,iEEG_filename, start_time_usec, stop_time_usec, ignore_electrodes, outputfile_EEG)


    

#%%












