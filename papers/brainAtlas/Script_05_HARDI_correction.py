"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

python3.6 Script_05_HARDI_correction.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import pandas as pd
#%% Paths and File names

inputfile_EEG_times = ospj( path, "data/data_raw/iEEG_times/EEG_times.xlsx")
inputpath_dwi =  "/mnt/data/data_processed/imaging"
outputpath_tractography = "/mnt/data/data_processed/tractography"

#may not be applicable if DSI Studio is already on your $PATH environment
#dsi_studio_path = "." # On a Mac: "/Applications/dsi_studio.app/Contents/MacOS/"

                             
#%%Load Data
data = pd.read_excel(inputfile_EEG_times)    

#%%
for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    print("\n\nSub-ID: {0}".format(sub_ID))
    inputfile_dwi =    "sub-{0}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz".format(sub_ID) 
    inputfile_dwi_fullpath = os.path.join(inputpath_dwi,"sub-{0}".format(sub_ID), inputfile_dwi)
    input_name = os.path.splitext(os.path.splitext(inputfile_dwi)[0])[0]
    outputpath_tractography_sub_ID = os.path.join(outputpath_tractography, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(outputpath_tractography_sub_ID)): os.mkdir(outputpath_tractography_sub_ID)
    output_src =  os.path.join(outputpath_tractography, "sub-{0}".format(sub_ID), "{0}.src.gz".format(input_name))
    output_fib = "{0}.odf8.f5.bal.012fy.rdi.gqi.1.25.fib.gz".format(output_src)
    output_trk =  os.path.join(outputpath_tractography, "sub-{0}".format(sub_ID), "{0}.trk.gz".format(input_name))
    os.path.exists(output_fib)
    
    if (os.path.exists(output_trk)):
        print("Tractography file already exists: {0}".format(output_trk))
    if not (os.path.exists(output_trk)):#if file already exists, don't run below
        print("Creating Source File in DSI Studio")
        cmd = "dsi_studio --action=src --source={0} --output={1}".format( inputfile_dwi_fullpath, output_src)
        os.system(cmd)
        print("Creating Reconstruction File in DSI Studio")
        cmd = "dsi_studio --action=rec --source={0} --method=4 --param0=1.25".format(output_src)
        os.system(cmd)
        parameter_id = '7C1D393C9A99193FF3B3513Fb803Fcb2041bC84340420Fca01cbaCDCC4C3Ec'
        print("Creating Tractography File in DSI Studio")
        cmd = "dsi_studio --action=trk --source={0} --parameter_id={1} --output={2}".format(output_fib, parameter_id, output_trk)
        os.system(cmd)
       


#%%












