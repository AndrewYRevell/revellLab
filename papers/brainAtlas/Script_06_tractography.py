"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 
    1. This is a wrapper script: Runs through meta-data to automatically calculate for all data
        Meta-data: data_raw/iEEG_times/EEG_times.xlsx
    2. Get tractography: computes tractography based on each patient's corrected diffusion imaging
    3. Calls dsi studio commands

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Import appropriate tools 
    2. Get appropriate input and output paths and file names
    3. Setting appropriate parameters and preprocessing of data
    4. Get tractography. See http://dsi-studio.labsolver.org/Manual/command-line-for-dsi-studio
        1. Dsi Studio command: src
            creates source file from patient's diffusion imaging and corresponding bvals and bvec files
            bvec and bval files must be in same directory with the same file name as diffusion imaging file
            DSI_studio Documentation - Find --action=src under sub-heading: Generate SRC files from DICOM/NIFTI/2dseq images
        2. DSI Studio command: rec
            Performs reconstruction using method 4 (GQI) See DSI Studio documentation for more info:
            DSI_studio Documentation - Find --action=rec under sub heading: Image reconstruction
        3. DSI Studio command: trk
            Performs tractography based on parameter ID:
            DSI_studio Documentation - Find --action=trk under sub heading: Tract-specific analysis, voxel-based analysis, connectivity matrix, and network measures
            Note, this does not make connectivity matrix files. This is done in the next wrapper script.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    None. This is a wrapper scipt that automatically runs based on meta-data file
    Meta-data: data_raw/iEEG_times/EEG_times.xlsx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    Saves tractography file for each patinet's imaging file in
    in appropriate directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python3.6 Script_06_tractography.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import os
from os.path import join as ospj
import pandas as pd
import numpy as np
#%% Input/Output Paths and File names

inputfile_EEG_times = ospj(path, "data/data_raw/iEEG_times/EEG_times.xlsx")
inputpath_dwi =  ospj(path, "data/data_processed/imaging")
outputpath_tractography = ospj(path, "data/data_processed/tractography")
                             
#%% Load Study Meta Data and parameters
data = pd.read_excel(inputfile_EEG_times)    

sub_ID_unique = np.unique(data.RID)

#%% Paramter ID
parameter_id = '7C1D393C9A99193FF3B3513Fb803Fcb2041bC84340420Fca01cbaCDCC4C3Ec'

"""
DSI_Studio parameter ID for calculating tractography. Related to number of streamlines, etc.
*Note from DSI_Studio documentation:
Assigning these tracking parameters takes a lot of time. To save the hassle, 
you can use "parameter_id" from the GUI (After running fiber tracking in GUI, 
the parameter ID will be shown in the method text in the right bottom window)
to override all parameters in one shot (e.g. --action=trk --source=some.fib.gz 
--parameter_id=c9A99193Fba3F2EFF013Fcb2041b96438813dcb). Please note that 
parameter ID does not overrider ROI settings. You may still need to assign 
ROI/ROA...etc. in the commnd line. 
"""


#%% Calculate tractography files from DSI sStudio
for i in range(len(sub_ID_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_ID_unique[i]
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
        print("Creating Tractography File in DSI Studio")
        cmd = "dsi_studio --action=trk --source={0} --parameter_id={1} --output={2}".format(output_fib, parameter_id, output_trk)
        os.system(cmd)
       


#%%












