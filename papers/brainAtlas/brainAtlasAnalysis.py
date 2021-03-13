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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
import sys
import os
import json
import copy
import pkg_resources
import pandas as pd
import numpy as np
from os.path import join, splitext, basename
from revellLab.packages.imaging.randomAtlas import randomAtlasGeneration as RAG
from revellLab.packages.imaging.regionMorphology import regionMophology
from revellLab.packages.dataClass import DataClassAtlases, DataClassCohorts
from revellLab.packages.atlasLocalization.atlasLocalizationFunctions import register_MNI_to_preopT1, show_slices

#%% Paths and File names

#BIDS directory
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
BIDSpenn = join(BIDS, "PIER")
BIDSmusc = join(BIDS, "MIER")
metadataDir = "/media/arevell/sharedSSD/linux/data/metadata"

freesurferLicense = "$FREESURFER_HOME/license.txt"
cohortsPath = join(metadataDir, "cohortData_brainAtlas.json")

#tools
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
atlasfilesPath = join(tools, "atlases", "atlasMetadata.json")
MNItemplatePath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz")
MNItemplateBrainPath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz")
randomAtlasesPath = join(tools, "atlases", "randomAtlasesWholeBrainMNI")


#BrainAtlas Project data analysis path
path= "/media/arevell/sharedSSD/linux/data/brainAtlas"
pathRegionMorphology = join(path, "regionMorphology")
pathQSIPREP = join(BIDS, "derivatives", "qsiprep")
pathTractography = join(BIDS, "derivatives", "tractography")

#%% Paramters

#Number of regions contained within the random atlas. 
randomAtlasRegions, permutations = [10, 30, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000, 10000], 5



#%% Read data

with open(atlasfilesPath) as f: atlasfiles = json.load(f)
atlases = DataClassAtlases.atlases(atlasfiles)

with open(cohortsPath) as f: cohortJson = json.load(f)
cohort = DataClassCohorts.brainAtlasCohort(cohortJson)


#%% 01 Atlases

#generate random atlases
RAG.batchGenerateRandomAtlases(randomAtlasRegions, permutations, MNItemplateBrainPath, randomAtlasesPath)

#measure volumes and sphericities
atlases.getAtlasMorphology(pathRegionMorphology)







#%% 02 DTI correction, atlas registration, tractography, and structural connectivity


#DTI correction
#get patietns with DTI data to be able to correct
patientsDTI = cohort.getWithDTI()

for i in range(len(patientsDTI)):
    sub = patientsDTI[i]
    cmd = f"qsiprep-docker {BIDSpenn} {pathQSIPREP} participant --output-resolution 1.5 --fs-license-file {freesurferLicense} -w {pathQSIPREP} --participant_label {sub}"
    print(cmd)
    #os.system(cmd) cannot run this docker command inside python interactive shell. Must run the printed command in terminal using qsiprep environment in mad in environment directory


#atlas registration
for i in range(len(patientsDTI)):
    sub = patientsDTI[i]

    preop3T = join(pathQSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-preproc_T1w")
    preop3Tmask = join(pathQSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-brain_mask")
    pathTractographySub = join(pathTractography, f"sub-{sub}")
    pathAtlasRegistration = join(pathTractographySub,"atlasRegistration")
    
    
    if not os.path.exists(pathTractography): raise IOError(f"Path does not exist: {pathTractography}")
    if not os.path.exists(pathAtlasRegistration): os.makedirs(pathAtlasRegistration)
    register_MNI_to_preopT1(preop3T, preop3Tmask, MNItemplatePath, MNItemplateBrainPath, "mni", pathAtlasRegistration)
    show_slices(preop3T + ".nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(pathAtlasRegistration, "preop3T.png"))
    show_slices(join(pathAtlasRegistration, "mni_flirt.nii.gz"), low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(pathAtlasRegistration, "mni_flirt.png"))
    show_slices(join(pathAtlasRegistration, "mni_fnirt.nii.gz"), low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(pathAtlasRegistration, "mni_fnirt.png"))














