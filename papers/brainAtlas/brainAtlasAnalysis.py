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
import time
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import multiprocessing
from itertools import repeat
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename
from revellLab.packages.atlasLocalization.atlasLocalizationFunctions import register_MNI_to_preopT1, applywarp_to_atlas, getBrainFromMask, getExpandedBrainMask
from revellLab.packages.imaging.randomAtlas import randomAtlasGeneration as RAG
from revellLab.packages.imaging.regionMorphology import regionMophology
from revellLab.packages.imaging.tractography import tractography
from revellLab.packages.dataClass import DataClassAtlases, DataClassCohortsBrainAtlas, DataClassJson
from revellLab.packages.utilities import utils
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
#%% Paths and File names

#BIDS directory
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
BIDSpenn = join(BIDS, "PIER")
BIDSmusc = join(BIDS, "MIER")
metadataDir = "/media/arevell/sharedSSD/linux/data/metadata"

dsiStudioSingularityPatah = "/home/arevell/singularity/dsistudio/dsistudio_latest.sif"
freesurferLicense = "$FREESURFER_HOME/license.txt"
cohortsPath = join(metadataDir, "cohortData_brainAtlas.json")
jsonFilePath = join(metadataDir, "iEEGdataRevell.json")
fnameiEEGusernamePassword  = join("/media/arevell/sharedSSD/linux/", "ieegorg.json")

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

#Atlas metadata
with open(atlasfilesPath) as f: atlasfiles = json.load(f)
atlases = DataClassAtlases.atlases(atlasfiles)

#Study cohort data
with open(cohortsPath) as f: cohortJson = json.load(f)
cohort = DataClassCohortsBrainAtlas.brainAtlasCohort(cohortJson)
#Get patietns with DTI data 
patientsDTI = cohort.getWithDTI()

#JSON metadata data
with open(jsonFilePath) as f: jsonFile = json.load(f)
metadata = DataClassJson.DataClassJson(jsonFile)


#Get iEEG.org username and password
with open(fnameiEEGusernamePassword) as f: usernameAndpassword = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]


#%% 01 Atlases

#generate random atlases
RAG.batchGenerateRandomAtlases(randomAtlasRegions, permutations, MNItemplateBrainPath, randomAtlasesPath)
#measure volumes and sphericities
atlases.getAtlasMorphology(pathRegionMorphology)


#%% 02 DTI correction and tractography


#01 DTI correction
for i in range(len(patientsDTI)):
    sub = patientsDTI[i]
    cmd = f"qsiprep-docker {BIDSpenn} {pathQSIPREP} participant --output-resolution 1.5 --fs-license-file {freesurferLicense} -w {pathQSIPREP} --participant_label {sub}"
    print(cmd)
    #os.system(cmd) cannot run this docker command inside python interactive shell. Must run the printed command in terminal using qsiprep environment in mad in environment directory

#02 Tractography
for i in range(len(patientsDTI)):
    sub = patientsDTI[i]
    print(f"\n\n\n\n{sub}")
    if utils.getSubType(sub) == "control": ses = "control3T"
    else: ses = "preop3T"
    #input/output paths from QSI prep
    pathDWI = join(pathQSIPREP, "qsiprep", f"sub-{sub}", f"ses-{ses}", "dwi", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
    pathTracts = join(pathTractography,f"sub-{sub}", "tracts")
    trkName = f"{join(pathTracts, utils.baseSplitextNiiGz(pathDWI)[2])}.trk.gz"
    utils.checkPathAndMake(pathTractography, pathTracts)
    
    if not utils.checkIfFileExists(trkName, returnOpposite=False):
        t0 = time.time()
        tractography.getTracts(BIDS, dsiStudioSingularityPatah, pathDWI, pathTracts)
        utils.calculateTimeToComplete(t0, time.time(), len(patientsDTI), i)
        

#%% 03 Structural Connectivity for all atlases

#01 Atlas Registration
def atlasRegistrationBatch(patientsDTI, i):
    #for i in range(len(patientsDTI)):
    sub = patientsDTI[i]
    
    #input/output paths from QSI prep
    preop3T = join(pathQSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-preproc_T1w.nii.gz")
    preop3Tmask = join(pathQSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-brain_mask.nii.gz")
    pathAtlasRegistration = join(pathTractography, f"sub-{sub}","atlasRegistration")
    preop3Tbrain = join(pathAtlasRegistration, "brain.nii.gz")
    preop3TSTD = join(pathAtlasRegistration, utils.baseSplitextNiiGz(preop3T)[2] + "_std.nii.gz")
    #preop3TmaskExpand = join(pathAtlasRegistration, "brain_maskExpanded.nii.gz")
    utils.checkPathAndMake(pathTractography, pathAtlasRegistration)
    
    #MNI registration
    if not utils.checkIfFileExists(join(pathAtlasRegistration, "mni_fnirt.nii.gz")):
        #extract brain
        getBrainFromMask(preop3T, preop3Tmask, preop3Tbrain)
        utils.show_slices( preop3Tbrain, save = True, saveFilename = join(pathAtlasRegistration, "brain.png"))
        #expand Brain mask
        register_MNI_to_preopT1(preop3T, preop3Tbrain, MNItemplatePath, MNItemplateBrainPath, "mni", pathAtlasRegistration)
        utils.show_slices(preop3T, save = True, saveFilename = join(pathAtlasRegistration, "preop3T.png"))
        utils.show_slices(join(pathAtlasRegistration, "mni_flirt.nii.gz"), save = True, saveFilename = join(pathAtlasRegistration, "mni_flirt.png"))
        utils.show_slices(join(pathAtlasRegistration, "mni_fnirt.nii.gz"), save = True, saveFilename = join(pathAtlasRegistration, "mni_fnirt.png"))

    #apply warp to atlases
    applywarp_to_atlas(atlases.getAllAtlasPaths(), preop3TSTD, join(pathAtlasRegistration, "mni_warp.nii.gz"), pathAtlasRegistration, isDir = False)

p = multiprocessing.Pool(8)
p.starmap(atlasRegistrationBatch, zip(repeat(patientsDTI), range(1, len(patientsDTI))    )   )

#02 Structural Connectivity

#only do the first N number of tractography generation. Random atlases > 1000 regions, you will have to terminate the network measures
def batchStructuralConnectivity(patientsDTI, i, batchBegin, batchEnd, pathTractography, pathQSIPREP, atlasList):
    #for i in range(len(patientsDTI)):
    sub = patientsDTI[i]
    print(f"\n\n\n\n{sub}")
    if utils.getSubType(sub) == "control": ses = "control3T"
    else: ses = "preop3T"
    #input/output paths from QSI prep
    pathTracts = join(pathTractography,f"sub-{sub}", "tracts")
    pathDWI = join(pathQSIPREP, "qsiprep", f"sub-{sub}", f"ses-{ses}", "dwi", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
    pathTRK = f"{join(pathTracts, utils.baseSplitextNiiGz(pathDWI)[2])}.trk.gz"
    pathFIB = f"{join(pathTracts, utils.baseSplitextNiiGz(pathDWI)[2])}.fib.gz"
    preop3T = join(pathQSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-preproc_T1w.nii.gz")
    pathAtlasRegistration = join(pathTractography, f"sub-{sub}","atlasRegistration")
    pathStructuralConnectivity = join(pathTractography, f"sub-{sub}","structuralConnectivity")
    utils.checkPathAndMake(pathTractography, pathStructuralConnectivity)

    for a in range(batchBegin, batchEnd):
        atlas = join(pathAtlasRegistration, utils.baseSplitextNiiGz(atlasList[a])[0])
        if utils.checkIfFileExists(atlas):
            output = join(pathStructuralConnectivity, f"sub-{sub}"  )
            if not utils.checkIfFileExists(output + f".{utils.baseSplitextNiiGz(atlasList[a])[2]}.count.pass.connectogram.txt",  returnOpposite=True):
                t0 = time.time()
                tractography.getStructuralConnectivity(BIDS, dsiStudioSingularityPatah, pathFIB, pathTRK, preop3T, atlas, output )
                utils.calculateTimeToComplete(t0, time.time(), len(patientsDTI), i)
atlasList = atlases.getAllAtlasPaths()


p = multiprocessing.Pool(8)
p.starmap(batchStructuralConnectivity, zip(repeat(patientsDTI), range(1, len(patientsDTI)), repeat(0), repeat(98), repeat(pathTractography), repeat(pathQSIPREP), repeat(atlasList)    )   )



batchStructuralConnectivity(patientsDTI, 0, 98, pathTractography, pathQSIPREP, atlases.getAllAtlasPaths())


#%% 04 Electrode and atlas localization


#get EEG times:
iEEGTimes = cohort.getiEEGdataKeys()

# Download iEEG data

for i in range(len(iEEGTimes)):
    metadata.get_precitalIctalPostictal(iEEGTimes["sub"][i] , "Ictal", iEEGTimes["iEEGdataKey"][i], username, password, 
                                        BIDS = BIDS, dataset= "derivatives/iEEGorgDownload", session = "implant01", secondsBefore = 180, secondsAfter = 180)
    
    #get intertical
    associatedInterictal = metadata.get_associatedInterictal(iEEGTimes["sub"][i],  iEEGTimes["iEEGdataKey"][i])
    
    metadata.get_iEEGData(iEEGTimes["sub"][i] , "Interictal", associatedInterictal, username, password, 
                                        BIDS = BIDS, dataset= "derivatives/iEEGorgDownload", session = "implant01", startKey = "Start")
    



























