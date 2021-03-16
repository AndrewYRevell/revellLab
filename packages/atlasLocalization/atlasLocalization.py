#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import sys
import os
import pandas as pd
import copy
from os import listdir
from  os.path import join, isfile
from os.path import splitext 
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.utilities import utils
#import seaborn as sns


#%% Input

    
electrodePreopT1Coordinates = sys.argv[1]
preopT1 = sys.argv[2]
preopT1bet = sys.argv[3]
MNItemplate = sys.argv[4]
MNItemplatebet = sys.argv[5]
atlasDirectory = sys.argv[6]
atlasLabelsPath = sys.argv[7]
outputDirectory = sys.argv[8]
outputName = str(sys.argv[9])

if not splitext(outputName)[1] == ".csv":
    raise IOError(f"\n\n\n\nOutput Name must end with .csv.\n\nWhat is given:\n{outputName}\n\n\n\n" )

print(f"\n\n\n\n\n\n\n\n\n\n{outputName} saving to: \n{join(outputDirectory, outputName)}\n\n\n\n")
fillerString = "\n###########################\n###########################\n###########################\n###########################\n"

"""
Examples:
    
#electrodePreopT1Coordinates = "/media/arevell/sharedSSD/linux/data/BIDS/PIER/sub-RID0648/ses-implant01/ieeg/sub-RID0648_ses-implant01_space-preimplantT1w_electrodes.tsv"
electrodePreopT1Coordinates = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/electrodenames_coordinates_native_and_T1.csv"
preopT1 = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/T00_RID0648_mprage.nii.gz"
preopT1bet = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/T00_RID0648_mprage_brainBrainExtractionBrain.nii.gz"
MNItemplate = "/media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz"
MNItemplatebet = "/media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz"
atlasDirectory = "/media/arevell/sharedSSD/linux/revellLab/tools/atlases/atlases"
atlasLabelsPath = "/media/arevell/sharedSSD/linux/revellLab/tools/atlases/atlasLabels"
outputDirectory = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648"
outputName = "sub-RID0648_atlasLocalization.csv"


utils.checkPathError(electrodePreopT1Coordinates)
utils.checkPathError(preopT1)
utils.checkPathError(preopT1bet)
utils.checkPathError(MNItemplate)
utils.checkPathError(MNItemplatebet)
utils.checkPathError(atlasDirectory)
utils.checkPathError(outputDirectory)

    
    
rid = "440"
    
    
print(\
    f"conda activate revellLab \n\
python /media/arevell/sharedSSD/linux/revellLab/packages/atlasLocalization/atlasLocalization.py  \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid}/electrodenames_coordinates_native_and_T1.csv \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid}/T00_RID{rid}_mprage.nii.gz \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid}/T00_RID{rid}_mprage_brainBrainExtractionBrain.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/atlases \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid} \
    sub-RID0{rid}_atlasLocalization.csv\n\n" \
    )
    
    
"""

#%% Input names


utils.checkPathError(electrodePreopT1Coordinates)
utils.checkPathError(preopT1)
utils.checkPathError(preopT1bet)
utils.checkPathError(MNItemplate)
utils.checkPathError(MNItemplatebet)
utils.checkPathError(atlasDirectory)
utils.checkPathError(atlasLabelsPath)
utils.checkPathError(outputDirectory)



#names of outputs output
outputDirectoryTMP = join(outputDirectory, "tmp")
atlasRegistrationTMP = join(outputDirectoryTMP, "atlasRegistration")


preopT1_basename =  os.path.basename( splitext(splitext(preopT1)[0])[0])
preopT1bet_basename =  os.path.basename( splitext(splitext(preopT1bet)[0])[0])
preopT1_output = f"{join(outputDirectoryTMP, preopT1_basename)}"
preopT1bet_output = f"{join(outputDirectoryTMP, preopT1bet_basename)}"


outputNameTissueSeg = join(outputDirectory, f"{preopT1_basename}_std1x1x1_tissueSegmentation.nii.gz")
FIRST = join(outputDirectory, f"{preopT1_basename}_std1x1x1_FIRST.nii.gz")
FAST = join(outputDirectory, f"{preopT1_basename}_std1x1x1_FAST.nii.gz")
outputMNIname = "mni"
MNIwarp = join(outputDirectory, outputMNIname + "_warp.nii.gz")


#Make temporary storage 
utils.checkPathAndMake(outputDirectoryTMP, outputDirectoryTMP, make = True)
utils.checkPathAndMake(atlasRegistrationTMP, atlasRegistrationTMP, make = True)

#%%Begin Pipeline: Orient all images to standard RAS

print(f"\n\n{fillerString}Part 1 of 4\nReorientation of Images\nEstimated time: 10-30 seconds{fillerString}\nReorient all images to standard RAS\n")
cmd = f"fslreorient2std {preopT1} {preopT1_output}_std.nii.gz"; print(cmd); os.system(cmd)
cmd = f"fslreorient2std {preopT1bet} {preopT1bet_output}_std.nii.gz"; print(cmd); os.system(cmd)

print("\n\n\nReslice all standard RAS images to 1x1x1mm voxels\n")
#Make images 1x1x1 mm in size (usually clinical images from electrodeLocalization pipelines are 0.98x0.98x1; others are even more different than 1x1x1)
cmd = f"flirt -in {preopT1_output}_std.nii.gz -ref {preopT1_output}_std.nii.gz -applyisoxfm 1.0 -nosearch -out {preopT1_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)
cmd = f"flirt -in {preopT1bet_output}_std.nii.gz -ref {preopT1bet_output}_std.nii.gz -applyisoxfm 1.0 -nosearch -out {preopT1bet_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)


#visualize
utils.show_slices(f"{preopT1_output}_std1x1x1.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(outputDirectory, "pic_T00.png")  )
utils.show_slices(f"{preopT1bet_output}_std1x1x1.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_T00bet.png"))

print(f"\n\nPictures are saved to {outputDirectory}\nPlease check for quality assurance")



#%%Tissue segmentation
print(f"\n\n{fillerString}Part 2 of 4\nTissue Segmentation\nEstimated time: 30+ min{fillerString}\nRUNNING FIRST SUBCORTICAL SEGMENTATION\n")

#FIRST: subcortical segmentation
if not os.path.exists(FIRST):
    cmd = f"run_first_all -i {preopT1bet_output}_std1x1x1.nii.gz -o {preopT1bet_output}_std1x1x1.nii.gz -b -v"; print(cmd); os.system(cmd)
    #clean up files
    cmd = f"rm -r {preopT1bet_output}_std1x1x1.logs"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*.bvars"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*.vtk"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*origsegs*"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1_to_std*"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*.com*"; print(cmd); os.system(cmd)
    cmd = f"mv {preopT1bet_output}_std1x1x1_all_fast_firstseg.nii.gz {FIRST}"; print(cmd); os.system(cmd)
    
else:
    print(f"File exists:\n{FIRST}")

print("\n\n\nRUNNING FAST SEGMENTATION\n")
    
#FAST: segmentation of cortex
if not os.path.exists(FAST):
    cmd = f"fast -n 3 -H 0.25 -t 1 -v {preopT1bet_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)
    #Clean up files
    cmd = f"rm -r {preopT1bet_output}_std1x1x1_*mixeltype*"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1_*pve*"; print(cmd); os.system(cmd)
    cmd = f"mv {preopT1bet_output}_std1x1x1_seg.nii.gz {FAST}"; print(cmd); os.system(cmd)
else:
    print(f"File exists:\n{FAST}")


#Combine FIRST and FAST images
atl.combine_first_and_fast(FIRST, FAST, outputNameTissueSeg)
utils.show_slices(outputNameTissueSeg, low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_tissueSegmentation.png"))
print(f"\nPictures of FIRST + FAST combined images (pic_tissueSegmentation.png) are saved to {outputDirectory}\nPlease check for quality assurance")


#%%Registration of MNI to patient space (atlases are all in MNI space, so using this warp to apply to the atlases)
print(f"\n\n{fillerString}Part 3 of 4\nMNI and atlas registration\nEstimated time: 1-2+ hours{fillerString}\nRegistration of MNI template to patient space\n")
if not os.path.exists(MNIwarp):
    atl.register_MNI_to_preopT1(f"{preopT1_output}_std1x1x1.nii.gz", f"{preopT1bet_output}_std1x1x1.nii.gz", MNItemplate, MNItemplatebet, outputMNIname, outputDirectory, convertToStandard = False)
utils.show_slices(f"{join(outputDirectory, outputMNIname)}_fnirt.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_mni_fnirt.png"))
print(f"\nPictures of MNI nonlinear registration is saved to {outputDirectory}\nPlease check for quality assurance")


print("\n\n\nUsing MNI template warp to register all atlases (these atlases are already in MNI space)")
#apply warp to all atlases
if not os.path.exists(join(outputDirectory, outputName)):
    atl.applywarp_to_atlas(atlasDirectory, f"{preopT1_output}_std1x1x1.nii.gz", MNIwarp, atlasRegistrationTMP, isDir = True)

#%%Electrode Localization
print(f"\n\n{fillerString}Part 4 of 4\nElectrode Localization\nEstimated time: 10-20 min{fillerString}\nPerforming Electrode Localization\n")

#do not run if electrode localization output file already exists. Part 3 atlas warp takes a while, so skip it. If need to re-run, delete old file.
if not os.path.exists(join(outputDirectory, outputName)):
    #localization by region to tissue segmentation 
    outputTissueCoordinates = join(outputDirectoryTMP, "tissueSegmentation.csv")
    atl.by_region(electrodePreopT1Coordinates, outputNameTissueSeg, join(atlasLabelsPath, "tissue_segmentation.csv"), outputTissueCoordinates, description = "tissue_segmentation", Labels=True)
    #rename channels to standard 4 characters (2 letters, 2 numbers)
    atl.channel2stdCSV(outputTissueCoordinates)

    
    #localization by region to atlases
    atlases = [f for f in listdir(atlasRegistrationTMP) if isfile(join(atlasRegistrationTMP, f))]
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlasLabels =  join(atlasLabelsPath, atlasName + ".csv")
        
        atlasInMNI = join(atlasRegistrationTMP, atlasName + ".nii.gz")
        utils.checkPathError(atlasInMNI)
        print(f"{atlasName}")
        if "RandomAtlas" in atlasName: 
            Labels=False
        else:
            utils.checkPathError(atlasLabels)
            Labels=True
        outputAtlasCoordinates = join(outputDirectoryTMP, f"{atlasName}" + "_localization.csv")
        atl.by_region(electrodePreopT1Coordinates, atlasInMNI, atlasLabels, outputAtlasCoordinates, description = atlasName, Labels=Labels)
        atl.channel2stdCSV(outputAtlasCoordinates)
    
    #localization of channel distance to tissue segmentation: White Matter electrodes distance to Gray Matter
    print("\n\n\n\nFinding the WM electrode contacts distance to GM")
    outputTissueCoordinatesDistanceGM = join(outputDirectory, "electrodeWM_DistanceToGM.csv")
    if not os.path.exists(outputTissueCoordinatesDistanceGM):
        atl.distance_from_label(electrodePreopT1Coordinates, outputNameTissueSeg, 2, join(atlasLabelsPath, "tissue_segmentation.csv"), outputTissueCoordinatesDistanceGM)
    atl.channel2stdCSV(outputTissueCoordinatesDistanceGM)
    
    #localization of channel distance to tissue segmentation: Gray Matter electrodes distance to White Matter
    print("\n\n\n\nFinding the GM electrode contacts distance to WM")
    outputTissueCoordinatesDistanceWM = join(outputDirectory, "electrodeGM_DistanceToWM.csv")
    if not os.path.exists(outputTissueCoordinatesDistanceWM):
        atl.distance_from_label(electrodePreopT1Coordinates, outputNameTissueSeg, 3, join(atlasLabelsPath, "tissue_segmentation.csv"), outputTissueCoordinatesDistanceWM)
    atl.channel2stdCSV(outputTissueCoordinatesDistanceWM)
    
    
    print("\n\n\n\nConcatenating all files")
    #Concatenate files into one
    dataTissue = pd.read_csv(outputTissueCoordinates, sep=",", header=0)
    dataGM = pd.read_csv(outputTissueCoordinatesDistanceGM, sep=",", header=0).iloc[:,4:]
    dataWM = pd.read_csv(outputTissueCoordinatesDistanceWM, sep=",", header=0).iloc[:,4:]
    data = pd.concat([dataTissue, dataGM, dataWM]  , axis = 1)
    atlases = [f for f in listdir(atlasRegistrationTMP) if isfile(join(atlasRegistrationTMP, f))]
    
    
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlas = join(atlasDirectory, atlases[i])
        atlasLabels =  join(atlasLabelsPath, atlasName + ".csv")
        if not ("RandomAtlas" in atlas):
            print(atlasName)
            outputAtlasCoordinates = join(outputDirectoryTMP, f"{atlasName}" + "_localization.csv")
            data= pd.concat([data, pd.read_csv(outputAtlasCoordinates, sep=",", header=0).iloc[:,4:] ] , axis = 1)
                            
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlas = join(atlasDirectory, atlases[i])
        atlasLabels =  join(atlasLabelsPath, atlasName + ".csv")
        if ("RandomAtlas" in atlas):
            print(atlasName)
            outputAtlasCoordinates = join(outputDirectoryTMP, f"{atlasName}" + "_localization.csv")
            data= pd.concat([data, pd.read_csv(outputAtlasCoordinates, sep=",", header=0).iloc[:,4:] ] , axis = 1)
                            
    electrodeLocalization = join(outputDirectory, f"{outputName}")
    pd.DataFrame.to_csv(data, electrodeLocalization, header=True, index=False)

    print(f"\n\n\nDone\n\nFind electrode localization file in {join(outputDirectory, outputName)}\n\n")
else:
    print(f"\n\n\n\n\n\n\n\n\n\nNote: Electrode localization file alread exists in \n{join(outputDirectory, outputName)}\n\
If you need to re-run pipeline, please delete this file (Part 3 atlas warp files will need to be re-made. They are large and only temporarily saved.\n\n\n\n\n")

                                                         

                                                         

#clean
print("\n\n\nCleaning Files")
cmd = f"mv {preopT1_output}_std1x1x1.nii.gz {outputDirectory}"; print(cmd); os.system(cmd)
cmd = f"mv {preopT1bet_output}_std1x1x1.nii.gz {outputDirectory}"; print(cmd); os.system(cmd)
cmd = f"rm -r {join(outputDirectory, 'tmp' )}"; print(cmd); os.system(cmd)


















