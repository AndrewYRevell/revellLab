#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import os
import sys
import copy
import json
import subprocess
import pkg_resources
import multiprocessing

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from scipy import ndimage
from glob import glob, iglob
from itertools import repeat
from scipy.ndimage import morphology
from os.path import join, isfile, splitext, basename, dirname

from revellLab.packages.utilities import utils


revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
atlasDirectory = join(tools, "atlases", "atlases" )
atlasLabelDirectory = join(tools, "atlases", "atlasLabels" )

MNItemplatePath = join( tools,"templates", "MNI", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz")
MNItemplateBrainPath = join( tools, "templates", "MNI", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz")

OASIStemplatePath = join( tools, "templates","OASIS" ,"T_template0.nii.gz")
OASISprobabilityPath = join( tools, "templates", "OASIS", "T_template0_BrainCerebellumProbabilityMask.nii.gz")

#%% functions




def atlasLocalizationBIDSwrapper(subList, BIDS, dataset, SesImplant, ieegSpace, acq, freesurferReconAllDir, atlasLocalizationDir, atlasDirectory, atlasLabelDirectory, MNItemplatePath, MNItemplateBrainPath, multiprocess = False, cores = 8, rerun = False):
    if not multiprocess:
        for i in range(len(subList)):
            atlasLocalizationBatchProccess(subList, i, BIDS, dataset, SesImplant, ieegSpace, acq, freesurferReconAllDir, atlasLocalizationDir, atlasDirectory, atlasLabelDirectory, MNItemplatePath, MNItemplateBrainPath, rerun = rerun )
    #atlasLocalizationBatchProccess(subList, i,  atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory)
    if multiprocess:
        p = multiprocessing.Pool(cores)
        p.starmap(atlasLocalizationBatchProccess, zip(repeat(subList), range(len(subList)),
                                     repeat(BIDS),
                                     repeat(dataset),
                                     repeat(SesImplant),
                                     repeat(ieegSpace),
                                     repeat(acq),
                                     repeat(freesurferReconAllDir),
                                     repeat(atlasLocalizationDir),
                                     repeat(atlasDirectory),
                                     repeat(atlasLabelDirectory),
                                     repeat(MNItemplatePath),
                                     repeat(MNItemplateBrainPath),
                                     repeat(rerun)
                                     )   )




def atlasLocalizationBatchProccess(subList, i, BIDS, dataset, SesImplant, ieegSpace, acq, freesurferReconAllDir, atlasLocalizationDir, atlasDirectory, atlasLabelDirectory, MNItemplatePath, MNItemplateBrainPath, rerun = False ):
    sub = subList[i][4:]
    utils.checkPathError(atlasDirectory)
    utils.checkPathError(atlasLabelDirectory)
    utils.checkPathError(MNItemplatePath)
    utils.checkPathError(MNItemplateBrainPath)
    electrodePreopT1Coordinates = join(BIDS, dataset, f"sub-{sub}", f"ses-{SesImplant}", "ieeg", f"sub-{sub}_ses-{SesImplant}_space-{ieegSpace}_electrodes.tsv")
    if utils.checkIfFileExists(electrodePreopT1Coordinates): #check if there are implant electrode coordinates

        #check if freesurfer recon-all has been run with aseg.mgz output
        if utils.checkIfFileExistsGlob(  join(freesurferReconAllDir, f"sub-{sub}", "*")):
            highresSession = glob( join(freesurferReconAllDir, f"sub-{sub}", "ses-research3Tv" + '[0-9][0-9]'))

            if len(highresSession)  < 1:  #grab implant image if there is no high resolution T1
                session = glob( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{SesImplant}" ))[0]
                freesurferDir =  join(session, "freesurfer")
                T1 =  join(freesurferDir,  "mri", "orig_nu.mgz")
                aseg =  join(freesurferDir,  "mri", "aseg.mgz")
                T1brain =  join(freesurferDir,  "mri", "brain.mgz")
                T00 =  join(BIDS, dataset, f"sub-{sub}", f"ses-{SesImplant}", "anat",   f"sub-{sub}_ses-{SesImplant}_acq-{ieegSpace}_T1w.nii.gz")
                sessionID = f"ses-{SesImplant}"
            else:
                freesurferDir =  join(highresSession[-1], "freesurfer")
                T1 =  join(freesurferDir,  "mri", "orig_nu.mgz")
                aseg =  join(freesurferDir,  "mri", "aseg.mgz")
                T1brain =  join(freesurferDir,  "mri", "brain.mgz")
                T00 =  join(BIDS, dataset, f"sub-{sub}", f"ses-{SesImplant}", "anat",   f"sub-{sub}_ses-{SesImplant}_acq-{ieegSpace}_T1w.nii.gz")
                sessionID = "ses-research3T"
            if utils.checkIfFileExistsGlob(aseg):
                utils.checkPathError(electrodePreopT1Coordinates)
                utils.checkPathError(T1)
                utils.checkPathError(T00)
                utils.checkPathError(T1brain)
                utils.checkPathError(aseg)

                utils.checkPathError(atlasLocalizationDir)
                outputpath = join(atlasLocalizationDir, f"sub-{sub}", f"ses-{SesImplant}")
                utils.checkPathAndMake(atlasLocalizationDir, outputpath)
                outputName =  f"sub-{sub}_ses-{SesImplant}_desc-atlasLocalization.csv"

                if utils.checkIfFileDoesNotExist(join(outputpath, outputName)) and not rerun: #do not run if atlas localization already exists


                    outputDirTMP = join(outputpath, "tmp")
                    utils.checkPathAndMake(outputpath, outputDirTMP)

                    if utils.checkIfFileDoesNotExist(f"{outputDirTMP}/coordinates_tissueSegmentation.csv"):
                        tissueLocalizationFromReconAll(T1,  T1brain, aseg, T00, electrodePreopT1Coordinates,  outputDirTMP)

                    #MNI registration
                    T1_basename =  os.path.basename( splitext(splitext(T1)[0])[0])
                    T1_tmp = f"{join(outputDirTMP, T1_basename)}"
                    T1brain_basename =  os.path.basename( splitext(splitext(T1brain)[0])[0])
                    T1brain_tmp = f"{join(outputDirTMP, T1brain_basename)}"
                    aseg_basename =  os.path.basename( splitext(splitext(aseg)[0])[0])
                    aseg_tmp = f"{join(outputDirTMP, aseg_basename)}"
                    coordinatesOrig_basename =  os.path.basename( splitext(splitext(electrodePreopT1Coordinates)[0])[0])
                    coordinatesOrig_tmp = f"{join(outputDirTMP, coordinatesOrig_basename)}"

                    if utils.checkIfFileDoesNotExist(f"{outputDirTMP}/{T1brain_basename}_std_to_MNI_FLIRT.nii.gz"):
                        utils.executeCommand(f"flirt -in {T1brain_tmp}_std.nii.gz -ref {MNItemplateBrainPath} -dof 12 -out {outputDirTMP}/{T1brain_basename}_std_to_MNI_FLIRT -omat {outputDirTMP}/{T1brain_basename}_std_to_MNI_FLIRT.mat -v")
                    utils.show_slices(f"{outputDirTMP}/{T1brain_basename}_std_to_MNI_FLIRT.nii.gz", save=True, saveFilename=f"{outputDirTMP}/pic_03_{T1brain_basename}_std_to_MNI_FLIRT.png" )
                    utils.show_slices(f"{MNItemplatePath}", save=True, saveFilename=f"{outputDirTMP}/pic_03_MNI.png" )
                    if utils.checkIfFileDoesNotExist(f"{outputDirTMP}/{T1brain_basename}_std_to_MNI_FNIRT.nii.gz"):
                        utils.executeCommand(f"fnirt --in={T1_tmp}_std.nii.gz --ref={MNItemplatePath} --aff={outputDirTMP}/{T1brain_basename}_std_to_MNI_FLIRT.mat --iout={outputDirTMP}/{T1brain_basename}_std_to_MNI_FNIRT -v --cout={outputDirTMP}/{T1brain_basename}_std_to_MNI_FNIRT_COUT --fout={outputDirTMP}/{T1brain_basename}_std_to_MNI_FNIRT_FOUT")
                        utils.sendEmail(subject = f"{sub} {sessionID} FNIRT to MNI")
                    utils.show_slices(f"{outputDirTMP}/{T1brain_basename}_std_to_MNI_FNIRT.nii.gz", save=True, saveFilename=f"{outputDirTMP}/pic_03_{T1brain_basename}_std_to_MNI_FNIRT.png" )

                    coordinatesOther = pd.read_csv(f"{outputDirTMP}/coordinates_tmp_xform.txt", sep = " ", skiprows=1, header=None)
                    coordinatesOther.drop(coordinatesOther.columns[[1,3]],axis=1,inplace=True)
                    coordinatesOther.to_csv(f"{outputDirTMP}/coordinates_tmp_xform_tmp.txt", header = False, index = False, sep = " ")
                    if utils.checkIfFileDoesNotExist(f"{coordinatesOrig_tmp}_to_MNI.txt"):
                        utils.executeCommand( f"img2imgcoord -src {T1_tmp}_std.nii.gz -dest {MNItemplatePath} -warp {outputDirTMP}/{T1brain_basename}_std_to_MNI_FNIRT_COUT.nii.gz -mm {outputDirTMP}/coordinates_tmp_xform_tmp.txt > {coordinatesOrig_tmp}_to_MNI.txt"    )

                    #convert coordinates to proper BIDS format
                    coordinatesMNI = pd.read_csv(f"{coordinatesOrig_tmp}_to_MNI.txt", sep = " ", skiprows=1, header=None)
                    coordinatesMNI.drop(coordinatesMNI.columns[[1,3]],axis=1,inplace=True)


                    coordinates = pd.read_csv(f"{coordinatesOrig_tmp}.tsv", sep = "\t")
                    coordinates["x"] = coordinatesMNI.iloc[:,0]
                    coordinates["y"] = coordinatesMNI.iloc[:,1]
                    coordinates["z"] = coordinatesMNI.iloc[:,2]

                    space_start = coordinatesOrig_basename.find("space")
                    space_end = space_start + coordinatesOrig_basename[space_start:].find("_")
                    before = coordinatesOrig_basename[:space_start-1]
                    after = coordinatesOrig_basename[space_end+1:]
                    coordinatesMNIPATH = f"{outputDirTMP}/{before}_space-MNI_{after}.tsv"
                    coordinates.to_csv(  coordinatesMNIPATH, index=False, header = True, sep = "\t")
                    coordinates.to_csv( join(BIDS, dataset, f"sub-{sub}", f"ses-{SesImplant}", "ieeg", f"{before}_space-MNI_{after}.tsv"), index=False, header = True, sep = "\t")

                    coordinatesT1 = pd.read_csv(f"{outputDirTMP}/coordinates_tmp_xform.txt", sep = " ", skiprows=1, header=None)
                    coordinatesT1.drop(coordinatesT1.columns[[1,3]],axis=1,inplace=True)
                    coordinates["x"] = coordinatesT1.iloc[:,0]
                    coordinates["y"] = coordinatesT1.iloc[:,1]
                    coordinates["z"] = coordinatesT1.iloc[:,2]
                    coordinatesT1PATH = f"{outputDirTMP}/{before}_space-T1w_{after}.tsv"
                    coordinates.to_csv(  coordinatesT1PATH, index=False, header = True, sep = "\t")


                    #atlas Localization
                    outputAtlasCoordnatesDir = join( outputDirTMP, "atlases" )
                    utils.checkPathAndMake(outputDirTMP, outputAtlasCoordnatesDir)

                    DATA = pd.read_csv(f"{outputDirTMP}/coordinates_tissueSegmentation.csv")


                    #localization of channel distance to tissue segmentation: White Matter electrodes distance to Gray Matter
                    tissueSegmentationPath = f"{outputDirTMP}/{T1brain_basename}_std_tissueSegmentation.nii.gz"
                    img = nib.load(tissueSegmentationPath)
                    imgData = img.get_fdata()
                    utils.show_slices(imgData, isPath=False)
                    tissueTransform = copy.deepcopy(imgData)
                    tissueTransform[np.where(imgData == 0)] = 1
                    tissueTransform[np.where(imgData == 1)] = 1
                    tissueTransform[np.where(imgData == 2)] = 0
                    tissueTransform[np.where(imgData == 3)] = 1
                    utils.show_slices(tissueTransform, isPath=False)
                    toGM = morphology.distance_transform_edt(tissueTransform)
                    utils.show_slices(toGM, isPath=False)
                    coordinatesOther = pd.read_csv(f"{outputDirTMP}/coordinates_tmp_xform.txt", sep = " ", skiprows=1, header=None)
                    coordinatesOther.drop(coordinatesOther.columns[[1,3]],axis=1,inplace=True)
                    coordinates = pd.read_csv(f"{coordinatesOrig_tmp}.tsv", sep = "\t")

                    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(img.affine), coordinatesOther)
                    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel
                    coordinates_voxels = coordinates_voxels.astype(int)
                    distance = toGM[coordinates_voxels[:,0],coordinates_voxels[:,1],coordinates_voxels[:,2]]
                    distanceDF = pd.concat( [coordinates,     pd.DataFrame(distance)] , axis=1  )

                    distanceDF.columns = ["name", "x","y","z","size","distance_to_GM_millimeters"]
                    DATA = pd.concat(  [DATA, distanceDF.iloc[:,5]    ], axis = 1)

                    #localization to FreeSurfer Atlases
                    DKT = join(freesurferDir, "mri", "aparc.DKTatlas+aseg" )
                    Destrieux = join(freesurferDir, "mri", "aparc.a2009s+aseg" )
                    DK = join(freesurferDir, "mri", "aparc+aseg" )
                    utils.executeCommand( f"mri_convert {DKT}.mgz {outputDirTMP}/aparc.DKTatlas+aseg.nii.gz" )
                    utils.executeCommand( f"mri_convert {Destrieux}.mgz {outputDirTMP}/aparc.a2009s+aseg.nii.gz" )
                    utils.executeCommand( f"mri_convert {DK}.mgz {outputDirTMP}/aparc+aseg.nii.gz" )
                    #DKT
                    reg = by_region(coordinatesT1PATH, f"{outputDirTMP}/aparc.DKTatlas+aseg.nii.gz" , join(atlasLabelDirectory, "FreeSurferLookUpTable.csv"), join(outputAtlasCoordnatesDir, "aparc.DKTatlas+aseg_localization_space-T1.csv"), description = "aparc.DKTatlas+aseg", Labels=True)
                    DATA = pd.concat([ DATA , reg.iloc[:,4:6] ], axis = 1)
                    #Destrieux
                    reg = by_region(coordinatesT1PATH, f"{outputDirTMP}/aparc.a2009s+aseg.nii.gz" , join(atlasLabelDirectory, "FreeSurferLookUpTable.csv"), join(outputAtlasCoordnatesDir, "aparc.a2009s+aseg_localization_space-T1.csv"), description = "aparc.a2009s+aseg", Labels=True)
                    DATA = pd.concat([ DATA , reg.iloc[:,4:6] ], axis = 1)
                    #DK
                    reg = by_region(coordinatesT1PATH, f"{outputDirTMP}/aparc+aseg.nii.gz" , join(atlasLabelDirectory, "FreeSurferLookUpTable.csv"), join(outputAtlasCoordnatesDir, "aparc+aseg_localization_space-T1.csv"), description = "aparc+aseg", Labels=True)
                    DATA = pd.concat([ DATA , reg.iloc[:,4:6] ], axis = 1)
                    #Aseg
                    reg =  by_region(coordinatesT1PATH, f"{aseg_tmp}.nii.gz" , join(atlasLabelDirectory, "FreeSurferLookUpTable.csv"), join(outputAtlasCoordnatesDir, "aseg_localization_space-T1.csv"), description = "aseg", Labels=True)
                    DATA = pd.concat([ DATA , reg.iloc[:,4:6] ], axis = 1)


                    #localization by region to atlases
                    atlases = [f for f in listdir(atlasDirectory) if isfile(join(atlasDirectory, f))]
                    for i in range(len(atlases)):
                        atlasPath = join(atlasDirectory, atlases[i] )
                        atlasName = splitext(splitext(atlases[i])[0])[0]
                        atlasLabelsPath =  join(atlasLabelDirectory, atlasName + ".csv")
                        print(f"{atlasName}")
                        if "RandomAtlas" in atlasName:
                            Labels=False
                        else:
                            utils.checkPathError(atlasLabelsPath)
                            Labels=True
                        outputAtlasCoordinates = join(outputAtlasCoordnatesDir, f"{atlasName}" + "_localization_space-MNI.csv")
                        reg = by_region(coordinatesMNIPATH, atlasPath, atlasLabelsPath, outputAtlasCoordinates, description = atlasName, Labels=Labels)
                        DATA = pd.concat([ DATA , reg.iloc[:,4:6] ], axis = 1)
                        #atl.channel2stdCSV(outputAtlasCoordinates)

                    #localization by region to Random atlases
                    atlases = [f for f in listdir(join(atlasDirectory, "randomAtlasesWholeBrainMNI")) if isfile(join(atlasDirectory, "randomAtlasesWholeBrainMNI", f))]
                    for i in range(len(atlases)):
                        atlasPath = join(atlasDirectory, "randomAtlasesWholeBrainMNI", atlases[i] )
                        atlasName = splitext(splitext(atlases[i])[0])[0]
                        atlasLabelsPath =  join(atlasLabelDirectory, atlasName + ".csv")
                        print(f"{atlasName}")
                        if "RandomAtlas" in atlasName:
                            Labels=False
                        else:
                            utils.checkPathError(atlasLabelsPath)
                            Labels=True
                        outputAtlasCoordinates = join(outputAtlasCoordnatesDir, f"{atlasName}" + "_localization_space-MNI.csv")
                        reg = by_region(coordinatesMNIPATH, atlasPath, atlasLabelsPath, outputAtlasCoordinates, description = atlasName, Labels=Labels)
                        DATA = pd.concat([ DATA , reg.iloc[:,4:6] ], axis = 1)
                        #atl.channel2stdCSV(outputAtlasCoordinates)



                    electrodeLocalization = join(f"{outputpath}", f"{outputName}")
                    pd.DataFrame.to_csv(DATA, electrodeLocalization, header=True, index=False)
                    get_electrodeTissueMetadata(join(f"{outputpath}", f"{outputName}"), join(f"{outputpath}", "electrodeTissueMetadata.json"))
                    print(f"\n\n\nDone\n\nFind electrode localization file in {join(outputpath, outputName)}\n\n")

                    #cleaning
                    utils.executeCommand(f"mv {join(outputDirTMP, 'pic_*' )} {outputpath}   ")




def tissueLocalizationFromReconAll(T1,  T1brain, aseg, T00, electrodePreopT1Coordinates,  outputpath):

    T1session = T1[T1.find("ses-"):]
    T1session = T1session[:T1session.find("/")]
    T00session = T00[T00.find("ses-"):]
    T00session = T00session[:T00session.find("/")]

    outputDirTMP = outputpath

    T1_basename =  os.path.basename( splitext(splitext(T1)[0])[0])
    T1_tmp = f"{join(outputDirTMP, T1_basename)}"
    T1brain_basename =  os.path.basename( splitext(splitext(T1brain)[0])[0])
    T1brain_tmp = f"{join(outputDirTMP, T1brain_basename)}"
    aseg_basename =  os.path.basename( splitext(splitext(aseg)[0])[0])
    aseg_tmp = f"{join(outputDirTMP, aseg_basename)}"
    T00_basename =  os.path.basename( splitext(splitext(T00)[0])[0])
    T00_tmp = f"{join(outputDirTMP, T00_basename)}"
    TissuePath = f"{join(outputDirTMP , T1brain_basename)}_std_tissueSegmentation.nii.gz"

    coordinatesOrig_basename =  os.path.basename( splitext(splitext(electrodePreopT1Coordinates)[0])[0])
    coordinatesOrig_tmp = f"{join(outputDirTMP, coordinatesOrig_basename)}"

    utils.executeCommand(  f"mri_convert {T1} {T1_tmp}.nii.gz")
    utils.executeCommand(  f"mri_convert {T1brain} {T1brain_tmp}.nii.gz")
    utils.executeCommand(  f"mri_convert {aseg} {aseg_tmp}.nii.gz")

    utils.executeCommand(  f"cp {T00} {T00_tmp}.nii.gz")
    utils.executeCommand(  f"cp {electrodePreopT1Coordinates} {coordinatesOrig_tmp}.tsv")

    utils.executeCommand( f"fslreorient2std {T1_tmp}.nii.gz {T1_tmp}_std.nii.gz "  )
    utils.executeCommand( f"fslreorient2std {T1brain_tmp}.nii.gz {T1brain_tmp}_std.nii.gz"  )
    utils.executeCommand( f"fslreorient2std {aseg_tmp}.nii.gz {aseg_tmp}_std.nii.gz"  )
    utils.executeCommand( f"fslreorient2std {T00_tmp}.nii.gz {T00_tmp}_std.nii.gz"  )

    #% register T00 to T1
    #brain extraction
    if utils.checkIfFileDoesNotExist(f"{outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain_INVERSE.mat"):
        if not T00session == T1session: #if there is alread a brain extraction image, don't do brain extraction
            utils.executeCommand(f"antsBrainExtraction.sh -d 3 -a {T00_tmp}_std.nii.gz -e {OASIStemplatePath} -m {OASISprobabilityPath} -o {outputDirTMP}/{T00_basename}_std_"    )
            utils.executeCommand(f"flirt -in {T1brain_tmp}_std.nii.gz -ref {outputDirTMP}/{T00_basename}_std_BrainExtractionBrain.nii.gz -omat {outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain.mat -dof 6 -out {outputDirTMP}/brain_std_to_{T00_basename}_std_BrainExtractionBrain.nii.gz"    )
            utils.executeCommand(f"convert_xfm -omat  {outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain_INVERSE.mat -inverse  {outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain.mat"    )
            utils.executeCommand(f"flirt -in {outputDirTMP}/{T00_basename}_std.nii.gz -ref {T1_tmp}_std.nii.gz -applyxfm -init {outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain_INVERSE.mat  -out {outputDirTMP}/{T00_basename}_std_to_{T1_basename}_std.nii.gz"    )
            utils.show_slices(f"{outputDirTMP}/{T00_basename}_std_to_{T1_basename}_std.nii.gz", save=True, saveFilename=f"{outputDirTMP}/pic_01_{T00_basename}_std_to_{T1_basename}_std.png")
            utils.show_slices(f"{T1_tmp}_std.nii.gz", save=True, saveFilename=f"{outputDirTMP}/pic_01_{T1_basename}_std.png")
        else:
            utils.executeCommand(f"cp {T1brain_tmp}_std.nii.gz {outputDirTMP}/{T00_basename}_std_BrainExtractionBrain.nii.gz")
            #making transformation matrix with diagonals of 1. Since FSL needs and textfile with separators of two white spaces, and pandas is one of the WORST datascience packages ever created, I have to do this manually
            xform = np.zeros(shape = (4,4))
            np.fill_diagonal(xform, 1)
            xform = xform.astype(int)
            xform = pd.DataFrame(xform)
            xform_string = xform.to_string( col_space=2, header=False, index= False)
            xform_string = xform_string.split('\n')
            for line in range(len(xform_string)):
                xform_string[line] = xform_string[line].strip()
            xform_string = '\n'.join(xform_string)
            with open(f"{outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain_INVERSE.mat", 'w') as file: print(xform_string, file = file)
        utils.show_slices(f"{outputDirTMP}/{T00_basename}_std_BrainExtractionBrain.nii.gz", save=True, saveFilename=f"{outputDirTMP}/pic_00_{T00_basename}_std_BrainExtractionBrain.png")
    coordinates = pd.read_csv(f"{coordinatesOrig_tmp}.tsv", sep = "\t")
    coordinatesNames = coordinates["name"]
    coordinates = coordinates.drop(["name", "size"], axis=1)
    coordinates.to_csv(  f"{outputDirTMP}/coordinates_tmp.txt",  sep=" ", index=False, header=False)

    utils.executeCommand( f"img2imgcoord -src {T00_tmp}_std_BrainExtractionBrain.nii.gz -dest {T1_tmp}_std.nii.gz -xfm {outputDirTMP}/xform_brain_std_to_{T00_basename}_std_BrainExtractionBrain_INVERSE.mat -mm {outputDirTMP}/coordinates_tmp.txt > {outputDirTMP}/coordinates_tmp_xform.txt"    )
    coordinatesOther = pd.read_csv(f"{outputDirTMP}/coordinates_tmp_xform.txt", sep = " ", skiprows=1, header=None)
    coordinatesOther.drop(coordinatesOther.columns[[1,3]],axis=1,inplace=True)
    print("\n\n\nTissue Segmentation\n\n\n")
    img = nib.load(f"{aseg_tmp}_std.nii.gz")
    imgData = img.get_fdata()
    imgBET = nib.load( f"{T1brain_tmp}_std.nii.gz")
    imgBETData = imgBET.get_fdata()
    imgBETData[np.where(imgBETData >0)] = -2  #anything inside the brain extraction will be labeled as CSF, and then replaced by classes below
    #utils.show_slices(imgBETData, isPath = False)
    imgData[np.where(imgData ==2)] = -1 #Left cerebral WM
    imgData[np.where(imgData ==4)] = -2 #Left lateral ventricle
    imgData[np.where(imgData ==5)] = -2 #Left inferior lateral ventricle
    imgData[np.where(imgData ==7)] = -1 #Left cerebellum WM
    imgData[np.where(imgData ==24)] = -2 #CSF
    imgData[np.where(imgData ==41)] = -1 #Right cerebral WM
    imgData[np.where(imgData ==43)] = -2 #Right lateral ventricle
    imgData[np.where(imgData ==44)] = -2 #Right inferior lateral ventricle
    imgData[np.where(imgData ==46)] = -1 #Right cerebellum WM
    imgData[np.where(imgData ==77)] = -1 #WM hypointensities
    imgData[np.where(imgData ==85)] = -1 #Optic Chiasm
    imgData[np.where(imgData ==251)] = -1 #Corpus Callosum
    imgData[np.where(imgData ==252)] = -1 #Corpus Callosum
    imgData[np.where(imgData ==253)] = -1 #Corpus Callosum
    imgData[np.where(imgData ==254)] = -1 #Corpus Callosum
    imgData[np.where(imgData ==255)] = -1 #Corpus Callosum
    imgData[np.where(imgData >0)] = 2 #GM

    imgBETData[np.where(imgData >0)] = 2 #GM
    imgBETData[np.where(imgData ==-2)] = 1 #CSF
    imgBETData[np.where(imgData ==-1)] = 3 #WM
    imgBETData[np.where(imgBETData ==-2)] = 1 #WM

    #utils.show_slices(imgBETData, isPath = False)

    imgTissue = nib.Nifti1Image(imgBETData, img.affine)
    nib.save(imgTissue, TissuePath)
    utils.show_slices(TissuePath, save=True, saveFilename= f"{outputDirTMP}/pic_02_{T1brain_basename}_std_tissueSegmentation.png")
    utils.show_slices(f"{T1brain_tmp}_std.nii.gz", save=True, saveFilename= f"{outputDirTMP}/pic_02_{T1brain_basename}_std.png")

    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(img.affine), coordinatesOther)
    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel
    coordinates_voxels = coordinates_voxels.astype(int)



    try:
        img_ROI = imgBETData[coordinates_voxels[:,0], coordinates_voxels[:,1], coordinates_voxels[:,2]]
    except: #checking to make sure coordinates are in the atlas. This happens usually for electrodes on the edge of the SEEG. For example, RID0420 electrodes LE11 and LE12 are outside the brain/skull, and thus are outside even the normal MNI space of 181x218x181 voxel dimensions
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>imgBETData.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = 0
            elif((coordinates_voxels[i,1]>imgBETData.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = 0
            elif((coordinates_voxels[i,2]>imgBETData.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = 0
            else:
                img_ROI[i] = imgBETData[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]





    img_percentGM = np.zeros(shape = img_ROI.shape)
    img_percentWM = np.zeros(shape = img_ROI.shape)
    img_percentCSF = np.zeros(shape = img_ROI.shape)
    img_percentOutside = np.zeros(shape = img_ROI.shape)

    #Get purity #"WM Depth"
    dilations = 5 #the distance around the channel coordinate we are looking at the tissue classes to get a consensus of the tissues
    for i in range(len(img_ROI)) :
        print(i)
        mask = np.zeros(shape = imgBETData.shape)
        mask[coordinates_voxels[i,0],coordinates_voxels[i,1] ,coordinates_voxels[i,2]    ]    = 1
        maskDilated = morphology.binary_dilation(mask, iterations = dilations).astype(mask.dtype)

        points = np.where(maskDilated > 0)
        classes = imgBETData[points]
        img_percentGM[i] = len(np.where(classes == 2)[0])/len(classes)
        img_percentWM[i] = len(np.where(classes == 3)[0])/len(classes)
        img_percentCSF[i] = len(np.where(classes == 1)[0])/len(classes)
        img_percentOutside[i]  = len(np.where(classes == 0)[0])/len(classes)


    tissueArgMax = np.argmax( np.column_stack((img_percentOutside,img_percentCSF, img_percentGM, img_percentWM)), axis = 1)

    tissue = pd.DataFrame( data=img_ROI.astype(int) )
    percentGM = pd.DataFrame( data=img_percentGM)
    percentWM = pd.DataFrame( data=img_percentWM )
    percentCSF = pd.DataFrame( data=img_percentCSF )
    percentOutside = pd.DataFrame( data=img_percentOutside )
    tissueArgMax = pd.DataFrame( data=tissueArgMax )
    coordinatesToSave = pd.concat([coordinatesNames, coordinatesOther, tissue,tissueArgMax, percentGM, percentWM, percentCSF, percentOutside ], axis=1, ignore_index=True)
    coordinatesToSave.columns = ["channel", "x", "y", "z", "tissue_number","tissue_numberArgMax", "percent_GM", "percent_WM", "percent_CSF", "percent_Outside"]
    coordinatesToSave.to_csv(  f"{outputDirTMP}/coordinates_tissueSegmentation.txt",  sep=" ", index=False, header=True)
    coordinatesToSave.to_csv(  f"{outputDirTMP}/coordinates_tissueSegmentation.csv", index=False)
    coordinatesHTML = pd.concat([coordinatesNames, coordinatesOther, tissueArgMax ], axis=1, ignore_index=True)
    coordinatesHTML.to_csv(  f"{outputDirTMP}/electrodes.txt",  sep=" ", index=False, header=False)







def by_region(coordinatesMNIPATH, atlasPath, atlasLabelsPath, ofname, sep="\t",  xColIndex=1, yColIndex=2, zColIndex=3, description = "unknown_atlas", Labels=True):
    # getting imaging data
    img = nib.load(atlasPath)
    imgData = img.get_fdata()  # getting actual image data array

    #utils.show_slices(atlasPath)
    affine = img.affine
    if Labels == True:
        #getting atlas labels file
        atlas_labels = pd.read_csv(atlasLabelsPath, sep=",", header=None)
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
        atlas_labels = atlas_labels.drop([0, 1], axis=0).reset_index(drop=True)
        atlas_regions_numbers = np.array(atlas_labels.iloc[:,0]).astype("float64")
        atlas_labels_descriptors = np.array(atlas_labels.iloc[:,1])
    if Labels == False:
        atlas_regions_numbers = np.arange(0,   np.max(imgData)+1 )
        atlas_labels_descriptors = np.arange(0,   np.max(imgData)+1 ).astype("int").astype("object")
        atlas_name = os.path.splitext(os.path.basename(atlasPath))[0]
        atlas_name = os.path.splitext(atlas_name)[0]
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
    # getting electrode coordinates data
    data = pd.read_csv(coordinatesMNIPATH, sep=sep, header=0)
    data = data.iloc[:, [0, xColIndex, yColIndex, zColIndex]]
    column_names = ['name', "x", "y", "z", column_description1,column_description2 ]
    channel_names = np.array(data["name"])

    coordinates = np.array((data.iloc[:, range(1, 4)]))  # get the scanner coordinates of electrodes
    # transform the real-world coordinates to the atals voxel space. Need to inverse the affine with np.linalg.inv(). To go from voxel to world, just input aff (dont inverse the affine)
    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(affine), coordinates)
    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel
    coordinates_voxels = coordinates_voxels.astype(int)


    try:
        img_ROI = imgData[coordinates_voxels[:,0], coordinates_voxels[:,1], coordinates_voxels[:,2]]
    except: #checking to make sure coordinates are in the atlas. This happens usually for electrodes on the edge of the SEEG. For example, RID0420 electrodes LE11 and LE12 are outside the brain/skull, and thus are outside even the normal MNI space of 181x218x181 voxel dimensions
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>imgData.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = 0
                print(f'{channel_names[i]} is outside image space: setting to zero')
            elif((coordinates_voxels[i,1]>imgData.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = 0
                print(f'{channel_names[i]} is outside image space: setting to zero')
            elif((coordinates_voxels[i,2]>imgData.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = 0
                print(f'{channel_names[i]} is outside image space: setting to zero')
            else:
                img_ROI[i] = imgData[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]


    #getting corresponding labels
    img_labels = np.zeros(shape =img_ROI.shape ).astype("object")
    for l in range(len(img_ROI)):
        ind = np.where( img_ROI[l] ==    atlas_regions_numbers)
        if len(ind[0]) >0: #if there is a correpsonding label, then fill in that label. If not, put "unknown"
            if img_ROI[l] ==0: #if label is 0, then outside atlas
                img_labels[l] = "OutsideAtlas"
            img_labels[l] = atlas_labels_descriptors[ind][0]
        else:
            img_labels[l] = "NotInAtlas"
    img_ROI = np.reshape(img_ROI, [img_ROI.shape[0], 1])
    img_ROI = img_ROI.astype(int)
    df_img_ROI = pd.DataFrame(img_ROI)
    df_img_ROI.columns = [column_names[4]]
    img_labels = np.reshape(img_labels, [img_labels.shape[0], 1])
    df_img_labels = pd.DataFrame( img_labels)
    df_img_labels.columns = [column_names[5]]
    data = pd.concat([data, df_img_ROI, df_img_labels], axis=1)
    pd.DataFrame.to_csv(data, ofname, header=True, index=False)
    return data




def get_electrodeTissueMetadata(atlasLocalizationFile, outputName):
    data = pd.read_csv(atlasLocalizationFile)
    tissueSegNumber = data["tissue_numberArgMax"]

    elecGM = data["channel"].iloc[np.where(tissueSegNumber == 2) ]
    elecWM = data["channel"].iloc[np.where(tissueSegNumber == 3) ]
    elecCSF = data["channel"].iloc[ np.where(tissueSegNumber == 1) ]
    elecOutside = data["channel"].iloc[np.where(tissueSegNumber == 0) ]

    metadataTissue = dict(grayMatter = elecGM.to_list() , whiteMatter = elecWM.to_list(), CSF = elecCSF.to_list(), outside = elecOutside.to_list()  )
    with open(outputName, 'w', encoding='utf-8') as f: json.dump(metadataTissue, f, ensure_ascii=False, indent=4)



def get_atlas_localization_file(sub, SESSION, paths):
    """
    :param sub: DESCRIPTION
    :type sub: string, example 'RID0194' (without the "sub-)
    :param SESSION: DESCRIPTION
    :type SESSION: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    file = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{sub}", f"ses-{SESSION}", f"sub-{sub}_ses-{SESSION}_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file, printBOOL=False):
        localization = pd.read_csv(file)
        localization_channels = localization["channel"]
        localization_channels = utils.channel2std(np.array(localization_channels))
        return localization, localization_channels




#%%


"""

BIDS = join("/media","arevell","sharedSSD","linux", "data", "BIDS")
dataset = "PIER"
derivatives = join(BIDS, "derivatives")

freesurferReconAllDir = join(BIDS, "derivatives", "freesurferReconAll")
atlasLocalizationDir = join(derivatives, "atlasLocalization")


SesImplant = "implant01"
acq = "3D"
ieegSpace = "T00"

subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]
subjects = [basename(item) for item in subDir ]



subList = subjects
atlasLocalizationBIDSwrapper(subList,  BIDS, dataset, SesImplant, ieegSpace, acq, freesurferReconAllDir, atlasLocalizationDir, atlasDirectory, atlasLabelDirectory, MNItemplatePath, MNItemplateBrainPath, multiprocess = True, cores = 12, rerun = False)
atlasLocalizationBIDSwrapper(["sub-RID0171"],  BIDS, dataset, SesImplant, ieegSpace, acq, freesurferReconAllDir, atlasLocalizationDir, atlasDirectory, atlasLabelDirectory, MNItemplatePath, MNItemplateBrainPath, multiprocess = True, cores = 12, rerun = False)
subList = "sub-RID0176"
"""


#%%






"""








def combine_first_and_fast(FIRST, FAST, outputName):
    img_first = nib.load(FIRST)
    data_first = img_first.get_fdata()
    img_fast= nib.load(FAST)
    data_fast = img_fast.get_fdata()
    #make all subcortical structures = 2 (2 is the GM category)
    data_first[np.where(data_first > 0)] = 2
    #replace fast images with first images where not zero
    data_fast[np.where(data_first > 0)] = data_first[np.where(data_first > 0)]
    #plot
    img_first_fast = nib.Nifti1Image(data_fast, img_fast.affine)
    nib.save(img_first_fast, outputName)































def atlasLocalizationBatchProccess2(subList, i,  atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory, rerun = False):
#for i in range(len(subList)):
    sub = subList[i]
    electrodePreopT1Coordinates = join(inputDirectory, f"sub-{sub}", "electrodenames_coordinates_native_and_T1.csv")
    preopT1 = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage.nii.gz")
    preopT1bet = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage_brainBrainExtractionBrain.nii.gz")
    outputDirectory = join(outputDirectory, f"sub-{sub}")
    outputName =  f"sub-{sub}_atlasLocalization.csv"
    cmd =  f"python {atlasLocalizationFunctionDirectory + '/atlasLocalization.py'} {electrodePreopT1Coordinates} {preopT1} {preopT1bet} \
        {MNItemplatePath} {MNItemplateBrainPath} {atlasDirectory} {atlasLabelsPath} {outputDirectory} {outputName} {rerun}"
    print(cmd); os.system(cmd)

def atlasLocalizationBIDSwrapper(subList, atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory, multiprocess = False, cores = 8, rerun = False):
    if not multiprocess:
        for i in range(len(subList)):
            sub = subList[i]
            electrodePreopT1Coordinates = join(inputDirectory, f"sub-{sub}", "electrodenames_coordinates_native_and_T1.csv")
            preopT1 = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage.nii.gz")
            preopT1bet = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage_brainBrainExtractionBrain.nii.gz")
            outputDirectorySub = join(outputDirectory, f"sub-{sub}")
            outputName =  f"sub-{sub}_atlasLocalization.csv"
            cmd =  f"python {atlasLocalizationFunctionDirectory + '/atlasLocalization.py'} {electrodePreopT1Coordinates} {preopT1} {preopT1bet} \
                {MNItemplatePath} {MNItemplateBrainPath} {atlasDirectory} {atlasLabelsPath} {outputDirectorySub} {outputName} {rerun}"
            print(cmd); os.system(cmd)
    #atlasLocalizationBatchProccess(subList, i,  atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory)
    if multiprocess:
        p = multiprocessing.Pool(cores)
        p.starmap(atlasLocalizationBatchProccess, zip(repeat(subList), range(len(subList)),
                                     repeat(atlasLocalizationFunctionDirectory),
                                     repeat(inputDirectory),
                                     repeat(atlasDirectory),
                                     repeat(atlasLabelsPath),
                                     repeat(MNItemplatePath),
                                     repeat(MNItemplateBrainPath),
                                     repeat(outputDirectory)  )   )


def executeAtlasLocalizationSingleSubject(atlasLocalizationFunctionDirectory, electrodePreopT1Coordinates, preopT1, preopT1bet, MNItemplatePath, MNItemplateBrainPath, atlasDirectory, atlasLabelsPath, outputDirectory, outputName, rerun = False):
    cmd =  f"python {atlasLocalizationFunctionDirectory + '/atlasLocalization.py'} {electrodePreopT1Coordinates} {preopT1} {preopT1bet} \
        {MNItemplatePath} {MNItemplateBrainPath} {atlasDirectory} {atlasLabelsPath} {outputDirectory} {outputName} {rerun}"
    utils.executeCommand(cmd)


def atlasLocalizationFromBIDS(BIDS, dataset, sub, ses, acq, sesImplant, acqImplant, electrodeCoordinatesPath, atlasLocalizationFunctionDirectory, MNItemplatePath , MNItemplateBrainPath, atlasDirectory, atlasLabelsPath, outputDirectory, rerun = False ):
    subject_T1 = join(BIDS, dataset, f"sub-{sub}", f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" )
    subject_outputDir = join(BIDS, "derivatives", "atlasLocalization", f"sub-{sub}" , f"ses-{sesImplant}")
    subject_preimplantT1 = join(subject_outputDir, "anat", f"sub-{sub}_ses-{sesImplant}_acq-{acqImplant}_T1w.nii.gz"  )
    subject_preimplantT1_brain = join(subject_outputDir, f"sub-{sub}_ses-{sesImplant}_acq-{acqImplant}brain_T1w.nii.gz" )
    subject_T1_to_preimplantT1 = join(subject_outputDir, f"T1_to_T00_{sub}_mprage.nii.gz" )
    subject_T1_to_preimplantT1_brain = join(subject_outputDir, f"T1_to_T00_{sub}_mprage_brain.nii.gz" )
    subject_biascorrected = join(subject_outputDir, f"sub-{sub}_biascorrected")
    subject_biascorrectedDirectory = join(subject_outputDir, f"sub-{sub}_biascorrected.anat")
    subject_biascorrectedT1 = join(subject_biascorrectedDirectory, "T1_biascorr.nii.gz")
    electrodePreopT1Coordinates = electrodeCoordinatesPath
    outputDirectorysubject = join(outputDirectory, f"sub-{sub}")
    outputName =  f"sub-{sub}_atlasLocalization.csv"
    ###Bias correction
    utils.executeCommand(cmd = f"fsl_anat -i {subject_T1} --noreorient --noreg --nononlinreg --noseg  --nosubcortseg --nocrop --clobber -o {subject_biascorrected}")
    ###Convert 3T preop to T00 space
    utils.executeCommand(cmd = f"flirt -in {subject_biascorrectedT1} -ref {subject_preimplantT1} -dof 6 -out {subject_T1_to_preimplantT1} -omat {subject_T1_to_preimplantT1}_flirt.mat -v")
    ###Getting brain extraction
    getBrainFromMask(subject_T1_to_preimplantT1, subject_preimplantT1_brain, subject_T1_to_preimplantT1_brain)
    executeAtlasLocalizationSingleSubject(atlasLocalizationFunctionDirectory, electrodePreopT1Coordinates, subject_T1_to_preimplantT1, subject_T1_to_preimplantT1_brain, MNItemplatePath, MNItemplateBrainPath, atlasDirectory, atlasLabelsPath, outputDirectorysubject, outputName, rerun = rerun)




def register_MNI_to_preopT1(preop3T, preop3Tbrain, MNItemplatePath, MNItemplateBrainPath, outputMNIname, outputDirectory, preop3Tmask = None, convertToStandard = True ):
    mniBase =  join(outputDirectory, outputMNIname)

    if convertToStandard:
        STDpreop3T = join(outputDirectory, utils.baseSplitextNiiGz(preop3T)[2] + "_std.nii.gz")
        STDpreop3Tbrain = join(outputDirectory, utils.baseSplitextNiiGz(preop3Tbrain)[2] + "_std.nii.gz")
        #convert to std space
        cmd = f"fslreorient2std {preop3T} {STDpreop3T}"; print(cmd);os.system(cmd)
        cmd = f"fslreorient2std {preop3Tbrain} {STDpreop3Tbrain}"; print(cmd);os.system(cmd)
        preop3T = STDpreop3T
        preop3Tbrain = STDpreop3Tbrain

    #linear reg of MNI to preopT1 space
    cmd = f"flirt -in {MNItemplateBrainPath} -ref {preop3Tbrain} -dof 12 -out {mniBase}_flirt -omat {mniBase}_flirt.mat -v"; print(cmd);os.system(cmd)
    #non linear reg of MNI to preopT1 space
    print("\n\nLinear registration of MNI template to image is done\n\nStarting Non-linear registration:\n\n\n")
    if preop3Tmask == None:
        cmd = f"fnirt --in={MNItemplatePath} --ref={preop3T} --aff={mniBase}_flirt.mat --iout={mniBase}_fnirt -v --cout={mniBase}_coef --fout={mniBase}_warp"; print(cmd); os.system(cmd)
        #cmd = f"applywarp -i {MNItemplatePath} -r {STDpreop3T} -w {mniBase}_warp --premat={mniBase}_flirt.mat --interp=nn -o {mniBase}_fnirtapplywarp"; print(cmd); os.system(cmd)
    else:
        STDpreop3Tmask = join(outputDirectory, utils.baseSplitextNiiGz(preop3Tmask)[2] + "_std.nii.gz")
        cmd = f"fslreorient2std {preop3Tmask} {STDpreop3Tmask}"; print(cmd);os.system(cmd)
        cmd = f"fnirt --in={MNItemplatePath} --ref={STDpreop3T} --refmask={preop3Tmask} --aff={mniBase}_flirt.mat --iout={mniBase}_fnirt -v --cout={mniBase}_coef --fout={mniBase}_warp"; print(cmd); os.system(cmd)



def getBrainFromMask(preop3T, preop3TMask, outputName):
    img = nib.load(preop3T)
    imgData = img.get_fdata()  # getting actual image data array

    mask = nib.load(preop3TMask)
    mask_data = mask.get_fdata()
    mask_data[np.where(mask_data >0 )] = imgData[np.where(mask_data >0)]

    brain = nib.Nifti1Image(mask_data, img.affine)
    print(f"Saving to {outputName}")
    nib.save(brain, outputName)

def getExpandedBrainMask(preop3Tmask, output, expansion = 10):
    img = nib.load(preop3Tmask)
    imgData = img.get_fdata()
    imgDataExpand = copy.deepcopy(imgData)
    for i in range(expansion):
        imgDataExpand = ndimage.binary_dilation(imgDataExpand).astype(imgDataExpand.dtype)
    imgExpand = nib.Nifti1Image(imgDataExpand, img.affine)
    nib.save(imgExpand, output)


def applywarp_to_atlas(atlasPaths, preop3T, MNIwarp, outputDirectory, isDir = True):
    if isDir: #crawling through directories
        utils.checkPathError(atlasPaths)
        allpaths = []
        for i,j,y in os.walk(atlasPaths):
            allpaths.append(i)
        #Find all atlases in the atlases folders and their subfolders
        atlasesList = []
        for s in range(len(allpaths)):
            atlasesList = atlasesList +  [f"{allpaths[s]}/" + st for st in [f for f in listdir(allpaths[s]) if isfile(join(allpaths[s], f))]]
    else:
        atlasesList = atlasPaths
    utils.checkPathError(MNIwarp)
    utils.checkPathError(preop3T)
    utils.checkPathError(outputDirectory)
    for i in range(len(atlasesList)):
        atlasName = basename(splitext(splitext(atlasesList[i])[0])[0])
        outputAtlasName = join(outputDirectory, atlasName + ".nii.gz")
        #if not os.path.exists(outputAtlasName):
        if utils.checkIfFileExists(outputAtlasName, returnOpposite=True):
            cmd = f"applywarp -i { atlasesList[i]} -r {preop3T} -w {MNIwarp} --interp=nn -o {outputAtlasName} --verbose"; print(cmd); os.system(cmd)
        #else: print(f"File exists: {outputAtlasName}")



def distance_from_label(electrodePreopT1Coordinates, atlasPath, label, atlasLabelsPath, ofname, sep=",", xColIndex=10, yColIndex=11, zColIndex=12):
    # getting imaging data
    img = nib.load(atlasPath)
    imgData = img.get_fdata()  # getting actual image data array
    affine = img.affine
    # getting electrode coordinates data
    data = pd.read_csv(electrodePreopT1Coordinates, sep=sep, header=None)
    data = data.iloc[:, [0, xColIndex, yColIndex, zColIndex]]

    atlas_labels = pd.read_csv(atlasLabelsPath, sep=",", header=None)
    column_description = "{0}_distance_from_label_{1}".format(atlas_labels.iloc[0,0], label)
    column_names = ['electrode_name', "x_coordinate", "y_coordinate", "z_coordinate", column_description]
    data = data.rename(
        columns={data.columns[0]: column_names[0], data.columns[1]: column_names[1], data.columns[2]: column_names[2],
                 data.columns[3]: column_names[3]})

    coordinates = np.array((data.iloc[:, range(1, 4)]))  # get the scanner coordinates of electrodes
    # transform the real-world coordinates to the atals voxel space. Need to inverse the affine with np.linalg.inv(). To go from voxel to world, just input aff (dont inverse the affine)
    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(affine), coordinates)
    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel
    coordinates_voxels = coordinates_voxels.astype(int)

    try:
        img_ROI = imgData[coordinates_voxels[:,0]-1, coordinates_voxels[:,1]-1, coordinates_voxels[:,2]-1]
    except:
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>imgData.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = -1
                print('Coordinate outside of MNI space: setting to zero')
            elif((coordinates_voxels[i,1]>imgData.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = -1
                print('Coordinate outside of MNI space: setting to zero')
            elif((coordinates_voxels[i,2]>imgData.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = -1
                print('Coordinate outside of MNI space: setting to zero')
            else:
                img_ROI[i] = imgData[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]

    img_ROI = np.reshape(img_ROI, [img_ROI.shape[0], 1])
    distances = copy.deepcopy(img_ROI)
    distances[(distances == 0)] = -1 #if coordinate equals to outside brain, then temporarily set to -1
    distances[(distances == label)] = 0 #if coordinate equals to the label, then it is zero distance

    # list of all points with label
    labelInds = np.where((imgData == label) )

    for i in range(0, distances.shape[0]):
        if ( int(img_ROI[i][0]) != int(label) ):
            point = coordinates_voxels[i, :] - 1 #coordinate trying to find distance to label
            minDist_coord = find_dist_to_label(point, labelInds)
            distances[i] = minDist_coord
            printProgressBar(i+1, img_ROI.shape[0], length = 20, suffix = 'Label: {0}. Point Label: {1} - {2}. Distance: {3} voxels'.format(label, data["electrode_name"][i],img_ROI[i][0] , np.round(minDist_coord,2) ))

    distances = pd.DataFrame(distances)
    data = pd.concat([data, distances], axis=1)
    data = data.rename(columns={data.columns[4]: column_names[4]})

    pd.DataFrame.to_csv(data, ofname, header=True, index=False)


def find_dist_to_label(point, labelInds):
    for i in range(0, labelInds[0].shape[0]):
        dist = np.sqrt((point[0] - labelInds[0][i]) ** 2 + (point[1] - labelInds[1][i]) ** 2 + (
                    point[2] - labelInds[2][i]) ** 2)
        if (i == 0):
            minDist = dist
        else:
            if (dist < minDist):
                minDist = dist
    return (minDist)



def channel2stdCSV(outputTissueCoordinates):
    df = pd.read_csv(outputTissueCoordinates, sep=",", header=0)
    for e in range(len( df  )):
        electrode_name = df.iloc[e]["electrode_name"]
        if (len(electrode_name) == 3): electrode_name = f"{electrode_name[0:2]}0{electrode_name[2]}"
        df.at[e, "electrode_name" ] = electrode_name
    pd.DataFrame.to_csv(df, outputTissueCoordinates, header=True, index=False)


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = "X", printEnd = "\r"):

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

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def show_slices(fname, low = 0.33, middle = 0.5, high = 0.66, save = False, saveFilename = None, isPath = True):

    if isPath:
        img = nib.load(fname)
        imgdata = img.get_fdata()
    else:
        imgdata = fname
     Function to display row of image slices
    slices1 = [   imgdata[:, :, int((imgdata.shape[2]*low)) ] , imgdata[:, :, int(imgdata.shape[2]*middle)] , imgdata[:, :, int(imgdata.shape[2]*high)]   ]
    slices2 = [   imgdata[:, int((imgdata.shape[1]*low)), : ] , imgdata[:, int(imgdata.shape[1]*middle), :] , imgdata[:, int(imgdata.shape[1]*high), :]   ]
    slices3 = [   imgdata[int((imgdata.shape[0]*low)), :, : ] , imgdata[int(imgdata.shape[0]*middle), :, :] , imgdata[int(imgdata.shape[0]*high), :, :]   ]
    slices = [slices1, slices2, slices3]
    plt.style.use('dark_background')
    fig = plt.figure(constrained_layout=False, dpi=300, figsize=(5, 5))
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0, right=1, bottom=0, top=1, wspace=0.00, hspace = 0.00)
    axes = []
    for r in range(3): #standard
        for c in range(3):
            axes.append(fig.add_subplot(gs1[r, c]))
    r = 0; c = 0
    for i in range(9):
        if (i%3 == 0 and i >0): r = r + 1; c = 0
        axes[i].imshow(slices[r][c].T, cmap="gray", origin="lower")
        c = c + 1
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].axis("off")

    if save:
        if saveFilename == None:
            raise Exception("No file name was given to save figures")
        plt.savefig(saveFilename, transparent=True)

#%% Input names

"""