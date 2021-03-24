#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:37:24 2021

@author: arevell
"""

import sys
import os
import pandas as pd
import copy
from os import listdir
from  os.path import join, isfile
from os.path import splitext, basename
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import ndimage 
import multiprocessing
from itertools import repeat
from revellLab.packages.utilities import utils
from revellLab.packages.imaging.electrodeLocalization import electrodeLocalizationFunctions as elLoc
import json

import smtplib, ssl

import numpy.linalg as npl
from nibabel.affines import apply_affine

import pkg_resources
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")

import time

#%%


#Free surfer recon all
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "implantRenders")





subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]


subjects = [basename(item) for item in subDir ]


for s in range(len(subjects)):

    #sub = "RID0213"
    sub = subjects[s][4:]
    print(sub)

    ses = "preop3T"
    acq = "3D"
    derivativesOutput = "implantRenders"
    
    T1path = join(BIDS, dataset, f"sub-{sub}", f"ses-{ses}", "anat" , f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz")
    T00 = join(BIDS, dataset, f"sub-{sub}", "ses-implant01", "anat", f"sub-{sub}_ses-implant01_acq-T00_T1w.nii.gz" )
    T00electrodes = join(BIDS, dataset, f"sub-{sub}", "ses-implant01", "ieeg", f"sub-{sub}_ses-implant01_space-T00_electrodes.tsv" )
    
    if utils.checkIfFileDoesNotExist(T1path):
        T1path = T00 #if there is no high resolution T1 image, then use the T00 for surface
    
    if utils.checkIfFileExists(T1path) and utils.checkIfFileExists(T00) and utils.checkIfFileExists(T00electrodes):
 
    
    
    #%
        
        outputpath = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{ses}")
        #_ ,outputpath = elLoc.getBIDSinputAndOutput(BIDS, dataset, sub, ses, acq, derivativesOutput)
        utils.checkPathAndMake(f"{outputpath}", f"{outputpath}/html")
        
        
        elLoc.freesurferReconAll(T1path, outputpath,  overwrite = False, threads=12)
        
        if utils.checkIfFileExists(f"{outputpath}/logfile"):
            utils.executeCommand(   f"rm {outputpath}/logfile"    )
        if utils.checkIfFileExists(f"{outputpath}/fsaverage"):
            utils.executeCommand(   f"mv {outputpath}/fsaverage {outputpath}/freesurfer"    )
        
        
        # combined surfs
        utils.executeCommand(   f"cp {outputpath}/freesurfer/surf/lh.pial {outputpath}/freesurfer/"    )
        utils.executeCommand(   f"cp {outputpath}/freesurfer/surf/rh.pial {outputpath}/freesurfer/"    )
        
        
        
        utils.executeCommand(   f"mris_convert --to-scanner {outputpath}/freesurfer/lh.pial {outputpath}/freesurfer/lh.pial"    )
        utils.executeCommand(   f"mris_convert --to-scanner {outputpath}/freesurfer/rh.pial {outputpath}/freesurfer/rh.pial"    )
        
        utils.executeCommand(   f"mris_convert --combinesurfs {outputpath}/freesurfer/rh.pial {outputpath}/freesurfer/lh.pial {outputpath}/freesurfer/combined.stl"    )
        
        cmd = f"blender --background --factory-startup --addons io_scene_gltf2 --python {revellLabPath}/packages/imaging/electrodeLocalization/blender_compress_mesh.py -- -i {outputpath}/freesurfer/combined.stl -o {outputpath}/html/brain.glb"
        utils.executeCommand(cmd)
        
        
        #%
        #register T00 to T1
        
        #brain extraction
        
        utils.executeCommand(   f"bet {T1path} {outputpath}/freesurfer/T1_bet.nii.gz"    )
        utils.executeCommand(   f"bet {T00} {outputpath}/freesurfer/T00_bet.nii.gz"    )
        utils.show_slices(f"{outputpath}/freesurfer/T1_bet.nii.gz", save=True, saveFilename=f"{outputpath}/freesurfer/pic_T1_bet.png")
        utils.show_slices(f"{outputpath}/freesurfer/T00_bet.nii.gz", save=True, saveFilename=f"{outputpath}/freesurfer/pic_T00_bet.png")
        
        if utils.checkIfFileDoesNotExist(f"{outputpath}/freesurfer/registeredImplant_bet.nii.gz"):
            utils.executeCommand(   f"flirt -in {outputpath}/freesurfer/T1_bet.nii.gz -ref {outputpath}/freesurfer/T00_bet.nii.gz -omat {outputpath}/freesurfer/xform.mat -dof 6 -out {outputpath}/freesurfer/registeredImplant_bet.nii.gz"    )
        utils.executeCommand(   f"flirt -in {T1path} -ref {T00} -applyxfm -init {outputpath}/freesurfer/xform.mat -out {outputpath}/freesurfer/registeredImplant.nii.gz"    )
        
        utils.show_slices(f"{outputpath}/freesurfer/registeredImplant.nii.gz", save=True, saveFilename=f"{outputpath}/freesurfer/pic_registeredImplant.png")
        utils.show_slices(f"{T00}", save=True, saveFilename=f"{outputpath}/freesurfer/pic_T00.png")
        
        utils.executeCommand(   f"convert_xfm -omat {outputpath}/freesurfer/xformInverse.mat -inverse {outputpath}/freesurfer/xform.mat"    )
        
        
        #% Coordinates
        
        
        coordinates = pd.read_csv(T00electrodes, sep = "\t")
        
        coordinatesNames = coordinates["name"]
        
        coordinates = coordinates.drop(["name", "size"], axis=1)
        
        coordinates.to_csv(  f"{outputpath}/freesurfer/sub-{sub}_ses-implant01_space-T00_electrodes.txt",  sep=" ", index=False, header=False)
        
        
        utils.executeCommand( f"img2imgcoord -src {T00} -dest {T1path} -xfm {outputpath}/freesurfer/xformInverse.mat -mm {outputpath}/freesurfer/sub-{sub}_ses-implant01_space-T00_electrodes.txt > {outputpath}/freesurfer/sub-{sub}_ses-implant01_space-other_electrodes.txt"    )
        
        
        
        
        coordinatesOther = pd.read_csv(f"{outputpath}/freesurfer/sub-{sub}_ses-implant01_space-other_electrodes.txt", sep = " ", skiprows=1, header=None)
        coordinatesOther.drop(coordinatesOther.columns[[1,3]],axis=1,inplace=True)
        
        
        
        
        
        atlasLocalizationPath = join(BIDS, "derivatives", "atlasLocalization", f"sub-{sub}", f"sub-{sub}_atlasLocalization.csv")
        if utils.checkIfFileExists(atlasLocalizationPath):
            atlasLocalization = pd.read_csv(atlasLocalizationPath)
            tissue = atlasLocalization["tissue_segmentation_region_number"]
        else:
            tissue = np.zeros(shape = (len(coordinatesOther)))
            tissue[:] = -1
            tissue = pd.DataFrame( data=tissue )
        
        
        
        
        coordinatesToSave = pd.concat([coordinatesNames, coordinatesOther, tissue ], axis=1, ignore_index=True)
        
        
        coordinatesToSave.to_csv(  f"{outputpath}/html/electrodes.txt",  sep=" ", index=False, header=False)
        
        
        
        
        #binary_stl_path = f"{outputpath}/freesurfer/combined.stl"
        #out_path = f"{outputpath}/html/brain.glb"
        
        #elLoc.stl_to_gltf(f"{outputpath}/freesurfer/combined.stl", f"{outputpath}/html/brain.glb", True)
        
        
        #%
        
        
        
        
        utils.executeCommand(   f"cp -r {tools}/threejs {outputpath}/html"    )
        utils.executeCommand(   f"mv {outputpath}/html/threejs/index.html {outputpath}/html"    )
        #python -m http.server
        #sudo lsof -t -i tcp:8000 | xargs kill -9
        
        implantRendersGithub = join("/home/arevell/Documents/implantRenders/renders")
        implantRendersGithubsub = join(implantRendersGithub, f"sub-{sub}")
        
        utils.checkPathAndMake(implantRendersGithub, implantRendersGithubsub)
        
        utils.executeCommand(   f"cp {outputpath}/html/brain.glb {implantRendersGithubsub}/"    )
        utils.executeCommand(   f"cp {outputpath}/html/electrodes.txt {implantRendersGithubsub}/"    )

        if utils.checkIfFileDoesNotExist(f"{implantRendersGithubsub}/index.html"):
            utils.executeCommand(   f"cp {outputpath}/html/index.html {implantRendersGithubsub}/"    )
        
        #send email done
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "andyrevell93@gmail.com"  # Enter your address
        receiver_email = "andyrevell93@gmail.com"  # Enter receiver address
        password = "leeunasrrijtfdth"
        message = f"""{sub} is done. {T1path}"""
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
  
        




#%%

#organization of BIDS directory
"""

BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"



atlasLocalizationDirivatives = join(BIDS, "derivatives", "atlasLocalization")






subDir = [os.path.join(atlasLocalizationDirivatives, o) for o in os.listdir(atlasLocalizationDirivatives)   if os.path.isdir(os.path.join(atlasLocalizationDirivatives,o))]


subjects = [basename(item) for item in subDir ]


for i in range(len(subjects)):
    
    sub = subjects[i]
    subRID = sub[4:]
    

    
    if utils.checkIfFileDoesNotExist( join(BIDS, dataset, sub, "ses-implant01" )  ):
        pathtomake = join(BIDS, dataset, "sub-RIDXXXX", "ses-implant01")
        pathtomake2 = join(BIDS, dataset, f"{sub}")
        
        utils.executeCommand(  f"cp -r {pathtomake} {pathtomake2} "    )
    
    
    T00 = join(atlasLocalizationDirivatives, f"{sub}", f"T00_{subRID}_mprage.nii.gz" )
    if utils.checkIfFileExists(T00):
        
        outname = join(BIDS, dataset, f"{sub}", "ses-implant01", "anat", f"{sub}_ses-implant01_acq-T00_T1w.nii.gz" )
        utils.executeCommand(  f"cp {T00} {outname}"    )
    
    coordinatesPath = join(atlasLocalizationDirivatives, f"{sub}", "electrodenames_coordinates_native_and_T1.csv" )
    
    if utils.checkIfFileExists(coordinatesPath):
        coordinates = pd.read_csv(coordinatesPath, sep = ",", header=None)
    
        
        coordinatesEdit = coordinates.iloc[:,[0, 10, 11, 12]]
        
        size = np.zeros(( len(coordinates)))
        size[:] = 1
        coordinatesEdit = pd.concat(    [coordinatesEdit,  pd.DataFrame(size  )  ] , axis= 1 )
        coordinatesEdit.columns = ["name", "x", "y", "z", "size"]
        outnameCoordinates = join(BIDS, dataset, f"{sub}", "ses-implant01", "ieeg", f"{sub}_ses-implant01_space-T00_electrodes.tsv" )
        coordinatesEdit.to_csv(  outnameCoordinates,  sep="\t", index=False, header=True)
    
    
    
    #deletepath = join(BIDS, dataset, f"{sub}", "ses-implant01Reconstruction")
    #if utils.checkIfFileExists(deletepath):
    #    utils.executeCommand(  f"rm -r {deletepath}"    )
    
    
    
    






"""




