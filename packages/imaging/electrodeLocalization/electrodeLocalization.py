#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:37:24 2021

@author: arevell
"""

import os
import sys
import time
import json
import copy
import pkg_resources
import multiprocessing
import smtplib, ssl

import numpy as np
import pandas as pd
import nibabel as nib
import numpy.linalg as npl
import matplotlib.pyplot as plt

from glob import glob
from os import listdir
from scipy import ndimage 
from itertools import repeat
from nibabel.affines import apply_affine
from os.path import join, isfile, splitext, basename

from revellLab.packages.utilities import utils
from revellLab.packages.imaging.electrodeLocalization import electrodeLocalizationFunctions as elLoc
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")



#%%
#Free surfer recon all
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "implantRenders")

subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]
subjects = [basename(item) for item in subDir ]

#%%
for s in range(50, 100):

    #sub = "RID0386"
    sub = subjects[s][4:]
    print(sub)
    ses = "research3T" #session for best quality mprage
    acq = "3D" #acquisition description for the mprage. 
    derivativesOutput = "implantRenders"
    
    T1highres= join(BIDS, dataset, f"sub-{sub}", f"ses-{ses}*", "anat" , f"sub-{sub}_ses-{ses}*_acq-3D_*T1w.nii.gz")
    T00 = join(BIDS, dataset, f"sub-{sub}", "ses-implant01", "anat", f"sub-{sub}_ses-implant01_acq-T00_*T1w.nii.gz" )
    T00electrodes = join(BIDS, dataset, f"sub-{sub}", "ses-implant01", "ieeg", f"sub-{sub}_ses-implant01_space-T00_electrodes.tsv" )
    
    #if there is no high resolution T1 image, then use the T00 for surface
    if utils.checkIfFileDoesNotExistGlob(T1highres):
        T1path = T00
    else:
        T1path = T1highres
    print(all([utils.checkIfFileExistsGlob(T1path) , utils.checkIfFileExistsGlob(T00) , utils.checkIfFileExistsGlob(T00electrodes)]))
    if utils.checkIfFileExistsGlob(T1path) and utils.checkIfFileExistsGlob(T00) and utils.checkIfFileExistsGlob(T00electrodes):
        
        session = basename(glob(T1path)[0]).partition("ses-")[2].partition("_")[0]
        outputpath = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{session}")
        utils.checkPathAndMake(f"{outputpath}", f"{outputpath}/html")
        
        if utils.checkIfFileDoesNotExist(f"{outputpath}/html/brain.glb"): 
            elLoc.freesurferReconAll(T1path, outputpath,  overwrite = False, threads=12)
            if utils.checkIfFileExists(f"{outputpath}/logfile"):
                utils.executeCommand(   f"mv {outputpath}/logfile {outputpath}/freesurfer"    )
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
            
            
        #% register T00 to T1
        #brain extraction
        if utils.checkIfFileDoesNotExist(f"{outputpath}/freesurfer/xformInverse.mat"):
            utils.executeCommand(   f"bet {T1path} {outputpath}/freesurfer/T1_bet.nii.gz"    )
            utils.executeCommand(   f"bet {T00} {outputpath}/freesurfer/T00_bet.nii.gz"    )
            utils.show_slices(f"{outputpath}/freesurfer/T1_bet.nii.gz", save=True, saveFilename=f"{outputpath}/freesurfer/pic_T1_bet.png")
            utils.show_slices(f"{outputpath}/freesurfer/T00_bet.nii.gz", save=True, saveFilename=f"{outputpath}/freesurfer/pic_T00_bet.png")
            
            utils.executeCommand(   f"flirt -in {outputpath}/freesurfer/T1_bet.nii.gz -ref {outputpath}/freesurfer/T00_bet.nii.gz -omat {outputpath}/freesurfer/xform.mat -dof 6 -out {outputpath}/freesurfer/registeredImplant_bet.nii.gz"    )
            utils.executeCommand(   f"flirt -in {T1path} -ref {T00} -applyxfm -init {outputpath}/freesurfer/xform.mat -out {outputpath}/freesurfer/registeredImplant.nii.gz"    )
        
            utils.show_slices(f"{outputpath}/freesurfer/registeredImplant.nii.gz", save=True, saveFilename=f"{outputpath}/freesurfer/pic_registeredImplant.png")
            utils.show_slices(f"{T00}", save=True, saveFilename=f"{outputpath}/freesurfer/pic_T00.png")
        
            utils.executeCommand(   f"convert_xfm -omat {outputpath}/freesurfer/xformInverse.mat -inverse {outputpath}/freesurfer/xform.mat"    )
              
        #% Coordinates
        if utils.checkIfFileDoesNotExist( f"{outputpath}/html/electrodes.txt"):
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

        #% HTML files
        utils.executeCommand(   f"cp -r {tools}/threejs {outputpath}/html"    )
        utils.executeCommand(   f"mv {outputpath}/html/threejs/index.html {outputpath}/html"    )
        #python -m http.server
        #sudo lsof -t -i tcp:8000 | xargs kill -9
        
        implantRendersGithub = join("/home/arevell/Documents/implantRenders/renders")
        implantRendersGithubsub = join(implantRendersGithub, f"sub-{sub}", f"ses-{session}")
        
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

#RNS

BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "implantRenders")



subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]


subjects = [basename(item) for item in subDir ]


for s in range(len(subjects)):

    #sub = "RID0206"
    sub = subjects[s][4:]
    print(sub)

    ses = "preop3T"
    acq = "3D"
    derivativesOutput = "implantRenders"
    
    T1path = join(BIDS, dataset, f"sub-{sub}", f"ses-{ses}", "anat" , f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz")
    
    T00implant = join(BIDS, dataset, f"sub-{sub}", "ses-implant01", "anat", f"sub-{sub}_ses-implant01_acq-T00_T1w.nii.gz" )
    
    T00 = join(BIDS, dataset, f"sub-{sub}", "ses-RNS01", "anat", f"sub-{sub}_ses-RNS01_acq-T00_T1w.nii.gz" )
    T00electrodes = join(BIDS, dataset, f"sub-{sub}", "ses-RNS01", "ieeg", f"sub-{sub}_ses-RNS01_space-T00_electrodes.tsv" )
    
    if utils.checkIfFileDoesNotExist(T1path):
        if utils.checkIfFileDoesNotExist(T00implant):
            T1path = T00 #if there is no high resolution T1 image, then use the T00 for surface
            T1pathImplant = T00implant
        else:
            T1path = T00implant
        #for sub-RID0186 T1path = join(BIDS, dataset,f"sub-{sub}", "ses-implant01", "anat", f"sub-{sub}_ses-implant01_acq-SAGGITAL_T1w.nii.gz")
    
    if utils.checkIfFileExists(T1path) and utils.checkIfFileExists(T00) and utils.checkIfFileExists(T00electrodes):

        outputpath = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", "ses-RNS01")
       
        #_ ,outputpath = elLoc.getBIDSinputAndOutput(BIDS, dataset, sub, ses, acq, derivativesOutput)
        utils.checkPathAndMake(f"{outputpath}", f"{outputpath}/html")
        utils.checkPathAndMake(f"{outputpath}", f"{outputpath}/freesurfer")
        
        
        
        
        if utils.checkIfFileDoesNotExist( join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{ses}", "html/brain.glb")):
            
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
        else:
            copyfile = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{ses}", "html/brain.glb")
            utils.executeCommand( f"cp {copyfile} {outputpath}/html")
            
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
            #%%Begin Pipeline: Orient all images to standard RAS

            print("\n\nReorientation of Images\nEstimated time: 10-30 seconds\nReorient all images to standard RAS\n")
            cmd = f"fslreorient2std {T1path} {outputpath}/freesurfer/T1_std.nii.gz"; print(cmd); os.system(cmd)
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
            
            cmd = f"run_first_all -i {preopT1bet_output}_std1x1x1.nii.gz -o {preopT1bet_output}_std1x1x1.nii.gz -b -v"; print(cmd); os.system(cmd)
            #clean up files
            cmd = f"rm -r {preopT1bet_output}_std1x1x1.logs"; print(cmd); os.system(cmd)
            cmd = f"rm -r {preopT1bet_output}_std1x1x1*.bvars"; print(cmd); os.system(cmd)
            cmd = f"rm -r {preopT1bet_output}_std1x1x1*.vtk"; print(cmd); os.system(cmd)
            cmd = f"rm -r {preopT1bet_output}_std1x1x1*origsegs*"; print(cmd); os.system(cmd)
            cmd = f"rm -r {preopT1bet_output}_std1x1x1_to_std*"; print(cmd); os.system(cmd)
            cmd = f"rm -r {preopT1bet_output}_std1x1x1*.com*"; print(cmd); os.system(cmd)
            cmd = f"mv {preopT1bet_output}_std1x1x1_all_fast_firstseg.nii.gz {FIRST}"; print(cmd); os.system(cmd)
            
            
            
            print("\n\n\nRUNNING FAST SEGMENTATION\n")
                
            #FAST: segmentation of cortex
            
            cmd = f"fast -n 3 -H 0.25 -t 1 -v {preopT1bet_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)
            #Clean up files
            cmd = f"rm -r {preopT1bet_output}_std1x1x1_*mixeltype*"; print(cmd); os.system(cmd)
            cmd = f"rm -r {preopT1bet_output}_std1x1x1_*pve*"; print(cmd); os.system(cmd)
            cmd = f"mv {preopT1bet_output}_std1x1x1_seg.nii.gz {FAST}"; print(cmd); os.system(cmd)
            
            print(f"File exists:\n{FAST}")
            
            
            #Combine FIRST and FAST images
            atl.combine_first_and_fast(FIRST, FAST, outputNameTissueSeg)
            utils.show_slices(outputNameTissueSeg, low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_tissueSegmentation.png"))
            print(f"\nPictures of FIRST + FAST combined images (pic_tissueSegmentation.png) are saved to {outputDirectory}\nPlease check for quality assurance")
            
            
                        
                        
                        
            
            
            
            
            
            
            
            
            
            
            
            tissue = np.zeros(shape = (len(coordinatesOther)))
            tissue[:] = -1
            tissue = pd.DataFrame( data=tissue )

        coordinatesToSave = pd.concat([coordinatesNames, coordinatesOther, tissue ], axis=1, ignore_index=True)

        coordinatesToSave.to_csv(  f"{outputpath}/html/electrodes.txt",  sep=" ", index=False, header=False)

        utils.executeCommand(   f"cp -r {tools}/threejs {outputpath}/html"    )
        utils.executeCommand(   f"mv {outputpath}/html/threejs/index.html {outputpath}/html"    )
        #python -m http.server
        #sudo lsof -t -i tcp:8000 | xargs kill -9
        
        implantRendersGithub = join("/home/arevell/Documents/implantRenders/renders")
        implantRendersGithubsub = join(implantRendersGithub, f"sub-{sub}_RNS")
        
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
      
            


