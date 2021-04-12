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
import subprocess
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
from scipy.ndimage import morphology
from nibabel.affines import apply_affine
from os.path import join, isfile, splitext, basename

from revellLab.packages.utilities import utils
from revellLab.packages.imaging.electrodeLocalization import electrodeLocalizationFunctions as elLoc
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")



#%
#Free surfer recon all
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "implantRenders")
freesurferReconAllDir = join(BIDS, "derivatives", "freesurferReconAll")

subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]
subjects = [basename(item) for item in subDir ]


OASIStemplatePath = join( tools, "templates","OASIS" ,"T_template0.nii.gz")
OASISprobabilityPath = join( tools, "templates", "OASIS", "T_template0_BrainCerebellumProbabilityMask.nii.gz")


#%%
implantList = ["implant01", "RNS01"]

for imp in range(len(implantList)):
    implantName = implantList[imp]
    preferredSurface = "research3T*" 
    
    for s in range(0,len(subjects)):
    
        #sub = "RID0030"
        sub = subjects[s][4:]
        
        print(sub)
        
        derivativesOutput = "implantRenders"
        
        T00 = join(BIDS, dataset, f"sub-{sub}", f"ses-{implantName}", "anat", f"sub-{sub}_ses-{implantName}_acq-T00_*T1w.nii.gz" )
        T00electrodes = join(BIDS, dataset, f"sub-{sub}", f"ses-{implantName}", "ieeg", f"sub-{sub}_ses-{implantName}_space-T00_electrodes.tsv" )
        if utils.checkIfFileExistsGlob(T00) and utils.checkIfFileExistsGlob(T00electrodes): 
            #check if freesurfer has been run on preferred session
            if utils.checkIfFileExistsGlob( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{preferredSurface}", "freesurfer", "surf", "lh.pial" ) ) and utils.checkIfFileExistsGlob( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{preferredSurface}", "freesurfer", "surf", "rh.pial" ) ):
                    sessionPath = np.array(glob( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{preferredSurface}")))[0]
                    session = basename(sessionPath)[4:]
                    outputpath = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{session}")
                    utils.checkPathAndMake(outputpath, join(outputpath, "freesurfer", 'mri' ))
                    utils.checkPathAndMake(outputpath, join(outputpath, "freesurfer", 'surf' ))
                    utils.checkPathAndMake(outputpath, join(outputpath, "html"))
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'mri' , 'orig_nu.mgz')}  {join(outputpath, 'freesurfer', 'mri' )} ")
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'mri' , 'brain.mgz')}  {join(outputpath, 'freesurfer', 'mri' )} ")
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'surf' , 'lh.pial')}  {join(outputpath, 'freesurfer', 'surf')} ")
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'surf' , 'rh.pial')}  {join(outputpath, 'freesurfer', 'surf')} ")
                    
            #check if freesurfer has been run on implant session
            elif utils.checkIfFileExists( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{implantName}", "freesurfer", "surf", "lh.pial" ) ) and utils.checkIfFileExists( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{implantName}", "freesurfer", "surf", "rh.pial" ) ):
                    sessionPath = np.array(glob( join(freesurferReconAllDir, f"sub-{sub}", f"ses-{implantName}")))[0]
                    session = basename(sessionPath)[4:]
                    outputpath = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{session}")
                    utils.checkPathAndMake(outputpath, join(outputpath, "freesurfer", 'mri' ))
                    utils.checkPathAndMake(outputpath, join(outputpath, "freesurfer", 'surf' ))
                    utils.checkPathAndMake(outputpath, join(outputpath, "html"))
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'mri' , 'orig_nu.mgz')}  {join(outputpath, 'freesurfer', 'mri' )} ")
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'mri' , 'brain.mgz')}  {join(outputpath, 'freesurfer', 'mri' )} ")
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'surf' , 'lh.pial')}  {join(outputpath, 'freesurfer', 'surf')} ")
                    utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'surf' , 'rh.pial')}  {join(outputpath, 'freesurfer', 'surf')} ")
            else:
                continue
            #check if lh and rh.pial have been moved to implantRenders derivatives path properly 
            if utils.checkIfFileDoesNotExist(f"{outputpath}/html/brain.glb", returnOpposite=False):
                if utils.checkIfFileExists(f"{join(outputpath, 'freesurfer', 'surf' ,'lh.pial')}" ) and utils.checkIfFileExists( f"{join(outputpath, 'freesurfer', 'surf' ,'rh.pial')}") :       
                        # combined surfs
                        utils.executeCommand(   f"cp {outputpath}/freesurfer/surf/lh.pial {outputpath}/freesurfer/"    )
                        utils.executeCommand(   f"cp {outputpath}/freesurfer/surf/rh.pial {outputpath}/freesurfer/"    )
                        utils.executeCommand(   f"mris_convert --to-scanner {outputpath}/freesurfer/lh.pial {outputpath}/freesurfer/lh.pial"    )
                        utils.executeCommand(   f"mris_convert --to-scanner {outputpath}/freesurfer/rh.pial {outputpath}/freesurfer/rh.pial"    )
                        utils.executeCommand(   f"mris_convert --combinesurfs {outputpath}/freesurfer/rh.pial {outputpath}/freesurfer/lh.pial {outputpath}/freesurfer/combined.stl"    )
                        cmd = f"blender --background --factory-startup --addons io_scene_gltf2 --python {revellLabPath}/packages/imaging/electrodeLocalization/blender_compress_mesh.py -- -i {outputpath}/freesurfer/combined.stl -o {outputpath}/html/brain.glb"
                        utils.executeCommand(cmd)          
                        utils.executeCommand(f"cp {join(sessionPath, 'freesurfer', 'mri' , 'aseg.mgz')}  {join(outputpath, 'freesurfer', 'mri' )} ")
                        outputDirTMP = join(outputpath, "freesurfer")      
                        T1 =join(outputpath, 'freesurfer', 'mri', 'orig_nu.mgz' )
                        T1brain =join(outputpath, 'freesurfer', 'mri', 'brain.mgz' )
                        aseg =join(outputpath, 'freesurfer', 'mri', 'aseg.mgz' )
                        T00 = glob(T00)[0]
    
                        if utils.checkIfFileDoesNotExist(f"{join(outputpath, 'freesurfer', 'mri', 'aseg.mgz')}"):
                            continue
                        ####      MAIN FUNCTION  #########
                        ##################################
                        ##################################
                        ##################################
                        atl.tissueLocalizationFromReconAll(T1, T1brain, aseg, T00, T00electrodes, outputDirTMP)
                        ##################################
                        ##################################
                        ##################################
                        #% HTML files
                        utils.executeCommand(   f"cp -r {tools}/threejs {outputpath}/html"    )
                        utils.executeCommand(   f"mv {outputpath}/html/threejs/index.html {outputpath}/html"    )
                        utils.executeCommand(   f"cp {outputDirTMP}/electrodes.txt {outputpath}/html/"    )
                        #to check locally:
                        #python -m http.server
                        #sudo lsof -t -i tcp:8000 | xargs kill -9
                        
                        implantRendersGithub = join("/home/arevell/Documents/implantRenders/renders")
                        implantRendersGithubsub = join(implantRendersGithub, f"sub-{sub}", f"ses-{session}")
                        utils.checkPathAndMake(implantRendersGithub, implantRendersGithubsub)
                        utils.executeCommand(   f"cp {outputpath}/html/brain.glb {implantRendersGithubsub}/"    )
                        utils.executeCommand(   f"cp {outputpath}/html/electrodes.txt {implantRendersGithubsub}/"    )
                
                        if utils.checkIfFileDoesNotExist(f"{implantRendersGithubsub}/index.html"):
                            utils.executeCommand(   f"cp {outputpath}/html/index.html {implantRendersGithubsub}/"    )
                          
    
        
                        
                        
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                












#%%
















#%%


