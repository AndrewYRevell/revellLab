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


import numpy.linalg as npl
from nibabel.affines import apply_affine




#%%


#Free surfer recon all
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "implantRenders")

sub = "RID0278"
ses = "preop3T"
acq = "3D"
derivativesOutput = "implantRenders"

T1path = join(BIDS, dataset, f"sub-{sub}", f"ses-{ses}", "anat" , f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz")
outputpath = join(BIDS, "derivatives", f"{derivativesOutput}", f"sub-{sub}", f"ses-{ses}")
T1path ,outputpath = elLoc.getBIDSinputAndOutput(BIDS, dataset, sub, ses, acq, derivativesOutput)

elLoc.freesurferReconAll(T1path, outputpath)


T00 = join(BIDS, dataset, f"sub-{sub}", "ses-implant01Reconstruction", "anat", f"sub-{sub}_ses-implant01Reconstruction_acq-T00_T1w.nii.gz" )
T00electrodes = join(BIDS, dataset, f"sub-{sub}", "ses-implant01Reconstruction", "ieeg", f"sub-{sub}_ses-implant01Reconstruction_space-T00_electrodes.tsv" )

utils.checkIfFileExists(T00)
utils.checkIfFileExists(T00electrodes)


#%%
#register T00 to T1

utils.executeCommand(   f"flirt -in {T1path} -ref  {T00} -omat {outputpath}/freesurfer/xform.mat -dof 6 -out {outputpath}/freesurfer/registeredImplant.nii.gz"    )
utils.executeCommand(   f"convert_xfm -omat {outputpath}/freesurfer/xformInverse.mat -inverse {outputpath}/freesurfer/xform.mat"    )

coordinates = pd.read_csv(T00electrodes, sep = "\t")

coordinatesNames = coordinates["name"]

coordinates = coordinates.drop(["name", "size"], axis=1)

coordinates.to_csv(  f"{outputpath}/freesurfer/sub-{sub}_ses-implant01Reconstruction_space-T00_electrodes.txt",  sep=" ", index=False, header=False)


utils.executeCommand( f"img2imgcoord -src {T00} -dest {T1path} -xfm {outputpath}/freesurfer/xformInverse.mat -mm {outputpath}/freesurfer/sub-{sub}_ses-implant01Reconstruction_space-T00_electrodes.txt > {outputpath}/freesurfer/sub-{sub}_ses-implant01Reconstruction_space-other_electrodes.txt"    )




coordinatesOther = pd.read_csv(f"{outputpath}/freesurfer/sub-{sub}_ses-implant01Reconstruction_space-other_electrodes.txt", sep = " ", skiprows=1, header=None)
coordinatesOther.drop(coordinatesOther.columns[[1,3]],axis=1,inplace=True)

coordinatesToSave = pd.concat([coordinatesNames, coordinatesOther ], axis=1, ignore_index=True)


coordinatesToSave.to_csv(  f"{outputpath}/html/electrodes.txt",  sep=" ", index=False, header=False)


#%% combined surfs
utils.executeCommand(   f"cp {outputpath}/freesurfer/surf/lh.pial {outputpath}/freesurfer/"    )
utils.executeCommand(   f"cp {outputpath}/freesurfer/surf/rh.pial {outputpath}/freesurfer/"    )



utils.executeCommand(   f"mris_convert --to-scanner {outputpath}/freesurfer/lh.pial {outputpath}/freesurfer/lh.pial"    )
utils.executeCommand(   f"mris_convert --to-scanner {outputpath}/freesurfer/rh.pial {outputpath}/freesurfer/rh.pial"    )

utils.executeCommand(   f"mris_convert --combinesurfs {outputpath}/freesurfer/rh.pial {outputpath}/freesurfer/lh.pial {outputpath}/freesurfer/combined.stl"    )


binary_stl_path = f"{outputpath}/freesurfer/combined.stl"
out_path = f"{outputpath}/freesurfer/brain.glb"

elLoc.stl_to_gltf(f"{outputpath}/freesurfer/combined.stl", f"{outputpath}/html/brain.glb", True)

import pkg_resources
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
utils.executeCommand(   f"cp -r {tools}/threejs {outputpath}/html"    )
utils.executeCommand(   f"mv -r {outputpath}/html/threejs/index.html {outputpath}/html"    )



#%%


path = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/html/sub-RID0278"


T1PATH = join(path, "sub-RID0278_ses-preop3T_acq-3D_T1w.nii.gz")
T00PATH = join(path, "T00_RID0278_mprage.nii.gz")
coordinatesPATH = join(path, "electrodenames_coordinates_native_and_T1.csv" )
xformPATH = join(path, "xformINV.mat" )



T1img = nib.load(T1PATH)
T1data = T1img.get_fdata() 

T00img = nib.load(T00PATH)
T00data = T00img.get_fdata() 


coordinates = pd.read_csv(coordinatesPATH, header=None)
coordinates = coordinates.iloc[:, [10,11,12]]
coordinates = np.array(coordinates)

utils.show_slices(T1data, isPath=False)


T1img.affine


xform = np.array(pd.read_csv(xformPATH, header=None, sep = "\s\s",engine='python'))





apply_affine(xform, coordinates)[0]






coordVoxels = apply_affine( npl.inv(T00img.affine), coordinates)[0]
apply_affine(xform, coordVoxels)








T00affine = T00img.affine


newaffine = T1img.affine.dot(xform)

apply_affine(newaffine, coordinates)[0]
apply_affine(newaffine, coordVoxels)














coordinatesPATH = join(path, "corrdinates.csv" )

coordinates = pd.read_csv(coordinatesPATH, header=None)
coordinates.to_csv( join(path, "corrdinates2.csv" ), sep=" ", index=False, header=False)

#img2imgcoord -src T00_RID0278_mprage.nii.gz -dest sub-RID0278_ses-preop3T_acq-3D_T1w.nii.gz -xfm xformINV.mat -mm corrdinates2.csv



import csv, os
path = os.path.join ("/media", "arevell", "sharedSSD", "linux", "data", "BIDS" , "derivatives", "html" , "sub-RID0278")
ifname = os.path.join (path, "cords.txt")
os.path.exists(ifname)


import csv, os



with open( ifname ) as csvfile:
    rdr = csv.reader( csvfile, delimiter=' ' )
    for i, row in enumerate( rdr ):
        if i == 0: continue # Skip column titles

        
        x, y, z = row[0], row[2], row[4]
        x = float(x); y = float(y); z = float(z)
        div = 1
        x = x/div
        y = y/div
        z = z/div
        print( (x,y,z  ))
        

    
    




from numpy import genfromtxt
my_data = genfromtxt(ifname, delimiter=' ')








