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
import json


import numpy.linalg as npl
from nibabel.affines import apply_affine
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








