#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:04:58 2022

@author: arevell
"""

import sys
import os
import pandas as pd
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib as plt
from scipy import ndimage
from skimage.measure import regionprops

from paths import constants_paths as paths

#%%
atlas_path = paths.ATLAS_AAL2
    
img = nib.load(atlas_path)
nib.aff2axcodes(img.affine)
img = nib.as_closest_canonical(img)
nib.aff2axcodes(img.affine)

data = img.get_fdata() 
data = data.astype(np.int)

label_properties = regionprops(data)
label_properties[0].centroid


coords = np.zeros(shape=(len(label_properties),4))
for l in range(len(label_properties)):
    #print("label: " + str(label_properties[l].label))
    coord = label_properties[l].centroid
    coords[l,0] = label_properties[l].label
    coords[l,1] = coord[0]
    coords[l,2] = coord[1]
    coords[l,3] = coord[2]
    


coordinates_mni_space = nib.affines.apply_affine(img.affine, coords[:,1:])

coords[:,1:] = coordinates_mni_space



    
centroids_df = pd.DataFrame(coords, columns=("label", "x", "y", "z"))


basename = os.path.basename(atlas_path)
basename = os.path.splitext(os.path.splitext(basename)[0])[0]
centroids_df.to_csv(os.path.join( paths.ATLAS_CENTROIDS, f"{basename}_centroid.csv"), index=False)
