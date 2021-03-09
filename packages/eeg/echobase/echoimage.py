#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:53:19 2020

@author: arevell
"""

#%%
path = "/media/arevell/sharedSSD/linux/papers/paper002" 
import sys
import os
import pandas as pd
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib as plt
from scipy import ndimage
from skimage.measure import regionprops


#%%
"""
ifname = ospj(path, "data", "raw","atlases","atlases","AAL2.nii.gz")
#ifname = ospj(path, "data", "data_raw","atlases","standard_atlases","AAL600.nii.gz")
#ifname = ospj(path, "data", "data_raw","atlases","standard_atlases","craddock_200.nii.gz")
print(os.path.exists(ifname))

#%%

    
img = nib.load(ifname)
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
    
    
centroids_df = pd.DataFrame(coords, columns=("label", "x", "y", "z"))

centroids_df.to_csv(ospj(path, "data", "raw","atlases", "atlasCentroids", "AAL2_centroid.csv"), index=False)
#centroids_df.to_csv(ospj(path, "data", "data_processed","atlas_centroids","AAL600", "AAL600_centroid.csv"), index=False)
#centroids_df.to_csv(ospj(path, "data", "data_processed","atlas_centroids","craddock_200", "craddock_200_centroid.csv"), index=False)
"""