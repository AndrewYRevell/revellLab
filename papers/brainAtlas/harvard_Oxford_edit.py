#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:52:16 2020

@author: arevell
"""


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import copy
path = "/mnt"
ifpath = os.path.join(path, "data/data_raw/atlases/original_atlases_from_source")
reference = os.path.join(ifpath, "MNI-maxprob-thr25-1mm.nii.gz")
identity_matrix_1 = os.path.join(path, "data/data_raw/atlases/atlas_labels_description", "identity_matrix_to_transform_atlases_to_1mm_MNI152_space_1.mat")#NOTE: the identity file is a txt file, NOT a matlab file. it is just a numpy.identity(4) matrix. 
identity_matrix_2 = os.path.join(path,"data/data_raw/atlases/atlas_labels_description", "identity_matrix_to_transform_atlases_to_1mm_MNI152_space_2.mat")#NOTE: the identity file is a txt file, NOT a matlab file. it is just a numpy.identity(4) matrix. 

ofpath = os.path.join(path,"data/data_raw/atlases/standard_atlases")
ofpath_canonical = os.path.join(path,"data/data_raw/atlases/original_atlases_in_RAS_orientation")
#%%

name_HO_cort = "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
name_HO_subcort = "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"



ifname_HO_cort = os.path.join(ifpath, name_HO_cort)
ifname_HO_subcort = os.path.join(ifpath, name_HO_subcort)


img_HO_cort = nib.load(ifname_HO_cort)
data_HO_cort = img_HO_cort.get_fdata() 
print(nib.aff2axcodes(img_HO_cort.affine))

img_HO_subcort = nib.load(ifname_HO_subcort)
data_HO_subcort = img_HO_subcort.get_fdata() 
print(nib.aff2axcodes(img_HO_subcort.affine))




tmp =data_HO_cort[:,:,100]
plt.imshow(tmp, cmap="gray_r",  origin="lower")

tmp =data_HO_subcort[:,:,100]
plt.imshow(tmp, cmap="gray_r",  origin="lower")



#removing gray and white matter from subcortical
data_HO_subcort[np.where(data_HO_subcort == 1)] =0
data_HO_subcort[np.where(data_HO_subcort == 2)] =0
data_HO_subcort[np.where(data_HO_subcort == 12)] =0
data_HO_subcort[np.where(data_HO_subcort == 13)] =0

img_HO_subcort_new = nib.Nifti1Image(data_HO_subcort, img_HO_subcort.affine)

nib.save(img_HO_subcort_new, os.path.join(ofpath, "HarvardOxford-sub-ONLY_maxprob-thr25-1mm.nii.gz"))


#making HO non-symmetric

data_HO_cort_nonsymm = copy.deepcopy(data_HO_cort)
np.max(data_HO_cort_nonsymm)
for i in range(91, data_HO_cort_nonsymm.shape[0] ):
    for j in range(data_HO_cort_nonsymm.shape[1]):
        for k in range(data_HO_cort_nonsymm.shape[2]):
            data_HO_cort_nonsymm[i,j,k] = data_HO_cort_nonsymm[i,j,k]+100
data_HO_cort_nonsymm[np.where(data_HO_cort_nonsymm == 100)] =0

tmp =data_HO_cort_nonsymm[:,:,100]
plt.imshow(tmp, cmap="gray_r",  origin="lower")
img_HO_cort_nonsymm_new = nib.Nifti1Image(data_HO_cort_nonsymm, img_HO_cort.affine)

nib.save(img_HO_cort_nonsymm_new, os.path.join(ofpath, "HarvardOxford-cort-NONSYMMETRIC-maxprob-thr25-1mm.nii.gz"))

            
            

#combining HO


data_HO_combined = copy.deepcopy(data_HO_cort_nonsymm)


data_HO_subcort[np.where(data_HO_subcort > 0)]
data_HO_combined[np.where(data_HO_subcort > 0)] = data_HO_subcort[np.where(data_HO_subcort > 0)] + 200

img_HO_Combined = nib.Nifti1Image(data_HO_combined, img_HO_cort.affine)
tmp =data_HO_combined[:,100,:]
plt.imshow(tmp, cmap="gray_r",  origin="lower")
nib.save(img_HO_Combined, os.path.join(ofpath, "HarvardOxford-combined.nii.gz"))



















