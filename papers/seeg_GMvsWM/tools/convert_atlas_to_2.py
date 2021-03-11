#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:23:28 2020

@author: arevell
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import copy
path = "/media/arevell/sharedSSD1/linux/papers/paper005" #Parent directory of project
ifpath = os.path.join(path, "data/data_raw/electrode_localization/sub-RID0278/ses-preop3T_registration")

ifname_atlas = os.path.join(ifpath, "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz")
identity_matrix_1 = os.path.join(path, "data/data_raw/atlases/atlas_labels_description", "identity_matrix_to_transform_atlases_to_1mm_MNI152_space_1.mat")#NOTE: the identity file is a txt file, NOT a matlab file. it is just a numpy.identity(4) matrix. 
identity_matrix_2 = os.path.join(path,"data/data_raw/atlases/atlas_labels_description", "identity_matrix_to_transform_atlases_to_1mm_MNI152_space_2.mat")#NOTE: the identity file is a txt file, NOT a matlab file. it is just a numpy.identity(4) matrix. 

ofpath = os.path.join(ifpath,"HO_sub_SEG_CLASS.nii.gz")



img = nib.load(ifname_atlas)
data = img.get_fdata() 

data_seg = copy.deepcopy(data)
data_seg[np.where(data_seg == 12)] = -1
data_seg[np.where(data_seg == 1)] = -1
data_seg[np.where(data_seg == 3)] = -2
data_seg[np.where(data_seg == 14)] = -2

data_seg[np.where(data_seg > 0)] = 2
data_seg[np.where(data_seg == -1)] = 3
data_seg[np.where(data_seg == -2)] = 1



tmp =data_seg[:,:,80]
plt.imshow(tmp, cmap="gray",  origin="lower")


img_HO_subcort_new = nib.Nifti1Image(data_seg, img.affine)

nib.save(img_HO_subcort_new, ofpath)




#combine First subcort and regular segmentation

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import copy
path = "/media/arevell/sharedSSD1/linux/papers/paper005" #Parent directory of project
ifpath = os.path.join(path, "data/data_raw/electrode_localization/sub-RID0278/ses-preop3T_registration")


ifname_first = os.path.join(ifpath, "preop3T_to_T00_nonlinear_bet_first_all_fast_firstseg.nii.gz")
ifname_fast = os.path.join(ifpath, "preop3T_to_T00_nonlinear_bet_seg.nii.gz")


img_first = nib.load(ifname_first)
data_first = img_first.get_fdata() 

img_fast= nib.load(ifname_fast)
data_fast = img_fast.get_fdata() 



tmp =data_first[:,:,80]
plt.imshow(tmp, cmap="gray",  origin="lower")



data_first[np.where(data_first > 0)] = 2


data_fast[np.where(data_first > 0)] = data_first[np.where(data_first > 0)] 


tmp =data_fast[:,:,80]
plt.imshow(tmp, cmap="gray",  origin="lower")

img_first_fast = nib.Nifti1Image(data_fast, img_fast.affine)

ofpath = os.path.join(ifpath,"preop3T_to_T00_nonlinear_bet_seg_FIST_FAST.nii.gz")

nib.save(img_first_fast, ofpath)

