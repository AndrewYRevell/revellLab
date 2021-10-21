#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:23:28 2020

@author: arevell
"""

#combine subcortical segmentation (FIST) and regular segmentation (FAST)

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


"""
path = "/media/arevell/sharedSSD1/linux/papers/paper005" #Parent directory of project
ifpath = os.path.join(path, "data/data_raw/electrode_localization/sub-RID0278/ses-preop3T_registration")


ifname_first = os.path.join(ifpath, "preop3T_to_T00_nonlinear_bet_first_all_fast_firstseg.nii.gz")
ifname_fast = os.path.join(ifpath, "preop3T_to_T00_nonlinear_bet_seg.nii.gz")
"""

def combine_first_and_fast(ifname_first, ifname_fast, ofname_FIRST_FAST_COMBINED):

    img_first = nib.load(ifname_first)
    data_first = img_first.get_fdata() 
    img_fast= nib.load(ifname_fast)
    data_fast = img_fast.get_fdata() 

    #make all subcortical structures = 2 (2 is the GM category)
    data_first[np.where(data_first > 0)] = 2
    #replace fast images with first images where not zero
    data_fast[np.where(data_first > 0)] = data_first[np.where(data_first > 0)] 
    #plot
    show_slices(data_fast)

    img_first_fast = nib.Nifti1Image(data_fast, img_fast.affine)
    nib.save(img_first_fast, ofname_FIRST_FAST_COMBINED)


def show_slices(img_data, low = 0.33, middle = 0.5, high = 0.66):
    """ Function to display row of image slices """
    slices1 = [   img_data[:, :, int((img_data.shape[2]*low)) ] , img_data[:, :, int(img_data.shape[2]*middle)] , img_data[:, :, int(img_data.shape[2]*high)]   ]
    slices2 = [   img_data[:, int((img_data.shape[1]*low)), : ] , img_data[:, int(img_data.shape[1]*middle), :] , img_data[:, int(img_data.shape[1]*high), :]   ]
    slices3 = [   img_data[int((img_data.shape[0]*low)), :, : ] , img_data[int(img_data.shape[0]*middle), :, :] , img_data[int(img_data.shape[0]*high), :, :]   ]
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





