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
path = "/media/arevell/sharedSSD/linux/papers/paper005" 
ifpath_atlases = os.path.join(path, "data/data_raw/atlases")
ifpath_MNI = os.path.join(path, "data/data_raw/MNI_brain_template")

ofpath_atlases =  ifpath_atlases

#%%


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
        
#%%
#Harvard Oxford
name_HO_cort = "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
name_HO_subcort = "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
name_AAL = "AAL2.nii.gz"
name_JHU = "JHU-ICBM-labels-1mm.nii.gz"



ifname_HO_cort = os.path.join(ifpath_atlases,"original_atlases_from_source", name_HO_cort)
ifname_HO_subcort = os.path.join(ifpath_atlases,"original_atlases_from_source", name_HO_subcort)
ifname_AAL = os.path.join(ifpath_atlases,"atlases", name_AAL)
ifname_JHU = os.path.join(ifpath_atlases,"atlases", name_JHU)


img_HO_cort = nib.load(ifname_HO_cort)
data_HO_cort = img_HO_cort.get_fdata() 
print(nib.aff2axcodes(img_HO_cort.affine))
show_slices(data_HO_cort)

img_HO_subcort = nib.load(ifname_HO_subcort)
data_HO_subcort = img_HO_subcort.get_fdata() 
print(nib.aff2axcodes(img_HO_subcort.affine))
show_slices(data_HO_subcort)



#removing gray and white matter from subcortical
data_HO_subcort[np.where(data_HO_subcort == 1)] =0
data_HO_subcort[np.where(data_HO_subcort == 2)] =0
data_HO_subcort[np.where(data_HO_subcort == 12)] =0
data_HO_subcort[np.where(data_HO_subcort == 13)] =0

img_HO_subcort_new = nib.Nifti1Image(data_HO_subcort, img_HO_subcort.affine)
show_slices(data_HO_subcort)
nib.save(img_HO_subcort_new, os.path.join(ofpath_atlases, "HarvardOxford-sub-ONLY_maxprob-thr25-1mm.nii.gz"))


#making HO non-symmetric

data_HO_cort_nonsymm = copy.deepcopy(data_HO_cort)
np.max(data_HO_cort_nonsymm)
for i in range(91, data_HO_cort_nonsymm.shape[0] ):
    for j in range(data_HO_cort_nonsymm.shape[1]):
        for k in range(data_HO_cort_nonsymm.shape[2]):
            data_HO_cort_nonsymm[i,j,k] = data_HO_cort_nonsymm[i,j,k]+100
data_HO_cort_nonsymm[np.where(data_HO_cort_nonsymm == 100)] =0
show_slices(data_HO_cort_nonsymm)

img_HO_cort_nonsymm_new = nib.Nifti1Image(data_HO_cort_nonsymm, img_HO_cort.affine)

nib.save(img_HO_cort_nonsymm_new, os.path.join(ofpath_atlases, "HarvardOxford-cort-NONSYMMETRIC-maxprob-thr25-1mm.nii.gz"))

            
            

#combining HO


data_HO_combined = copy.deepcopy(data_HO_cort_nonsymm)


data_HO_subcort[np.where(data_HO_subcort > 0)]
data_HO_combined[np.where(data_HO_subcort > 0)] = data_HO_subcort[np.where(data_HO_subcort > 0)] + 200

img_HO_Combined = nib.Nifti1Image(data_HO_combined, img_HO_cort.affine)
show_slices(data_HO_combined)
nib.save(img_HO_Combined, os.path.join(ofpath_atlases, "HarvardOxford-combined.nii.gz"))




#%%

#making AAL-JHU

img_AAL = nib.load(ifname_AAL)
data_AAL = img_AAL.get_fdata() 
print(nib.aff2axcodes(img_AAL.affine))

img_JHU = nib.load(ifname_JHU)
data_JHU = img_JHU.get_fdata() 
print(nib.aff2axcodes(img_JHU.affine))


data_AALJHU = copy.deepcopy(data_AAL)


np.max(data_JHU)

np.max(data_AAL)

data_JHU[np.where(data_JHU >0) ] = data_JHU[np.where(data_JHU >0) ]+10000


data_AALJHU[np.where(data_JHU >0) ] = data_JHU[np.where(data_JHU >0) ]


show_slices(data_AALJHU)

img_AALJHU = nib.Nifti1Image(data_AALJHU, img_AAL.affine)
nib.save(img_AALJHU, os.path.join(ofpath_atlases, "AAL_JHU_combined.nii.gz"))














#%%

#editing MNI and CerebrA




ifname_MNI2009 = os.path.join(ifpath_MNI, "mni_icbm152_t1_tal_nlin_asym_09c.nii")
ifname_MNI2009_mask = os.path.join(ifpath_MNI, "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii")
ifname_MNI2006 = os.path.join(ifpath_MNI, "MNI152_T1_1mm.nii.gz")
ifname_MNI2006_brain = os.path.join(ifpath_MNI, "MNI152_T1_1mm_brain.nii.gz")
ofname_MNI2006_std = os.path.join(ifpath_MNI, "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182")
ofname_MNI2006_std_brain = os.path.join(ifpath_MNI, "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain")
ofname_MNI2006_brain = os.path.join(ifpath_MNI, "mni_icbm152_t1_tal_nlin_asym_09c_brain.nii")
ifname_identity = os.path.join(ifpath_MNI, "identity_matrix_to_transform_atlases_to_1mm_MNI152_182x218x182.mat")
os.path.join(ifpath_atlases,"original_atlases_from_source", "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii")


ifname_cerebrA = os.path.join(ifpath_atlases,"original_atlases_from_source", "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii")
ifname_cerebrA_std = os.path.join(ifpath_atlases,"atlases", "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii")

#reshape 2009 to 2006 182x218x182

img_MNI2009 = nib.load(ifname_MNI2009)
img_MNI2009_mask = nib.load(ifname_MNI2009_mask)
data_MNI2009 = img_MNI2009.get_fdata() 
data_MNI2009_mask = img_MNI2009_mask.get_fdata() 

data_MNI2009[np.where(data_MNI2009_mask ==0)] = 0

print(nib.aff2axcodes(img_MNI2009.affine))
show_slices(data_MNI2009)
img_MNI2009brain= nib.Nifti1Image(data_MNI2009, img_MNI2009.affine)
nib.save(img_MNI2009brain, os.path.join(ifpath_MNI, ofname_MNI2006_brain))




cmd = "flirt -verbose 2 -in {0} -ref {1} -out {2} -omat {2}.mat".format(ofname_MNI2006_brain, ifname_MNI2006_brain, ofname_MNI2006_std_brain)
os.system(cmd)

show_slices( nib.load(ifname_MNI2006_brain).get_fdata() )
show_slices( nib.load(ofname_MNI2006_brain).get_fdata() )




cmd = "flirt -verbose 2 -in {0} -ref {1} -out {2} -applyxfm -init {3}.mat".format(ifname_MNI2009, ifname_MNI2006, ofname_MNI2006_std, ofname_MNI2006_std_brain)
os.system(cmd)


#Put CerebrA into std space

cmd = "flirt -verbose 2 -in {0} -ref {1} -out {2} -interp nearestneighbour -applyxfm -init {3}.mat".format(ifname_cerebrA, ofname_MNI2006_std, ifname_cerebrA_std, ofname_MNI2006_std_brain)
os.system(cmd)








