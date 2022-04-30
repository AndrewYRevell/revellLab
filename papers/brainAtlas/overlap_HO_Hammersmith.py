#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 21:55:19 2022

@author: arevell
"""

import nibabel as nib
import numpy as np
import seaborn as sns
import copy
# %%

HO_path = "tools/atlases/atlases/HarvardOxford-cort-NONSYMMETRIC-maxprob-thr25-1mm.nii.gz"
ham_path = "tools/atlases/atlases/Hammersmith_atlas_n30r83_SPM5.nii.gz"

HO_img = nib.load(HO_path)
ham_img  = nib.load(ham_path)


HO = HO_img.get_fdata()
ham = ham_img.get_fdata()


# %%

"""
superior temporal gyrus labels

HO: out of the box  symmetric
9,"Superior Temporal Gyrus, anterior division"

NON symmetric
9,"Right Superior Temporal Gyrus, anterior division"
109,"Left Superior Temporal Gyrus, anterior division"


Hammersmith
82,"Left Superior temporal gyrus, anterior part"
83,"Right Superior temporal gyrus, anterior part"


11,"Right Superior temporal gyrus, posterior part"
12,"Left Superior temporal gyrus, posterior part"
"""
# =============================================================================
# superior temporal gyrus
# Hammersmith = 
# HO
# =============================================================================


np.where(HO == 109)
np.where(ham == 12)



# %%
ho_stg =  copy.deepcopy(HO)
ham_stg = copy.deepcopy(ham)

ho_stg[np.where(ho_stg != 109)] = 0
ham_stg[np.where(ham_stg != 12)] = 0

ho_stg[np.where(ho_stg == 109)] = 1
ham_stg[np.where(ham_stg == 12)] = 2


overlap  =  copy.deepcopy(ho_stg)

sns.heatmap(ho_stg[:, 120,:] , square= True)
sns.heatmap(ham_stg[:, 120,:] , square= True)



# %%

ho_1 = np.where( ho_stg == 1 )
len(ho_1[0])
ham_2 = np.where( ham_stg == 2 )
len(ham_2[0])
ho_overlap = copy.deepcopy(ho_stg)

overlap[(ho_stg ==1 ) & (ham_stg == 2)] = 3



sns.heatmap(HO[:, 121,:], square= True )

len(np.where(overlap == 3)[0]) / len(ho_1[0])

ho_overlap[np.where(overlap == 3)] = 2

sns.heatmap(overlap[:, 121,:], square= True )


np.unique(np.where(HO == 109)[1])

sns.heatmap(ho_overlap[:, 130,:], square= True )


sns.heatmap(overlap[:, 113,:], square= True )
np.unique(overlap)

# %%
# The overlap percentage

len(np.where(overlap == 3)[0]) / len(ho_1[0])


np.unique(ho_overlap)
len(np.where(ho_overlap == 2)[0])
len(np.where(ho_overlap == 1)[0])


len(np.where(ho_overlap == 2)[0]) / (len(np.where(ho_overlap == 2)[0]) +  len(np.where(ho_overlap == 1)[0]))
