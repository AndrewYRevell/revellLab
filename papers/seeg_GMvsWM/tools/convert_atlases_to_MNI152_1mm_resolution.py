#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:29:50 2020

@author: andyrevell
"""
#%%

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
path = "/mnt"
ifpath = os.path.join(path, "data/data_raw/atlases/original_atlases_from_source")
reference = os.path.join(ifpath, "MNI-maxprob-thr25-1mm.nii.gz")
identity_matrix_1 = os.path.join(path, "data/data_raw/atlases/atlas_labels_description", "identity_matrix_to_transform_atlases_to_1mm_MNI152_space_1.mat")#NOTE: the identity file is a txt file, NOT a matlab file. it is just a numpy.identity(4) matrix. 
identity_matrix_2 = os.path.join(path,"data/data_raw/atlases/atlas_labels_description", "identity_matrix_to_transform_atlases_to_1mm_MNI152_space_2.mat")#NOTE: the identity file is a txt file, NOT a matlab file. it is just a numpy.identity(4) matrix. 

ofpath = os.path.join(path,"data/data_raw/atlases/standard_atlases")
ofpath_canonical = os.path.join(path,"data/data_raw/atlases/original_atlases_in_RAS_orientation")
#%%

atlases = [f for f in sorted(os.listdir(ifpath))]


for i in range(len(atlases)):
    atlas = atlases[i]
    print("\n\n")
    print(atlas)
    ifname = os.path.join(ifpath, atlas)
    ofname = os.path.join(ofpath, atlas )
    
    ofname_canonical = os.path.join(ofpath_canonical,atlas )

     
    img = nib.load(ifname)
    data = img.get_fdata() 
    print(nib.aff2axcodes(img.affine))
    img_canonical = nib.as_closest_canonical(img)
    print(nib.aff2axcodes(img_canonical.affine))

    
    
    #Yeo images are 256 x 256x 256 x 1 --> removing that extra dimension
    if len(data.shape) ==4:
        print("reshaping image from {0} dimension to {1}".format(data.shape, (data.shape[0],data.shape[1],data.shape[2]) ))
        img_3D = nib.funcs.four_to_three(img_canonical)[0]
        print(nib.aff2axcodes(img_3D.affine))
        print("saving {0}".format(ofname_canonical))
        nib.save(img_3D, ofname_canonical)
    else:
        print("saving {0}".format(ofname_canonical))
        nib.save(img_canonical, ofname_canonical)

    cmd = "flirt -verbose 2 -in {0} -ref {1} -out {2} -interp nearestneighbour -applyxfm -init {3}".format(ofname_canonical, reference, ofname, identity_matrix_2)
    os.system(cmd)
    
    
    cmd = "c3d {0} -info-full".format(reference)
    os.system(cmd)

    cmd = "c3d {0} -info-full".format(ofname_canonical)
    os.system(cmd)

    cmd = "c3d {0} -origin-voxel -90x-126x-72vox -o {1}".format(ofname_canonical, ofname)
    os.system(cmd)
    
    cmd = "c3d {0} -info-full".format(ofname)
    os.system(cmd)

    
    cmd = "flirt -verbose 2 -in {0} -ref {1} -out {2} -interp nearestneighbour -interp nearestneighbour -applyxfm -init {3}".format(ofname_canonical, reference, ofname, identity_matrix_2)
    os.system(cmd)
    
    cmd = "c3d {0} -info-full".format(ofname + ".gz")
    os.system(cmd)



img_standard = nib.load(ofname + ".gz")
data_s = img_standard.get_fdata() 
tmp =data_s[:,:,100]
plt.imshow(tmp, cmap="gray",  origin="lower")


cmd = "c3d {0} -info-full".format(os.path.join(ofpath, atlases[42]))
os.system(cmd)





for i in range(len(atlases)):
    atlas = atlases[i]
    print(atlas)
    ifname = os.path.join(ifpath, atlas)
    cmd = "c3d {0} -info-full".format(ifname)
    os.system(cmd)
    i=i+1
    print(atlas)

img_c = nib.load(ofname_canonical)
data_c = img_c.get_fdata() 
print(nib.aff2axcodes(img_c.affine))

img_standard = nib.load(ofname)
print(nib.aff2axcodes(img_standard.affine))
data_s = img_standard.get_fdata() 
tmp =data_s[:,:,100]
plt.imshow(tmp, cmap="gray",  origin="lower")


img_standard_canonical = nib.as_closest_canonical(img_standard)
data_s_c = img_standard_canonical.get_fdata() 
#nib.save(img_standard_canonical, ofname)

#fslreorient2std MMP_in_MNI_corr.nii ./transform


tmp_img = nib.load(os.path.join(ifpath, "tmp", "transform.nii.gz"))
tmp_data = tmp_img.get_fdata() 
tmp =tmp_data[:,:,100]
plt.imshow(tmp, cmap="gray",  origin="lower")
print(nib.aff2axcodes(tmp_img.affine))


transformed = os.path.join(ifpath, "tmp", "transform.nii.gz")
cmd = "flirt -verbose 2 -in {0} -ref {1} -out {2} -dof 6  -interp nearestneighbour -applyisoxfm 1 ".format(transformed, reference, ofname, identity_matrix_2)
os.system(cmd)

img_standard = nib.load(ofname + ".gz")
data_s = img_standard.get_fdata() 
tmp =data_s[:,:,100]
plt.imshow(tmp, cmap="gray",  origin="lower")



mni = os.path.join(ofpath, atlases[14] )
mni_img = nib.load(mni)
data_mni = mni_img.get_fdata() 
data_mni_tmp =data_mni[:,:,100]
plt.imshow(data_mni_tmp, cmap="gray",  origin="lower")






