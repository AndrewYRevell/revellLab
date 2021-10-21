#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:23:27 2021

@author: arevell
"""

import os
import glob
import numpy as np
import nibabel as nib
from os.path import join, splitext, basename
from revellLab.packages.utilities import utils
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl



def make_spherical_regions(patient_list, SESSION, paths, radius = 7, rerun = False, show_slices = False):
    N = len(patient_list)
    for i in range(N):
        sub = patient_list[i]
        print(sub)
        localization, localization_channels = atl.get_atlas_localization_file(sub, SESSION, paths)


        #t1_image = join(paths['BIDS'], "PIER", f"sub-{sub}", f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
        t1_image = glob.glob(join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{sub}", f"ses-{SESSION}", "tmp", "orig_nu_std.nii.gz" ))[0]
        img = nib.load(t1_image)
        if show_slices: utils.show_slices(img, data_type = "img")
        img_data = img.get_fdata()
        affine = img.affine
        shape = img_data.shape

        coordinates = np.array(localization[["x", "y", "z"]])

        coordinates_voxels = utils.transform_coordinates_to_voxel(coordinates, affine)

        path_spheres = join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", "electrodeContactSphereROIs")
        utils.checkPathAndMake(path_spheres, path_spheres, printBOOL = False)

        for e in range(len(localization_channels)):

            fname_ROIs_sub_ID = join(path_spheres, f"sub-{sub}_ses-implant01_desc-{localization_channels[e]}.nii.gz")
            if utils.checkIfFileDoesNotExist(fname_ROIs_sub_ID, printBOOL = False) or rerun:

                img_data_sphere = copy.deepcopy(img_data)
                img_data_sphere[  np.where(img_data_sphere != 0)  ] = 0
                x = coordinates_voxels[e][0]
                y = coordinates_voxels[e][1]
                z = coordinates_voxels[e][2]
                img_data_sphere = utils.make_sphere_from_point(img_data_sphere, x, y, z, radius = radius) #radius = 7mm
                img_sphere = nib.Nifti1Image(img_data_sphere, img.affine)
                nib.save(img_sphere, fname_ROIs_sub_ID)
                #utils.show_slices(img_data_sphere, data_type = "data")
            utils.printProgressBar(e+1, len(localization_channels))