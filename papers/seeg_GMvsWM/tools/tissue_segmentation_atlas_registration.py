#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:23:28 2020

@author: arevell
"""

#combine subcortical segmentation (FIST) and regular segmentation (FAST)
import os
import nibabel as nib
#import imagingToolsRevell as iTR
import numpy as np

def first_and_fast_segmentation(ifname_preop3T, ifname_T00, ofpath_segmentation_sub_ID):
    
    #copy files to temporary
    cmd = "cp {0}.nii.gz {1}".format(ifname_preop3T, ofpath_segmentation_sub_ID); print(cmd); os.system(cmd)
    
    #copy files to temporary
    cmd = "cp {0}.nii.gz {1}".format(ifname_T00, ofpath_segmentation_sub_ID); print(cmd); os.system(cmd)
    
    
    ifname_preop3T =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_preop3T))
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    
    
    #Orient all images to standard RAS
    ##Orient the preop3T image
    cmd = "fslreorient2std {0}.nii.gz {1}.nii.gz".format(ifname_preop3T, "{0}_std".format(ifname_preop3T)); print(cmd); os.system(cmd)
    ##Orient the T00  image
    cmd = "fslreorient2std {0}.nii.gz {1}.nii.gz".format(ifname_T00, "{0}_std".format(ifname_T00)); print(cmd); os.system(cmd)
   

    
    #brain extraction of preop. Brain Extraction parameters matter to get FIRST to work properly
    #setting g and f parameters for bet
    f = 0.5; g = -0.3
    if (os.path.basename(ifname_preop3T) == "sub-RID0194_ses-preop3T_acq-3D_T1w"):
        f = 0.5; g= -0.3
    if (os.path.basename(ifname_preop3T) == "sub-RID0278_ses-preop3T_acq-3D_T1w"):
        f = 0.3; g= -0.4
    if (os.path.basename(ifname_preop3T) == "sub-RID0320_ses-preop3T_acq-3D_T1w"):
        f = 0.5; g= -0.3
    if (os.path.basename(ifname_preop3T) == "sub-RID0420_ses-preop3T_acq-3D_T1w"):
        f = 0.5; g= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0502_ses-preop3T_acq-3D_T1w"):
        f = 0.3; g= -0.4
    if (os.path.basename(ifname_preop3T) == "sub-RID0508_ses-preop3T_acq-3D_T1w"):
        f = 0.4; g= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0459_ses-preop3T_acq-3D_T1w"):
        f = 0.4; g= -0.5
        
    f_T00 = 0.15; g_T00 = -0.0
    if (os.path.basename(ifname_preop3T) == "sub-RID0365_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.3; g_T00= -0.1
    if (os.path.basename(ifname_preop3T) == "sub-RID0459_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.4; g_T00= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0490_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.4; g_T00= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0520_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.4; g_T00= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0522_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.4; g_T00= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0572_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.4; g_T00= -0.5
    if (os.path.basename(ifname_preop3T) == "sub-RID0595_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.3; g_T00= -0.1
    if (os.path.basename(ifname_preop3T) == "sub-RID0037_ses-preop3T_acq-3D_T1w"):
        f_T00 = 0.3; g_T00= -0.4
    
    #default:f=0.5, g=-0.3 #RID0194:f ,g  #RID0320:f=0.5'g=-0.3    #RID0420:f=0.5,g=-0.5    #RID502:f=0.3,g=-0.4  #RID0508 f=0.4 g=-0.5;
    cmd = "bet {0}_std.nii.gz {0}_std_bet.nii.gz -f {1} -g {2}".format(ifname_preop3T, f, g); print(cmd); os.system(cmd)
    
    #brain extraction of T00
    cmd = "bet {0}_std.nii.gz {0}_std_bet.nii.gz -f {1} -g {2}".format(ifname_T00, f_T00, g_T00); print(cmd); os.system(cmd)
    
    #subcortical segmentation (FIRST)
    cmd = "run_first_all -i {0}_std_bet.nii.gz -o {0}_std_bet_subcort.nii.gz -b -v".format(ifname_preop3T); print(cmd); os.system(cmd)
    
    #seg of preop img  (FAST)
    cmd = "fast -n 3 -H 0.25 -t 1 -v {0}_std_bet.nii.gz".format(ifname_preop3T); print(cmd); os.system(cmd)
    
    
def register_preop3T_to_T00(ifname_preop3T, ifname_T00, ofpath_segmentation_sub_ID, ofbase_flirt, ofbase_fnirt):
    
    ifname_preop3T =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_preop3T))
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    
    #linear reg of preop to T00 space
    cmd = "flirt -in {0}_std_bet.nii.gz -ref {1}_std_bet.nii.gz -dof 12 -out {2} -omat {2}.mat -v".format(ifname_preop3T, ifname_T00, ofbase_flirt)
    print(cmd)
    os.system(cmd)
    
    #Do Not do non-linear registration
    #non linear reg of preop to T00 space
    #cmd = "fnirt --in={0}_std.nii.gz --ref={1}_std.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_preop3T, ifname_T00, ofbase_flirt, ofbase_fnirt)
    #print(cmd)
    #os.system(cmd)
    
    
def applywarp_to_combined_first_fast(ofname_FIRST_FAST_COMBINED, ifname_T00, ofbase_flirt, ofpath_segmentation_sub_ID, ofname_FIRST_FAST_COMBINED_to_T00):
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    #warp combined_FIRST_and_FAST image to T00 space
    
    
    cmd = "flirt -in {0} -ref {1}_std_bet.nii.gz -dof 12 -init {2}.mat -v -applyxfm -out {3} -interp nearestneighbour ".format(ofname_FIRST_FAST_COMBINED, ifname_T00, ofbase_flirt, ofname_FIRST_FAST_COMBINED_to_T00)

    #cmd = "applywarp -i {0} -r {1}_std.nii.gz -w {2}_warp.nii.gz --interp=nn -o {3}".format(ofname_FIRST_FAST_COMBINED, ifname_T00, ofbase_fnirt, ofname_FIRST_FAST_COMBINED_to_T00)
    print(cmd)
    os.system(cmd)






def register_MNI_to_T00(ifname_T00,ifname_MNI, ofpath_segmentation_sub_ID, ofbase_flirt_MNI, ofbase_fnirt_MNI):
    
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    
    #linear reg of preop to T00 space
    cmd = "flirt -in {0}_brain.nii.gz -ref {1}_std_bet.nii.gz -dof 12 -out {2} -omat {2}.mat -v".format(ifname_MNI, ifname_T00, ofbase_flirt_MNI)
    print(cmd)
    os.system(cmd)
    
    #Do Not do non-linear registration
    #non linear reg of preop to T00 space
    cmd = "fnirt --in={0}.nii.gz --ref={1}_std.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_MNI, ifname_T00, ofbase_flirt_MNI, ofbase_fnirt_MNI)
    print(cmd)
    os.system(cmd)
    
    
def applywarp_to_atlas(ifname_atlas, ifname_T00, ofbase_fnirt_MNI, ofpath_segmentation_sub_ID, ofname_atlas_to_T00):
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    #warp combined_FIRST_and_FAST image to T00 space
    cmd = "applywarp -i {0} -r {1}_std.nii.gz -w {2}_warp.nii.gz --interp=nn -o {3}".format(ifname_atlas, ifname_T00, ofbase_fnirt_MNI, ofname_atlas_to_T00)
    print(cmd)
    os.system(cmd)


def register_MNI_to_3Tpreop(ifname_3Tpreop, ifname_3Tpreop_mask, ifname_MNI, ofpath_atlas_registration_sub_ID, ofbase_flirt_MNI, ofbase_fnirt_MNI):
    
    cmd = "cp {0}.nii.gz {1}".format(ifname_3Tpreop, ofpath_atlas_registration_sub_ID); print(cmd); os.system(cmd)
    cmd = "cp {0}.nii.gz {1}".format(ifname_3Tpreop_mask, ofpath_atlas_registration_sub_ID); print(cmd); os.system(cmd)
    
    ifname_3Tpreop =  os.path.join(ofpath_atlas_registration_sub_ID, os.path.basename(ifname_3Tpreop))
    ifname_3Tpreop_mask =  os.path.join(ofpath_atlas_registration_sub_ID, os.path.basename(ifname_3Tpreop_mask))
    
    #Orient all images to standard RAS
    ##Orient the preop3T image
    cmd = "fslreorient2std {0}.nii.gz {1}.nii.gz".format(ifname_3Tpreop, "{0}_std".format(ifname_3Tpreop)); print(cmd); os.system(cmd)
    cmd = "fslreorient2std {0}.nii.gz {1}.nii.gz".format(ifname_3Tpreop_mask, "{0}_std".format(ifname_3Tpreop_mask)); print(cmd); os.system(cmd)
    
    #brain extraction
    img_3Tpreop = nib.load("{0}_std.nii.gz".format(ifname_3Tpreop))
    data_3Tpreop = img_3Tpreop.get_fdata() 
    img_3Tpreop_mask = nib.load("{0}_std.nii.gz".format(ifname_3Tpreop_mask))
    data_3Tpreop_mask = img_3Tpreop_mask.get_fdata() 
    
    #iTR.show_slices(data_3Tpreop)
    #iTR.show_slices(data_3Tpreop_mask)
    
    

    data_3Tpreop[np.where(data_3Tpreop_mask == 0)] = 0
    #iTR.show_slices(data_3Tpreop)
    
    img_data_3Tpreop = nib.Nifti1Image(data_3Tpreop, img_3Tpreop.affine)
    nib.save(img_data_3Tpreop, "{0}_std_brain.nii.gz".format(ifname_3Tpreop))
    
    
    
    #linear reg of preop to T00 space
    cmd = "flirt -in {0}_brain.nii.gz -ref {1}_std_brain.nii.gz -dof 12 -out {2} -omat {2}.mat -v".format(ifname_MNI, ifname_3Tpreop, ofbase_flirt_MNI)
    print(cmd)
    os.system(cmd)
    
    #Do Not do non-linear registration
    #non linear reg of preop to T00 space
    cmd = "fnirt --in={0}.nii.gz --ref={1}_std.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_MNI, ifname_3Tpreop, ofbase_flirt_MNI, ofbase_fnirt_MNI)
    print(cmd)
    os.system(cmd)
    








