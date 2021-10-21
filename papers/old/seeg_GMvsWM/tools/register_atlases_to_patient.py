#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:23:28 2020

@author: arevell
"""

#combine subcortical segmentation (FIST) and regular segmentation (FAST)
import os

def register_MNI_to_T00(ifname_T00,ifname_MNI, ofpath_segmentation_sub_ID, ofbase_flirt, ofbase_fnirt):
    
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    
    #linear reg of preop to T00 space
    cmd = "flirt -in {0}_std_bet.nii.gz -ref {1}_std_bet.nii.gz -dof 12 -out {2} -omat {2}.mat -v".format(ifname_MNI, ifname_T00, ofbase_flirt)
    print(cmd)
    os.system(cmd)
    
    #Do Not do non-linear registration
    #non linear reg of preop to T00 space
    cmd = "fnirt --in={0}_std.nii.gz --ref={1}_std.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_MNI, ifname_T00, ofbase_flirt, ofbase_fnirt)
    print(cmd)
    os.system(cmd)
    
    
def applywarp_to_atlas(ofname_FIRST_FAST_COMBINED, ifname_T00, ofbase_fnirt, ofpath_segmentation_sub_ID, ofname_FIRST_FAST_COMBINED_to_T00):
    ifname_T00 =  os.path.join(ofpath_segmentation_sub_ID, os.path.basename(ifname_T00))
    #warp combined_FIRST_and_FAST image to T00 space
    cmd = "applywarp -i {0} -r {1}_std.nii.gz -w {2}_warp.nii.gz --interp=nn -o {3}".format(ofname_FIRST_FAST_COMBINED, ifname_T00, ofbase_fnirt, ofname_FIRST_FAST_COMBINED_to_T00)
    print(cmd)
    os.system(cmd)

"""


#For RID0309 because images are not in correct orientation

##Orient the preop3T image
#fslreorient2std sub-RID0309_ses-preop3T_acq-3D_T1w.nii.gz sub-RID0309_ses-preop3T_acq-3D_T1w_std.nii.gz
#fslreorient2std sub-RID0309_ses-preop3T_acq-3D_T1w_bet.nii.gz sub-RID0309_ses-preop3T_acq-3D_T1w_bet_std.nii.gz
##Orient the T00  image
#fslreorient2std sub-RID0309_T00_mprage.nii.gz sub-RID0309_T00_mprage_std.nii.gz
#fslreorient2std sub-RID0309_T00_mprage_bet.nii.gz sub-RID0309_T00_mprage_bet_std.nii.gz
#flirt -in sub-RID0309_ses-preop3T_acq-3D_T1w_bet_std.nii.gz -ref sub-RID0309_T00_mprage_bet_std.nii.gz -dof 12 -out sub-RID0309_preop3T_to_T00 -omat sub-RID0309_preop3T_to_T00.mat -v
#ifname_preop3T_std = "{0}_std".format(ifname_preop3T)
#ifname_T00_std = "{0}_std".format(ifname_T00)
#cmd = "fnirt --in={0}.nii.gz --ref={1}.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_preop3T_std, ifname_T00_std, ofbase_flirt, ofbase_fnirt)
#print(cmd)
#os.system(cmd)
#cmd = "applywarp -i {0} -r {1}.nii.gz -w {2}_warp.nii.gz --interp=nn -o {3}".format(ofname_FIRST_FAST_COMBINED, ifname_T00_std, ofbase_fnirt, ofname_FIRST_FAST_COMBINED_to_T00)
#print(cmd)
#os.system(cmd)


#For RID0194

##Orient the preop3T image
fslreorient2std sub-RID0194_ses-preop3T_acq-3D_T1w.nii.gz sub-RID0194_ses-preop3T_acq-3D_T1w_std.nii.gz
fslreorient2std sub-RID0194_ses-preop3T_acq-3D_T1w_bet.nii.gz sub-RID0194_ses-preop3T_acq-3D_T1w_bet_std.nii.gz
##Orient the T00  image
fslreorient2std sub-RID0194_T00_mprage.nii.gz sub-RID0194_T00_mprage_std.nii.gz
fslreorient2std sub-RID0194_T00_mprage_bet.nii.gz sub-RID0194_T00_mprage_bet_std.nii.gz
#Orient GM_WM
fslreorient2std sub-RID0194_ses-preop3T_acq-3D_T1w_bet_GM_WM_CSF.nii.gz sub-RID0194_ses-preop3T_acq-3D_T1w_bet_GM_WM_CSF_std.nii.gz
flirt -in sub-RID0194_ses-preop3T_acq-3D_T1w_bet_std.nii.gz -ref sub-RID0194_T00_mprage_bet_std.nii.gz -dof 12 -out sub-RID0194_preop3T_to_T00 -omat sub-RID0194_preop3T_to_T00.mat -v
ifname_preop3T_std = "{0}_std".format(ifname_preop3T)
ifname_T00_std = "{0}_std".format(ifname_T00)
cmd = "fnirt --in={0}.nii.gz --ref={1}.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_preop3T_std, ifname_T00_std, ofbase_flirt, ofbase_fnirt)
print(cmd)
os.system(cmd)
ofname_FIRST_FAST_COMBINED = "{0}_std.nii.gz".format(os.path.splitext(os.path.splitext(ofname_FIRST_FAST_COMBINED)[0])[0])
cmd = "applywarp -i {0} -r {1}.nii.gz -w {2}_warp.nii.gz --interp=nn -o {3}".format(ofname_FIRST_FAST_COMBINED, ifname_T00_std, ofbase_fnirt, ofname_FIRST_FAST_COMBINED_to_T00)
print(cmd)
os.system(cmd)



#For RID0320

##Orient the preop3T image
fslreorient2std sub-RID0320_ses-preop3T_acq-3D_T1w.nii.gz sub-RID0320_ses-preop3T_acq-3D_T1w_std.nii.gz
fslreorient2std sub-RID0320_ses-preop3T_acq-3D_T1w_bet.nii.gz sub-RID0320_ses-preop3T_acq-3D_T1w_bet_std.nii.gz
##Orient the T00  image
fslreorient2std sub-RID0320_T00_mprage.nii.gz sub-RID0320_T00_mprage_std.nii.gz
fslreorient2std sub-RID0320_T00_mprage_bet.nii.gz sub-RID0320_T00_mprage_bet_std.nii.gz
flirt -in sub-RID0320_ses-preop3T_acq-3D_T1w_bet_std.nii.gz -ref sub-RID0320_T00_mprage_bet_std.nii.gz -dof 12 -out sub-RID0320_preop3T_to_T00 -omat sub-RID0320_preop3T_to_T00.mat -v
ifname_preop3T_std = "{0}_std".format(ifname_preop3T)
ifname_T00_std = "{0}_std".format(ifname_T00)
cmd = "fnirt --in={0}.nii.gz --ref={1}.nii.gz --aff={2}.mat --iout={3} -v --cout={3}_coef --fout={3}_warp".format(ifname_preop3T_std, ifname_T00_std, ofbase_flirt, ofbase_fnirt)
print(cmd)
os.system(cmd)
cmd = "applywarp -i {0} -r {1}.nii.gz -w {2}_warp.nii.gz --interp=nn -o {3}".format(ofname_FIRST_FAST_COMBINED, ifname_T00_std, ofbase_fnirt, ofname_FIRST_FAST_COMBINED_to_T00)
print(cmd)
os.system(cmd)




"""










