#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:11:58 2022

@author: arevell
"""
#%% 1/4 Imports
import sys
import os
import json
import copy
import glob
import numpy as np
from os.path import join, splitext, basename

#revellLab
from revellLab.packages.utilities import utils
from revellLab.paths import constants_paths as paths

#package functions
from revellLab.packages.imaging.tractography import tractography

#%%

#Imaging parameters and metadata
SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"
with open(paths.ATLAS_FILES_PATH) as f: atlas_metadata_json = json.load(f)
#%%get list of all patients in BIDS data directory
 
#get list of all patients in data directory
#paths.BIDS #This is the data directory where all patient imaging is stored
#paths.BIDS_DERIVATIVES_QSIPREP

subs = os.listdir(os.path.join(paths.BIDS, "PIER"))
subs = [w.replace('sub-', '') for w in subs]

# %% Get patients where we can do QSI prep (those who have dwi)

#these are all patients with DWI correctable images thru QSI prep
subs_wtih_tractography = tractography.get_patients_with_dwi(subs, paths, dataset = "PIER")
subs_wtih_tractography = list(np.sort(np.array(subs_wtih_tractography)))
#Prints command in terminal for running QSI prep
tractography.print_dwi_image_correction_QSIprep(subs_wtih_tractography, paths, dataset = "PIER")

#%% Computing tractography for DSI studio ()
d= os.path.join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep")
subs =  [idx for idx in os.listdir(d) if "sub" in idx and "html" not in idx]
subs_wtih_qsiprep = [os.path.join(o) for o in subs if os.path.isdir(os.path.join(d,o))]
subs_wtih_qsiprep = [w.replace('sub-', '') for w in subs_wtih_qsiprep]
subs_wtih_qsiprep = list(np.sort(np.array(subs_wtih_qsiprep)))

tractography.get_tracts_loop_through_patient_list(subs_wtih_qsiprep, paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)


#%% Get structural connectivity from atlases 
N = len(subs_wtih_qsiprep)
print(subs_wtih_qsiprep)
tmp= np.array(subs_wtih_qsiprep)

tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 0, stop = 10)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 10, stop = 20)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 20, stop = 30)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 30, stop = 40)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 40, stop = 50)

tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 50, stop = 60)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 60, stop = 70)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 70, stop = 75)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 75, stop = 80)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 80, stop = 95)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 95, stop = 100)
tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 100, stop = 105)


tractography.batch_get_structural_connectivity_with_mni_registration(subs_wtih_qsiprep, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = 105)





# %% MNI registration of patient's T1

i = 0

sub = subs_wtih_qsiprep[i]
tractography.mni_registration_to_T1(sub, paths, SESSION_RESEARCH3T = "research3Tv[0-9][0-9]")
# %% Atlas registration

# Get relevant paths
atlas_paths =  tractography.get_atlases_from_priority(atlas_metadata_json, priority_type = "structure", priority_level_max = 2)
atlas_registration, structural_matrices, mni_images, mni_warp, ses, preop_T1, preop_T1_std, path_fib, path_trk = tractography.get_atlas_registration_and_tractography_paths(sub = sub, paths = paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)

#Apply warp to atlases
tractography.applywarp_to_atlas(atlas_directory = paths.REVELLLAB, atlas_paths = atlas_paths, preop_T1_std = preop_T1_std, mni_warp = mni_warp, atlas_registration = atlas_registration)

# %% Calculate structural connectivity
tractography.parse_get_structural_connectivity(sub, atlas_paths, paths, path_fib, path_trk, preop_T1, atlas_registration, structural_matrices)


# %%


fillerString = f"\n###########################"*3


i=16
sub = patient_list[i]
ses = basename(glob.glob( join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep",  f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}"))[0])[4:]

atlas_registration = os.path.join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}" , f"ses-{ses}", "atlas_registration")
structural_matrices = os.path.join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}" , f"ses-{ses}", "matrices")
utils.checkPathAndMake(atlas_registration, atlas_registration,  printBOOL = False)
utils.checkPathAndMake(structural_matrices, structural_matrices,  printBOOL = False)

preop_T1 = os.path.join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-preproc_T1w.nii.gz")
preop_T1_mask = os.path.join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-brain_mask.nii.gz")
preop_T1_bet = os.path.join(atlas_registration, f"sub-{sub}_desc-preproc_T1w_brain.nii.gz")


utils.checkPathError(preop_T1)
utils.checkPathError(preop_T1_mask)
utils.checkPathError(paths.MNI_TEMPLATE)
utils.checkPathError(paths.MNI_TEMPLATE_BRAIN)
utils.checkPathError(paths.ATLASES)
utils.checkPathError(paths.ATLAS_FILES_PATH)
utils.checkPathError(structural_matrices)

cmd = f"cp  {preop_T1} {os.path.join(atlas_registration, os.path.basename(preop_T1))}"
os.system(cmd)
cmd = f"cp  {preop_T1_mask} {os.path.join(atlas_registration, os.path.basename(preop_T1_mask))}"
os.system(cmd)
cmd = f"fslmaths  {preop_T1} -mul {preop_T1_mask} {preop_T1_bet}"
os.system(cmd)

preop_T1 = os.path.join(atlas_registration, os.path.basename(preop_T1))
preop_T1_mask = os.path.join(atlas_registration, os.path.basename(preop_T1_mask))

preop_T1_std = f"{splitext(splitext(preop_T1)[0])[0]}_std.nii.gz"
preop_T1_bet_std = f"{splitext(splitext(preop_T1_bet)[0])[0]}_std.nii.gz"

mni_base = "from_mni_to_T1"
mni_images = os.path.join(atlas_registration, f"{mni_base}")
#%%Begin Pipeline: Orient all images to standard RAS

print(f"\n\n{fillerString}Part 1 of 4\nReorientation of Images\nEstimated time: 10-30 seconds{fillerString}\nReorient all images to standard RAS\n")
cmd = f"fslreorient2std {preop_T1} {preop_T1_std}"; print(cmd); os.system(cmd)
cmd = f"fslreorient2std {preop_T1_bet} {preop_T1_bet_std}"; print(cmd); os.system(cmd)

#visualize
utils.show_slices(f"{preop_T1_std}", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(atlas_registration, "pic_T1.png")  )
utils.show_slices(f"{preop_T1_bet_std}", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(atlas_registration, "pic_T1_brain.png"))


#%%Registration of MNI to patient space (atlases are all in MNI space, so using this warp to apply to the atlases)
print(f"\n\n{fillerString}Part 3 of 4\nMNI and atlas registration\nEstimated time: 1-2+ hours{fillerString}\nRegistration of MNI template to patient space\n")


#linear reg of MNI to preopT1 space
cmd = f"flirt -in {paths.MNI_TEMPLATE_BRAIN} -ref {preop_T1_bet_std} -dof 12 -out {mni_images}_flirt -omat {mni_images}_flirt.mat -v"; print(cmd);os.system(cmd)
#non linear reg of MNI to preopT1 space
utils.show_slices(f"{mni_images}_flirt.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(atlas_registration, "pic_mni_to_T1_flirt.png"))
print("\n\nLinear registration of MNI template to image is done\n\nStarting Non-linear registration:\n\n\n")

cmd = f"fnirt --in={paths.MNI_TEMPLATE} --ref={preop_T1_std} --aff={mni_images}_flirt.mat --iout={mni_images}_fnirt -v --cout={mni_images}_coef --fout={mni_images}_warp"; print(cmd); os.system(cmd)
utils.show_slices(f"{mni_images}_fnirt.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(atlas_registration, "pic_mni_to_T1_fnirt.png"))


print("\n\n\nUsing MNI template warp to register all atlases (these atlases are already in MNI space)")
#apply warp to all atlases

#atl.applywarp_to_atlas(atlasDirectory, f"{preopT1_output}_std1x1x1.nii.gz", MNIwarp, atlasRegistrationTMP, isDir = True)


atlas_directory = paths.ATLASES
paths.ATLAS_FILES_PATH


with open(paths.ATLAS_FILES_PATH) as f: atlas_metadata_json = json.load(f)

def get_atlases_from_priority(atlas_metadata_json, priority_type = "structure", priority_level_max = 2):
    atlas_paths = []
    standard = list(atlas_metadata_json["STANDARD"].keys())
    standard_common_path = atlas_metadata_json["PATHS"]["STANDARD"]
    for s in range(len(standard)):
        priority_level = atlas_metadata_json["STANDARD"][standard[s]]["priority_level"][priority_type]
        if priority_level <= priority_level_max:
            atlas_paths.append( os.path.join(standard_common_path, atlas_metadata_json["STANDARD"][standard[s]]["name"])  )
            
    #Random atlases
    random = list(atlas_metadata_json["RANDOM"].keys())
    for r in range(len(random)):
        random_common_path = atlas_metadata_json["PATHS"][random[r]]
        
        random_sublevel = list(atlas_metadata_json["RANDOM"][random[r]].keys())
        for s in range(len(random_sublevel)):
            priority_level = atlas_metadata_json["RANDOM"][random[r]][random_sublevel[s]]["priority_level"][priority_type]
            if priority_level <= priority_level_max:
                permutations = atlas_metadata_json["RANDOM"][random[r]][random_sublevel[s]]["permutations"]
                for p in range(1,permutations+1):                
                    atlas_paths.append( os.path.join(random_common_path,  f'{atlas_metadata_json["RANDOM"][random[r]][random_sublevel[s]]["name"]}_v{p:04}.nii.gz'  ))
    return atlas_paths
            
atlas_paths =    get_atlases_from_priority(atlas_metadata_json, priority_type = "structure", priority_level_max = 2)

atlas_directory = paths.REVELLLAB
mni_warp = f"{mni_images}_warp.nii.gz"
atlas_registration_output_path = atlas_registration
def applywarp_to_atlas(atlas_directory, atlas_paths, preop_T1_std, mni_warp, atlas_registration_output_path):
    full_atlas_paths = []
    for a in range(len(atlas_paths)):
        utils.checkPathError( f"{os.path.join(atlas_directory, atlas_paths[a])}" )
        full_atlas_paths.append( f"{os.path.join(atlas_directory, atlas_paths[a])}")
    utils.checkPathError(mni_warp)
    utils.checkPathError(preop_T1_std)
    utils.checkPathError(atlas_registration_output_path)
    for a in range(len(full_atlas_paths)):
        atlasName = basename(splitext(splitext(full_atlas_paths[a])[0])[0])
        outputAtlasName = join(atlas_registration_output_path, atlasName + ".nii.gz")
        #if not os.path.exists(outputAtlasName):
        print(f"\r{np.round((a+1)/len(full_atlas_paths)*100,2)}%   {atlasName[0:20]}                         ", end = "\r")
        if not utils.checkIfFileExists(outputAtlasName, printBOOL=False ):
            cmd = f"applywarp -i { full_atlas_paths[a]} -r {preop_T1_std} -w {mni_warp} --interp=nn -o {outputAtlasName}"; os.system(cmd)
        #else: print(f"File exists: {outputAtlasName}")


applywarp_to_atlas(atlas_directory, atlas_paths, preop_T1_std, mni_warp, atlas_registration)

sub
path_fib = glob.glob(os.path.join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", f"ses-{ses}", "*.fib.*"))[0]
path_trk = glob.glob(os.path.join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", f"ses-{ses}", "*.trk.*"))[0]

for a in range(len(atlas_paths)):
    atlasName = basename(splitext(splitext(atlas_paths[a])[0])[0])
    print(f"\n\n\n\n\n\n{np.round((a+1)/len(atlas_paths)*100,2)}%   {atlasName[0:20]}  \n\n\n\n\n\n"   )
    atlas = join(atlas_registration, atlasName + ".nii.gz")
    
    structural_matrices_output = join(structural_matrices, f"sub-{sub}")
    tractography.get_structural_connectivity(paths.BIDS, paths.DSI_STUDIO_SINGULARITY, path_fib, path_trk, preop_T1, atlas, structural_matrices_output )
