#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import os
import glob
import numpy as np
from os.path import join,  splitext, basename
from revellLab.packages.utilities import utils



#%% functions


def dwi_image_correction_QSIprep(sub, paths, dataset = "PIER"):
    cmd = f"qsiprep-docker {join(paths.BIDS, dataset)} {paths.BIDS_DERIVATIVES_QSIPREP} participant --output-resolution 1.5 --fs-license-file {paths.FREESURFER_LICENSE} -w {paths.BIDS_DERIVATIVES_QSIPREP} --participant_label {sub}"
    #print(f"\n\n{cmd}\n\n")
    return cmd

def get_tracts(singularity_bind_path, path_dsiStudioSingularity, path_dwi, output_directory):
    utils.checkPathError(singularity_bind_path)
    utils.checkPathError(path_dsiStudioSingularity)
    utils.checkPathError(path_dwi)
    utils.checkPathError(output_directory)

    base, split, basesplit = utils.baseSplitextNiiGz(path_dwi)
    source_name = f"{join(output_directory, basesplit)}.src.gz"
    fib_name = f"{join(output_directory, basesplit)}.fib.gz"
    trk_name = f"{join(output_directory, basesplit)}.trk.gz"
    print("\n\nCreating Source File in DSI Studio\n\n")
    cmd = f"singularity exec --bind {singularity_bind_path} {path_dsiStudioSingularity} dsi_studio --action=src --source={path_dwi} --output={source_name}"
    os.system(cmd)
    print("\n\nCreating Reconstruction File in DSI Studio\n\n")
    cmd = f"singularity exec --bind {singularity_bind_path} {path_dsiStudioSingularity} dsi_studio --action=rec --source={source_name} --method=4 --param0=1.25"
    os.system(cmd)
    #rename dsistudio output of .fib file because there is no way to output a specific name on cammnad line
    cmd = f"mv {source_name}*.fib.gz {fib_name}"
    os.system(cmd)
    print("\n\nCreating Tractography File in DSI Studio\n\n")
    cmd = f"singularity exec --bind {singularity_bind_path} {path_dsiStudioSingularity} dsi_studio --action=trk --source={fib_name} --min_length=10 --max_length=800 --thread_count=16 --fiber_count=1000000 --output={trk_name}"
    os.system(cmd)



def get_structural_connectivity(singularity_bind_path, path_dsiStudioSingularity, path_fib, path_trk, preop3T, atlas, output ):
    utils.checkPathError(singularity_bind_path)
    utils.checkPathError(path_dsiStudioSingularity)
    utils.checkPathError(path_fib)
    utils.checkPathError(path_trk)
    utils.checkPathError(preop3T)
    utils.checkPathError(atlas)
    cmd = f"singularity exec --bind {singularity_bind_path} {path_dsiStudioSingularity} dsi_studio --action=ana --source={path_fib} --tract={path_trk} --t1t2={preop3T} --connectivity={atlas} --connectivity_type=pass --connectivity_threshold=0 --output={output}"
    os.system(cmd)


# %% MNI template registration to T1 image (to eventually warp atlases to same space)

def get_atlas_registration_and_tractography_paths(sub, paths, SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"):
    ses = basename(glob.glob( join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep",  f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}"))[0])[4:]
    atlas_registration = os.path.join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}" , f"ses-{ses}", "atlas_registration")
    structural_matrices = os.path.join(paths.BIDS_DERIVATIVES_STRUCTURAL_MATRICES, f"sub-{sub}" , f"ses-{ses}", "matrices")
    utils.checkPathAndMake(atlas_registration, atlas_registration,  printBOOL = False)
    utils.checkPathAndMake(structural_matrices, structural_matrices,  printBOOL = False)
    mni_base = "from_mni_to_T1"
    mni_images = os.path.join(atlas_registration, f"{mni_base}")
    mni_warp = f"{mni_images}_warp.nii.gz"
    preop_T1 = os.path.join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-preproc_T1w.nii.gz")
    path_fib = glob.glob(os.path.join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", f"ses-{ses}", "*.fib.*"))[0]
    path_trk = glob.glob(os.path.join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", f"ses-{ses}", "*.trk.*"))[0]
    preop_T1 = os.path.join(atlas_registration, os.path.basename(preop_T1))
    preop_T1_std = f"{splitext(splitext(preop_T1)[0])[0]}_std.nii.gz"
    return atlas_registration, structural_matrices, mni_images, mni_warp, ses, preop_T1, preop_T1_std, path_fib, path_trk

def mni_registration_to_T1(sub, paths, SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"):
    
    #getting directories and files
    
    #structural connectivity derivatives
    atlas_registration, structural_matrices, mni_images, mni_warp, ses, preop_T1, preop_T1_std, path_fib, path_trk = get_atlas_registration_and_tractography_paths(sub = sub, paths = paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)

    if not utils.checkIfFileExists(f"{mni_images}_fnirt.nii.gz", printBOOL=False ):
    
        #QSI prep outputs
        preop_T1 = os.path.join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-preproc_T1w.nii.gz")
        preop_T1_mask = os.path.join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep", f"sub-{sub}", "anat", f"sub-{sub}_desc-brain_mask.nii.gz")
        preop_T1_bet = os.path.join(atlas_registration, f"sub-{sub}_desc-preproc_T1w_brain.nii.gz")
        
        #checking all images exist
        utils.checkPathError(preop_T1)
        utils.checkPathError(preop_T1_mask)
        utils.checkPathError(paths.MNI_TEMPLATE)
        utils.checkPathError(paths.MNI_TEMPLATE_BRAIN)
    
        
        #copying relevant images from QSI prep to registration folder
        cmd = f"cp  {preop_T1} {os.path.join(atlas_registration, os.path.basename(preop_T1))}"
        os.system(cmd)
        cmd = f"cp  {preop_T1_mask} {os.path.join(atlas_registration, os.path.basename(preop_T1_mask))}"
        os.system(cmd)
        cmd = f"fslmaths  {preop_T1} -mul {preop_T1_mask} {preop_T1_bet}"
        os.system(cmd)
        
        #intemediary files output names
        preop_T1 = os.path.join(atlas_registration, os.path.basename(preop_T1))
        preop_T1_mask = os.path.join(atlas_registration, os.path.basename(preop_T1_mask))
        
        preop_T1_std = f"{splitext(splitext(preop_T1)[0])[0]}_std.nii.gz"
        preop_T1_bet_std = f"{splitext(splitext(preop_T1_bet)[0])[0]}_std.nii.gz"
        
        #Begin Pipeline: Orient all images to standard RAS
        
        fillerString = "\n###########################"*3
        print(f"\n\n{fillerString}\nPart 1 of 2\nReorientation of Images\nEstimated time: 10-30 seconds{fillerString}\nReorient all images to standard RAS\n")
        cmd = f"fslreorient2std {preop_T1} {preop_T1_std}"; print(cmd); os.system(cmd)
        cmd = f"fslreorient2std {preop_T1_bet} {preop_T1_bet_std}"; print(cmd); os.system(cmd)
        
        #visualize
        utils.show_slices(f"{preop_T1_std}", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(atlas_registration, "pic_T1.png")  )
        utils.show_slices(f"{preop_T1_bet_std}", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(atlas_registration, "pic_T1_brain.png"))
        
        
        #Registration of MNI to patient space (atlases are all in MNI space, so using this warp to apply to the atlases)
        print(f"\n\n{fillerString}\nPart 2 of 2\nMNI and atlas registration\nEstimated time: 1-2+ hours{fillerString}\nRegistration of MNI template to patient space\n")
        
        
        #linear reg of MNI to preopT1 space
        if not utils.checkIfFileExists(f"{mni_images}_flirt.nii.gz", printBOOL=False ):
            cmd = f"flirt -in {paths.MNI_TEMPLATE_BRAIN} -ref {preop_T1_bet_std} -dof 12 -out {mni_images}_flirt -omat {mni_images}_flirt.mat -v"; print(cmd);os.system(cmd)
        #non linear reg of MNI to preopT1 space
        utils.show_slices(f"{mni_images}_flirt.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(atlas_registration, "pic_mni_to_T1_flirt.png"))
        print("\n\nLinear registration of MNI template to image is done\n\nStarting Non-linear registration:\n\n\n")
        if not utils.checkIfFileExists(f"{mni_images}_fnirt.nii.gz", printBOOL=False ):
            cmd = f"fnirt --in={paths.MNI_TEMPLATE} --ref={preop_T1_std} --aff={mni_images}_flirt.mat --iout={mni_images}_fnirt -v --cout={mni_images}_coef --fout={mni_images}_warp"; print(cmd); os.system(cmd)
        utils.show_slices(f"{mni_images}_fnirt.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(atlas_registration, "pic_mni_to_T1_fnirt.png"))
        print(f"\n\n{fillerString}\nDone{fillerString}\n\n\n\n")
    else:
        print(f"\n\n\n\nMNI registration already performed\n\n\n\n")
        

# %% Apply warp to relevant atlases



def get_atlases_from_priority(atlas_metadata_json, priority_type = "structure", priority_level_max = 2):
    """
    Get list of atlases from the metadata file. The atlases that are output are based on their priority rating (made by me, Revell). A priority of 1 indicates we definitely want to use that atlas. A priority of 3 is almost never use. A priority of 4 means never use it (e.g. random atlas with 10k regions. Was used just for brain atlas paper as proof of concept)
    """
    
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


def applywarp_to_atlas(atlas_directory, atlas_paths, preop_T1_std, mni_warp, atlas_registration):
    full_atlas_paths = []
    for a in range(len(atlas_paths)):
        utils.checkPathError( f"{os.path.join(atlas_directory, atlas_paths[a])}" )
        full_atlas_paths.append( f"{os.path.join(atlas_directory, atlas_paths[a])}")
    utils.checkPathError(mni_warp)
    utils.checkPathError(preop_T1_std)
    utils.checkPathError(atlas_registration)
    print("\n\n\nApplying xform to atlases (MNI space to patient T1 space)")
    for a in range(len(full_atlas_paths)):
        atlasName = basename(splitext(splitext(full_atlas_paths[a])[0])[0])
        outputAtlasName = join(atlas_registration, atlasName + ".nii.gz")
        #if not os.path.exists(outputAtlasName):
        print(f"\r{np.round((a+1)/len(full_atlas_paths)*100,2)}%   {atlasName[0:25]}                       ", end = "\r")
        if not utils.checkIfFileExists(outputAtlasName, printBOOL=False ):
            cmd = f"applywarp -i { full_atlas_paths[a]} -r {preop_T1_std} -w {mni_warp} --interp=nn -o {outputAtlasName}"; os.system(cmd)
        #else: print(f"File exists: {outputAtlasName}")
    print("\n\n\nDone\n\n\n")

def parse_get_structural_connectivity(sub, atlas_paths, paths, path_fib, path_trk, preop_T1, atlas_registration, structural_matrices):
    for a in range(len(atlas_paths)):
        atlasName = basename(splitext(splitext(atlas_paths[a])[0])[0])
        print(f"\n\n\n\n\n\n{np.round((a+1)/len(atlas_paths)*100,2)}%   {atlasName[0:25]}  \n\n\n\n\n\n"   )
        atlas = join(atlas_registration, atlasName + ".nii.gz")
        
        structural_matrices_output = join(structural_matrices, f"sub-{sub}")
        get_structural_connectivity(paths.BIDS, paths.DSI_STUDIO_SINGULARITY, path_fib, path_trk, preop_T1, atlas, structural_matrices_output )
    print("\n\n\nDone\n\n\n")


#%%
#parsing thru data

def get_patients_with_dwi(subject_list, paths, dataset = "PIER", SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"):
    sfc_patient_list = []
    N = len(subject_list)
    for i in range(N):
        sub = subject_list[i]

        dwi_path = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "dwi", "*_dwi.nii.gz")
        dwi_path_json = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "dwi", "*_dwi.json")
        dwi_path_bvec = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "dwi", "*_dwi.bvec")
        dwi_path_bval = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "dwi", "*_dwi.bval")
        dwi_path_fmap_phasediff = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "fmap", "*_phasediff.nii.gz")
        dwi_path_fmap_topup = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "fmap", "*_epi.nii.gz")
        dwi_path_fmap_T1 = join(paths.BIDS, "PIER", f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}", "anat", f"*acq-3D_T1w.nii.gz")
        exists = all([utils.checkIfFileExistsGlob(dwi_path, printBOOL = False), utils.checkIfFileExistsGlob(dwi_path_json, printBOOL = False) ,
                      utils.checkIfFileExistsGlob(dwi_path_bvec, printBOOL = False), utils.checkIfFileExistsGlob(dwi_path_bval, printBOOL = False) ,
                      utils.checkIfFileExistsGlob(dwi_path_fmap_T1, printBOOL = False)])
        #check if all the necessary imaging files for correction exists
        if exists:
            #check if phasediff or topup exists
            if any([utils.checkIfFileExistsGlob(dwi_path_fmap_phasediff, printBOOL = False) , utils.checkIfFileExistsGlob(dwi_path_fmap_topup, printBOOL = False)]):
                sfc_patient_list.append(sub)
    sfc_patient_list = list(np.unique(sfc_patient_list))
    return sfc_patient_list


def print_dwi_image_correction_QSIprep(dwi_correction_patient_list, paths, dataset = "PIER"):
    cmd = []
    for i in range(len(dwi_correction_patient_list)):
        sub = dwi_correction_patient_list[i]
        cmd.append(dwi_image_correction_QSIprep(sub, paths, dataset = dataset))
    return cmd


def get_tracts_loop_through_patient_list(patient_list, paths, singularity_bind_path = None, path_dsiStudioSingularity = None, SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"):
    if singularity_bind_path == None:
        singularity_bind_path = paths.BIDS
    if path_dsiStudioSingularity == None:
        path_dsiStudioSingularity = paths.DSI_STUDIO_SINGULARITY

    N = len(patient_list)
    for i in range(N):
        sub = patient_list[i]
        ses = basename(glob.glob( join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep",  f"sub-{sub}", f"ses-{SESSION_RESEARCH3T}"))[0])[4:]
        path_dwi = join(paths.BIDS_DERIVATIVES_QSIPREP, "qsiprep", f"sub-{sub}", f"ses-{ses}", "dwi", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
        path_tracts = join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", f"ses-{ses}")
        utils.checkPathAndMake(path_tracts, path_tracts,  printBOOL = False)
        utils.checkIfFileExistsGlob(  path_dwi, printBOOL = False)

        trk_name = f"{join(path_tracts, utils.baseSplitextNiiGz(path_dwi)[2])}.trk.gz"
        if utils.checkIfFileDoesNotExist(trk_name):
            get_tracts(singularity_bind_path, path_dsiStudioSingularity, path_dwi, path_tracts)




def batch_get_structural_connectivity_with_mni_registration(patient_list, atlas_metadata_json, paths, SESSION_RESEARCH3T, start = None, stop = None):
    # %% MNI registration of patient's T1
    if start == None:
        start = 0
    if stop == None:
        stop = len(patient_list)
    for i in range(start, stop):
        sub = patient_list[i]
        print(f"Part 1/3: MNI registration \n{(i+1-start)}/{stop-start}; {np.round((i+1)/(stop-start)*100,2)}% \n\n\n\n\n")
        mni_registration_to_T1(sub, paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)
        # Atlas registration
        print(f"Part 2/3: Atlas registration (Apply warps)\n{sub}:  {(i+1-start)}/{stop-start}; {np.round((i+1)/(stop-start)*100,2)}% \n\n\n\n\n")
        # Get relevant paths
        atlas_paths =  get_atlases_from_priority(atlas_metadata_json, priority_type = "structure", priority_level_max = 2)
        atlas_registration, structural_matrices, mni_images, mni_warp, ses, preop_T1, preop_T1_std, path_fib, path_trk = get_atlas_registration_and_tractography_paths(sub = sub, paths = paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)
        #Apply warp to atlases
        applywarp_to_atlas(atlas_directory = paths.REVELLLAB, atlas_paths = atlas_paths, preop_T1_std = preop_T1_std, mni_warp = mni_warp, atlas_registration = atlas_registration)
        print(f"Part 3/3: Calculate Structural Connectivity\ n{sub}:  {(i+1-start)}/{stop-start}; {np.round((i+1)/(stop-start)*100,2)}% \n\n\n\n\n")
        # Calculate structural connectivity
        parse_get_structural_connectivity(sub, atlas_paths, paths, path_fib, path_trk, preop_T1, atlas_registration, structural_matrices)
        print(f"Done\n{sub}:  {(i+1)}/{stop-start}; {np.round((i+1-start)/(stop-start)*100,2)}% \n\n\n\n\n")