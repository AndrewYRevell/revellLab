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
    cmd = f"singularity exec --bind {singularity_bind_path} {path_dsiStudioSingularity} dsi_studio --action=trk --source={fib_name} --min_length=30 --max_length=800 --thread_count=16 --fiber_count=1000000 --output={trk_name}"
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
        path_tracts = join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", "tracts")
        utils.checkPathAndMake(path_tracts, path_tracts,  printBOOL = False)
        utils.checkIfFileExistsGlob(  path_dwi, printBOOL = False)

        trk_name = f"{join(path_tracts, utils.baseSplitextNiiGz(path_dwi)[2])}.trk.gz"
        if utils.checkIfFileDoesNotExist(trk_name):
            get_tracts(singularity_bind_path, path_dsiStudioSingularity, path_dwi, path_tracts)




