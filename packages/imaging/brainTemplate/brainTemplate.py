#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import os
import sys
import copy
import json
import subprocess
import pkg_resources
import multiprocessing

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir
from scipy import ndimage 
from glob import glob, iglob
from itertools import repeat
from os.path import join, isfile, splitext, basename

from revellLab.packages.utilities import utils


revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
atlasDirectory = join(tools, "atlases", "atlases" )
atlasLabelDirectory = join(tools, "atlases", "atlasLabels" )

MNItemplatePath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz")
MNItemplateBrainPath = join( tools, "mniTemplate", "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz")

OASIStemplatePath = join( tools, "templates","OASIS" ,"T_template0.nii.gz")
OASISprobabilityPath = join( tools, "templates", "OASIS", "T_template0_BrainCerebellumProbabilityMask.nii.gz")

ANTSPATH="/Users/andyrevell/ANTS/install/bin"


BIDS = "/Users/andyrevell/research/data/BIDS"
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "brainTemplates")


subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]
subjects = [basename(item) for item in subDir ]

#%% functions

ControlsList = ["RID0285",
                "RID0286",
                "RID0287",
                "RID0288",
                "RID0289",
                "RID0290",
                "RID0291",
                "RID0292",
                "RID0297",
                "RID0599",
                "RID0600",
                "RID0505",
                "RID0602",
                "RID0603",
                "RID0604",
                "RID0615",
                "RID0682",
                "RID0683"]


T1wDir = join(outputDir , "T1w")
T1wImagesDir = join(T1wDir , "images")
T1wImagesPATEINTSDir = join(T1wImagesDir , "patients")
T1wImagesCONTROLSDir = join(T1wImagesDir , "controls")
utils.checkPathAndMake(outputDir, T1wImagesPATEINTSDir)
utils.checkPathAndMake(outputDir, T1wImagesCONTROLSDir)

T1wDirTemplate = join(T1wDir , "template")
utils.checkPathAndMake(outputDir, T1wDirTemplate)
T1wDirTemplate2 = join(T1wDir , "templateIteration2")
utils.checkPathAndMake(outputDir, T1wDirTemplate2)


T1wDirTemplatePatients = join(T1wDir , "patients")
utils.checkPathAndMake(outputDir, T1wDirTemplatePatients)
T1wDirTemplatePatients01 = join(T1wDirTemplatePatients , "templateIteration01")
utils.checkPathAndMake(outputDir, T1wDirTemplatePatients01)
T1wDirTemplatePatients02 = join(T1wDirTemplatePatients , "templateIteration02")
utils.checkPathAndMake(outputDir, T1wDirTemplatePatients02)


acq = "3D"
ses = "research3T*"


#going thru BIDS directory to get paths of controls and patients

patients = []
controls = []
for s in range(len(subDir)):
    #sub = "RID0588"
    subjectPath = subDir[s]
    sub = basename(subjectPath)[4:]
    print(sub)
    if utils.checkIfFileExistsGlob( join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" ) ):
        if sub not in ControlsList:
            patients.append(glob(join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" )))
            file = glob(join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" ))[0]
            if utils.checkIfFileDoesNotExistGlob(f"{file}"):
                utils.executeCommand(f"cp {file} {T1wImagesPATEINTSDir}")
            
        if sub in ControlsList:
            controls.append(glob(join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" )))
            file = glob(join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" ))[0]
            if utils.checkIfFileDoesNotExistGlob(f"{file}"):    
                utils.executeCommand(f"cp {file} {T1wImagesCONTROLSDir}")



patients = pd.DataFrame(np.array(patients))
controls = pd.DataFrame(np.array(controls))

patients.to_csv(  join(T1wDir, "subjectsPatientsEpilepsyAll.csv"), header = False , index = False   )
controls.to_csv(  join(T1wDir, "subjectsControls.csv"), header = False , index = False   )




utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 6 -o {join(T1wDirTemplate, 'antsBTP')} {join(T1wImagesCONTROLSDir, '*_T1w.nii.gz')}"  )
utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 12 -z {join(T1wDirTemplate, 'antsBTPtemplate0.nii.gz')} -o {join(T1wDirTemplate2, 'antsBTP')} {join(T1wImagesCONTROLSDir, '*_T1w.nii.gz')}"  )


utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 24 -z {join(T1wDirTemplate2, 'antsBTPtemplate0.nii.gz')} -o {join(T1wDirTemplatePatients01, 'PIER_')} {join(T1wImagesPATEINTSDir, '*_T1w.nii.gz')}"  )
utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 24 -z {join(T1wDirTemplatePatients01, 'PIER_template0.nii.gz')} -o {join(T1wDirTemplatePatients02, 'PIER_')} {join(T1wImagesPATEINTSDir, '*_T1w.nii.gz')}"  )



"""
antsBrainExtraction.sh -d 3 -a antsBTPtemplate0.nii.gz -e /media/arevell/sharedSSD/linux/revellLab/tools/templates/OASIS/T_template0.nii.gz -m /media/arevell/sharedSSD/linux/revellLab/tools/templates/OASIS/T_template0_BrainCerebellumProbabilityMask.nii.gz -k true -o ./template_
"""

#%%Analysis

Tctrl_img = nib.load(join(T1wDirTemplate2, 'antsBTPtemplate0.nii.gz'))
Tctrl_Data = Tctrl_img.get_fdata()

Tpt_img = nib.load(join(T1wDirTemplatePatients01, 'PIER_template0.nii.gz'))
Tpt_Data = Tpt_img.get_fdata()

mask = nib.load(join(T1wDir, "brainExtraction", 'template_BrainExtractionMask.nii.gz'))
mask_data = mask.get_fdata()

utils.show_slices(Tctrl_Data, isPath=False)
utils.show_slices(Tpt_Data, isPath=False)
utils.show_slices(mask_data, isPath=False)





Tctrl_Data[np.where(mask_data <= 0)] = 0
Tpt_Data[np.where(mask_data <= 0)] = 0
utils.show_slices(Tctrl_Data, isPath=False)
utils.show_slices(Tpt_Data, isPath=False)
#normalize





Tctrl_DataNORM = Tctrl_Data/np.max(Tctrl_Data)
Tpt_DataNORM = Tpt_Data/np.max(Tpt_Data)

utils.show_slices(Tctrl_DataNORM, isPath=False)
utils.show_slices(Tpt_DataNORM, isPath=False)


difference = Tpt_DataNORM - Tctrl_DataNORM
difference[np.where(mask_data <= 0)] = np.min(difference)

utils.show_slices(difference, isPath=False)

inbrain_diff = difference[np.where(mask_data > 0)] 
sns.histplot(inbrain_diff   )

diffIMG = nib.Nifti1Image(difference, Tpt_img.affine)
nib.save(diffIMG, join(T1wDirTemplatePatients01, 'difference.nii.gz'))

diffThreshold = copy.deepcopy(difference)
diffThreshold[np.where(diffThreshold <= 0.17)] = 0
diffthrehsoldIMG = nib.Nifti1Image(diffThreshold, Tpt_img.affine)
nib.save(diffthrehsoldIMG, join(T1wDirTemplatePatients01, 'difference_threshold.nii.gz'))




#%%
patientList = []

patientPaths = glob(join(T1wDirTemplatePatients01, "PIER_template0sub-RID*_ses-research3Tv*_T1w*WarpedToTemplate.nii.gz"))

for i in range(len(patientPaths)):
    print(basename(patientPaths[i]))
    img = nib.load(patientPaths[i])
    imgData = img.get_fdata()
    imgData[np.where(mask_data <= 0)] = 0
    imgDataNORM = imgData/np.max(imgData)
    #utils.show_slices(imgDataNORM, isPath=False)
    patientList.append( imgDataNORM  )
    
patientIMGs = np.stack(patientList,axis=0)

patientsstd = np.std(patientIMGs, axis=0)

utils.show_slices(patientsstd, isPath=False, cmap="BuGn_r")
patientsstdIMG = nib.Nifti1Image(patientsstd, Tpt_img.affine)
nib.save(patientsstdIMG, join(T1wDirTemplatePatients01, 'std.nii.gz'))





controlList = []

controlPaths = glob(join(T1wDirTemplate2, "antsBTPtemplate0sub-RID*_ses-research3Tv*_T1w*WarpedToTemplate.nii.gz"))

for i in range(len(controlPaths)):
    print(basename(controlPaths[i]))
    img = nib.load(controlPaths[i])
    imgData = img.get_fdata()
    imgData[np.where(mask_data <= 0)] = 0
    imgDataNORM = imgData/np.max(imgData)
    #utils.show_slices(imgDataNORM, isPath=False)
    controlList.append( imgDataNORM  )


controlIMGs = np.stack(controlList,axis=0)

controlsstd = np.std(controlIMGs, axis=0)

utils.show_slices(controlsstd, isPath=False, cmap="BuGn_r")
controlsstdIMG = nib.Nifti1Image(controlsstd, Tpt_img.affine)
nib.save(controlsstdIMG, join(T1wDirTemplatePatients01, 'std_control.nii.gz'))


stdDIFF = patientsstd - controlsstd
stdDIFF[np.where(mask_data <= 0)] = np.min(stdDIFF)
utils.show_slices(stdDIFF, isPath=False, cmap="BuGn_r")
stdDIFFIMG = nib.Nifti1Image(stdDIFF, Tpt_img.affine)
nib.save(stdDIFFIMG, join(T1wDirTemplatePatients01, 'std_diff.nii.gz'))

inbrain = stdDIFF[np.where(mask_data > 0)]

sns.histplot( inbrain    )

threshold = 0.05
stdDIFFThreshold = copy.deepcopy(stdDIFF)
stdDIFFThreshold[np.where(stdDIFFThreshold < threshold)] = 0
utils.show_slices(stdDIFFThreshold, isPath=False, cmap="BuGn_r")

stdDIFFTHRESHOLDIMG = nib.Nifti1Image(stdDIFFThreshold, Tpt_img.affine)
nib.save(stdDIFFTHRESHOLDIMG, join(T1wDirTemplatePatients01, 'std_diff_threshold.nii.gz'))







from scipy.stats import kstest, ks_2samp, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection as fdr

pvalIMG2 = np.zeros(  Tpt_DataNORM.shape  )



x,y,z = np.where(mask_data > 0)

for i in range(len(x)):
    print(i)
    a,b,c = x[i], y[i], z[i]
    
    pvalIMG2[a,b,c] = mannwhitneyu( controlIMGs[:,a,b,c], patientIMGs[:,a,b,c]   )[1]
    #pvalIMG[a,b,c] = ks_2samp( controlIMGs[:,a,b,c], patientIMGs[:,a,b,c]   )[1]


sns.histplot( pvalIMG2[np.where(mask_data > 0)]    )

np.min(pvalIMG2[np.where(mask_data > 0)])

inv = copy.deepcopy(pvalIMG2)
inv[np.where(mask_data > 0)]  = 1. / inv[np.where(mask_data > 0)] 
inv10 = np.log10(inv)
utils.show_slices(np.log10(inv), isPath=False)
inv10IMG = nib.Nifti1Image(inv10, Tpt_img.affine)
nib.save(inv10IMG, join(T1wDirTemplatePatients01, 'pval_diff_log10inv.nii.gz'))

pvalIMG2_threshold = copy.deepcopy(pvalIMG2)
pvalIMG2_threshold[np.where(pvalIMG2_threshold > 0.5/len(x)    )] = 0
utils.show_slices(pvalIMG2_threshold, isPath=False)


trues = fdr(pvalIMG2[np.where(mask_data > 0)],  alpha=0.05)[0]
any(trues)
pvalIMG2_fdr = copy.deepcopy(pvalIMG2)
pvalIMG2_fdr[np.where(mask_data > 0)] = fdr(pvalIMG2[np.where(mask_data > 0)])[1]
pvalIMG2_fdr_threshold = copy.deepcopy(pvalIMG2_fdr)
pvalIMG2_fdr_threshold[np.where(pvalIMG2_fdr_threshold > 0.05)] = 0
utils.show_slices(pvalIMG2_fdr_threshold, isPath=False)
sns.histplot( pvalIMG2_fdr_threshold[np.where(mask_data > 0)]    )

invFDR = copy.deepcopy(pvalIMG2_fdr)
invFDR[np.where(mask_data > 0)]  = 1. / invFDR[np.where(mask_data > 0)] 
utils.show_slices(pvalIMG2_fdr_threshold, isPath=False)


(pvalIMG2_fdr_threshold>0).any()






#%%



ControlsList = ["RID0285",
                "RID0286",
                "RID0287",
                "RID0288",
                "RID0289",
                "RID0290",
                "RID0291",
                "RID0292",
                "RID0297",
                "RID0599",
                "RID0600",
                "RID0505",
                "RID0602",
                "RID0603",
                "RID0604",
                "RID0615",
                "RID0682",
                "RID0683"]


multimodal = join(outputDir , "multimodal")
imagesDir = join(multimodal , "images")
imagesPatients = join(imagesDir , "patients")
imagesControls = join(imagesDir , "controls")
utils.checkPathAndMake(outputDir, imagesPatients)
utils.checkPathAndMake(outputDir, imagesControls)

templates = join(multimodal , "template")
utils.checkPathAndMake(outputDir, templates)

templatesPatients= join(templates , "patients")
templatesControls= join(templates , "controls")
utils.checkPathAndMake(templatesPatients, templatesPatients)
utils.checkPathAndMake(templatesPatients, templatesControls)


iteration1Patients = join(templatesPatients , "interation01")
iteration1Controls = join(templatesControls , "interation01")
iteration2Controls = join(templatesControls , "interation02")

utils.checkPathAndMake(outputDir, iteration1Patients)
utils.checkPathAndMake(outputDir, iteration1Controls)
utils.checkPathAndMake(outputDir, iteration2Controls)

acq = "3D"
ses = "research3T*"


#going thru BIDS directory to get paths of controls and patients

patients = []
controls = []
for s in range(len(subDir)):
    #sub = "RID0588"
    subjectPath = subDir[s]
    sub = basename(subjectPath)[4:]
    T1 = join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" )
    T2 = join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T2w.nii.gz" )
    FLAIR = join(subjectPath,  f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_FLAIR.nii.gz" )
    if utils.checkIfFileExistsGlob(T1, printBOOL=False) and utils.checkIfFileExistsGlob(T2, printBOOL=False) and utils.checkIfFileExistsGlob(FLAIR, printBOOL=False):
        print(sub)
        if utils.checkIfFileExistsGlob(T1, printBOOL=False): T1name = glob(T1)[0]
        if utils.checkIfFileExistsGlob(T2, printBOOL=False): T2name = glob(T2)[0]
        if utils.checkIfFileExistsGlob(FLAIR, printBOOL=False): FLAIRname = glob(FLAIR)[0]
        if sub not in ControlsList: 
            patients.append(f"{ join(imagesPatients, basename(T1name))},{join(imagesPatients, basename(T2name))},{join(imagesPatients, basename(FLAIRname))}")
            utils.executeCommand(f"cp {T1name} {imagesPatients}")
            utils.executeCommand(f"cp {T2name} {imagesPatients}")
            utils.executeCommand(f"cp {FLAIRname} {imagesPatients}")
        if sub in ControlsList: 
            #controls.append(f"{ join(imagesControls, basename(T1name))},{join(imagesControls, basename(T2name))},{join(imagesControls, basename(FLAIRname))}")
            controls.append(f"{ basename(T1name)},{ basename(T2name)},{ basename(FLAIRname)}")
            utils.executeCommand(f"cp {T1name} {imagesControls}")
            utils.executeCommand(f"cp {T2name} {imagesControls}")
            utils.executeCommand(f"cp {FLAIRname} {imagesControls}")
        

patients = pd.DataFrame(np.array(patients))
controls = pd.DataFrame(np.array(controls))

patients.to_csv(  join(multimodal, "subjectsPatientsEpilepsyAll.csv"), header = False , index = False   )
controls.to_csv(  join(multimodal, "subjectsControls.csv"), header = False , index = False   )

cmd = f"{ANTSPATH}/antsMultivariateTemplateConstruction2.sh -d 3 -k 3 -v 16 -c 2 -j 12 -r 1 -o {join(iteration1Controls, 'PIER_controls_')} {join(imagesControls, '*_T1w.nii.gz')},{join(imagesControls, '*_T2w.nii.gz')},{join(imagesControls, '*_FLAIR.nii.gz')}" 
print(cmd)
utils.executeCommand( cmd )

utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 12 -z {join(T1wDirTemplate, 'antsBTPtemplate0.nii.gz')} -o {join(T1wDirTemplate2, 'antsBTP')} {join(T1wImagesCONTROLSDir, '*_T1w.nii.gz')}"  )


utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 24 -z {join(T1wDirTemplate2, 'antsBTPtemplate0.nii.gz')} -o {join(T1wDirTemplatePatients01, 'PIER_')} {join(T1wImagesPATEINTSDir, '*_T1w.nii.gz')}"  )
utils.executeCommand( f"{ANTSPATH}antsMultivariateTemplateConstruction2.sh -d 3 -v 32 -c 2 -j 24 -z {join(T1wDirTemplatePatients01, 'PIER_template0.nii.gz')} -o {join(T1wDirTemplatePatients02, 'PIER_')} {join(T1wImagesPATEINTSDir, '*_T1w.nii.gz')}"  )





































































