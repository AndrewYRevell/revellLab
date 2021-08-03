#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import sys
import os
import time
import pandas as pd
import copy
from os import listdir
from os.path import join, isfile
from os.path import splitext, basename
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
#import seaborn as sns
import glob
import dirsync
from revellLab.packages.utilities import utils

# %% To be run on python 3.6 on Borel itself:

import sys
import json
import os
from os import listdir
from os.path import join, isfile
from os.path import splitext, basename
import glob


def executeCommand(cmd, printBOOL = True):
    if printBOOL: print(f"\n\nExecuting Command Line: \n{cmd}\n\n")
    os.system(cmd)

def checkPathAndMake(pathToCheck, pathToMake, make = True, printBOOL = True, exist_ok = True):
    """
    Check if pathToCheck exists. If so, then option to make a second directory pathToMake (may be same as pathToCheck)
    """
    if not os.path.exists(pathToCheck):
        if printBOOL: print(f"\nFile or Path does not exist:\n{pathToCheck}" )
    if make:
        if os.path.exists(pathToMake):
            if printBOOL: print(f"Path already exists\n{pathToMake}")
        else:
            os.makedirs(pathToMake, exist_ok=exist_ok)
            if printBOOL: print("Making Path")

def checkPathError(path):
    """
    Check if path exists
    """
    if not os.path.exists(path):
        raise IOError(f"\n\n\n\nPath or file does not exist:\n{path}\n\n\n\n" )

def checkIfFileDoesNotExist(path, returnOpposite = False, printBOOL = True):
    if not (os.path.exists(path)):
        if printBOOL: print(f"\nFile does not exist:\n    {path}\n\n")
        if returnOpposite:
            return False
        else:
            return True
    else:
        if printBOOL: print(f"\nFile exists:\n    {path}\n\n")
        if returnOpposite:
            return True
        else:
            return False


LINUX_BIDSserver = join("/gdrive/public/DATA/Human_Data/BIDS", "PIER")



CORNBLATH = join("/gdrive/public/USERS/elicorn/projects/BIDSfMRI")

checkPathError(CORNBLATH)

directories = os.listdir(CORNBLATH)


for i in range(len(directories)):
    file = directories[i]
    if "sub-RID"  in file :
        sub = file
        ses = os.listdir(join(CORNBLATH,sub))
        if len(ses) > 1:
            print(f"\n\n\n{sub} has more than one session")
        else:
            ses = ses[0]
            if checkIfFileDoesNotExist(join(CORNBLATH,sub, ses, "func"),  returnOpposite = True,printBOOL = False):
                images = os.listdir(join(CORNBLATH,sub, ses, "func"))
                for x in range(len(images)):
                    image_file = join(CORNBLATH,sub, ses, "func", images[x])
                    PIER_folder_sub_ses = join(LINUX_BIDSserver, sub, ses)
                    #checkPathError(PIER_folder_sub_ses)
                    PIER_folder_sub_ses_func =  join(PIER_folder_sub_ses, "func")
                    checkPathAndMake(PIER_folder_sub_ses_func, PIER_folder_sub_ses_func,printBOOL = False)
                    #command to copy files
                    #if file already exists, dont copy
                    if checkIfFileDoesNotExist(join(PIER_folder_sub_ses_func, images[x])):
                        cmd = f"cp {image_file} {PIER_folder_sub_ses_func}"
                        executeCommand(cmd)


#updating intended for in fmap json files:





#%%
#on local

LINUX_BIDSserver = join("/home/arevell/borel/DATA/Human_Data/BIDS", "PIER")

#
#updating intended for in fmap json files:

directories = os.listdir(LINUX_BIDSserver)

ses_research3T = "research3Tv[0-9][0-9]"

for i in range(len(directories)):
    file = directories[i]
    if "sub-RID"  in file :
        sub = file


        if utils.checkIfFileExistsGlob( join(LINUX_BIDSserver,sub, f"ses-{ses_research3T}")  , printBOOL=False):

            ses_folder = glob.glob(join(LINUX_BIDSserver,sub, f"ses-{ses_research3T}"))[0]
            ses = basename(ses_folder)[4:]


            if utils.checkIfFileExistsGlob(join(LINUX_BIDSserver,sub, f"ses-{ses}", "func", "*rest_bold.nii.gz"), printBOOL=False): #if fMRI exists
                fMRI_path = glob.glob(join(LINUX_BIDSserver,sub, f"ses-{ses}", "func", "*rest_bold.nii.gz"))[0]
                fMRI_path_basename = basename(fMRI_path)
                fMRI_IntendedFor = join(f"ses-{ses}", "func", f"{fMRI_path_basename}")

                fmap_names = ["*_epi.json", "*_phasediff.json"]
                for ff in range(len(fmap_names)):
                    if utils.checkIfFileExistsGlob(join(LINUX_BIDSserver,sub, f"ses-{ses}", "fmap", fmap_names[ff]) , printBOOL=False): #if fmap json TOPUP exists
                        fmap_file = glob.glob(join(LINUX_BIDSserver,sub, f"ses-{ses}", "fmap",  fmap_names[ff]))[0]
                        with open(fmap_file) as f: fmap = json.load(f)
                        if 'IntendedFor' in fmap: #if IntendedFor already exists, then add to it, else, write into it
                            intendedFor = fmap["IntendedFor"]
                            if not fMRI_IntendedFor in intendedFor:
                                if type(intendedFor) is list: #if list, meaning there are multiple intended fors already
                                    intendedFor = intendedFor + [fMRI_IntendedFor]
                                else:
                                    intendedFor = [intendedFor] + [fMRI_IntendedFor]
                                fmap["IntendedFor"] = intendedFor
                        else:
                            fmap["IntendedFor"] = intendedFor
                        with open(fmap_file, 'w', encoding='utf-8') as f: json.dump(fmap, f, ensure_ascii=False, indent=4)
                        print(f'writing {sub}, ses-{ses}, {fmap_names[ff][2:]}')














