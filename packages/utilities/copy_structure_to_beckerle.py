#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:31:02 2022

@author: arevell
"""

import os
from revellLab.packages.utilities import utils
import numpy as np
import glob

path1 = "/media/arevell/data/linux/data/BIDS/derivatives/structural_connectivity/structural_matrices"
path2 = "/home/arevell/Desktop/Berckerle_structure"


subs = np.sort(os.listdir(path1))
for i in range(len(subs)):
    sub = subs[i]
    
    print(f"\r{sub}   {i+1}/{len(subs)},    {np.round((i+1)/len(subs)*100,1)}       ", end = "\r")
    path_sub = os.path.join(path2, sub)
    utils.checkPathAndMake(path_sub, path_sub, printBOOL=False )
    
    sc_path = os.path.join(path1, sub, "ses-research3Tv[0-9][0-9]", "matrices", "*connectogram.txt")

    utils.checkPathErrorGlob(sc_path)
    
    sc_paths_all = glob.glob(sc_path)    
    
    for sc in range(len(sc_paths_all)):
        sc_p = sc_paths_all[sc]
        cmd = f"cp {sc_p} {path_sub}"
        utils.executeCommand(cmd, printBOOL=False)
        
