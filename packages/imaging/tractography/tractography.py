#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import os
from os.path import join
from revellLab.packages.utilities import utils

#%% Input


#%% functions

def getTracts(pathBind, dsiStudioSingularityPatah, pathDWI, pathOutDir):
    utils.checkPathError(pathBind)
    utils.checkPathError(dsiStudioSingularityPatah)
    utils.checkPathError(pathDWI)
    utils.checkPathError(pathOutDir)
    
    base, split, basesplit = utils.baseSplitextNiiGz(pathDWI)
    sourceName = f"{join(pathOutDir, basesplit)}.src.gz"
    fibName = f"{join(pathOutDir, basesplit)}.fib.gz"
    trkName = f"{join(pathOutDir, basesplit)}.trk.gz"
    print("\n\nCreating Source File in DSI Studio\n\n")
    cmd = f"singularity exec --bind {pathBind} {dsiStudioSingularityPatah} dsi_studio --action=src --source={pathDWI} --output={sourceName}"
    os.system(cmd)
    print("\n\nCreating Reconstruction File in DSI Studio\n\n")
    cmd = f"singularity exec --bind {pathBind} {dsiStudioSingularityPatah} dsi_studio --action=rec --source={sourceName} --method=4 --param0=1.25"
    os.system(cmd)
    #rename dsistudio output of .fib file because there is no way to output a specific name on cammnad line
    cmd = f"mv {sourceName}*.fib.gz {fibName}"
    os.system(cmd)
    print("\n\nCreating Tractography File in DSI Studio\n\n")
    cmd = f"singularity exec --bind {pathBind} {dsiStudioSingularityPatah} dsi_studio --action=trk --source={fibName} --min_length=30 --max_length=800 --thread_count=16 --fiber_count=1000000 --output={trkName}"
    os.system(cmd)

    

def getStructuralConnectivity(pathBind, dsiStudioSingularityPatah, pathFIB, pathTRK, preop3T, atlas, output ):
    utils.checkPathError(pathBind)
    utils.checkPathError(dsiStudioSingularityPatah)
    utils.checkPathError(pathFIB)
    utils.checkPathError(pathTRK)
    utils.checkPathError(preop3T)
    utils.checkPathError(atlas)
    cmd = f"singularity exec --bind {pathBind} {dsiStudioSingularityPatah} dsi_studio --action=ana --source={pathFIB} --tract={pathTRK} --t1t2={preop3T} --connectivity={atlas} --connectivity_type=pass --connectivity_threshold=0 --output={output}"
    os.system(cmd)

#%% Input names

