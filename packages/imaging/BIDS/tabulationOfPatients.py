#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:37:24 2021

@author: arevell
"""

import sys
import os
import pandas as pd
import copy
from os import listdir, walk
from  os.path import join, isfile, isdir, splitext, basename, relpath, dirname
import numpy as np
from itertools import repeat
from revellLab.packages.utilities import utils
import json
import smtplib, ssl
import pkg_resources
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
import time
from glob import glob

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles

#%% Getting directories


dataset= "PIER"

BIDSlocal = join("/media/arevell/sharedSSD/linux/data/BIDS", dataset)

LOCAdir = [join(BIDSlocal, o) for o in listdir(BIDSlocal) if isdir(join(BIDSlocal, o)) and basename(join(BIDSlocal, o))[0:3] == "sub"  ]

LOCAsubj = [basename(item) for item in LOCAdir ]



#%% Copy from server to local


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
RNSlist =      ["RID0015",
                "RID0030",
                "RID0165",
                "RID0171",
                "RID0186",
                "RID0206",
                "RID0252",
                "RID0272",
                "RID0280",
                "RID0328",
                "RID0334",
                "RID0337"]


implantTypeFile = join("/media/arevell/sharedSSD/linux/data/metadata", "patientsImplantType.csv")

implantType = pd.read_csv(implantTypeFile)


chart = pd.DataFrame( columns =  ["RID", "implant", "types", "laterality", "RNS", "research3T", "v00", "v01", "v02", "v03", "v04", "dwi", "control"])
ver = ["v00", "v01", "v02", "v03", "v04"]
for s in range(0,len(LOCAdir)):
#sub = "RID0030"
    sub = LOCAsubj[s][4:]
    print(sub)
    
    if sub == "RIDXXXX":
        continue
    rid = int(sub[3:])
    
    
    implantTypeRID = np.array(implantType["RID"])
    
    if any(rid == implantTypeRID):
        ind = np.where( rid == implantTypeRID )[0][0]
        implant = "yes"
        implantECoGorSEEG = implantType["Type"][ind]
        implantLaterality = implantType["Laterality"][ind]
    else:
        implant = "no"
        implantECoGorSEEG = "NA"
        implantLaterality = "NA"
        
    if any(sub == np.array(RNSlist)):
        RNS = "yes"
    else:
        RNS = "no"
    
    
    imaging =  join( LOCAdir[s], "ses-research3Tv[0-9][0-9]", "anat", "*acq-3D_T1w.nii.gz"  ) 
    utils.checkIfFileExistsGlob( imaging)
    if utils.checkIfFileExistsGlob( imaging, printBOOL=False):
        research3T = "yes"
        version = glob(imaging)[0]
  
        st = basename(version).find( "research3Tv")
        v = basename(version)[st+10:st+13]
        if v == "v00":
            v00 = "yes"; v01 = "no"; v02 = "no"; v03 = "no"; v04 = "no"
        elif v == "v01":
            v00 = "no"; v01 = "yes"; v02 = "no"; v03 = "no"; v04 = "no"
        elif v == "v02":
            v00 = "no"; v01 = "no"; v02 = "yes"; v03 = "no"; v04 = "no"
        elif v == "v03":
            v00 = "no"; v01 = "no"; v02 = "no"; v03 = "yes"; v04 = "no"
        elif v == "v04":
            v00 = "no"; v01 = "no"; v02 = "no"; v03 = "no"; v04 = "yes"
        else:
            v00 = "unknown"; v01 = "unknown"; v02 = "unknown"; v03 = "unknown"; v04 = "unknown"
    else:
        research3T = "no"
        v00 = "NA"; v01 = "NA"; v02 = "NA"; v03 = "NA"; v04 = "NA"
        
        
    diffusion = join( LOCAdir[s], "ses-research3Tv[0-9][0-9]", "dwi", "*.gz"  ) 
    if utils.checkIfFileExistsGlob( diffusion, printBOOL=False):
        dwi = "yes"
    else:
        dwi = "no"
         
    if any(sub == np.array(ControlsList)):
        control = "yes"
    else:
        control = "no"
   
    entry = dict( RID = sub, implant = implant, types  = implantECoGorSEEG, 
                 laterality= implantLaterality, RNS=RNS , research3T = research3T, 
                 v00=v00  , v01= v01, v02=v02 , v03=v03 , v04=v04 , dwi = dwi, control=  control )
    chart = chart.append(entry, ignore_index=True)




chart['implant'].value_counts()
chart['types'].value_counts()
chart['laterality'].value_counts()
chart['RNS'].value_counts()
chart['control'].value_counts()

chart['research3T'].value_counts()
chart['v00'].value_counts()
chart['v01'].value_counts()
chart['v02'].value_counts()
chart['v03'].value_counts()
chart['v04'].value_counts()





chart.value_counts(["implant", "types"])
chart.value_counts(["implant", "types", "research3T","v00", "control"])











