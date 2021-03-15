#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:21:07 2021

@author: arevell
"""
import os
import pandas as pd
import numpy as np
from os.path import join, splitext, basename
from dataclasses import dataclass
from revellLab.packages.imaging.regionMorphology import regionMophology


@dataclass
class brainAtlasCohort:
    cohortJson: dict = "unknown"

    def getWithDTI(self):
        patients = []
        ctrl = self.cohortJson["IMAGING_CONTROLS"]
        sub = self.cohortJson["SUBJECTS"]
        
        
        ctrolNames = list(ctrl.keys())
        subNames = list(sub.keys())
        for i in range(len(ctrolNames)):
            if ctrl[ctrolNames[i]]["DTI"] == "yes":
                patients.append(ctrolNames[i])
        for i in range(len(subNames)):
            if sub[subNames[i]]["DTI"] == "yes":
                patients.append(subNames[i])
        return patients
    
    def getIctalandInterictalTimes(self):
        sub = self.cohortJson["SUBJECTS"]
        subNames = list(sub.keys())
        times = pd.DataFrame(columns= ["sub", "IctalStart", "IctalStop", "InterictalStart", "InterictalStop"] )
        for i in range(len(subNames)):
            if sub[subNames[i]]["SFC"] == "yes":
                if any("Ictal" == np.array(list(sub[subNames[i]].keys()) )): #make sure data actually has ictal
                    ictal = sub[subNames[i]]["Ictal"]
                    interictal =sub[subNames[i]]["Interictal"]
                    ictalKeys = list(ictal.keys())
                    for ic in range(len(ictalKeys)):
                        associatedInterictalKey = ictal[ictalKeys[ic]]["associatedInterictal"]
                        times = times.append( dict( sub  = subNames[i] , 
                                           IctalStart = ictal[ictalKeys[ic]]["start"] , 
                                           IctalStop = ictal[ictalKeys[ic]]["stop"],
                                           InterictalStart = interictal[associatedInterictalKey]["start"], 
                                           InterictalStop = interictal[associatedInterictalKey]["stop"]), 
                                     ignore_index=True  )
                else:
                    raise IOError(f"Data indicates SFC times are present, but none exists\nID: {subNames[i]}")
        return times
    
    def getiEEGdataKeys(self):
        sub = self.cohortJson["SUBJECTS"]
        subNames = list(sub.keys())
        times = pd.DataFrame(columns= ["sub", "iEEGdataKey"] )
        for i in range(len(subNames)):
            if sub[subNames[i]]["SFC"] == "yes":
                if any("Ictal" == np.array(list(sub[subNames[i]].keys()) )): #make sure data actually has ictal
                    ictal = sub[subNames[i]]["Ictal"]
                    ictalKeys = list(ictal.keys())
                    for ic in range(len(ictalKeys)):
                        iEEGdataKey = ictal[ictalKeys[ic]]["iEEGdataKey"]
                        times = times.append( dict( sub  = subNames[i] , 
                                           iEEGdataKey = iEEGdataKey),
                                     ignore_index=True  )
                else:
                    raise IOError(f"Data indicates SFC times are present, but iEEGdataKey does not exist\nID: {subNames[i]}")
        return times
























