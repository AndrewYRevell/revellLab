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