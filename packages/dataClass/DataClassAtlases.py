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
class atlases:
    atlasfiles: dict = "unknown"

    def getAllAtlasPaths(self):
        paths = []
        PATH_standardAtlas = self.atlasfiles["PATHS"]["STANDARD"]
        PATH_randomAtlasWB = self.atlasfiles["PATHS"]["wholeBrainMNI"]
        
        standard = self.atlasfiles["STANDARD"]
        standardNames = list(standard.keys()  )
        for i in range(len(standard)):
            paths.append(  join(PATH_standardAtlas, standard[standardNames[i]]["name"]   ) )
        
        random = self.atlasfiles["RANDOM"]["wholeBrainMNI"]
        randomNames = list(random.keys()  )
        for i in range(len(random)):
            
            name = random[randomNames[i]]["name"]  
            permutations = random[randomNames[i]]["permutations"]  
            for p in range(1, permutations + 1):
                namePath = f"{name}_v{p:04d}.nii.gz"
            
                paths.append(  join(PATH_randomAtlasWB, namePath  ) )
        return paths
    
    def getAllAtlasNames(self):
        names = []
        standard = self.atlasfiles["STANDARD"]
        standardNames = list(standard.keys()  )
        for i in range(len(standard)):
            names.append(   splitext(splitext(    standard[standardNames[i]]["name"] )[0])[0] )
                
     
        
        random = self.atlasfiles["RANDOM"]["wholeBrainMNI"]
        randomNames = list(random.keys()  )
        for i in range(len(random)):
            
            name = random[randomNames[i]]["name"]  
            permutations = random[randomNames[i]]["permutations"]  
            for p in range(1, permutations + 1):
                namePath = f"{name}_v{p:04d}.nii.gz"
            
                names.append(   splitext(splitext(  namePath)[0])[0] )
        return names
        
    def getAtlasMorphology(self, outpath):
        atlasPaths = self.getAllAtlasPaths()
        means = pd.DataFrame(np.zeros(shape=(len(atlasPaths),2)), columns=["volume_voxels", "sphericity"], index=self.getAllAtlasNames()  ) 
        
        for i in range(len(atlasPaths)):
            atlasName =  splitext(splitext(basename(atlasPaths[i]))[0])[0] 
            fnameMorphology = join(outpath, atlasName + "_morphology.csv")
            fnameMorphologyMean = join(outpath, "morphologyMeans.csv")
            print(f"{atlasName}")
            if os.path.exists(fnameMorphology):
                print(f"Morphology data exists. Reading data \n{fnameMorphology}")
                morphology = pd.read_csv(fnameMorphology, header=0)
            else:
                volume = regionMophology.get_region_volume(atlasPaths[i])
                sphericity = regionMophology.get_region_sphericity(atlasPaths[i])
                morphology = pd.concat([ volume,  sphericity.drop(sphericity.columns[0], axis = 1)], axis=1)
                pd.DataFrame.to_csv(morphology, fnameMorphology, header=True, index=False)
            
            means.iloc[i] = np.nanmean( morphology.drop(0, axis=0), axis=0)[1:]
            pd.DataFrame.to_csv(means, fnameMorphologyMean, header=True, index=True)
            
        
