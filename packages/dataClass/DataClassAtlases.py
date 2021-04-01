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
            
    def create_feature_matrix(self):
        # Feature matrix with each element containing an NxN array
        feature_matrix = []

        # EDGE WEIGHT (Depth 0)
        structural_connectivity_array = self.get_structure_and_function()
        feature_matrix.append(structural_connectivity_array)

        # DEGREE (Depth 1 & 2)
        deg = bct.degrees_und(structural_connectivity_array)
        self.fill_array_2D(feature_matrix, deg)

        # Conversion of connection weights to connection lengths
        connection_length_matrix = bct.weight_conversion(structural_connectivity_array, 'lengths')

        # SHORTEST PATH LENGTH (Depth 3 & 4)
        shortest_path = bct.distance_wei(connection_length_matrix)
        feature_matrix.append(shortest_path[0])  # distance (shortest weighted path) matrix
        feature_matrix.append(shortest_path[1])  # matrix of number of edges in shortest weighted path

        # BETWEENNESS CENTRALITY (Depth 5 & 6)
        bc = bct.betweenness_wei(connection_length_matrix)
        from python_files.create_feature_matrix import fill_array_2D
        self.fill_array_2D(feature_matrix, bc)

        # CLUSTERING COEFFICIENTS (Depth 7 & 8)
        cl = bct.clustering_coef_wu(connection_length_matrix)
        self.fill_array_2D(feature_matrix, cl)

        return feature_matrix

    # turns 2D feature into 3D
    def fill_array_2D(feature_matrix, feature_array):
        feature_array_level1 = []
        for row in range(len(feature_array)):
            one_row = []
            for col in range(len(feature_array)):
                one_row.append(feature_array[row])
            feature_array_level1.append(one_row)
        feature_matrix.append(np.array(feature_array_level1))

        feature_array_level2 = []
        for row in range(len(feature_array)):
            one_row = []
            for col in range(len(feature_array)):
                one_row.append(feature_array[row])
            feature_array_level2.append(one_row)
        feature_matrix.append(np.array(feature_array_level2))
            
        
