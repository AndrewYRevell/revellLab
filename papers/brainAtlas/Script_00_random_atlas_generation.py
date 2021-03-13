"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: script to get electrode localization

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python3.6 Script_00_random_atlas_generation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import random_atlas_generation as RAG

#%% Paths and File names
ofpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ifname_MNI_template = ospj( path, "data/data_raw/MNI_brain_template/MNI152_T1_1mm_brain.nii.gz")

#%% Paramters

#Number of regions contained within the random atlas. This code will loop through this list to generate a random atlas of these sizes
number_of_regions_list = [10, 30, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000, 10000]
#number of random atlas permutations to make
permutations = 5
#%%

#random atlases
for a in range(len(number_of_regions_list)):
    number_of_regions = number_of_regions_list[a]
    atlas_names_random = "RandomAtlas{:07}".format(number_of_regions)
    for p in range(1, permutations+1):
        ofpath_atlases_random_atlas = ospj(ofpath_atlases_random, atlas_names_random)
        if not (os.path.isdir(ofpath_atlases_random_atlas)): os.mkdir(ofpath_atlases_random_atlas)
        atlas_name_version = "{0}_v{1}.nii.gz".format(atlas_names_random, '{:04}'.format(p))
        ofname_atlases_random = ospj(ofpath_atlases_random_atlas, atlas_name_version )
        print("\nGenerating Atlas: {0}".format(atlas_name_version))
        #Volumes
        if not (os.path.exists(ofname_atlases_random)):#check if file exists
            RAG.generateRandomAtlases_wholeBrain(number_of_regions, ofname_atlases_random, ifname_MNI_template)
        else:
            print("File exists: {0}".format(ofname_atlases_random))


