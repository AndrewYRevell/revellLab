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

python3.6 Script_01_volumes_vs_sphericity.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "brainAtlas/code/tools"))
import volumes_sphericity_surface_area as vsSA
import pandas as pd
import numpy as np
#%% Paths and File names
ifpath_atlases_standard = ospj( path, "data/data_raw/atlases/standard_atlases")
ifpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ofpath_volumes = ospj( path, "data/data_processed/volumes_and_sphericity/volumes")
ofpath_sphericity = ospj( path, "data/data_processed/volumes_and_sphericity/sphericity")

#%% Paramters
#number of random atlas permutations to run on
permutations = 1#Only do the first permutation

atlas_names_standard = [f for f in sorted(os.listdir(ifpath_atlases_standard))]
atlas_names_random = [f for f in sorted(os.listdir(ifpath_atlases_random))]


#%%

#standard atlases
for a in range(len(atlas_names_standard)):
    ifname_atlases_standard = ospj(ifpath_atlases_standard, atlas_names_standard[a] )
    atlas_name = os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0]
    ofpath_volumes_atlas =  ospj(ofpath_volumes, atlas_name )
    ofpath_sphericity_atlas =  ospj(ofpath_sphericity, atlas_name )
    if not (os.path.isdir(ofpath_volumes_atlas)): os.mkdir(ofpath_volumes_atlas)
    if not (os.path.isdir(ofpath_sphericity_atlas)): os.mkdir(ofpath_sphericity_atlas)
    ofname_volumes = "{0}/{1}_volumes.csv".format(ofpath_volumes_atlas, atlas_name)
    ofname_sphericity = "{0}/{1}_sphericity.csv".format(ofpath_sphericity_atlas, atlas_name)
    print("\nAtlas: {0}".format(atlas_name))
    #Volumes
    if not (os.path.exists(ofname_volumes)):#check if file exists
        print("Calculating Volumes: {0}".format(ofname_volumes))
        volumes = vsSA.get_region_volume(ifname_atlases_standard)
        pd.DataFrame.to_csv(volumes, ofname_volumes, header=True, index=False)
    else:
        print("File exists: {0}".format(ofname_volumes))
    #Sphericity
    if not (os.path.exists(ofname_sphericity)):#check if file exists
        print("Calculating Sphericity: {0}".format(ofname_sphericity))
        sphericity = vsSA.get_region_sphericity(ifname_atlases_standard)
        pd.DataFrame.to_csv(sphericity, ofname_sphericity, header=True, index=False)
        print("")#print new line due to overlap of next line with progress bar
    else:
        print("File exists: {0}".format(ofname_sphericity))
        
#%%  
#random atlases
for a in range(len(atlas_names_random)):
    for p in range(1, permutations+1):
        ifname_atlases_random = ospj(ifpath_atlases_random, atlas_names_random[a], "{0}_v{1}.nii.gz".format(atlas_names_random[a], '{:04}'.format(p))  )
        atlas_name = os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0]
        ofpath_volumes_atlas =  ospj(ofpath_volumes, atlas_name )
        ofpath_sphericity_atlas =  ospj(ofpath_sphericity, atlas_name )
        if not (os.path.isdir(ofpath_volumes_atlas)): os.mkdir(ofpath_volumes_atlas)
        if not (os.path.isdir(ofpath_sphericity_atlas)): os.mkdir(ofpath_sphericity_atlas)
        ofname_volumes = "{0}/{1}_volumes.csv".format(ofpath_volumes_atlas, atlas_name)
        ofname_sphericity = "{0}/{1}_sphericity.csv".format(ofpath_sphericity_atlas, atlas_name)
        print("\nAtlas: {0}".format(atlas_name))
        #Volumes
        if not (os.path.exists(ofname_volumes)):#check if file exists
            print("Calculating Volumes: {0}".format(ofname_volumes))
            volumes = vsSA.get_region_volume(ifname_atlases_random)
            pd.DataFrame.to_csv(volumes, ofname_volumes, header=True, index=False)
        else:
            print("File exists: {0}".format(ofname_volumes))
        #Sphericity
        if not (os.path.exists(ofname_sphericity)):#check if file exists
            print("Calculating Sphericity: {0}".format(ofname_sphericity))
            sphericity = vsSA.get_region_sphericity(ifname_atlases_random)
            pd.DataFrame.to_csv(sphericity, ofname_sphericity, header=True, index=False)
            print("")#print new line due to overlap of next line with progress bar
        else:
            print("File exists: {0}".format(ofname_sphericity))
            
#%%
#aggregate data into single spreadsheet for plotting
print("\n\nCalculating Means")

ifpath_volumes = ospj(path, "data/data_processed/volumes_and_sphericity/volumes")
ifpath_sphericity = ospj(path, "data/data_processed/volumes_and_sphericity/sphericity")
ofpath_volumes_and_sphericity_means = ospj(path, "data/data_processed/volumes_and_sphericity_means")

#%%

#extracting data into lists
atlas_names = [f for f in sorted(os.listdir(ifpath_volumes))]

print("Crawling thru directories and loading volumes data")
data_volumes = list()
for i in range(len(atlas_names)):
  data =  pd.read_csv(ospj(ifpath_volumes, atlas_names[i],  "{0}_volumes.csv".format(atlas_names[i])  ))
  data = np.array(data)[:,1]
  data = np.delete( data, 0, None)#removing first rentry because this is region label = 0, which is outside the brain
  data_volumes.append(data)
  
  
print("Crawling thru directories and loading sphericity data")
data_sphericity = list()
for i in range(len(atlas_names)):
  data =  pd.read_csv(ospj(ifpath_sphericity, atlas_names[i],  "{0}_sphericity.csv".format(atlas_names[i])  ))
  data = np.array(data)[:,1]
  data = np.delete( data, 0, None)#removing first rentry because this is region label = 0, which is outside the brain
  data_sphericity.append(data)
  
  
#%%  
#Calculating Means
print("Calculating volumes and sphericity means")
#initializing dataframe
means = pd.DataFrame(np.zeros(shape=(len(atlas_names),2)), columns=["volume_voxels", "sphericity"], index=[atlas_names]) 

#means
for i in range(len(atlas_names)):
    means.iloc[i,0] =  np.mean(data_volumes[i]  )
    means.iloc[i,1] =  np.mean(data_sphericity[i]  )
    
#%%
#saving
print("Saving: {0}".format(ospj(ofpath_volumes_and_sphericity_means, "volumes_and_sphericity.csv")))
pd.DataFrame.to_csv(means,  ospj(ofpath_volumes_and_sphericity_means, "volumes_and_sphericity.csv"), header=True, index=True)
  












