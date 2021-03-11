"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
#%%
path = "/media/arevell/sharedSSD/linux/papers/paper005" 
#path = "/mnt"
import sys
import os
import pandas as pd
import numpy as np
from os.path import join as ospj
import time
import multiprocessing
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
import volumes_sphericity_surface_area as vsSA

#% Input/Output Paths and File names
fname_diffusion_imaging = ospj( path, "data/data_raw/iEEG_times/diffusion_imaging.csv")
fname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
fpath_atlases = ospj( path, "data/data_raw/atlases/atlases")

fpath_morphology = ospj( path, "data/data_processed/atlas_morphology")
fpath_volume = ospj( fpath_morphology, "volume")
fpath_sphericity = ospj( fpath_morphology, "sphericity")

if not (os.path.isdir(fpath_sphericity)): os.makedirs(fpath_sphericity, exist_ok=True)
if not (os.path.isdir(fpath_volume)): os.makedirs(fpath_volume, exist_ok=True)

#% Load Study Meta Data
data = pd.read_csv(fname_diffusion_imaging) 
atlases = pd.read_csv(fname_atlases_csv)     
#% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]




#%%
#calculate volume
for a in range(len(atlases)):
    t0 = time.time()
    fname_atlas = ospj(fpath_atlases, atlases["atlas_filename"][a] )
    atlas_name = atlases["atlas_name"][a]
    fname_volume = ospj(fpath_volume, f"{atlas_name}_volume.csv")
    print(f"Atlas: {atlas_name}")
    #Volumes
    if not (os.path.exists(fname_volume)):#check if file exists
        print(f"Calculating Volumes: {fname_volume}")
        volumes = vsSA.get_region_volume(fname_atlas)
        pd.DataFrame.to_csv(volumes, fname_volume, header=True, index=False)
    else:
        print(f"File exists: {fname_volume}")
    t1 = time.time(); td = t1-t0; tr = np.round(td*(len(atlases)-a)/60)
    print(f"Time remaining: {tr} min")
        
#%%
#calculate Sphericity
for a in range(len(atlases)):
    t0 = time.time()
    fname_atlas = ospj(fpath_atlases, atlases["atlas_filename"][a] )
    atlas_name = atlases["atlas_name"][a]
    fname_sphericity = ospj(fpath_sphericity, f"{atlas_name}_sphericity.csv")
    print(f"Atlas: {atlas_name}")
    if not (os.path.exists(fname_sphericity)):#check if file exists
        print(f"Calculating Sphericity: {fname_sphericity}")
        sphericity = vsSA.get_region_sphericity(fname_atlas)
        pd.DataFrame.to_csv(sphericity, fname_sphericity, header=True, index=False)
        print("")#print new line due to overlap of next line with progress bar
    else:
        print("File exists: {0}".format(fname_sphericity))
    t1 = time.time(); td = t1-t0; tr = np.round(td*(len(atlases)-a)/60)
    print(f"Time remaining: {tr} min")
            
#%%
#aggregate data into single spreadsheet for plotting
print("\n\nCalculating Means")

#%%

print("loading volumes data")
data_volumes = list()
for a in range(len(atlases)):
    fname_atlas = ospj(fpath_atlases, atlases["atlas_filename"][a] )
    atlas_name = atlases["atlas_name"][a]
    fname_volume = ospj(fpath_volume, f"{atlas_name}_volume.csv")
    
    data =  pd.read_csv(fname_volume)
    data = np.array(data)[:,1]
    data = np.delete( data, 0, None)#removing first rentry because this is region label = 0, which is outside the brain
    data_volumes.append(data)

  
print("loading sphericity data")
data_sphericity = list()
for a in range(len(atlases)):
    fname_atlas = ospj(fpath_atlases, atlases["atlas_filename"][a] )
    atlas_name = atlases["atlas_name"][a]
    fname_sphericity = ospj(fpath_sphericity, f"{atlas_name}_sphericity.csv")
    
    data =  pd.read_csv(fname_sphericity)
    data = np.array(data)[:,1]
    data = np.delete( data, 0, None)#removing first rentry because this is region label = 0, which is outside the brain
    data_sphericity.append(data)

  
#%%  
#Calculating Means
print("Calculating volumes and sphericity means")
#initializing dataframe
means = pd.DataFrame(np.zeros(shape=(len(atlases),2)), columns=["volume_voxels", "sphericity"], index=[atlases["atlas_name"]]) 

#means
for i in range(len(atlases)):
    means.iloc[i,0] =  np.mean(data_volumes[i]  )
    means.iloc[i,1] =  np.mean(data_sphericity[i]  )
    
#%%
#saving
fname_means = ospj(fpath_morphology, "volumes_and_sphericity.csv")
print(f"Saving: {fname_means}")
pd.DataFrame.to_csv(means,  fname_means, header=True, index=True)
  












