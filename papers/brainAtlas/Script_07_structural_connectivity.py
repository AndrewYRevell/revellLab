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

python3.6 Script_07_structural_connectivity.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%%
path = "/mnt" #/mnt is the directory in the Docker or Singularity Continer where this study is mounted
import os
from os.path import join as ospj
import pandas as pd
import numpy as np
#%% Paths and File names

ifname_EEG_times = ospj(path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_tractography = ospj(path, "data/data_processed/tractography")
ifpath_atlases_standard = ospj( path, "data/data_raw/atlases/standard_atlases")
ifpath_atlases_random = ospj( path, "data/data_raw/atlases/random_atlases")
ofpath_connectivity = ospj( path, "data/data_processed/connectivity_matrices/structure/end")
                             
#%%Load Data
data = pd.read_excel(ifname_EEG_times)    
sub_ID_unique = np.unique(data.RID)

#%%
for i in range(len(sub_ID_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_ID_unique[i]
    print("\n\nSub-ID: {0}".format(sub_ID))
    atlas_names_standard = [f for f in sorted(os.listdir(ifpath_atlases_standard))]
    atlas_names_random = [f for f in sorted(os.listdir(ifpath_atlases_random))]
    ifname_base = "sub-{0}_ses-preop3T_dwi-eddyMotionB0Corrected".format(sub_ID) 
    ifname_src =    "{0}.src.gz".format(ifname_base) 
    ifname_src = os.path.join(ifpath_tractography,"sub-{0}".format(sub_ID), ifname_src)
    ifname_fib = "{0}.odf8.f5.bal.012fy.rdi.gqi.1.25.fib.gz".format(ifname_src)
    ifname_trk =    "{0}.trk.gz".format(ifname_base) 
    ifname_trk = os.path.join(ifpath_tractography,"sub-{0}".format(sub_ID), ifname_trk)
    ofpath_connectivity_sub_ID = os.path.join(ofpath_connectivity, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofpath_connectivity_sub_ID)): os.mkdir(ofpath_connectivity_sub_ID)
    os.path.exists(ofpath_connectivity_sub_ID)
    #standard atlases: Calculating Connectivity per atlas
    for a in range(len(atlas_names_standard)):
        ifname_atlases_standard = ospj(ifpath_atlases_standard, atlas_names_standard[a] )
        ofname_connectivity_atlas = ospj(ofpath_connectivity_sub_ID,  os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0])
        if not (os.path.isdir(ofname_connectivity_atlas)): os.mkdir(ofname_connectivity_atlas)
        ofname_connectivity = ospj(ofname_connectivity_atlas, ifname_base)
        ofname_connectivity_long_name = "{0}.{1}.count.end.connectivity.mat".format(ofname_connectivity, os.path.splitext(os.path.splitext(atlas_names_standard[a] )[0])[0] )
        if (os.path.exists(ofname_connectivity_long_name)):
            print("File exists: {0}".format(ofname_connectivity_long_name))
        else:
            print("\n\n\nSubject: {0}  Atlas: {1}".format(sub_ID, atlas_names_standard[a]))
            cmd = "dsi_studio --action=ana --source={0}  --tract={1} --connectivity={2} --connectivity_type=end --connectivity_threshold=0 --output={3}".format(ifname_fib, ifname_trk , ifname_atlases_standard, ofname_connectivity)
            os.system(cmd)
    #random atlases: Calculating Connectivity per atlas
    for a in range(len(atlas_names_random)):
        versions = [f for f in sorted(os.listdir(ospj(ifpath_atlases_random, atlas_names_random[a])))]  
        for p in range(len(versions)):
            
           ifname_atlases_random = ospj(ifpath_atlases_random,atlas_names_random[a] , versions[p]  )
           ofname_connectivity_atlas = ospj(ofpath_connectivity_sub_ID,  os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0])
           if not (os.path.isdir(ofname_connectivity_atlas)): os.mkdir(ofname_connectivity_atlas)
           ofname_connectivity = ospj(ofname_connectivity_atlas, ifname_base)
           ofname_connectivity_long_name = "{0}.{1}.count.end.connectivity.mat".format(ofname_connectivity,  "{0}_v{1}".format(os.path.splitext(os.path.splitext(atlas_names_random[a] )[0])[0], '{:04}'.format(p+1)) )
           if (os.path.exists(ofname_connectivity_long_name)):
               print("File exists: {0}".format(ofname_connectivity_long_name))
           else:
               print("\n\n\nSubject: {0}  Atlas: {1}".format(sub_ID, atlas_names_random[a]))
               cmd = "dsi_studio --action=ana --source={0}  --tract={1} --connectivity={2} --connectivity_type=end --connectivity_threshold=0 --output={3}".format(ifname_fib, ifname_trk , ifname_atlases_random, ofname_connectivity)
               os.system(cmd)
               

#%%












