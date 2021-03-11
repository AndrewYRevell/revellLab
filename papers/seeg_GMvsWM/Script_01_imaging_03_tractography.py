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
path = "/media/arevell/sharedSSD/linux/papers/paper005" 
#path = "/mnt"
import sys
import os
import pandas as pd
import numpy as np
from os.path import join as ospj
import seaborn as sns
import matplotlib.pyplot as plt
import time
import multiprocessing
from itertools import repeat
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))

#% Input/Output Paths and File names
ifname_diffusion_imaging = ospj( path, "data/data_raw/iEEG_times/diffusion_imaging.csv")
ifname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")
ifpath_imaging_qsiprep = ospj( path, "data/data_processed/imaging/qsiprep/")
ifpath_atlas_registration = ospj( path, "data/data_processed/atlas_registration/preop3T")
ifpath_atlases = ospj( path, "data/data_raw/atlases/atlases")

ofpath_tractography = ospj( path, "data/data_processed/tractography")
ofpath_connectivity = ospj( path, "data/data_processed/connectivity_matrices/structure")

if not (os.path.isdir(ofpath_tractography)): os.makedirs(ofpath_tractography, exist_ok=True)
if not (os.path.isdir(ofpath_connectivity)): os.makedirs(ofpath_connectivity, exist_ok=True)

#% Load Study Meta Data
data = pd.read_csv(ifname_diffusion_imaging) 
atlases = pd.read_csv(ifname_atlases_csv)     
#% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]

#% Paramter ID
parameter_id = '7C1D393C9A99193FF3B3513Fb803Fcb2041bC84340420Fca01cbaCDCC4C3Ec'
parameter_id = 'A323393C9A99193FF3B3513Fb803Fcb2041bC84340420Fca01cbaCDCC4C3Ec'
parameter_id = 'D467313C9A99193FF3B3513Fb803Fcb2041b4844404B4Cca01cbaCDCC4C3Ec'



"""
DSI_Studio parameter ID for calculating tractography. Related to number of streamlines, etc.
*Note from DSI_Studio documentation:
Assigning these tracking parameters takes a lot of time. To save the hassle, 
you can use "parameter_id" from the GUI (After running fiber tracking in GUI, 
the parameter ID will be shown in the method text in the right bottom window)
to override all parameters in one shot (e.g. --action=trk --source=some.fib.gz 
--parameter_id=c9A99193Fba3F2EFF013Fcb2041b96438813dcb). Please note that 
parameter ID does not overrider ROI settings. You may still need to assign 
ROI/ROA...etc. in the commnd line. 
"""


#%% Calculate tractography files from DSI Studio
for i in range(len(sub_IDs_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print(f"\n\nSub-ID: {sub_ID}")
    if "control" in data["type"][i]: ses = "control3T"
    else: ses = "preop3T"
    
    ifname_dwi_base = f"sub-{sub_ID}_ses-{ses}_space-T1w_desc-preproc_dwi" 
    ifname_dwi =  ospj(ifpath_imaging_qsiprep,  f"sub-{sub_ID}/ses-{ses}/dwi/{ifname_dwi_base}")
    os.path.exists(ifname_dwi + ".nii.gz")
    
    ifname_T1_sub_ID = ospj(ifpath_atlas_registration, f"sub-{sub_ID}", f"sub-{sub_ID}_desc-preproc_T1w_std.nii.gz")
    os.path.exists(ifname_T1_sub_ID)
    
    ifpath_atlas_registration_sub_ID = ospj(ifpath_atlas_registration, f"sub-{sub_ID}")

    ofpath_tractography_subID = os.path.join(ofpath_tractography, f"sub-{sub_ID}")
    if not (os.path.isdir(ofpath_tractography_subID)): os.mkdir(ofpath_tractography_subID)
    ofname_dwi =  os.path.join(ofpath_tractography_subID, ifname_dwi_base)
    os.path.exists(ofname_dwi + ".trk.gz")
    
    ofpath_connectivity_subID = os.path.join(ofpath_connectivity, f"sub-{sub_ID}")
    if not (os.path.isdir(ofpath_connectivity_subID)): os.mkdir(ofpath_connectivity_subID)
    
    if (os.path.exists(ofname_dwi + ".trk.gz")):
        print("Tractography file already exists: {0}".format(ofname_dwi + ".trk.gz"))
    if not (os.path.exists(ofname_dwi + ".trk.gz")):#if file already exists, don't run below
        t0 = time.time()
        print("Creating Source File in DSI Studio")
        cmd = f"singularity exec --bind {path} ~/singularity/dsistudio_latest.sif dsi_studio --action=src --source={ifname_dwi}.nii.gz --output={ofname_dwi}.src.gz"
        os.system(cmd)
        print("Creating Reconstruction File in DSI Studio")
        cmd = f"singularity exec --bind {path} ~/singularity/dsistudio_latest.sif dsi_studio --action=rec --source={ofname_dwi}.src.gz --method=4 --param0=1.25"
        os.system(cmd)
        #rename dsistudio output of .fib file because there is no way to output a specific name on cammnad line
        cmd = "mv {0}.src*.fib.gz {0}.fib.gz".format(ofname_dwi)
        os.system(cmd)
        print("Creating Tractography File in DSI Studio")
        cmd = f"singularity exec --bind /media ~/singularity/dsistudio_latest.sif dsi_studio --action=trk --source={ofname_dwi}.fib.gz --parameter_id={parameter_id} --output={ofname_dwi}.trk.gz"
        cmd = f"singularity exec --bind /media ~/singularity/dsistudio_latest.sif dsi_studio --action=trk --source={ofname_dwi}.fib.gz --min_length=30 --max_length=300 --thread_count=24 --fiber_count=5000000 --output={ofname_dwi}.trk.gz"
        cmd = f"singularity exec --bind /media ~/singularity/dsistudio_latest.sif dsi_studio --action=trk --source={ofname_dwi}.fib.gz --min_length=10 --max_length=800 --thread_count=24 --fiber_count=1000000 --output={ofname_dwi}.trk.gz"
        os.system(cmd)
        t1 = time.time()
        td = np.round(t1-t0,2)
        tr = np.round((len(sub_IDs_unique) - i- 1) * td,2)
        print(f"Took {td} seconds. Estimated time remaining: {tr} seconds")



#%% Calculate get connectivity
for i in range(len(sub_IDs_unique)):
    #def multiproc_conn(i):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print(f"\n\nSub-ID: {sub_ID}")
    
    if "control" in data["type"][i]: ses = "control3T"
    else: ses = "preop3T"

    ifpath_atlas_registration_sub_ID = ospj(ifpath_atlas_registration, "sub-{0}".format(sub_ID))
    ifname_T1_sub_ID = ospj(ifpath_atlas_registration, f"sub-{sub_ID}", "sub-{0}_desc-preproc_T1w_std.nii.gz".format(sub_ID))
    ifname_dwi_base = f"sub-{sub_ID}_ses-{ses}_space-T1w_desc-preproc_dwi"
    
    ifpath_tractography_subID = os.path.join(ofpath_tractography, "sub-{0}".format(sub_ID))
    ifname_dwi =  os.path.join(ifpath_tractography_subID, ifname_dwi_base)
    os.path.exists(ifname_dwi + ".trk.gz")
    
    ofpath_connectivity_subID = os.path.join(ofpath_connectivity, "sub-{0}".format(sub_ID))
    if not (os.path.isdir(ofpath_connectivity_subID)): os.mkdir(ofpath_connectivity_subID)


    #getting atlases
    for a in range(90,108):
    #def multiproc_atlas(a):
        ifname_atlas = ospj(ifpath_atlas_registration_sub_ID, f"sub-{sub_ID}_preop3T_{np.array( atlases['atlas_filename'])[a]}")
        ofname_connectivity = ospj(ofpath_connectivity_subID, f"sub-{sub_ID}" )
        basename = os.path.splitext(os.path.splitext(f"sub-{sub_ID}_preop3T_{np.array( atlases['atlas_filename'])[a]}")[0])[0]
        basename2 = f"{ofname_connectivity}.{basename}.count.pass.connectogram.txt"
        if not (os.path.exists( basename2 )):
            print( f"\nNew: {np.array(atlases['atlas_filename'])[a]}" )
            t0 = time.time()
            cmd = f"singularity exec --bind {path} ~/singularity/dsistudio_latest.sif dsi_studio --action=ana --source={ifname_dwi}.fib.gz --tract={ifname_dwi}.trk.gz --t1t2={ifname_T1_sub_ID} --connectivity={ifname_atlas} --connectivity_type=pass --connectivity_threshold=0 --output={ofname_connectivity}"
            os.system(cmd)
            t1 = time.time(); td = int(np.round(t1-t0,0)); #tr = int(np.round((37 - a- 2) * td/60,0))
            print( f"{np.array(atlases['atlas_filename'])[a]}; time: {td} sec" )

    #p = multiprocessing.Pool(10)
    #p.map(multiproc_atlas, range(70, 90)   )
    #p.close()
    
#multiprocess over multiple cores
#p = multiprocessing.Pool(4)
#p.map(multiproc_conn, range(20)   )
#p.close()



#%%visualize connectivity 

"""
import bct        

for i in [0]:
    sub_ID = sub_IDs_unique[i]
    ofname_connectivity = ospj(ofpath_connectivity, f"sub-{sub_ID}", f"sub-{sub_ID}" )
    for a in [80]:   
        basename = os.path.splitext(os.path.splitext(f"sub-{sub_ID}_preop3T_{np.array( atlases['atlas_filename'])[a]}")[0])[0]
        df = pd.read_csv(f"{ofname_connectivity}.{basename}.count.pass.connectogram.txt", sep="\t", header=None)
        df = df.drop([0,1], axis=0)
        df = df.drop([0,1], axis =1)
        df = df.drop(df.columns[-1], axis =1)
        data = np.array(df).astype("float64")
        fig,axes = plt.subplots(1,1,figsize=(8,4), dpi = 300)
        sns.heatmap(data, square=True, ax=axes)
        fig.suptitle(f"sub-{sub_ID}  {np.array( atlases['atlas_name'])[a]}")        
        
        
for i in [0]:
    sub_ID = sub_IDs_unique[i]
    ofname_connectivity = ospj(ofpath_connectivity, f"sub-{sub_ID}", f"sub-{sub_ID}" )
    for a in [150]:   
        basename = os.path.splitext(os.path.splitext(f"sub-{sub_ID}_preop3T_{np.array( atlases['atlas_filename'])[a]}")[0])[0]
        if os.path.exists(f"{ofname_connectivity}.{basename}.count.pass.connectogram.txt"):
            
            df = pd.read_csv(f"{ofname_connectivity}.{basename}.count.pass.connectogram.txt", sep="\t", header=None)
            df = df.drop([0,1], axis=0)
            df = df.drop([0,1], axis =1)
            df = df.drop(df.columns[-1], axis =1)
            data = np.array(df).astype("float64")
            print(f"{basename}\n{bct.density_und(data)}")
            
              
        
"""    
        
        
        
        
        
        
        
        
        
        
        