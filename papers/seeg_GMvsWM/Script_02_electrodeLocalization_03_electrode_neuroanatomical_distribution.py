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
path = "/media/arevell/sharedSSD/linux/papers/paper005" #Parent directory of project
import sys
import os
import pandas as pd
import numpy as np
from os.path import join as ospj
from os.path import splitext as splitext
import seaborn as sns
import matplotlib.pyplot as plt

#import custom
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))


#%% Input/Output Paths and File names
fname_elec_loc_csv = ospj( path, "data/data_raw/electrode_localization/electrode_localization_subjects.csv")

fname_atlases_csv = ospj( path, "data/data_raw/atlases/atlas_names.csv")

fpath_figure = ospj(path, "seeg_GMvsWM", "figures", "coverage")
os.path.exists(fpath_figure)

fpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization")
if not (os.path.isdir(fpath_electrode_localization)): os.makedirs(fpath_electrode_localization, exist_ok=True)
#%% Load Study Meta Data
data = pd.read_csv(fname_elec_loc_csv)    
atlases = pd.read_csv(fname_atlases_csv)  
#%% Processing Meta Data: extracting sub-IDs

sub_IDs_unique = np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]


#%% Get Electrode Localization. Find which region each electrode is in for each atlas
for i in range(len(sub_IDs_unique)-2):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    print(f"\nSubject: {sub_ID}")
   
    #getting electrode localization file
    fpath_electrode_localization_sub_ID = ospj(fpath_electrode_localization, f"sub-{sub_ID}")
    if not os.path.exists(fpath_electrode_localization_sub_ID): print(f"Path does not exist: {fpath_electrode_localization_sub_ID}")
    fname_electrode_localization_sub_ID = ospj(fpath_electrode_localization_sub_ID, f"sub-{sub_ID}_electrodenames_coordinates_native_and_T1.csv")
    os.path.isfile(fname_electrode_localization_sub_ID)
    #getting electrode localization files

    fname_electrode_localization_concatenated = ospj(fpath_electrode_localization_sub_ID, f"sub-{sub_ID}_electrode_localization.csv")

    el_localization = pd.read_csv(fname_electrode_localization_concatenated)

    for a in range(len(atlases)):
        atlas_name = splitext(splitext(atlases["atlas_filename"][a])[0])[0]
        in_brain = el_localization["tissue_segmentation_region_number"]>=2
        
        region_numbers = el_localization[atlas_name + "_region_number"][in_brain]
        
        len_total = len(region_numbers)
        len_inside = len(np.where(region_numbers > 0)[0])
        len_outside = len(np.where(region_numbers ==0 )[0])
        coverage = len_inside/ len_total*100
        
        if atlas_name[0:11] == "RandomAtlas":
            atlas_name2 = "RandomAtlas"
        elif atlas_name[0:8] == "Schaefer":
            atlas_name2 = "Schaefer"
        elif atlas_name == "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask_resliced":
            atlas_name2 = "Yeo_liberal"
        elif atlas_name == "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_resliced":
            atlas_name2 = "Yeo"
        elif atlas_name == "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask_resliced":
            atlas_name2 = "Yeo_liberal"
        elif atlas_name== "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_resliced":
            atlas_name2 = "Yeo"
        elif atlas_name== "HarvardOxford-cort-maxprob-thr25-1mm":
            atlas_name2 = "HarvardOxford-cort"
        elif atlas_name== "HarvardOxford-cort-NONSYMMETRIC-maxprob-thr25-1mm":
            atlas_name2 = "HarvardOxford-cort"
        elif atlas_name== "cc200_roi_atlas":
            atlas_name2 = "Craddock"
        elif atlas_name== "cc400_roi_atlas":
            atlas_name2 = "Craddock"
        else:
            atlas_name2 = atlas_name
        #intialize
        if i ==0 and a ==0:
            df = pd.DataFrame([dict(subID = sub_ID, atlas = atlas_name2, len_total = len_total, len_inside = len_inside,  len_outside = len_outside, coverage = coverage )])
        else:
            df1 = pd.DataFrame([dict(subID = sub_ID, atlas = atlas_name2, len_total = len_total, len_inside = len_inside,  len_outside = len_outside, coverage = coverage )])
            df = pd.concat([df, df1], axis=0)
        print(f"{sub_ID}; {atlas_name}; {np.round(coverage, 2)}")
        
        
        
#%% Graph



df_mean = df.groupby(['atlas'], as_index=False).mean()

ind_order = np.argsort(  np.array(df_mean["coverage"]) )
#ind_order = np.sort(  np.array(df_mean["coverage"]) )


df_mean_order = df_mean.iloc[ind_order]
order = np.array(df_mean_order["atlas"])
order
labels = [
        "HO Sub only", 
        "JHU Tracts", 
        "JHU Labels",
        "Gordon Petersen",
        "Yeo",
        "MMP",
        "Julich",
        "DKT31 OASIS",
        "CerebrA",
        "HO Cort Only",
        "Schaefer",
        "Yeo Liberal",
        "Craddock",
        "HO combined",
        "BNA",
        "MNI structural",
        "AICHA",
        "AAL2",
        "AAL3",
        "AAL600",
        "AAL1",
        "AAL-JHU",
        "Brodmann",
        "Hammersmith",
        "Talairach",
        "Random",
            ]



df_drop = df.drop_duplicates()

means = np.array(df_mean_order["coverage"])
means = np.round(means, 0)
#%%

fig, ax = plt.subplots(1,1, dpi=300, figsize=(7, 3))
sns.violinplot(x = "atlas", y = "coverage", data = df_drop, order = order, ax = ax,  inner=None, scale ="count", color = "#0277bd")
sns.swarmplot(x="atlas", y="coverage", data=df_drop, order = order, size = 3, color = "#add8e6")


xticks = ax.get_xticks()
fs = 11
for x in range(len(xticks)):
    ax.text(x = xticks[x], y = 116, s = int(means[x]), fontsize = fs, ha = "center", va = "top")

ax.text(x =-0.15, y = 100, s ="N = 22 ", fontsize = fs, ha = "left", va = "top")
ax.set_ylim([0, 106])

sns.despine()
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("Coverage (%)", fontsize = fs)
ax.set_xlabel("")

plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.31)
plt.savefig(ospj(fpath_figure, "coverage.pdf"))




