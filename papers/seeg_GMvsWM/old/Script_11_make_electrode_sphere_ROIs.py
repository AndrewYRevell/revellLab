"""
2020.08.01
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Compute spectrogram and interpolate each segment to a length of 200

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

interictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_394423190000_394512890000_EEG_filtered.pickle"
preictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_415933490000_416023190000_EEG_filtered.pickle"
ictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_416023190000_416112890000_EEG_filtered.pickle"
postictal_file = "./data_processed/eeg/montage/referential/filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_416112890000_416292890000_EEG_filtered.pickle"

~~~~~~~
"""


import pickle
import numpy as np
import os
import sys
from os.path import join as ospj
path = ospj("/media","arevell","sharedSSD","linux","papers","paper005") #Parent directory of project
#path = ospj("E:\\","linux","papers","paper005") #Parent directory of project
sys.path.append(ospj(path, "seeg_GMvsWM", "code", "tools"))
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import mannwhitneyu
import numpy as np
import copy
import pingouin as pg
import statsmodels.api as sm
import bct
import networkx as nx
import matplotlib.colors
np.seterr(divide = 'ignore')
import nibabel as nib



#%% Input/Output Paths and File names
ifname_EEG_times = ospj( path, "data/data_raw/iEEG_times/EEG_times.xlsx")
ifpath_electrode_localization = ospj( path, "data/data_processed/electrode_localization")
ifpath_electrode_images = ospj( path, "data/data_raw/electrode_localization")

ofpath_ROIs = ospj(path, "data/data_processed/electrode_sphere_ROIs")

if not (os.path.isdir(ofpath_ROIs)): os.makedirs(ofpath_ROIs, exist_ok=True)

#% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#% Processing Meta Data: extracting sub-IDs

sub_IDs_unique =  np.unique(data.RID)[np.argsort( np.unique(data.RID, return_index=True)[1])]
#%   

descriptors = ["interictal","preictal","ictal","postictal"]
#%%



    

def show_slices(img_data, low = 0.33, middle = 0.5, high = 0.66):
    """ Function to display row of image slices """
    slices1 = [   img_data[:, :, int((img_data.shape[2]*low)) ] , img_data[:, :, int(img_data.shape[2]*middle)] , img_data[:, :, int(img_data.shape[2]*high)]   ]
    slices2 = [   img_data[:, int((img_data.shape[1]*low)), : ] , img_data[:, int(img_data.shape[1]*middle), :] , img_data[:, int(img_data.shape[1]*high), :]   ]
    slices3 = [   img_data[int((img_data.shape[0]*low)), :, : ] , img_data[int(img_data.shape[0]*middle), :, :] , img_data[int(img_data.shape[0]*high), :, :]   ]
    slices = [slices1, slices2, slices3]
    plt.style.use('dark_background')
    fig = plt.figure(constrained_layout=False, dpi=300, figsize=(5, 5))
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0, right=1, bottom=0, top=1, wspace=0.00, hspace = 0.00)
    axes = []
    for r in range(3): #standard
        for c in range(3):
            axes.append(fig.add_subplot(gs1[r, c]))
    r = 0; c = 0
    for i in range(9):
        if (i%3 == 0 and i >0): r = r + 1; c = 0
        axes[i].imshow(slices[r][c].T, cmap="gray", origin="lower")
        c = c + 1
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].axis("off")

    


# Progress bar function
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = "X", printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()



#%%
radii = [1, 2,5, 10, 15, 25, 40]
for i in range(len(sub_IDs_unique)):
    #parsing data DataFrame to get iEEG information
    sub_ID = sub_IDs_unique[i]
    sub_RID = "sub-{0}".format(sub_ID)

    print( "\n\n{0}".format(sub_ID) )

    
    #Inputs and OUtputs
    #input filename EEG
  
    ifpath_electrode_localization_sub_ID =  ospj(ifpath_electrode_localization, sub_RID)
    ifname_electrode_localization = ospj(ifpath_electrode_localization_sub_ID, "sub-{0}_electrode_localization.csv".format(sub_ID))
    ifpath_electrode_image_sub_ID =  ospj(ifpath_electrode_images, sub_RID)
    ifname_electrode_image = ospj(ifpath_electrode_image_sub_ID, "sub-{0}_T00_mprageelectrodelabels_spheres.nii.gz".format(sub_ID))
    
    ofpath_ROIs_sub_ID = ospj(ofpath_ROIs, sub_RID)
    if not (os.path.isdir(ofpath_ROIs_sub_ID)): os.mkdir(ofpath_ROIs_sub_ID)
    
    os.path.isdir(ifname_electrode_image)
    os.path.isfile(ifname_electrode_image)
         
    
    
    #GET DATA
    #get localization and FC files
    electrode_localization = pd.read_csv(ifname_electrode_localization)
    
    img = nib.load(ifname_electrode_image)
    data = img.get_fdata() 
    affine = img.affine
    print(nib.aff2axcodes(affine))
    

    shape = data.shape
   
    
   
    x,y,z = electrode_localization.iloc[0]["x_coordinate"],  electrode_localization.iloc[0]["y_coordinate"],  electrode_localization.iloc[0]["z_coordinate"]
   
    coordinates = np.array((electrode_localization.iloc[:, range(1, 4)]))  # get the scanner coordinates of electrodes
    # transform the real-world coordinates to the atals voxel space. Need to inverse the affine with np.linalg.inv(). To go from voxel to world, just input aff (dont inverse the affine)
    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(affine), coordinates)
    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel  
    coordinates_voxels = coordinates_voxels.astype(int)  


    e = 0    
    
    

    
    for rr in radii:
        radius = rr #in mm
        
        ofpath_ROIs_radius = ospj(ofpath_ROIs_sub_ID, f"radius_{radius:02}")
        if not (os.path.isdir(ofpath_ROIs_radius)): os.makedirs(ofpath_ROIs_radius, exist_ok=True)
        
        for e in range(len(electrode_localization)):
            x0, y0, z0 = coordinates_voxels[e][0], coordinates_voxels[e][1], coordinates_voxels[e][2]
            electrode_name = electrode_localization["electrode_name"][e]    
           
            data[  np.where(data != 0)  ] = 0
           
            
           
            for x in range(x0-radius, x0+radius+1):
                for y in range(y0-radius, y0+radius+1):
                    for z in range(z0-radius, z0+radius+1):   
                        if ( radius - np.sqrt(abs(x0-x)**2 + abs(y0-y)**2 + abs(z0-z)**2 ))>=0:
                            #check to make sure bubble is inside image
                            if x > data.shape[0]-1:
                                x1 =  data.shape[0] - 1
                            elif x < 0:
                                x1 =  0
                            else:
                                x1 = x
                            if y > data.shape[1]-1:
                                y1 =  data.shape[1] - 1
                            elif y < 0:
                                y1 =  0
                            else:
                                y1 = y
                            if z > data.shape[2] -1:
                                z1 =  data.shape[2] - 1
                            elif z < 0:
                                z1 =  0
                            else:
                                z1 = z
                            data[x1,y1,z1] = 1
                            printProgressBar(e+1, len(electrode_localization), prefix = '', suffix = '{0} / {1}'.format(e, len(electrode_localization)), decimals = 1, length = 25, fill = "X", printEnd = "\r")
                
            #show_slices(data)
            
            ofname_ROIs_sub_ID = ospj(ofpath_ROIs_radius, "{0}_{1}.nii.gz".format(sub_RID, electrode_name))
            img_sphere = nib.Nifti1Image(data, img.affine)
            nib.save(img_sphere, ofname_ROIs_sub_ID)
            
    
    
    
    
    
#%%
    
    
    



    
    
    
    
    
    
    
    

    
    
    
    
    
    
   