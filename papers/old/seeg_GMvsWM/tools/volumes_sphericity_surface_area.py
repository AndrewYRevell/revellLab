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

python3.6 Script_02_electrode_localization.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


#assumes current working directory is scripts directory
#%%
path = "/mnt"
import sys
from os.path import join as ospj
sys.path.append(ospj(path, "paper001/code/tools"))
import nibabel as nib
import numpy as np
import pandas as pd

#%% Paths and File names
"""
atlas_path = ospj( path, "data_raw/atlases/standard_atlases/aal_res-1x1x1.nii.gz")


img = nib.load(atlas_path)
img_data = img.get_fdata() 

"""


#%% volume
#(1) - count the number of voxels within a given region

def get_region_volume(atlas_path):
    
    img = nib.load(atlas_path)
    img_data = img.get_fdata() 
    regions = np.unique(img_data)
    volumes =  pd.DataFrame(np.zeros(shape = (len(regions), 2)))  #intialize
    volumes.iloc[:,0] = regions
    for i in range(len(regions)):
        volumes.iloc[i,1] = len(np.where(img_data == regions[i])[0])
    volumes.columns = ["region_label", "voxels"]
    return volumes

#%%Surface Area

"""
Surface Area Calculation:
(1) Take a voxel, and see if a neighboring voxel in all 6 directions is in the same region. 
(2) Subtract that from 6 to get the number of square faces without a neighbor. 
(3) iterate that over all voxels to get the total number of square faces with no neighbor --> This equals Surface Area


"""
def get_region_surface_area(atlas_path):
    img = nib.load(atlas_path)
    img_data = img.get_fdata()
    regions = np.unique(img_data)
    SAs = pd.DataFrame(np.zeros(shape = (len(regions), 2)));#intialize. SAs = surface areas
    SAs.iloc[:,0] = regions
    dims = img_data.shape
    count = 0
    total = (dims[0]*dims[1]*dims[2])
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                printProgressBar(count, total,length = 50 )
                count = count + 1
                if (img_data[i,j,k] > 0):
                    region = img_data[i,j,k]
                    iloc = np.where(SAs.iloc[:,0] == region )[0][0]
                    nextTo = 0
                    if (i+1 <= dims[0]):#If the neighboring voxel is still whithin the confines of the image dimensions
                        if (img_data[i+1,j,k]==region):
                            nextTo = nextTo+1
                    if (i-1 >= 1):
                        if (img_data[i-1,j,k]==region):
                            nextTo = nextTo+1
                    if (j+1 <=dims[1]):
                        if (img_data[i,j+1,k]==region):
                            nextTo = nextTo+1
                    if (j-1 >=1):
                        if (img_data[i,j-1,k]==region):
                            nextTo = nextTo+1
                    if (k+1 <= dims[2]):
                        if (img_data[i,j,k+1]==region):
                            nextTo = nextTo+1
                    if (k-1 >= 1):
                        if (img_data[i,j,k-1]==region):
                            nextTo = nextTo+1
                    if (nextTo<6):
                        SAs.iloc[iloc, 1] = SAs.iloc[iloc, 1] + ((6-nextTo)*0.001*0.001);
    SAs.columns = ["region_label", "surface_area"]
    return SAs

#%% Sphericity

def get_region_sphericity(atlas_path):
    img = nib.load(atlas_path)
    img_data = img.get_fdata()
    regions = np.unique(img_data)
    sphericity = pd.DataFrame(np.zeros(shape = (len(regions), 2)));#intialize
    sphericity.iloc[:,0] = regions
    volumes = get_region_volume(atlas_path)*(0.001)*(0.001)*(0.001)
    surface_areas = get_region_surface_area(atlas_path)
    sphericity.iloc[:,1] = np.cbrt(np.power(6*volumes.iloc[:,1],2)) * (np.cbrt(np.pi)) / surface_areas.iloc[:,1]
    sphericity.columns = ["region_label", "sphericity"]
    return sphericity


#%%
# Progress bar function
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = "X", printEnd = "\r"):
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





