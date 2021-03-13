#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import sys
import os
import pandas as pd
import copy
from os import listdir
from  os.path import join, isfile
from os.path import splitext 
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
#import seaborn as sns


#%% Input


#%% functions
def check_path(path):
    '''
    Check if path exists

    Parameters
    ----------
        path: str
            Check if valid path
    '''
    if not os.path.exists(path):
        raise IOError(f"\n\n\n\nPath or file does not exist:\n{path}\n\n\n\n" )


def register_MNI_to_preopT1(fname_preopT1, fname_preopT1bet, MNItemplate, MNItemplatebet, outputMNIname, outputDirectory):
    mniBase =  join(outputDirectory, outputMNIname)
    #linear reg of MNI to preopT1 space
    cmd = f"flirt -in {MNItemplatebet} -ref {fname_preopT1bet} -dof 12 -out {mniBase}_flirt -omat {mniBase}_flirt.mat -v"; print(cmd);os.system(cmd)
    #non linear reg of MNI to preopT1 space
    print("\n\nLinear registration of MNI template to image is done\n\nStarting Non-linear registration:\n\n\n")
    cmd = f"fnirt --in={MNItemplate} --ref={fname_preopT1} --aff={mniBase}_flirt.mat --iout={mniBase}_fnirt -v --cout={mniBase}_coef --fout={mniBase}_warp"
    print(cmd)
    os.system(cmd)

def combine_first_and_fast(FIRST, FAST, outputName):
    img_first = nib.load(FIRST)
    data_first = img_first.get_fdata() 
    img_fast= nib.load(FAST)
    data_fast = img_fast.get_fdata() 
    #make all subcortical structures = 2 (2 is the GM category)
    data_first[np.where(data_first > 0)] = 2
    #replace fast images with first images where not zero
    data_fast[np.where(data_first > 0)] = data_first[np.where(data_first > 0)] 
    #plot
    img_first_fast = nib.Nifti1Image(data_fast, img_fast.affine)
    nib.save(img_first_fast, outputName)
    
def applywarp_to_atlas(atlasDirectory, fname_preopT1, MNIwarp, outputDirectory):
    atlases = [f for f in listdir(atlasDirectory) if isfile(join(atlasDirectory, f))]
    check_path(MNIwarp)
    check_path(atlasDirectory)
    check_path(fname_preopT1)
    check_path(outputDirectory)
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlas = join(atlasDirectory, atlases[i])
        if (".nii" in atlas):
            outputAtlasName = join(outputDirectory, atlasName + ".nii.gz")
            if not os.path.exists(outputAtlasName):
                cmd = f"applywarp -i {atlas} -r {fname_preopT1} -w {MNIwarp} --interp=nn -o {outputAtlasName}"; print(cmd); os.system(cmd)
            else: print(f"File exists: {outputAtlasName}")
            
            
def by_region(electrodePreopT1Coordinates, atlasPath, atlasLabelsPath, ofname, sep=",",  xColIndex=10, yColIndex=11, zColIndex=12, description = "unknown_atlas", Labels=True):
    # getting imaging data
    img = nib.load(atlasPath)
    img_data = img.get_fdata()  # getting actual image data array

    #show_slices(atlasPath)
    affine = img.affine
    if Labels == True:
        #getting atlas labels file
        atlas_labels = pd.read_csv(atlasLabelsPath, sep=",", header=None)
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
        atlas_labels = atlas_labels.drop([0, 1], axis=0).reset_index(drop=True)
        atlas_regions_numbers = np.array(atlas_labels.iloc[:,0]).astype("float64")
        atlas_labels_descriptors = np.array(atlas_labels.iloc[:,1])
    if Labels == False:
        atlas_regions_numbers = np.arange(0,   np.max(img_data)+1 )
        atlas_labels_descriptors = np.arange(0,   np.max(img_data)+1 ).astype("int").astype("object")
        atlas_name = os.path.splitext(os.path.basename(atlasPath))[0]
        atlas_name = os.path.splitext(atlas_name)[0]
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
    # getting electrode coordinates data
    data = pd.read_csv(electrodePreopT1Coordinates, sep=sep, header=0)
    data = data.iloc[:, [0, xColIndex, yColIndex, zColIndex]]
    column_names = ['electrode_name', "x_coordinate", "y_coordinate", "z_coordinate", column_description1,column_description2 ]
    data = data.rename(
        columns={data.columns[0]: column_names[0], data.columns[1]: column_names[1], data.columns[2]: column_names[2],
                 data.columns[3]: column_names[3]})

    coordinates = np.array((data.iloc[:, range(1, 4)]))  # get the scanner coordinates of electrodes
    # transform the real-world coordinates to the atals voxel space. Need to inverse the affine with np.linalg.inv(). To go from voxel to world, just input aff (dont inverse the affine)
    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(affine), coordinates)
    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel  
    coordinates_voxels = coordinates_voxels.astype(int)  
     
    try:
        img_ROI = img_data[coordinates_voxels[:,0]-1, coordinates_voxels[:,1]-1, coordinates_voxels[:,2]-1]
    except: #checking to make sure coordinates are in the atlas. This happens usually for electrodes on the edge of the SEEG. For example, RID0420 electrodes LE11 and LE12 are outside the brain/skull, and thus are outside even the normal MNI space of 181x218x181 voxel dimensions
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>img_data.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = 0
                print('Coordinate outside of atlas image space: setting to zero')
            elif((coordinates_voxels[i,1]>img_data.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = 0  
                print('Coordinate outside of atlas image space: setting to zero')
            elif((coordinates_voxels[i,2]>img_data.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = 0   
                print('Coordinate outside of atlas image space: setting to zero')
            else:
                img_ROI[i] = img_data[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]

    #getting corresponding labels
    img_labels = np.zeros(shape =img_ROI.shape ).astype("object")
    for l in range(len(img_ROI)):
        ind = np.where( img_ROI[l] ==    atlas_regions_numbers)
        if len(ind[0]) >0: #if there is a correpsonding label, then fill in that label. If not, put "unknown"
            if img_ROI[l] ==0: #if label is 0, then outside atlas
                img_labels[l] = "OutsideAtlas"
            img_labels[l] = atlas_labels_descriptors[ind][0]
        else:
            img_labels[l] = "NotInAtlas"
    img_ROI = np.reshape(img_ROI, [img_ROI.shape[0], 1])
    img_ROI = img_ROI.astype(int)
    df_img_ROI = pd.DataFrame(img_ROI)
    df_img_ROI.columns = [column_names[4]]
    img_labels = np.reshape(img_labels, [img_labels.shape[0], 1])
    df_img_labels = pd.DataFrame( img_labels)
    df_img_labels.columns = [column_names[5]]
    data = pd.concat([data, df_img_ROI, df_img_labels], axis=1)
    pd.DataFrame.to_csv(data, ofname, header=True, index=False)


def distance_from_label(electrodePreopT1Coordinates, atlasPath, label, atlasLabelsPath, ofname, sep=",", xColIndex=10, yColIndex=11, zColIndex=12):
    # getting imaging data
    img = nib.load(atlasPath)
    img_data = img.get_fdata()  # getting actual image data array
    affine = img.affine
    # getting electrode coordinates data
    data = pd.read_csv(electrodePreopT1Coordinates, sep=sep, header=None)
    data = data.iloc[:, [0, xColIndex, yColIndex, zColIndex]]
    
    atlas_labels = pd.read_csv(atlasLabelsPath, sep=",", header=None)
    column_description = "{0}_distance_from_label_{1}".format(atlas_labels.iloc[0,0], label)
    column_names = ['electrode_name', "x_coordinate", "y_coordinate", "z_coordinate", column_description]
    data = data.rename(
        columns={data.columns[0]: column_names[0], data.columns[1]: column_names[1], data.columns[2]: column_names[2],
                 data.columns[3]: column_names[3]})

    coordinates = np.array((data.iloc[:, range(1, 4)]))  # get the scanner coordinates of electrodes
    # transform the real-world coordinates to the atals voxel space. Need to inverse the affine with np.linalg.inv(). To go from voxel to world, just input aff (dont inverse the affine)
    coordinates_voxels = nib.affines.apply_affine(np.linalg.inv(affine), coordinates)
    coordinates_voxels = np.round(coordinates_voxels)  # round to nearest voxel  
    coordinates_voxels = coordinates_voxels.astype(int)  
    
    try:
        img_ROI = img_data[coordinates_voxels[:,0]-1, coordinates_voxels[:,1]-1, coordinates_voxels[:,2]-1]
    except:
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>img_data.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = -1
                print('Coordinate outside of MNI space: setting to zero')
            elif((coordinates_voxels[i,1]>img_data.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = -1 
                print('Coordinate outside of MNI space: setting to zero')
            elif((coordinates_voxels[i,2]>img_data.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = -1   
                print('Coordinate outside of MNI space: setting to zero')
            else:
                img_ROI[i] = img_data[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]
                
    img_ROI = np.reshape(img_ROI, [img_ROI.shape[0], 1])
    distances = copy.deepcopy(img_ROI)
    distances[(distances == 0)] = -1 #if coordinate equals to outside brain, then temporarily set to -1
    distances[(distances == label)] = 0 #if coordinate equals to the label, then it is zero distance
    
    # list of all points with label
    labelInds = np.where((img_data == label) )

    for i in range(0, distances.shape[0]):
        if ( int(img_ROI[i][0]) != int(label) ):
            point = coordinates_voxels[i, :] - 1 #coordinate trying to find distance to label
            minDist_coord = find_dist_to_label(point, labelInds)
            distances[i] = minDist_coord
            printProgressBar(i+1, img_ROI.shape[0], length = 20, suffix = 'Label: {0}. Point Label: {1} - {2}. Distance: {3} voxels'.format(label, data["electrode_name"][i],img_ROI[i][0] , np.round(minDist_coord,2) ))

    distances = pd.DataFrame(distances)
    data = pd.concat([data, distances], axis=1)
    data = data.rename(columns={data.columns[4]: column_names[4]})

    pd.DataFrame.to_csv(data, ofname, header=True, index=False)


def find_dist_to_label(point, labelInds):
    for i in range(0, labelInds[0].shape[0]):
        dist = np.sqrt((point[0] - labelInds[0][i]) ** 2 + (point[1] - labelInds[1][i]) ** 2 + (
                    point[2] - labelInds[2][i]) ** 2)
        if (i == 0):
            minDist = dist
        else:
            if (dist < minDist):
                minDist = dist
    return (minDist)



def channel2stdCSV(outputTissueCoordinates):
    df = pd.read_csv(outputTissueCoordinates, sep=",", header=0)
    for e in range(len( df  )):
        electrode_name = df.iloc[e]["electrode_name"]
        if (len(electrode_name) == 3): electrode_name = f"{electrode_name[0:2]}0{electrode_name[2]}"
        df.at[e, "electrode_name" ] = electrode_name
    pd.DataFrame.to_csv(df, outputTissueCoordinates, header=True, index=False)
    
    
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

def show_slices(fname, low = 0.33, middle = 0.5, high = 0.66, save = False, saveFilename = None):
    
    img = nib.load(fname)
    imgdata = img.get_fdata()  
    """ Function to display row of image slices """
    slices1 = [   imgdata[:, :, int((imgdata.shape[2]*low)) ] , imgdata[:, :, int(imgdata.shape[2]*middle)] , imgdata[:, :, int(imgdata.shape[2]*high)]   ]
    slices2 = [   imgdata[:, int((imgdata.shape[1]*low)), : ] , imgdata[:, int(imgdata.shape[1]*middle), :] , imgdata[:, int(imgdata.shape[1]*high), :]   ]
    slices3 = [   imgdata[int((imgdata.shape[0]*low)), :, : ] , imgdata[int(imgdata.shape[0]*middle), :, :] , imgdata[int(imgdata.shape[0]*high), :, :]   ]
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

    if save:
        if saveFilename == None:
            raise Exception("No file name was given to save figures")
        plt.savefig(saveFilename, transparent=True)

#%% Input names

