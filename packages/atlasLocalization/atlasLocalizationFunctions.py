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
from os.path import splitext, basename
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import ndimage 
import multiprocessing
from itertools import repeat
from revellLab.packages.utilities import utils

#%% functions

def atlasLocalizationBatchProccess(subList, i,  atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory):
#for i in range(len(subList)):
    sub = subList[i]
    electrodePreopT1Coordinates = join(inputDirectory, f"sub-{sub}", "electrodenames_coordinates_native_and_T1.csv")   
    preopT1 = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage.nii.gz")
    preopT1bet = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage_brainBrainExtractionBrain.nii.gz")
    outputDirectory = join(outputDirectory, f"sub-{sub}")
    outputName =  f"sub-{sub}_atlasLocalization.csv"
    cmd =  f"python {atlasLocalizationFunctionDirectory + '/atlasLocalization.py'} {electrodePreopT1Coordinates} {preopT1} {preopT1bet} \
        {MNItemplatePath} {MNItemplateBrainPath} {atlasDirectory} {atlasLabelsPath} {outputDirectory} {outputName}"
    print(cmd); os.system(cmd)
    
def atlasLocalizationBIDSwrapper(subList, atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory, multiprocess = False, cores = 8):
    if not multiprocess:
        for i in range(len(subList)):
            sub = subList[i]
            electrodePreopT1Coordinates = join(inputDirectory, f"sub-{sub}", "electrodenames_coordinates_native_and_T1.csv")   
            preopT1 = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage.nii.gz")
            preopT1bet = join(inputDirectory, f"sub-{sub}", f"T00_{sub}_mprage_brainBrainExtractionBrain.nii.gz")
            outputDirectory = join(outputDirectory, f"sub-{sub}")
            outputName =  f"sub-{sub}_atlasLocalization.csv"
            cmd =  f"python {atlasLocalizationFunctionDirectory + '/atlasLocalization.py'} {electrodePreopT1Coordinates} {preopT1} {preopT1bet} \
                {MNItemplatePath} {MNItemplateBrainPath} {atlasDirectory} {atlasLabelsPath} {outputDirectory} {outputName}"
            print(cmd); os.system(cmd)
    atlasLocalizationBatchProccess(subList, i,  atlasLocalizationFunctionDirectory, inputDirectory, atlasDirectory, atlasLabelsPath, MNItemplatePath, MNItemplateBrainPath, outputDirectory)
    if multiprocess:
        p = multiprocessing.Pool(cores)
        p.starmap(atlasLocalizationBatchProccess, zip(repeat(subList), range(len(subList)), 
                                     repeat(atlasLocalizationFunctionDirectory), 
                                     repeat(inputDirectory), 
                                     repeat(atlasDirectory), 
                                     repeat(atlasLabelsPath), 
                                     repeat(MNItemplatePath),
                                     repeat(MNItemplateBrainPath),
                                     repeat(outputDirectory)  )   )
        
        
def executeAtlasLocalizationSingleSubject(atlasLocalizationFunctionDirectory, electrodePreopT1Coordinates, preopT1, preopT1bet, MNItemplatePath, MNItemplateBrainPath, atlasDirectory, atlasLabelsPath, outputDirectory, outputName): 
    cmd =  f"python {atlasLocalizationFunctionDirectory + '/atlasLocalization.py'} {electrodePreopT1Coordinates} {preopT1} {preopT1bet} \
        {MNItemplatePath} {MNItemplateBrainPath} {atlasDirectory} {atlasLabelsPath} {outputDirectory} {outputName}"
    utils.executeCommand(cmd)
    
    
def atlasLocalizationFromBIDS(BIDS, dataset, sub, ses, acq, electrodeCoordinatesPath,atlasLocalizationFunctionDirectory, MNItemplatePath , MNItemplateBrainPath, atlasDirectory, atlasLabelsPath, outputDirectory ):
    subject_T1 = join(BIDS, dataset, f"sub-{sub}", f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_acq-{acq}_T1w.nii.gz" )
    subject_outputDir = join(BIDS, "derivatives", "atlasLocalization", f"sub-{sub}")
    subject_preimplantT1 = join(subject_outputDir, f"T00_{sub}_mprage.nii.gz" )
    subject_preimplantT1_brain = join(subject_outputDir, f"T00_{sub}_mprage_brainBrainExtractionBrain.nii.gz" )
    subject_T1_to_preimplantT1 = join(subject_outputDir, f"T1_to_T00_{sub}_mprage.nii.gz" )
    subject_T1_to_preimplantT1_brain = join(subject_outputDir, f"T1_to_T00_{sub}_mprage_brain.nii.gz" )
    subject_biascorrected = join(subject_outputDir, f"sub-{sub}_biascorrected")
    subject_biascorrectedDirectory = join(subject_outputDir, f"sub-{sub}_biascorrected.anat")
    subject_biascorrectedT1 = join(subject_biascorrectedDirectory, "T1_biascorr.nii.gz")
    electrodePreopT1Coordinates = electrodeCoordinatesPath   
    outputDirectorysubject = join(outputDirectory, f"sub-{sub}")
    outputName =  f"sub-{sub}_atlasLocalization.csv"
    ###Bias correction
    utils.executeCommand(cmd = f"fsl_anat -i {subject_T1} --noreorient --noreg --nononlinreg --noseg  --nosubcortseg --nocrop --clobber -o {subject_biascorrected}")
    ###Convert 3T preop to T00 space
    utils.executeCommand(cmd = f"flirt -in {subject_biascorrectedT1} -ref {subject_preimplantT1} -dof 6 -out {subject_T1_to_preimplantT1} -omat {subject_T1_to_preimplantT1}_flirt.mat -v")
    ###Getting brain extraction
    getBrainFromMask(subject_T1_to_preimplantT1, subject_preimplantT1_brain, subject_T1_to_preimplantT1_brain)
    executeAtlasLocalizationSingleSubject(atlasLocalizationFunctionDirectory, electrodePreopT1Coordinates, subject_T1_to_preimplantT1, subject_T1_to_preimplantT1_brain, MNItemplatePath, MNItemplateBrainPath, atlasDirectory, atlasLabelsPath, outputDirectorysubject, outputName)
    


    
def register_MNI_to_preopT1(preop3T, preop3Tbrain, MNItemplatePath, MNItemplateBrainPath, outputMNIname, outputDirectory, preop3Tmask = None, convertToStandard = True ):
    mniBase =  join(outputDirectory, outputMNIname)
    
    if convertToStandard:
        STDpreop3T = join(outputDirectory, utils.baseSplitextNiiGz(preop3T)[2] + "_std.nii.gz")
        STDpreop3Tbrain = join(outputDirectory, utils.baseSplitextNiiGz(preop3Tbrain)[2] + "_std.nii.gz")
        #convert to std space
        cmd = f"fslreorient2std {preop3T} {STDpreop3T}"; print(cmd);os.system(cmd)
        cmd = f"fslreorient2std {preop3Tbrain} {STDpreop3Tbrain}"; print(cmd);os.system(cmd)
        preop3T = STDpreop3T
        preop3Tbrain = STDpreop3Tbrain
        
    #linear reg of MNI to preopT1 space
    cmd = f"flirt -in {MNItemplateBrainPath} -ref {preop3Tbrain} -dof 12 -out {mniBase}_flirt -omat {mniBase}_flirt.mat -v"; print(cmd);os.system(cmd)
    #non linear reg of MNI to preopT1 space
    print("\n\nLinear registration of MNI template to image is done\n\nStarting Non-linear registration:\n\n\n")
    if preop3Tmask == None:
        cmd = f"fnirt --in={MNItemplatePath} --ref={preop3T} --aff={mniBase}_flirt.mat --iout={mniBase}_fnirt -v --cout={mniBase}_coef --fout={mniBase}_warp"; print(cmd); os.system(cmd)
        #cmd = f"applywarp -i {MNItemplatePath} -r {STDpreop3T} -w {mniBase}_warp --premat={mniBase}_flirt.mat --interp=nn -o {mniBase}_fnirtapplywarp"; print(cmd); os.system(cmd)
    else:
        STDpreop3Tmask = join(outputDirectory, utils.baseSplitextNiiGz(preop3Tmask)[2] + "_std.nii.gz")
        cmd = f"fslreorient2std {preop3Tmask} {STDpreop3Tmask}"; print(cmd);os.system(cmd)
        cmd = f"fnirt --in={MNItemplatePath} --ref={STDpreop3T} --refmask={preop3Tmask} --aff={mniBase}_flirt.mat --iout={mniBase}_fnirt -v --cout={mniBase}_coef --fout={mniBase}_warp"; print(cmd); os.system(cmd)
        
        
        
def getBrainFromMask(preop3T, preop3TMask, outputName):
    img = nib.load(preop3T)
    imgData = img.get_fdata()  # getting actual image data array
    
    mask = nib.load(preop3TMask)
    mask_data = mask.get_fdata()  
    mask_data[np.where(mask_data >0 )] = imgData[np.where(mask_data >0)]

    brain = nib.Nifti1Image(mask_data, img.affine)
    print(f"Saving to {outputName}")
    nib.save(brain, outputName)

def getExpandedBrainMask(preop3Tmask, output, expansion = 10):
    img = nib.load(preop3Tmask)
    imgData = img.get_fdata()
    imgDataExpand = copy.deepcopy(imgData)
    for i in range(expansion):
        imgDataExpand = ndimage.binary_dilation(imgDataExpand).astype(imgDataExpand.dtype)
    imgExpand = nib.Nifti1Image(imgDataExpand, img.affine)
    nib.save(imgExpand, output)
    
    
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

def applywarp_to_atlas(atlasPaths, preop3T, MNIwarp, outputDirectory, isDir = True):
    if isDir: #crawling through directories
        utils.checkPathError(atlasPaths)
        allpaths = []
        for i,j,y in os.walk(atlasPaths):
            allpaths.append(i)
        #Find all atlases in the atlases folders and their subfolders
        atlasesList = []
        for s in range(len(allpaths)):
            atlasesList = atlasesList +  [f"{allpaths[s]}/" + st for st in [f for f in listdir(allpaths[s]) if isfile(join(allpaths[s], f))]]
    else:
        atlasesList = atlasPaths
    utils.checkPathError(MNIwarp)
    utils.checkPathError(preop3T)
    utils.checkPathError(outputDirectory)
    for i in range(len(atlasesList)):
        atlasName = basename(splitext(splitext(atlasesList[i])[0])[0])
        outputAtlasName = join(outputDirectory, atlasName + ".nii.gz")
        #if not os.path.exists(outputAtlasName):
        if utils.checkIfFileExists(outputAtlasName, returnOpposite=True):
            cmd = f"applywarp -i { atlasesList[i]} -r {preop3T} -w {MNIwarp} --interp=nn -o {outputAtlasName} --verbose"; print(cmd); os.system(cmd)
        #else: print(f"File exists: {outputAtlasName}")
        
            
def by_region(electrodePreopT1Coordinates, atlasPath, atlasLabelsPath, ofname, sep=",",  xColIndex=10, yColIndex=11, zColIndex=12, description = "unknown_atlas", Labels=True):
    # getting imaging data
    img = nib.load(atlasPath)
    imgData = img.get_fdata()  # getting actual image data array

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
        atlas_regions_numbers = np.arange(0,   np.max(imgData)+1 )
        atlas_labels_descriptors = np.arange(0,   np.max(imgData)+1 ).astype("int").astype("object")
        atlas_name = os.path.splitext(os.path.basename(atlasPath))[0]
        atlas_name = os.path.splitext(atlas_name)[0]
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
    # getting electrode coordinates data
    data = pd.read_csv(electrodePreopT1Coordinates, sep=sep, header=None)
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
        img_ROI = imgData[coordinates_voxels[:,0]-1, coordinates_voxels[:,1]-1, coordinates_voxels[:,2]-1]
    except: #checking to make sure coordinates are in the atlas. This happens usually for electrodes on the edge of the SEEG. For example, RID0420 electrodes LE11 and LE12 are outside the brain/skull, and thus are outside even the normal MNI space of 181x218x181 voxel dimensions
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>imgData.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = 0
                print('Coordinate outside of atlas image space: setting to zero')
            elif((coordinates_voxels[i,1]>imgData.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = 0  
                print('Coordinate outside of atlas image space: setting to zero')
            elif((coordinates_voxels[i,2]>imgData.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = 0   
                print('Coordinate outside of atlas image space: setting to zero')
            else:
                img_ROI[i] = imgData[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]

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
    imgData = img.get_fdata()  # getting actual image data array
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
        img_ROI = imgData[coordinates_voxels[:,0]-1, coordinates_voxels[:,1]-1, coordinates_voxels[:,2]-1]
    except:
        img_ROI = np.zeros((coordinates_voxels.shape[0],))
        for i in range(0,coordinates_voxels.shape[0]):
            if((coordinates_voxels[i,0]>imgData.shape[0]) or (coordinates_voxels[i,0]<1)):
                img_ROI[i] = -1
                print('Coordinate outside of MNI space: setting to zero')
            elif((coordinates_voxels[i,1]>imgData.shape[1]) or (coordinates_voxels[i,1]<1)):
                img_ROI[i] = -1 
                print('Coordinate outside of MNI space: setting to zero')
            elif((coordinates_voxels[i,2]>imgData.shape[2]) or (coordinates_voxels[i,2]<1)):
                img_ROI[i] = -1   
                print('Coordinate outside of MNI space: setting to zero')
            else:
                img_ROI[i] = imgData[coordinates_voxels[i,0]-1, coordinates_voxels[i,1]-1, coordinates_voxels[i,2]-1]
                
    img_ROI = np.reshape(img_ROI, [img_ROI.shape[0], 1])
    distances = copy.deepcopy(img_ROI)
    distances[(distances == 0)] = -1 #if coordinate equals to outside brain, then temporarily set to -1
    distances[(distances == label)] = 0 #if coordinate equals to the label, then it is zero distance
    
    # list of all points with label
    labelInds = np.where((imgData == label) )

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

def show_slices(fname, low = 0.33, middle = 0.5, high = 0.66, save = False, saveFilename = None, isPath = True):
    
    if isPath:
        img = nib.load(fname)
        imgdata = img.get_fdata()  
    else:
        imgdata = fname
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

