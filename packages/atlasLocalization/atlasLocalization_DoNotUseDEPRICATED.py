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

    
electrodePreopT1Coordinates = sys.argv[1]
preopT1 = sys.argv[2]
preopT1bet = sys.argv[3]
MNItemplate = sys.argv[4]
MNItemplatebet = sys.argv[5]
atlasDirectory = sys.argv[6]
outputDirectory = sys.argv[7]
outputName = str(sys.argv[8])

if not splitext(outputName)[1] == ".csv":
    raise IOError(f"\n\n\n\nOutput Name must end with .csv.\n\nWhat is given:\n{outputName}\n\n\n\n" )

print(f"\n\n\n\n\n\n\n\n\n\n{outputName} saving to: \n{join(outputDirectory, outputName)}\n\n\n\n")
fillerString = "\n###########################\n###########################\n###########################\n###########################\n"
"""
Examples:
    
#electrodePreopT1Coordinates = "/media/arevell/sharedSSD/linux/data/BIDS/PIER/sub-RID0648/ses-implant01/ieeg/sub-RID0648_ses-implant01_space-preimplantT1w_electrodes.tsv"
electrodePreopT1Coordinates = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/electrodenames_coordinates_native_and_T1.csv"
preopT1 = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/T00_RID648_mprage.nii.gz"
preopT1bet = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/T00_RID648_mprage_brainBrainExtractionBrain.nii.gz"
MNItemplate = "/media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz"
MNItemplatebet = "/media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz"
atlasDirectory = "/media/arevell/sharedSSD/linux/revellLab/tools/atlases"

outputDirectory = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648"
outputName = "sub-RID0648_atlasLocalization.csv"


check_path(electrodePreopT1Coordinates)
check_path(preopT1)
check_path(preopT1bet)
check_path(MNItemplate)
check_path(MNItemplatebet)
check_path(atlasDirectory)
check_path(outputDirectory)

    
conda activate revellLab

python /media/arevell/sharedSSD/linux/revellLab/packages/atlasLocalization/atlasLocalization.py \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/electrodenames_coordinates_native_and_T1.csv \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/T00_RID648_mprage.nii.gz \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648/T00_RID648_mprage_brainBrainExtractionBrain.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/atlases \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0648
    sub-RID0648_atlasLocalization.csv
    
    
rid = "652"
    
    
print(\
    f"conda activate revellLab \n\
python /media/arevell/sharedSSD/linux/revellLab/packages/atlasLocalization/atlasLocalization.py  \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid}/electrodenames_coordinates_native_and_T1.csv \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid}/T00_RID{rid}_mprage.nii.gz \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid}/T00_RID{rid}_mprage_brainBrainExtractionBrain.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz \
    /media/arevell/sharedSSD/linux/revellLab/tools/atlases \
    /media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization/sub-RID0{rid} \
    sub-RID0{rid}_atlasLocalization.csv\n\n" \
    )
    
    
"""

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


check_path(electrodePreopT1Coordinates)
check_path(preopT1)
check_path(preopT1bet)
check_path(MNItemplate)
check_path(MNItemplatebet)
check_path(atlasDirectory)
check_path(outputDirectory)



#names of outputs output
outputDirectoryTMP = join(outputDirectory, "tmp")


preopT1_basename =  os.path.basename( splitext(splitext(preopT1)[0])[0])
preopT1bet_basename =  os.path.basename( splitext(splitext(preopT1bet)[0])[0])
preopT1_output = f"{join(outputDirectoryTMP, preopT1_basename)}"
preopT1bet_output = f"{join(outputDirectoryTMP, preopT1bet_basename)}"


outputNameTissueSeg = join(outputDirectory, f"{preopT1_basename}_std1x1x1_tissueSegmentation.nii.gz")
FIRST = join(outputDirectory, f"{preopT1_basename}_std1x1x1_FIRST.nii.gz")
FAST = join(outputDirectory, f"{preopT1_basename}_std1x1x1_FAST.nii.gz")
outputMNIname = "mni"
MNIwarp = join(outputDirectory, outputMNIname + "_warp.nii.gz")



#Make temporary storage 
if not os.path.exists(outputDirectoryTMP):
    cmd = f"mkdir {outputDirectoryTMP}"; print(f"\n{cmd}"); os.system(cmd)

#%%Begin Pipeline: Orient all images to standard RAS

print(f"\n\n{fillerString}Part 1 of 4\nReorientation of Images\nEstimated time: 10-30 seconds{fillerString}\nReorient all images to standard RAS\n")
cmd = f"fslreorient2std {preopT1} {preopT1_output}_std.nii.gz"; print(cmd); os.system(cmd)
cmd = f"fslreorient2std {preopT1bet} {preopT1bet_output}_std.nii.gz"; print(cmd); os.system(cmd)

print("\n\n\nReslice all standard RAS images to 1x1x1mm voxels\n")
#Make images 1x1x1 mm in size (usually clinical images from electrodeLocalization pipelines are 0.98x0.98x1; others are even more different than 1x1x1)
cmd = f"flirt -in {preopT1_output}_std.nii.gz -ref {preopT1_output}_std.nii.gz -applyisoxfm 1.0 -nosearch -out {preopT1_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)
cmd = f"flirt -in {preopT1bet_output}_std.nii.gz -ref {preopT1bet_output}_std.nii.gz -applyisoxfm 1.0 -nosearch -out {preopT1bet_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)


#visualize
show_slices(f"{preopT1_output}_std1x1x1.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename = join(outputDirectory, "pic_T00.png")  )
show_slices(f"{preopT1bet_output}_std1x1x1.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_T00bet.png"))

print(f"\n\nPictures are saved to {outputDirectory}\nPlease check for quality assurance")



#%%Tissue segmentation
print(f"\n\n{fillerString}Part 2 of 4\nTissue Segmentation\nEstimated time: 30+ min{fillerString}\nRUNNING FIRST SUBCORTICAL SEGMENTATION\n")

#FIRST: subcortical segmentation
if not os.path.exists(FIRST):
    cmd = f"run_first_all -i {preopT1bet_output}_std1x1x1.nii.gz -o {preopT1bet_output}_std1x1x1.nii.gz -b -v"; print(cmd); os.system(cmd)
    #clean up files
    cmd = f"rm -r {preopT1bet_output}_std1x1x1.logs"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*.bvars"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*.vtk"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*origsegs*"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1_to_std*"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1*.com*"; print(cmd); os.system(cmd)
    cmd = f"mv {preopT1bet_output}_std1x1x1_all_fast_firstseg.nii.gz {FIRST}"; print(cmd); os.system(cmd)
    
else:
    print(f"File exists:\n{FIRST}")
show_slices(f"{FIRST}", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_FIRST.png"))
print(f"\nPictures of FIRST are saved to {outputDirectory}\nPlease check for quality assurance")
    
print("\n\n\nRUNNING FAST SEGMENTATION\n")
    
#FAST: segmentation of cortex
if not os.path.exists(FAST):
    cmd = f"fast -n 3 -H 0.25 -t 1 -v {preopT1bet_output}_std1x1x1.nii.gz"; print(cmd); os.system(cmd)
    #Clean up files
    cmd = f"rm -r {preopT1bet_output}_std1x1x1_*mixeltype*"; print(cmd); os.system(cmd)
    cmd = f"rm -r {preopT1bet_output}_std1x1x1_*pve*"; print(cmd); os.system(cmd)
    cmd = f"mv {preopT1bet_output}_std1x1x1_seg.nii.gz {FAST}"; print(cmd); os.system(cmd)
else:
    print(f"File exists:\n{FAST}")
show_slices(f"{FAST}", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_FAST.png"))
print(f"\nPictures of FAST are saved to {outputDirectory}\nPlease check for quality assurance")


#Combine FIRST and FAST images
combine_first_and_fast(FIRST, FAST, outputNameTissueSeg)
show_slices(outputNameTissueSeg, low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_tissueSegmentation.png"))
print(f"\nPictures of FIRST + FAST combined images (pic_tissueSegmentation.png) are saved to {outputDirectory}\nPlease check for quality assurance")


#%%Registration of MNI to patient space (atlases are all in MNI space, so using this warp to apply to the atlases)
print(f"\n\n{fillerString}Part 3 of 4\nMNI and atlas registration\nEstimated time: 1-2+ hours{fillerString}\nRegistration of MNI template to patient space\n")
if not os.path.exists(MNIwarp):
    register_MNI_to_preopT1(f"{preopT1_output}_std1x1x1.nii.gz", f"{preopT1bet_output}_std1x1x1.nii.gz", MNItemplate, MNItemplatebet, outputMNIname, outputDirectory)
show_slices(f"{join(outputDirectory, outputMNIname)}_fnirt.nii.gz", low = 0.33, middle = 0.5, high = 0.66, save = True, saveFilename =  join(outputDirectory, "pic_mni_fnirt.png"))
print(f"\nPictures of MNI nonlinear registration is saved to {outputDirectory}\nPlease check for quality assurance")


print("\n\n\nUsing MNI template warp to register all atlases (these atlases are already in MNI space)")
#apply warp to all atlases
if not os.path.exists(join(outputDirectory, outputName)):
    applywarp_to_atlas(atlasDirectory, f"{preopT1_output}_std1x1x1.nii.gz", MNIwarp, outputDirectoryTMP)


#%%Electrode Localization
print(f"\n\n{fillerString}Part 4 of 4\nElectrode Localization\nEstimated time: 10-20 min{fillerString}\nPerforming Electrode Localization\n")

#do not run if electrode localization output file already exists. Part 3 atlas warp takes a while, so skip it. If need to re-run, delete old file.
if not os.path.exists(join(outputDirectory, outputName)):
    #localization by region to tissue segmentation 
    outputTissueCoordinates = join(outputDirectoryTMP, "tissueSegmentation.csv")
    by_region(electrodePreopT1Coordinates, outputNameTissueSeg, join(atlasDirectory, "tissue_segmentation.csv"), outputTissueCoordinates, description = "tissue_segmentation", Labels=True)
    #rename channels to standard 4 characters (2 letters, 2 numbers)
    channel2stdCSV(outputTissueCoordinates)

    
    #localization by region to atlases
    atlases = [f for f in listdir(atlasDirectory) if isfile(join(atlasDirectory, f))]
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlas = join(atlasDirectory, atlases[i])
        atlasLabels =  join(atlasDirectory, atlasName + ".csv")
        
        if (".nii" in atlas):
            atlasInMNI = join(outputDirectoryTMP, atlasName + ".nii.gz")
            check_path(atlasInMNI)
            print(f"{atlasName}")
            if "RandomAtlas" in atlasName: 
                Labels=False
            else:
                check_path(atlasLabels)
                Labels=True
            outputAtlasCoordinates = join(outputDirectoryTMP, f"{atlasName}" + "_localization.csv")
            by_region(electrodePreopT1Coordinates, atlasInMNI, atlasLabels, outputAtlasCoordinates, description = atlasName, Labels=Labels)
            channel2stdCSV(outputAtlasCoordinates)
    
    
    #localization of channel distance to tissue segmentation: White Matter electrodes distance to Gray Matter
    print("\n\n\n\nFinding the WM electrode contacts distance to GM")
    outputTissueCoordinatesDistanceGM = join(outputDirectory, "electrodeWM_DistanceToGM.csv")
    if not os.path.exists(outputTissueCoordinatesDistanceGM):
        distance_from_label(electrodePreopT1Coordinates, outputNameTissueSeg, 2, join(atlasDirectory, "tissue_segmentation.csv"), outputTissueCoordinatesDistanceGM)
    channel2stdCSV(outputTissueCoordinatesDistanceGM)
    
    #localization of channel distance to tissue segmentation: Gray Matter electrodes distance to White Matter
    print("\n\n\n\nFinding the GM electrode contacts distance to WM")
    outputTissueCoordinatesDistanceWM = join(outputDirectory, "electrodeGM_DistanceToWM.csv")
    if not os.path.exists(outputTissueCoordinatesDistanceWM):
        distance_from_label(electrodePreopT1Coordinates, outputNameTissueSeg, 3, join(atlasDirectory, "tissue_segmentation.csv"), outputTissueCoordinatesDistanceWM)
    channel2stdCSV(outputTissueCoordinatesDistanceWM)
    
    
    print("\n\n\n\nConcatenating all files")
    #Concatenate files into one
    dataTissue = pd.read_csv(outputTissueCoordinates, sep=",", header=0)
    dataGM = pd.read_csv(outputTissueCoordinatesDistanceGM, sep=",", header=0).iloc[:,4:]
    dataWM = pd.read_csv(outputTissueCoordinatesDistanceWM, sep=",", header=0).iloc[:,4:]
    data = pd.concat([dataTissue, dataGM, dataWM]  , axis = 1)
    atlases = [f for f in listdir(atlasDirectory) if isfile(join(atlasDirectory, f))]
    
    
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlas = join(atlasDirectory, atlases[i])
        atlasLabels =  join(atlasDirectory, atlasName + ".csv")
        if (".nii" in atlas):
            if not ("RandomAtlas" in atlas):
                print(atlasName)
                outputAtlasCoordinates = join(outputDirectoryTMP, f"{atlasName}" + "_localization.csv")
                data= pd.concat([data, pd.read_csv(outputAtlasCoordinates, sep=",", header=0).iloc[:,4:] ] , axis = 1)
                                
    for i in range(len(atlases)):
        atlasName = splitext(splitext(atlases[i])[0])[0]
        atlas = join(atlasDirectory, atlases[i])
        atlasLabels =  join(atlasDirectory, atlasName + ".csv")
        if (".nii" in atlas):
            if ("RandomAtlas" in atlas):
                print(atlasName)
                outputAtlasCoordinates = join(outputDirectoryTMP, f"{atlasName}" + "_localization.csv")
                data= pd.concat([data, pd.read_csv(outputAtlasCoordinates, sep=",", header=0).iloc[:,4:] ] , axis = 1)
                                
    electrodeLocalization = join(outputDirectory, f"{outputName}")
    pd.DataFrame.to_csv(data, electrodeLocalization, header=True, index=False)

#clean
print("\n\n\nCleaning Files")

cmd = f"mv {preopT1_output}_std1x1x1.nii.gz {outputDirectory}"; print(cmd); os.system(cmd)
cmd = f"mv {preopT1bet_output}_std1x1x1.nii.gz {outputDirectory}"; print(cmd); os.system(cmd)


cmd = f"rm {join(outputDirectory, 'mni_flirt*' )}"; print(cmd); os.system(cmd)
cmd = f"rm -r {join(outputDirectory, 'tmp' )}"; print(cmd); os.system(cmd)



if not os.path.exists(join(outputDirectory, outputName)):
    print(f"\n\n\nDone\n\nFind electrode localization file in {join(outputDirectory, outputName)}\n\n")
else:
    print(f"\n\n\n\n\n\n\n\n\n\nNote: Electrode localization file alread exists in \n{join(outputDirectory, outputName)}\n\
If you need to re-run pipeline, please delete this file (Part 3 atlas warp files will need to be re-made. They are large and only temporarily saved.\n\n\n\n\n")
