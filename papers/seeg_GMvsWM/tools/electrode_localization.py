"""
Date 2020.05.10
updated 2020.11.10
Andy Revell 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    1. To find the region in which an x, y, z coordinate is given - usually for electrode localizetion

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. by atlas: Given a csv file with electrode coordinates, the atlas path, an mni template, it will output the corresponding region label in the atlas
        1. Load atlas
        2. load MNI template
        3. load coorniate
        4. Tranformation of world to voxel corrdinates system
        5. Check to make sure coordinates are actually in the image space
        6. Find the region label corresponding to the coordinate
    2. inside or outside atlas: find whether or not the electrode coordinate is inside or outside the atlas
    3. distance_from_grayMatter: finds the distance from gray matter tissue

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    1. Electrode_coordinates_mni_path. An N x 4 csv file. N = number of electrodes. Col 1: Electrode label. Col 2-4: x, y, z coordinate
    2. atlas_path: the full path of the atlas you want 
    3. MNI path: the 1x1x1 MNI template

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    4. An N x 5 csv file saved in specified path. 
    N = number of electrodes. Col 1: Electrode label. Col 2-4: x, y, z coordinate. Col 5: the ROI in the atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    from os.path import join as ospj
    path = "/mnt" Where paper/study top-level directory is stored
    #path = "/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001"
    electrode_coordinates_mni_path= ospj(path, 'data_raw/electrode_localization/sub-RID0194/sub-RID0194_electrode_coordinates_mni.cs')
    atlas_path=ospj(path, 'data_raw/atlases/standard_atlases/AAL600.nii.gz')
    outputfile=ospj(path, 'data_processed/electrode_localization_atlas_region/sub-RID0194/AAL600/sub-RID0194_electrode_coordinates_mni_AAL600.csv')
    mni_template_path =ospj(path, 'data_raw/MNI_brain_template/MNI152_T1_1mm_brain.nii.gz')
    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please use this naming convention

example:

"""
import numpy as np
import pandas as pd
import nibabel as nib
import copy
import matplotlib.pyplot as plt
import os


def by_region(ifname_electrode_localization_sub_ID, ifname_atlas_path, ifname_atlas_labels_path, ofname, description = "unknown_atlas", noLabels=False):
    """
    electrode_coordinates_path = ifname_electrode_localization_sub_ID
    ifname_atlas_path = ifname_seg_sub_ID
    outputfile = ospj(ofpath_localization_files, "sub-{0}_GM_WM_CSF.csv".format(sub_ID))
    """
    # getting imaging data
    img = nib.load(ifname_atlas_path)
    img_data = img.get_fdata()  # getting actual image data array
    #aff = img.affine  # get affine transformation in atals. Helps us convert real-world coordinates to voxel locations

    #show_slices(img_data)
    
    affine = img.affine
    print(nib.aff2axcodes(affine))

    if noLabels == False:
        #getting atlas labels file
        atlas_labels = pd.read_csv(ifname_atlas_labels_path, sep=",", header=None)
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
        atlas_labels = atlas_labels.drop([0, 1], axis=0).reset_index(drop=True)
        atlas_regions_numbers = np.array(atlas_labels.iloc[:,0]).astype("float64")
        atlas_labels_descriptors = np.array(atlas_labels.iloc[:,1])
    if noLabels == True:
        atlas_regions_numbers = np.arange(0,   np.max(img_data)+1 )
        atlas_labels_descriptors = np.arange(0,   np.max(img_data)+1 ).astype("int").astype("object")
        atlas_name = os.path.splitext(os.path.basename(ifname_atlas_path))[0]
        atlas_name = os.path.splitext(atlas_name)[0]
        column_description1 = f"{description}_region_number"
        column_description2 = f"{description}_label"
    # getting electrode coordinates data
    data = pd.read_csv(ifname_electrode_localization_sub_ID, sep=",", header=None)
    data = data.iloc[:, [0, 10, 11, 12]]
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


def distance_from_label(ifname_electrode_localization_sub_ID, ifname_atlas_path, label, ifname_atlas_labels_path, ofname):
    """
    electrode_coordinates_path = ifname_electrode_localization_sub_ID
    atlas_path = ifname_seg_sub_ID
    label = 2
    outputfile =  ospj(ofpath_localization_files, "sub-{0}_WM_distance.csv".format(sub_ID))
    description = "tissue_segmentation"

    """

    # getting imaging data
    img = nib.load(ifname_atlas_path)
    img_data = img.get_fdata()  # getting actual image data array
    #aff = img.affine  # get affine transformation in atals. Helps us convert real-world coordinates to voxel locations

    show_slices(img_data)
    
    affine = img.affine
    print(nib.aff2axcodes(affine))

    # getting electrode coordinates data
    data = pd.read_csv(ifname_electrode_localization_sub_ID, sep=",", header=None)
    data = data.iloc[:, [0, 10, 11, 12]]
    
    atlas_labels = pd.read_csv(ifname_atlas_labels_path, sep=",", header=None)
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


