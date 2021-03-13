"""
2020.05.06
Andy Revell and Alex Silva
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
import numpy as np
import nibabel as nib
import os
from os.path import join

def grassfireAlgorithm(edgePoints,atlas,vols):
    dims = atlas.shape
    if((edgePoints.shape[0]*7) < (dims[0]*dims[1]*dims[2])):
        newEdgePoints = np.zeros((edgePoints.shape[0]*7, 4))
    else:
        newEdgePoints = np.zeros((dims[0]*dims[1]*dims[2],4))
    counter = 1
    dimsEdge = edgePoints.shape
    for i in range(0,dimsEdge[0]):
        point = edgePoints[i,:].astype(int)
        # step in positive x
        if((point[0]+1)<dims[0]):
            if(atlas[point[0]+1,point[1],point[2]]==0):
                newEdgePoints[counter,:] = [point[0]+1,point[1],point[2], point[3]]
                atlas[point[0]+1,point[1],point[2]]= point[3]
                counter = counter + 1
                vols[point[3]-1] = vols[point[3]-1]+1
        # step in negative x
        if((point[0]-1)>=0):
            if(atlas[point[0]-1,point[1],point[2]]==0):
                newEdgePoints[counter,:] = [point[0]-1,point[1],point[2], point[3]]
                atlas[point[0]-1,point[1],point[2]]= point[3]
                counter = counter + 1
                vols[point[3]-1] = vols[point[3]-1]+1
        # step in positive y
        if((point[1]+1)<dims[1]):
            if(atlas[point[0],point[1]+1,point[2]]==0):
                newEdgePoints[counter,:] = [point[0],point[1]+1,point[2], point[3]]
                atlas[point[0],point[1]+1,point[2]]= point[3]
                counter = counter + 1
                vols[point[3]-1] = vols[point[3]-1]+1
        # step in negative y
        if((point[1]-1)>=0):
            if(atlas[point[0],point[1]-1,point[2]]==0):
                newEdgePoints[counter,:] = [point[0],point[1]-1,point[2], point[3]]
                atlas[point[0],point[1]-1,point[2]]= point[3]
                counter = counter + 1
                vols[point[3]-1] = vols[point[3]-1]+1
        # step in positive z
        if((point[2]+1)<dims[2]):
            if(atlas[point[0],point[1],point[2]+1]==0):
                newEdgePoints[counter,:] = [point[0],point[1],point[2]+1, point[3]]
                atlas[point[0],point[1],point[2]+1]= point[3]
                counter = counter + 1
                vols[point[3]-1] = vols[point[3]-1]+1
        # step in negative x
        if((point[2]-1)>=0):
            if(atlas[point[0],point[1],point[2]-1]==0):
                newEdgePoints[counter,:] = [point[0],point[1],point[2]-1, point[3]]
                atlas[point[0],point[1],point[2]-1]= point[3]
                counter = counter + 1
                vols[point[3]-1] = vols[point[3]-1]+1
                
    newEdgePoints = newEdgePoints[~np.all(newEdgePoints == 0, axis=1)]
    extraCol = np.zeros((newEdgePoints.shape[0],1))
    newEdgePoints = np.concatenate((newEdgePoints,extraCol),axis=1)
    if(newEdgePoints.shape[0]>1):
        for i in range(0,newEdgePoints.shape[0]):
            newEdgePoints[i,4] = vols[int(newEdgePoints[i,3]-1)]                                          
                                      
    newEdgePoints = newEdgePoints[newEdgePoints[:,4].argsort()]       
    return(newEdgePoints,atlas,vols)

def generateRandomAtlases_wholeBrain(numberOfRegions, outputPath, MNItemplateBrainPath):
    print("Loading Template")
    MNItemp_path = nib.load(MNItemplateBrainPath)
    T1_data = MNItemp_path.get_fdata() #getting actual image data array
    atlas = np.zeros(T1_data.shape)
    dimOrig = atlas.shape
    print("Geting all the points in the template that are not outside of brain")
    brainPoints = np.zeros((dimOrig[0]*dimOrig[1]*dimOrig[2],4))
    rowCount = 1
    for i in range(0,dimOrig[0]):
        for j in range(0,dimOrig[1]):
            for k in range(0,dimOrig[2]):
                if(T1_data[i,j,k]>0):
                    brainPoints[rowCount,:] = [i,j,k,T1_data[i,j,k]]
                    rowCount = rowCount + 1
                else: 
                    atlas[i,j,k] = -1
    brainPoints = brainPoints[~np.all(brainPoints == 0, axis=1)]
    currentAtlas = atlas.copy()
    # choose n random start points inside brain
    print("Seeding Brain Regions with {0} seeds".format(numberOfRegions))
    indexstartPoints = np.random.choice(brainPoints.shape[0],numberOfRegions,replace=False)
    startPoints = brainPoints[indexstartPoints,0:3]
    extraCols = np.zeros((startPoints.shape[0],2))
    startPoints = np.concatenate((startPoints,extraCols),axis=1)
    for i in range(0,startPoints.shape[0]):
        startPoints[i,3] = i+1
        startPoints[i,4] = 1
        point = startPoints[i,:]
        currentAtlas[int(point[0]),int(point[1]),int(point[2])] = point[3]
    
    vols = np.ones((startPoints.shape[0],1))
    print("Expanding seeds with Grassfire Algorithm")
    while(startPoints.size !=0):
        startPoints,currentAtlas,vols=grassfireAlgorithm(startPoints,currentAtlas,vols)
        
    currentAtlas[currentAtlas == -1] = 0  
    # write out the atlas to file \
    cur_nifti_img = nib.Nifti1Image(currentAtlas, MNItemp_path.affine)
    print("Saving {0}".format(outputPath))
    nib.save(cur_nifti_img, outputPath)


def batchGenerateRandomAtlases(regionNumbersList, permutations, MNItemplateBrainPath, savePath):
    if not os.path.exists(MNItemplateBrainPath): raise IOError(f"{MNItemplateBrainPath} does not exists")
    if not os.path.exists(savePath): raise IOError(f"{savePath} does not exists")
    
    for a in range(len(regionNumbersList)):
        numberOfRegions = regionNumbersList[a]
        atlasNamesRandom = "RandomAtlas{:07}".format(numberOfRegions)
        for p in range(1, permutations+1):
            atlasNameVersion = "{0}_v{1}.nii.gz".format(atlasNamesRandom, '{:04}'.format(p))
            fnameAtlasesRandom = join(savePath, atlasNameVersion )
            print("\nGenerating Atlas: {0}".format(atlasNameVersion))
            #Volumes
            if not (os.path.exists(fnameAtlasesRandom)):#check if file exists
                generateRandomAtlases_wholeBrain(numberOfRegions, fnameAtlasesRandom, MNItemplateBrainPath)
            else:
                print("File exists: {0}".format(fnameAtlasesRandom))





#Not used for study:
def generateRandomAtlases_wholeBrain_tissue_seg_based(numberOfRegions, output_file, MNItemplateBrainPath, tissue_seg_path):
    print("Loading Template")
    MNItemp_path = nib.load(MNItemplateBrainPath)
    T1_data = MNItemp_path.get_fdata() #getting actual image data array
    atlas = np.zeros(T1_data.shape)
    dimOrig = atlas.shape
    # load in the tissue segmentation 
    tissue_seg = nib.load(tissue_seg_path)
    #tissue_data = tissue_seg.get_fdata()
    print("Geting all the points in the template that are not CSF or outside of brain")
    brainPoints = np.zeros((dimOrig[0]*dimOrig[1]*dimOrig[2],4))
    rowCount = 1
    for i in range(0,dimOrig[0]):
        for j in range(0,dimOrig[1]):
            for k in range(0,dimOrig[2]):
                if(tissue_seg[i,j,k]>0):
                    #This was the cutoff we found optimal
                    brainPoints[rowCount,:] = [i,j,k,T1_data[i,j,k]]
                    rowCount = rowCount + 1
                else: 
                    atlas[i,j,k] = -1
    
    brainPoints = brainPoints[~np.all(brainPoints == 0, axis=1)]
    currentAtlas = atlas.copy()
    # choose n random start points inside brain
    print("Seeding Brain Regions with {0} seeds".format(numberOfRegions))
    indexstartPoints = np.random.choice(brainPoints.shape[0],numberOfRegions,replace=False)
    startPoints = brainPoints[indexstartPoints,0:3]
    extraCols = np.zeros((startPoints.shape[0],2))
    startPoints = np.concatenate((startPoints,extraCols),axis=1)
    for i in range(0,startPoints.shape[0]):
        startPoints[i,3] = i+1
        startPoints[i,4] = 1
        point = startPoints[i,:]
        currentAtlas[int(point[0]),int(point[1]),int(point[2])] = point[3]
    
    vols = np.ones((startPoints.shape[0],1))
    print("Expanding seeds with Grassfire Algorithm")
    while(startPoints.size !=0):
        startPoints,currentAtlas,vols=grassfireAlgorithm(startPoints,currentAtlas,vols)
    currentAtlas[currentAtlas == -1] = 0  
    # write out the atlas to file \
    cur_nifti_img = nib.Nifti1Image(currentAtlas, MNItemp_path.affine)
    nib.save(cur_nifti_img, output_file)
    

#%%

        

               