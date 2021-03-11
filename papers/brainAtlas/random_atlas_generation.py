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

def generateRandomAtlases_wholeBrain(number_of_regions, output_file, ifname_MNI_template):
    print("Loading Template")
    MNItemp_path = nib.load(ifname_MNI_template)
    T1_data = MNItemp_path.get_fdata() #getting actual image data array
    atlas = np.zeros(T1_data.shape)
    dimOrig = atlas.shape
    print("Geting all the points in the template that are not CSF or outside of brain")
    brainPoints = np.zeros((dimOrig[0]*dimOrig[1]*dimOrig[2],4))
    rowCount = 1
    for i in range(0,dimOrig[0]):
        for j in range(0,dimOrig[1]):
            for k in range(0,dimOrig[2]):
            
                if(T1_data[i,j,k]>2000):
                    #This was the cutoff we found optimal
                    brainPoints[rowCount,:] = [i,j,k,T1_data[i,j,k]]
                    rowCount = rowCount + 1
                else: 
                    atlas[i,j,k] = -1
    brainPoints = brainPoints[~np.all(brainPoints == 0, axis=1)]
    currentAtlas = atlas.copy()
    # choose n random start points inside brain
    print("Seeding Brain Regions with {0} seeds".format(number_of_regions))
    indexstartPoints = np.random.choice(brainPoints.shape[0],number_of_regions,replace=False)
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
    print("Saving {0}".format(output_file))
    nib.save(cur_nifti_img, output_file)

#Not used for study:
def generateRandomAtlases_wholeBrain_tissue_seg_based(number_of_regions, output_file, ifname_MNI_template, tissue_seg_path):
    print("Loading Template")
    MNItemp_path = nib.load(ifname_MNI_template)
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
    print("Seeding Brain Regions with {0} seeds".format(number_of_regions))
    indexstartPoints = np.random.choice(brainPoints.shape[0],number_of_regions,replace=False)
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

        

               