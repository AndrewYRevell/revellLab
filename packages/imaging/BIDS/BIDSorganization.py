#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:37:24 2021

@author: arevell
"""

import sys
import os
import pandas as pd
import copy
from os import listdir, walk
from  os.path import join, isfile, isdir, splitext, basename, relpath, dirname
import numpy as np
from itertools import repeat
from revellLab.packages.utilities import utils
import json
import smtplib, ssl
import pkg_resources
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
import time
from glob import glob

#%% Getting directories


dataset= "PIER"

BIDSlocal = join("/media/arevell/sharedSSD/linux/data/BIDS", dataset)
BIDSserver = join("/home/arevell/borel/DATA/Human_Data/BIDS",dataset)

SERVdir = [join(BIDSserver, o) for o in listdir(BIDSserver) if isdir(join(BIDSserver, o)) and basename(join(BIDSserver, o))[0:3] == "sub"  ]
LOCAdir = [join(BIDSlocal, o) for o in listdir(BIDSlocal) if isdir(join(BIDSlocal, o)) and basename(join(BIDSlocal, o))[0:3] == "sub"  ]

SERVsubj = [basename(item) for item in SERVdir ]
LOCAsubj = [basename(item) for item in LOCAdir ]



#%% Copy from server to local


listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(BIDSserver):
    listOfFiles += [join(dirpath, file) for file in filenames]


for f in range(len(listOfFiles)):
    print(f)
    SERVERfile =  listOfFiles[f]
    LOCALfile = join( BIDSlocal, relpath(SERVERfile, BIDSserver))
    #ignore git
    if not ".git" in relpath(SERVERfile, BIDSserver):
        if utils.checkIfFileDoesNotExist(LOCALfile, printBOOL = False):
            print(f"\n{LOCALfile}")
            utils.checkPathAndMake( dirname(LOCALfile), dirname(LOCALfile), printBOOL = False  )
            utils.executeCommand( f"cp -r {SERVERfile} {LOCALfile}", printBOOL = False  )

#%% Copy from local to server


listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(BIDSlocal):
    listOfFiles += [join(dirpath, file) for file in filenames]


for f in range(len(listOfFiles)):
    print(f)
    LOCALfile =  listOfFiles[f]
    SERVERfile = join( BIDSserver, relpath(LOCALfile, BIDSlocal))
    #ignore git
    if not ".git" in relpath(LOCALfile, BIDSlocal):
        if utils.checkIfFileDoesNotExist(SERVERfile, printBOOL = False):
            print(f"\n{SERVERfile}")
            utils.checkPathAndMake( dirname(SERVERfile), dirname(SERVERfile), printBOOL = False  )
            utils.executeCommand( f"cp -r {LOCALfile} {SERVERfile}", printBOOL = False  )
    
    


#%% Changing names of preop 3T


subFolders = glob(BIDSlocal + "/*/")


for l in range(len(subFolders)):

    folders = glob(subFolders[l]+ "/*/")
    for f in range(len(folders)):
        if folders[f].find("preop3T") > -1:
             newName = folders[f].replace("preop3T", "research3Tv03")
             utils.executeCommand(f"mv {folders[f]} {newName}")
             listOfFiles = list()
             for (dirpath, dirnames, filenames) in os.walk(newName):
                 listOfFiles += [join(dirpath, file) for file in filenames]
             for fi in range(len(listOfFiles)):
                 newNameFi = listOfFiles[fi].replace("preop3T", "research3Tv03")
                 utils.executeCommand(f"mv {listOfFiles[fi]} {newNameFi}")
                 
#%% Changing to different versions

BIDSdir = BIDSlocal
subjects = ["RID0659",
            "RID0660",
            "RID0646"]

old = "research3Tv04"
new = "research3Tv03"

for l in range(len(subjects)):
    subFolder = join(BIDSdir, "sub-"+ subjects[l])
    folders = glob(subFolder+ "/*/")
    for f in range(len(folders)):
        if folders[f].find(old) > -1:
            newName = folders[f].replace(old, new)
            utils.executeCommand(f"mv {folders[f]} {newName}")
            listOfFiles = list()
            for (dirpath, dirnames, filenames) in os.walk(newName):
                listOfFiles += [join(dirpath, file) for file in filenames]
            for fi in range(len(listOfFiles)):
                 newNameFi = listOfFiles[fi].replace(old, new)
                 utils.executeCommand(f"mv {listOfFiles[fi]} {newNameFi}")
                 
                 
                 
#%% add Intended for to fmaps
BIDSdir = BIDSserver

subFolders = glob(BIDSdir + "/*/")


for l in range(len(subFolders)):

    folders = glob(subFolders[l]+ "/*/")
    for f in range(len(folders)):
        if utils.checkIfFileExists( join( folders[f], "fmap")  ):
            session = basename(folders[f][:-1])
            subject = basename(subFolders[l][:-1])
            
            files = glob( join( folders[f], "fmap/*.json")   )
            for fi in range(len(files)):
                if "phasediff" in files[fi] or "_epi" in files[fi]:
                    with open(files[fi]) as fas: jsonfile = json.load(fas)
                    
                   
                    if utils.checkIfFileExists(join(join( folders[f], "dwi", f"{subject}_{session}_dwi.nii.gz") )):
                        jsonfile['IntendedFor'] = f"{session}/dwi/{subject}_{session}_dwi.nii.gz"
                        with open(files[fi], 'w', encoding='utf-8') as fas: json.dump(jsonfile, fas, ensure_ascii=False, indent=4)
                    else:
                        if 'IntendedFor' in jsonfile:
                            del jsonfile['IntendedFor']
                        with open(files[fi], 'w', encoding='utf-8') as fas: json.dump(jsonfile, fas, ensure_ascii=False, indent=4)
                       

#%% Curate from RAW to BIDS

BIDSdir = BIDSserver
ses = "research3Tv04"
subjects = ["RID0659",
            "RID0660",
            "RID0646",
            "RID0661",
            "RID0662",
            "RID0619",
            "RID0663",
            "RID0652",
            "RID0664",
            "RID0665",
            "RID0648",
            "RID0666",
            "RID0667",
            "RID0668",
            "RID0669",
            "RID0670",
            "RID0671",
            "RID0672",
            "RID0673",
            "RID0656"]

subjects = [
            "RID0679",#
            "RID0681",#
            "RID0684",#
            "RID0683"]#

subjectsP = ["3T_P091",
            "3T_P092",
            "3T_P093",
            "3T_P094",
            "3T_P095",
            "3T_P096",
            "3T_P097",
            "3T_P098",
            "3T_P099",
            "3T_P100",
            "3T_P101",
            "3T_P102",
            "3T_P103",
            "3T_P104",
            "3T_P105",
            "3T_P106",
            "3T_P107",
            "3T_P108",
            "3T_P109",
            "3T_P110"]

subjectsP = [
            "3T_P111",#
            "3T_P112",#
            "3T_P113",#
            "3T_C018"]#


subjects = ["RID0682"]#
subjectsP = ["3T_C017"]#

for l in range(len(subjects)):
    subFolder = join(BIDSdir, "sub-"+ subjects[l])
    utils.checkPathAndMake( BIDSserver, subFolder  )
    
    anat =   join(subFolder, f"ses-{ses}", "anat")
    dwi = join(subFolder, f"ses-{ses}", "dwi") 
    fmap =  join(subFolder, f"ses-{ses}", "fmap") 
    func =  join(subFolder, f"ses-{ses}", "func") 
    utils.checkPathAndMake( subFolder, anat)
    utils.checkPathAndMake( subFolder, dwi  )
    utils.checkPathAndMake( subFolder, fmap  )
    utils.checkPathAndMake( subFolder, func  )
    
    raw = join(dirname(BIDSserver), "sourcedata", "3T_Subjects", subjectsP[l])
    utils.checkIfFileExists(raw)
    folderT1 = glob(join(raw, "T1w_MPR*"))
    folderdwi =  glob(join(raw, "MULTISHELL_b2000_117dir*"))
    folderTOP =  glob(join(raw, "MULTISHELL_TOPUP*"))
    folderB0map =  glob(join(raw, "B0map_v4*"))
  
    
    cmd = f"dcm2niix -z y -f sub-{subjects[l]}_ses-{ses}_acq-3D_T1w -w 1 -o {anat} {folderT1[0]}/"
    utils.executeCommand(cmd)
    
    
    cmd = f"dcm2niix -z y -f sub-{subjects[l]}_ses-{ses}_dwi -w 1 -o {dwi} {folderdwi[0]}/"
    utils.executeCommand(cmd)
    
    
    cmd = f"dcm2niix -z y -f sub-{subjects[l]}_ses-{ses}_dir-AP_epi -w 1 -o {fmap} {folderTOP[0]}/"
    utils.executeCommand(cmd)
    utils.executeCommand(f"rm {fmap}/*epi.bval")
    utils.executeCommand(f"rm {fmap}/*epi.bvec")
    
    rawMag = sorted(folderB0map)[0]
    rawPhasediff = sorted(folderB0map)[1]
    
    cmd = f"dcm2niix -z y -f sub-{subjects[l]}_ses-{ses}_magnitude -w 1 -o {fmap} {rawMag}/"
    utils.executeCommand(cmd)
    
    utils.executeCommand(f"mv {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude_e1.nii.gz  {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude1.nii.gz")
    utils.executeCommand(f"mv {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude_e2.nii.gz  {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude2.nii.gz")
    
    utils.executeCommand(f"mv {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude_e1.json  {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude1.json")
    utils.executeCommand(f"mv {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude_e2.json  {fmap}/sub-{subjects[l]}_ses-{ses}_magnitude2.json")
    
    
    cmd = f"dcm2niix -z y -f sub-{subjects[l]}_ses-{ses}_phasediff -w 1 -o {fmap} {rawPhasediff}/"
    utils.executeCommand(cmd)
    
    utils.executeCommand(f"mv {fmap}/sub-{subjects[l]}_ses-{ses}_phasediff_*.nii.gz  {fmap}/sub-{subjects[l]}_ses-{ses}_phasediff.nii.gz")
    utils.executeCommand(f"mv {fmap}/sub-{subjects[l]}_ses-{ses}_phasediff_*.json  {fmap}/sub-{subjects[l]}_ses-{ses}_phasediff.json")
    
    with open( f"{fmap}/sub-{subjects[l]}_ses-{ses}_phasediff.json") as fas: jsonfile = json.load(fas)
    if utils.checkIfFileExists(join(f"{dwi}", f"sub-{subjects[l]}_ses-{ses}_dwi.nii.gz") ):
        jsonfile['IntendedFor'] = f"ses-{ses}/dwi/sub-{subjects[l]}_ses-{ses}_dwi.nii.gz"
        with open(f"{fmap}/sub-{subjects[l]}_ses-{ses}_phasediff.json", 'w', encoding='utf-8') as fas: json.dump(jsonfile, fas, ensure_ascii=False, indent=4)
 
    with open( f"{fmap}/sub-{subjects[l]}_ses-{ses}_dir-AP_epi.json") as fas: jsonfile2 = json.load(fas)
    if utils.checkIfFileExists(join(f"{dwi}", f"sub-{subjects[l]}_ses-{ses}_dwi.nii.gz") ):
        jsonfile2['IntendedFor'] = f"ses-{ses}/dwi/sub-{subjects[l]}_ses-{ses}_dwi.nii.gz"
        with open(f"{fmap}/sub-{subjects[l]}_ses-{ses}_dir-AP_epi.json", 'w', encoding='utf-8') as fas: json.dump(jsonfile2, fas, ensure_ascii=False, indent=4)
 



#%%




                       
#%%


# Download data programmatically from box.com (Python >= 3.5)

# Note: You will have to create a box App first, using the following steps:
# Go to developer.box.com, log into your box account (or create one) and create a new App
# Find the Client ID, Client Secret, and access token (or developer token if you keep your app in developer mode) under your app's "Development" tab
# In your online box.com account, navigate to the folder to download. The folder ID can be found at the end of the URL (the numbers after the last slash).


from boxsdk import Client, OAuth2, DevelopmentClient
import os
from os.path import join

from fuzzywuzzy import fuzz

# Define client ID, client secret, and developer token.
# Note, Access token  (aka developer token) expires after 1 hour, go to developer.box.com and get new key
ACCESS_TOKEN = 'LlutB466K4MJcGj2r1JkoLZLeqR40DqL'       # box app access token (or developer token, in configuration tab)
CLIENT_ID = '66zcvd8k07vlxvuuokqwq0h8rihgksuk'          # OAuth2.0 client ID for box app (in configuration tab)
CLIENT_SECRET = '8eMkGQBVhuR0F0l5zSWNewYcwGXqcPKE'      # OAuth2.0 client secret (in configuration tab)

# Define Box folder ID (can find from URL), and path to deposit folder
folderID = '104816745525' # ID of box folder to download on box
path = '/media/arevell/sharedSSD/linux/data/BIDS/PIER/'


auth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    access_token=ACCESS_TOKEN,
)

client = Client(auth)


folderID = "102659909076" #CNT reconstruction folder name
folder = client.folder(folder_id=folderID).get()
subfolders = client.folder(folder_id=folderID).get_items() #get all patient folder IDS
for subfolder in subfolders:
    
    if not "OLD" in subfolder.name:
        if not "RNS" in subfolder.name :
            #print(f"{subfolder.name}         {subfolder.id}")
            subBOX = subfolder.name[0:6] #get RID
            subfolderID = subfolder.id
            if subBOX[0:3] == "RID": #If folder name actually begins with RID
                #get actaul RID number with 4 digits
                sub = subBOX[0:3] + "0" + subBOX[3:]
                #print(sub)
                savepath = join(path , f"sub-{sub}")
                utils.checkPathAndMake(path, join(savepath, "ses-implant01", "anat"))
                utils.checkPathAndMake(path, join(savepath, "ses-implant01", "ct"))
                utils.checkPathAndMake(path, join(savepath, "ses-implant01", "ieeg"))
                
                items = client.folder(folder_id=subfolderID).get_items() 
                for item in items:
                    
                    if item.name == "electrodenames_coordinates_native_and_T1.csv":
                        newName = f"sub-{sub}_ses-implant01_electrodes.csv"
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "ieeg", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                        coordinates = pd.read_csv(join(savepath,"ses-implant01",  "ieeg", newName), sep = ",", header=None)
    
        
                        coordinatesT1 = coordinates.iloc[:,[0, 10, 11, 12]]
                        size = np.zeros(( len(coordinates)))
                        size[:] = 1
                        coordinatesT1 = pd.concat(    [coordinatesT1,  pd.DataFrame(size  )  ] , axis= 1 )
                        coordinatesT1.columns = ["name", "x", "y", "z", "size"]
                        outnameCoordinates = join(savepath, "ses-implant01", "ieeg", f"sub-{sub}_ses-implant01_space-T00_electrodes.tsv" )
                        coordinatesT1.to_csv(  outnameCoordinates,  sep="\t", index=False, header=True)
                    
                        coordinatesCT = coordinates.iloc[:,[0, 2, 3, 4]]
                        size = np.zeros(( len(coordinates)))
                        size[:] = 1
                        coordinatesCT = pd.concat(    [coordinatesCT,  pd.DataFrame(size  )  ] , axis= 1 )
                        coordinatesCT.columns = ["name", "x", "y", "z", "size"]
                        outnameCoordinates = join(savepath, "ses-implant01", "ieeg", f"sub-{sub}_ses-implant01_space-CT_electrodes.tsv" )
                        coordinatesCT.to_csv(  outnameCoordinates,  sep="\t", index=False, header=True)
                    
                        utils.executeCommand(f"rm {join(savepath,'ses-implant01',  'ieeg', newName)}", printBOOL=False)
                                    
                    if item.name == f"T00_{subBOX}_mprage.nii.gz" or item.name == f"T00_{subBOX}rev_mprage.nii.gz" or item.name == f"T00_{subBOX}new_mprage.nii.gz" or item.name == f"T00_{subBOX}revision_mprage.nii.gz":                 
                        newName = f"sub-{sub}_ses-implant01_acq-T00_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                    if item.name == f"T00_{subBOX}_mprage_brainBrainExtractionBrain.nii.gz" or item.name == f"T00_{subBOX}rev_mprage_brainBrainExtractionBrain.nii.gz" or item.name == f"T00_{subBOX}new_mprage_brainBrainExtractionBrain.nii.gz" or item.name == f"T00_{subBOX}revision_mprage_brainBrainExtractionBrain.nii.gz":                 
                        newName = f"sub-{sub}_ses-implant01_acq-T00brain_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()

                    if item.name == f"T01_{subBOX}_mprage.nii.gz" or item.name == f"T01_{subBOX}rev_mprage.nii.gz" or item.name == f"T01_{subBOX}new_mprage.nii.gz" or item.name == f"T01_{subBOX}revision_mprage.nii.gz":                 
                        newName = f"sub-{sub}_ses-implant01_acq-T01_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == "T01_mprage_to_T00_mprageANTs.nii.gz":
                        newName = f"sub-{sub}_ses-implant01_acq-T01toT00_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == f"T01_{subBOX}_CT.nii.gz":
                        newName = f"sub-{sub}_ses-implant01_acq-CTnative_ct.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "ct", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == "T01_CT_to_T00_mprageANTs.nii.gz":
                        newName = f"sub-{sub}_ses-implant01_acq-CTtoT00_ct.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "ct", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == "T01_CT_to_T01_mprageANTs.nii.gz":
                        newName = f"sub-{sub}_ses-implant01_acq-CTtoT01_ct.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-implant01",  "ct", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
#%% RNS                            

folderID = "102659909076" #CNT reconstruction folder name
folder = client.folder(folder_id=folderID).get()
subfolders = client.folder(folder_id=folderID).get_items() #get all patient folder IDS
for subfolder in subfolders:
    
    if not "OLD" in subfolder.name:
        if "RNS" in subfolder.name :
            print(f"{subfolder.name}         {subfolder.id}")
            subBOX = subfolder.name[0:6] #get RID
            subfolderID = subfolder.id
            if subBOX[0:3] == "RID": #If folder name actually begins with RID
                #get actaul RID number with 4 digits
                sub = subBOX[0:3] + "0" + subBOX[3:]
                #print(sub)
                savepath = join(path , f"sub-{sub}")
                utils.checkPathAndMake(path, join(savepath, "ses-RNS01", "anat"))
                utils.checkPathAndMake(path, join(savepath, "ses-RNS01", "ct"))
                utils.checkPathAndMake(path, join(savepath, "ses-RNS01", "ieeg"))
                

                items = client.folder(folder_id=subfolderID).get_items() 
                for item in items:
                    
                    if item.name == "electrodenames_coordinates_native_and_T1.csv":
                        newName = f"sub-{sub}_ses-RNS01_electrodes.csv"
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "ieeg", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                        coordinates = pd.read_csv(join(savepath,"ses-RNS01",  "ieeg", newName), sep = ",", header=None)
    
        
                        coordinatesT1 = coordinates.iloc[:,[0, 10, 11, 12]]
                        size = np.zeros(( len(coordinates)))
                        size[:] = 1
                        coordinatesT1 = pd.concat(    [coordinatesT1,  pd.DataFrame(size  )  ] , axis= 1 )
                        coordinatesT1.columns = ["name", "x", "y", "z", "size"]
                        outnameCoordinates = join(savepath, "ses-RNS01", "ieeg", f"sub-{sub}_ses-RNS01_space-T00_electrodes.tsv" )
                        coordinatesT1.to_csv(  outnameCoordinates,  sep="\t", index=False, header=True)
                    
                        coordinatesCT = coordinates.iloc[:,[0, 2, 3, 4]]
                        size = np.zeros(( len(coordinates)))
                        size[:] = 1
                        coordinatesCT = pd.concat(    [coordinatesCT,  pd.DataFrame(size  )  ] , axis= 1 )
                        coordinatesCT.columns = ["name", "x", "y", "z", "size"]
                        outnameCoordinates = join(savepath, "ses-RNS01", "ieeg", f"sub-{sub}_ses-RNS01_space-CT_electrodes.tsv" )
                        coordinatesCT.to_csv(  outnameCoordinates,  sep="\t", index=False, header=True)
                    
                        utils.executeCommand(f"rm {join(savepath,'ses-RNS01',  'ieeg', newName)}", printBOOL=False)
                                    
                    if item.name == f"T00_{subBOX}_mprage.nii.gz" or item.name == f"T00_{subBOX}rev_mprage.nii.gz" or item.name == f"T00_{subBOX}new_mprage.nii.gz" or item.name == f"T00_{subBOX}rns_mprage.nii.gz":                 
                        newName = f"sub-{sub}_ses-RNS01_acq-T00_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                    if item.name == f"T00_{subBOX}_mprage_brainBrainExtractionBrain.nii.gz" or item.name == f"T00_{subBOX}rev_mprage_brainBrainExtractionBrain.nii.gz" or item.name == f"T00_{subBOX}new_mprage_brainBrainExtractionBrain.nii.gz" or item.name == f"T00_{subBOX}rns_mprage_brainBrainExtractionBrain.nii.gz":                 
                        newName = f"sub-{sub}_ses-RNS01_acq-T00brain_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()

                    if item.name == f"T01_{subBOX}_mprage.nii.gz" or item.name == f"T01_{subBOX}rev_mprage.nii.gz" or item.name == f"T01_{subBOX}new_mprage.nii.gz" or item.name == f"T01_{subBOX}rns_mprage.nii.gz":                 
                        newName = f"sub-{sub}_ses-RNS01_acq-T01_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == "T01_mprage_to_T00_mprageANTs.nii.gz":
                        newName = f"sub-{sub}_ses-RNS01_acq-T01toT00_T1w.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "anat", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == f"T01_{subBOX}_CT.nii.gz" or item.name == f"T01_{subBOX}rns_CT.nii.gz":
                        newName = f"sub-{sub}_ses-RNS01_acq-CTnative_ct.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "ct", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == "T01_CT_to_T00_mprageANTs.nii.gz":
                        newName = f"sub-{sub}_ses-RNS01_acq-CTtoT00_ct.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "ct", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            
                    if item.name == "T01_CT_to_T01_mprageANTs.nii.gz":
                        newName = f"sub-{sub}_ses-RNS01_acq-CTtoT01_ct.nii.gz"  
                        print(join(item.name))
                        with open(join(savepath,"ses-RNS01",  "ct", newName), 'wb') as open_file:
                            client.file(file_id=item.id).download_to(open_file); open_file.close()
                            
                            

#%%










subBOX = "RID032"
BOX_coordinates = "electrodenames_coordinates_native_and_T1.csv"
BOX_T00 = f"T00_{subBOX}_mprage.nii.gz"
BOX_bet = f"T00_{subBOX}_mprage_brainBrainExtractionBrain.nii.gz"
BOX_T01 = f"T01_{subBOX}_mprage.nii.gz"
BOX_T01_to_T00 = "T01_mprage_to_T00_mprageANTs.nii.gz"
BOX_CT = f"T01_{subBOX}_CT.nii.gz"
BOX_CT_to_T00 = "T01_CT_to_T00_mprageANTs.nii.gz"
BOX_CT_to_T01 = "T01_CT_to_T01_mprageANTs.nii.gz"


BOX_files = [BOX_coordinates, BOX_T00, BOX_bet, BOX_T01, BOX_T01_to_T00, BOX_CT, BOX_CT_to_T00, BOX_CT_to_T01]


folder = client.folder(folder_id=folderID).get()
items = client.folder(folder_id=folderID).get_items()
for item in items:
    #print(item.name)
    if item.name in BOX_files:
        print('Downloading' + join(path, item.name))
        
        with open(os.path.join(path, item.name), 'wb') as open_file:
            client.file(file_id=item.id).download_to(open_file)
            open_file.close()
    else:
        print(f"DOES NOT EXIST\n{item.name}")











# recursively download all files in folder
def download(folderID, path):
    folder = client.folder(folder_id=folderID).get()
    items = client.folder(folder_id=folderID).get_items()

    if ~os.path.exists(os.path.join(path, folder.name)):
        os.makedirs(os.path.join(path, folder.name))

    # mkdir folder name
    for item in items:
        # If item is a folder
        if item.type == 'folder':
            print(item.name)
            download(item.id, os.path.join(path, folder.name))
        # output_file = open('file.pdf', 'wb')
        elif item.type == 'file':
            if item.name[0] == '.':
                continue
            print('Downloading' + os.path.join(path, folder.name, item.name))
            with open(os.path.join(path, folder.name, item.name), 'wb') as open_file:
                client.file(file_id=item.id).download_to(open_file)
                open_file.close()

download(folderID, path)




#%%



#organization of BIDS directory
"""

BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset= "PIER"



atlasLocalizationDirivatives = join(BIDS, "derivatives", "atlasLocalization")






subDir = [os.path.join(atlasLocalizationDirivatives, o) for o in os.listdir(atlasLocalizationDirivatives)   if os.path.isdir(os.path.join(atlasLocalizationDirivatives,o))]


subjects = [basename(item) for item in subDir ]


for i in range(len(subjects)):
    
    sub = subjects[i]
    subRID = sub[4:]
    

    
    if utils.checkIfFileDoesNotExist( join(BIDS, dataset, sub, "ses-implant01" )  ):
        pathtomake = join(BIDS, dataset, "sub-RIDXXXX", "ses-implant01")
        pathtomake2 = join(BIDS, dataset, f"{sub}")
        
        utils.executeCommand(  f"cp -r {pathtomake} {pathtomake2} "    )
    
    
    T00 = join(atlasLocalizationDirivatives, f"{sub}", f"T00_{subRID}_mprage.nii.gz" )
    if utils.checkIfFileExists(T00):
        
        outname = join(BIDS, dataset, f"{sub}", "ses-implant01", "anat", f"{sub}_ses-implant01_acq-T00_T1w.nii.gz" )
        utils.executeCommand(  f"cp {T00} {outname}"    )
    
    coordinatesPath = join(atlasLocalizationDirivatives, f"{sub}", "electrodenames_coordinates_native_and_T1.csv" )
    
    if utils.checkIfFileExists(coordinatesPath):
        coordinates = pd.read_csv(coordinatesPath, sep = ",", header=None)
    
        
        coordinatesEdit = coordinates.iloc[:,[0, 10, 11, 12]]
        
        size = np.zeros(( len(coordinates)))
        size[:] = 1
        coordinatesEdit = pd.concat(    [coordinatesEdit,  pd.DataFrame(size  )  ] , axis= 1 )
        coordinatesEdit.columns = ["name", "x", "y", "z", "size"]
        outnameCoordinates = join(BIDS, dataset, f"{sub}", "ses-implant01", "ieeg", f"{sub}_ses-implant01_space-T00_electrodes.tsv" )
        coordinatesEdit.to_csv(  outnameCoordinates,  sep="\t", index=False, header=True)
    
    
    
    #deletepath = join(BIDS, dataset, f"{sub}", "ses-implant01Reconstruction")
    #if utils.checkIfFileExists(deletepath):
    #    utils.executeCommand(  f"rm -r {deletepath}"    )
    
    
    
    






"""




