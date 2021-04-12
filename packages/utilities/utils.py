#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import sys, os, time, copy, glob, smtplib, ssl
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile, splitext, basename
from email.header import Header
from email.mime.text import MIMEText
#import seaborn as sns

#%% Input


#%% functions
def checkPathError(path):
    """
    Check if path exists
    """
    if not os.path.exists(path):
        raise IOError(f"\n\n\n\nPath or file does not exist:\n{path}\n\n\n\n" )
        

def checkPathErrorGlob(path):
    """
    Check if path exists
    """
    files = glob.glob(path)
    files = glob.glob(path)
    if len(files) > 1:
        IOError(f"More than 1 file exists: \n{files}")
    if len(files) == 0:
        raise IOError(f"\n\n\n\nPath or file does not exist:\n{path}\n\n\n\n" )        
        
        
def checkPathAndMake(pathToCheck, pathToMake, make = True, printBOOL = True):
    """
    Check if pathToCheck exists. If so, then option to make a second directory pathToMake (may be same as pathToCheck)
    """
    if not os.path.exists(pathToCheck):
        if printBOOL: print(f"\nFile or Path does not exist:\n{pathToCheck}" )
    if make:
        if os.path.exists(pathToMake):
            if printBOOL: print(f"Path already exists\n{pathToMake}")
        else: 
            os.makedirs(pathToMake)
            if printBOOL: print("Making Path")

def checkIfFileExists(path, returnOpposite = False, printBOOL = True):
    if (os.path.exists(path)): 
        if printBOOL: print(f"\nFile exists:\n    {path}\n\n")
        if returnOpposite: 
            return False
            if printBOOL: print("\nHowever, re-writing over file\n\n")
        else: 
            return True
    else: 
        if printBOOL: print(f"\nFile does not exists:\n    {path}\n\n")
        if returnOpposite: 
            return True
        else: 
            return False




def checkIfFileDoesNotExist(path, returnOpposite = False, printBOOL = True):
    if not (os.path.exists(path)): 
        if printBOOL: print(f"\nFile does not exist:\n    {path}\n\n")
        if returnOpposite: 
            return False
        else:
            return True
    else: 
        if printBOOL: print(f"\nFile exists:\n    {path}\n\n")
        if returnOpposite: 
            return True
        else: 
            return False
        
def checkIfFileExistsGlob(path, returnOpposite = False, printBOOL = True):
    files = glob.glob(path)
    if len(files) > 1:
        print(f"More than 1 file exists: \n{files}")
    else:
        if len(files) == 0: file = ""
        else:
            file = files[0]
        
        if (os.path.exists(file)): 
            if printBOOL: print(f"\nFile exists:\n    {path}\n\n")
            if returnOpposite: 
                return False
                if printBOOL: print("\nHowever, re-writing over file\n\n")
            else: 
                return True
        else: 
            if printBOOL: print(f"\nFile does not exists:\n    {path}\n\n")
            if returnOpposite: 
                return True
            else: 
                return False

def checkIfFileDoesNotExistGlob(path, returnOpposite = False, printBOOL = True):
    files = glob.glob(path)
    if len(files) > 1:
        print(f"More than 1 file exists: \n{files}")
    else:
        if len(files) == 0: file = ""
        else:
            file = files[0]
        if not (os.path.exists(file)): 
            if printBOOL: print(f"\nFile does not exist:\n    {path}\n\n")
            if returnOpposite: 
                return False
            else:
                return True
        else: 
            if printBOOL: print(f"\nFile exists:\n    {file}\n\n")
            if returnOpposite: 
                return True
            else: 
                return False

def executeCommand(cmd, printBOOL = True):
    if printBOOL: print(f"\n\nExecuting Command Line: \n{cmd}\n\n") 
    os.system(cmd)

def channel2stdCSV(outputTissueCoordinates):
    df = pd.read_csv(outputTissueCoordinates, sep=",", header=0)
    for e in range(len( df  )):
        electrode_name = df.iloc[e]["electrode_name"]
        if (len(electrode_name) == 3): electrode_name = f"{electrode_name[0:2]}0{electrode_name[2]}"
        df.at[e, "electrode_name" ] = electrode_name
    pd.DataFrame.to_csv(df, outputTissueCoordinates, header=True, index=False)

def baseSplitextNiiGz(path):
    base = basename(path)
    split = splitext(splitext(path)[0])[0]
    basesplit = basename(split)
    return base, split, basesplit

def calculateTimeToComplete(t0, t1, total, completed):
    td = np.round(t1-t0,2)
    tr = np.round((total - completed- 1) * td,2)
    print(f"Took {td} seconds. Estimated time remaining: {tr} seconds")


def getSubType(name):
    if "C" in name:
        return "control"
    if "RID" in name:
        return "subjects"

    

def sendEmail(receiver_email = "andyrevell.python@gmail.com", subject ="Process is done", text = "done", port = 465, smtp_server = "smtp.gmail.com" , sender_email = "andyrevell.python@gmail.com"):
    
    AppPassword_twoAuthentication = "vgkpolyhiwzzifcc"
    email_message = MIMEText(text, 'plain', 'utf-8')
    email_message['From'] = sender_email
    email_message['To'] = receiver_email
    email_message['Subject'] = Header(subject, 'utf-8')
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, AppPassword_twoAuthentication)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
    print(f"Email sent to {receiver_email}")
    
    
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = "X", printEnd = "\r"):
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

def show_slices(fname, low = 0.33, middle = 0.5, high = 0.66, save = False, saveFilename = None, isPath = True, cmap="gray"):
    
    if isPath:
        if checkIfFileExistsGlob(fname):
            fname = glob.glob(fname)[0]
            img = nib.load(fname)
            imgdata = img.get_fdata()  
        else:
            return print(f"File does not exist: \n{fname}")
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
        axes[i].imshow(slices[r][c].T, cmap=cmap, origin="lower")
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





#%%
"""
# Download data programmatically from box.com (Python >= 3.5)

# Note: You will have to create a box App first, using the following steps:
# Go to developer.box.com, log into your box account (or create one) and create a new App
# Find the Client ID, Client Secret, and access token (or developer token if you keep your app in developer mode) under your app's "Development" tab
# In your online box.com account, navigate to the folder to download. The folder ID can be found at the end of the URL (the numbers after the last slash).

# Last Updated: 3/19/2021
# bscheid@seas.upenn.edu

from boxsdk import Client, OAuth2
import os

# Define client ID, client secret, and developer token.
# Note, Access token  (aka developer token) expires after 1 hour, go to developer.box.com and get new key
CLIENT_ID = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'          # OAuth2.0 client ID for box app (in configuration tab)
CLIENT_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'      # OAuth2.0 client secret (in configuration tab)
ACCESS_TOKEN = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'       # box app access token (or developer token, in configuration tab)

# Define Box folder ID (can find from URL), and path to deposit folder
folderID = 'xxxxxxxx' # ID of box folder to download on box
path = 'local/path/to/download/folder'

auth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    access_token=ACCESS_TOKEN,
)

client = Client(auth)

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
            output_file = open(os.path.join(path, folder.name, item.name), 'wb')
            client.file(file_id=item.id).download_to(output_file)


download(folderID, path)

"""
#%% structural connectivity


def readDSIstudioTxtSC(path):
    C = pd.read_table(path, header=None, dtype=object)
    #cleaning up structural data 
    C = C.drop([0,1], axis=1)
    C = C.drop([0], axis=0)
    C = C.iloc[:, :-1]
    C = np.array(C.iloc[1:, :]).astype('float64')  
    return C


def getUpperTri(C, k = 1):
    return C[np.triu_indices( len(C), k=k)]








