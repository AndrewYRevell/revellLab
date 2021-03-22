#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:01:56 2021

@author: arevell
"""

import sys
import os
import time
import pandas as pd
import copy
from os import listdir
from  os.path import join, isfile
from os.path import splitext, basename
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
#import seaborn as sns


#%% Input


#%% functions
def checkPathError(path):
    """
    Check if path exists
    """
    if not os.path.exists(path):
        raise IOError(f"\n\n\n\nPath or file does not exist:\n{path}\n\n\n\n" )
def checkPathAndMake(pathToCheck, pathToMake, make = True):
    """
    Check if pathToCheck exists. If so, then option to make a second directory pathToMake (may be same as pathToCheck)
    """
    if not os.path.exists(pathToCheck):
        print(f"\nFile or Path does not exist:\n{pathToCheck}" )
    if make:
        if os.path.exists(pathToMake):
            print(f"Path already exists\n{pathToMake}")
        else: 
            os.makedirs(pathToMake)
            print("Making Path")

def checkIfFileExists(path, returnOpposite = False):
    if (os.path.exists(path)): 
        print(f"\nFile exists:\n    {path}\n\n")
        if returnOpposite: 
            return False
            print("\nHowever, re-writing over file\n\n")
        else: 
            return True
    else: 
        print(f"\nFile does not exists:\n    {path}\n\n")
        if returnOpposite: 
            return True
        else: 
            return False
    
def checkIfFileDoesNotExist(path, returnOpposite = False):
    if not (os.path.exists(path)): 
        print(f"\nFile does not exist:\n    {path}\n\n")
        if returnOpposite: 
            return False
        else:
            return True
    else: 
        print(f"\nFile exists:\n    {path}\n\n")
        if returnOpposite: 
            return True
        else: 
            return False

def executeCommand(cmd):
    print(f"\n\nExecuting Command Line: \n{cmd}\n\n"); os.system(cmd)

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








