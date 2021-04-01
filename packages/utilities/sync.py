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
import glob
import dirsync

#%% Input

syncFolder = "PIER"

MAC_BIDSserver = join("/Users/andyrevell/mount/DATA/Human_Data/BIDS", syncFolder)
MAC_BIDSlocal = join("/Users/andyrevell/research/data/BIDS", syncFolder)
LINUX_BIDSserver =  join("/home/arevell/borel/DATA/Human_Data/BIDS", syncFolder)
LINUX_BIDSlocal =  join("/media/arevell/sharedSSD/linux/data/BIDS", syncFolder)


BIDSserver = MAC_BIDSserver
BIDSlocal = MAC_BIDSlocal
#%%


dirsync.sync(BIDSserver, BIDSlocal, 'sync', verbose=True)




