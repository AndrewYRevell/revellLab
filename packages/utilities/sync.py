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



BIDSserver = "/Users/andyrevell/mount/DATA/Human_Data/BIDS/PIER"
BIDSlocal = "/Users/andyrevell/research/data/BIDS/PIER"



#%%


dirsync.sync(BIDSserver, BIDSlocal, 'sync', verbose=True)




