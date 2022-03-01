#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:11:58 2022

@author: arevell
"""
#%% 1/4 Imports
import sys
import os
import json
import copy
import time
import bct
import glob
import math
import random
import pickle
import pingouin
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
import statsmodels.api as sm
from scipy import signal, stats
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename

#revellLab
#utilities, constants/parameters, and thesis helper functions
from revellLab.packages.utilities import utils
from revellLab.paths import constants_paths as paths

#package functions
from revellLab.packages.dataclass import dataclass_atlases, dataclass_iEEG_metadata
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.imaging.tractography import tractography


#%%

#get patients in patientsWithseizures that have dti
sfc_patient_list = tractography.get_patients_with_dwi(np.unique(patientsWithseizures["subject"]), paths, dataset = "PIER", SESSION_RESEARCH3T = SESSION_RESEARCH3T)
cmd = tractography.print_dwi_image_correction_QSIprep(sfc_patient_list, paths, dataset = "PIER")
tractography.get_tracts_loop_through_patient_list(sfc_patient_list, paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)

