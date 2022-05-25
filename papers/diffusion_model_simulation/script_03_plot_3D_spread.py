#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:35:18 2022

@author: arevell
"""


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
import scipy
import re
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from os.path import join, splitext, basename

from packages.utilities import utils
from packages.diffusionModels import functions as dmf 
from paths import constants_paths as paths

sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False, 'figure.dpi': 300, "legend.frameon": False, "savefig.transparent": True},  palette="pastel")
sns.set_context("talk")


#%%
#Get data

centroids = pd.read_csv(paths.AAL2_CENTROIDS )
xyz = np.array(centroids[["x","y","z"]])


adj = utils.read_DSI_studio_Txt_files_SC(paths.SUB01_AAL2)
region_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(paths.SUB01_AAL2)
adj_hat = adj/adj.max()

#%
N= len(adj_hat)
adj_hat_select = 0.1 < adj_hat
adj_hat[~adj_hat_select]=0


#%%
nx.from_numpy_array(A[, parallel_edges, ...])
G = nx.newman_watts_strogatz_graph(N, 2, p=0.5)













































