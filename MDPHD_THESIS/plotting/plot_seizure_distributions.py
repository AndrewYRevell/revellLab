#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:02:58 2021

@author: arevell
"""


import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy.integrate import simps
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


#%%
# how many seizures per patient

def plot_distribution_seizures_per_patient(patientsWithseizures):
    seizureCounts = patientsWithseizures.groupby(['subject']).count()
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    sns.histplot(data=seizureCounts, x="idKey", binwidth=1, kde=True, ax=axes)
    axes.set_xlim(1, None)
    axes.set(xlabel='Number of Seizures', ylabel='Number of Patients',
             title="Distribution of Seizure Occurrences")
#utils.savefig(f"{paths['figures']}/seizureSummaryStats/seizureCounts.pdf", saveFigures=saveFigures)

def plot_distribution_seizure_length(patientsWithseizures):
    # get time distributions
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    sns.histplot(data=patientsWithseizures, x="length",
                 bins=range(0, 1080, 60), kde=True, ax=axes[0])
    sns.histplot(data=patientsWithseizures, x="length",
                 bins=range(0, 1080, 10), kde=True, ax=axes[1])
    for i in range(2):
        axes[i].set_xlim(0, None)
        axes[i].set(ylabel='Number of Seizures', xlabel='Length (s)',
                    title="Distribution of Seizure Lengths")
        #axes[i].set_xticklabels(axes[i].get_xticks(), rotation = 45)














