#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:23:28 2020

@author: arevell
"""

#combine subcortical segmentation (FIST) and regular segmentation (FAST)

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns


def show_slices(img_data, low = 0.33, middle = 0.5, high = 0.66):
    """ Function to display row of image slices """
    slices1 = [   img_data[:, :, int((img_data.shape[2]*low)) ] , img_data[:, :, int(img_data.shape[2]*middle)] , img_data[:, :, int(img_data.shape[2]*high)]   ]
    slices2 = [   img_data[:, int((img_data.shape[1]*low)), : ] , img_data[:, int(img_data.shape[1]*middle), :] , img_data[:, int(img_data.shape[1]*high), :]   ]
    slices3 = [   img_data[int((img_data.shape[0]*low)), :, : ] , img_data[int(img_data.shape[0]*middle), :, :] , img_data[int(img_data.shape[0]*high), :, :]   ]
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


def show_eeg_bysec(data, fs, channel = 0, start_sec = 0, stop_sec = 2):
    data_ch = data[:,channel]

    fig,axes = plt.subplots(1,1,figsize=(4,4), dpi = 300)
    sns.lineplot(x =  np.array(range(fs*start_sec,fs*stop_sec))/1e6*fs, y = data_ch[range(fs*start_sec,fs*stop_sec)], ax = axes , linewidth=0.5 )
    plt.show()


def show_eeg(data, fs, channel = 0):
    data_ch = data[:,channel]

    fig,axes = plt.subplots(1,1,figsize=(4,4), dpi = 300)
    sns.lineplot(x =  np.array(range(len(data_ch)))/1e6*fs, y = data_ch, ax = axes , linewidth=0.5 )
    plt.show()


def plot_adj(adj, vmin = -1, vmax = 1, cbar = True ):
    fig,axes = plt.subplots(1,1,figsize=(4,4), dpi = 300)
    sns.heatmap(adj, square=True, ax = axes, vmin = vmin, vmax = vmax, cbar = cbar)

def plot_adj_allbands(adj_list, cbar=True, vmin = -1, vmax = 1, titles = ["Broadband", "Delta", "Theta", "Alpha", "Beta", "Gamma - Low", "Gamma - Mid", "Gamma - High"] ):
    fig,axes = plt.subplots(2,4,figsize=(14,7), dpi = 80)
    count = 0
    for x in range(2):
        for y in range(4):
            sns.heatmap(adj_list[count], square=True, ax = axes[x][y], vmin = vmin, vmax = vmax, cbar=cbar)
            axes[x][y].set_title(titles[count], size=10)
            count = count+1



# Progress bar function
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 20, fill = "X", printEnd = "\r"):
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


