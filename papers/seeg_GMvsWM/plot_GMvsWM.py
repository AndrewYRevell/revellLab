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
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap, LinearSegmentedColormap




def plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll):
    powerGM = [powerGMmean[:,0:200], powerGMmean[:,200:400], powerGMmean[:,400:600], powerGMmean[:,600:800]]
    powerWM = [powerWMmean[:,0:200], powerWMmean[:,200:400], powerWMmean[:,400:600], powerWMmean[:,600:800]]
    

    #general plotting parameters
    plt.rcParams['figure.figsize'] = (20.0, 15.0)
    plt.rcParams['font.family'] = "serif"
    sns.set_context("talk")
    fontsize = 30
    vminSpec = -1; vmaxSpec = 2.5
    
    #making plot layout
    fig1 = plt.figure(constrained_layout=False)
    left = 0.07; right = 0.8
    gs1 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.72, top = 0.88 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs2 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.52, top = 0.69 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs3 = fig1.add_gridspec(nrows=1, ncols=4, left=left, right=right, bottom = 0.25, top = 0.45 ,wspace=0.1, width_ratios=[1, 1, 1, 1])
    gs4 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.062, top = 0.18 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    
    gs = [gs1, gs2, gs3, gs4]
    axes = [None] * len(gs)
    for a in range(len(gs)):
        axes[a] = [None] * gs[a]._ncols
    
    for a in range(len(axes)):
        for b in range(len(axes[a])):
            axes[a][b] = fig1.add_subplot(gs[a][:, b])
    
    #Spectrogram
    cbar_ax1 = fig1.add_axes([.85, .72, .02, .16])
    cbar_ax2 = fig1.add_axes([.85, .52, .02, .16])
    ylim = 100
    for i in range(2): #i= 0 GM, i = 1 WM
        k = 0
        for j in range(5): #interictal space preictal ictal postictal
            if i == 0:
                plot = np.log10(powerGM[k])
            if i == 1:
                plot = np.log10(powerWM[k])
            #plot = np.delete(plot, range(ylim, np.shape(plot)[0]), axis=0)
            #plot = np.delete(plot, range(0, 1), axis=0)
            if j == 1:#create a blank space between interictal and preictal
                axes[i][j].remove()
            if j != 1:
                if i == 0:
                    sns.heatmap(plot[1:ylim,:], cmap=sns.color_palette("coolwarm", n_colors=40, desat=0.8), vmin=vminSpec,vmax=vmaxSpec, ax = axes[i][j], yticklabels=10, center = 0.5,
                                cbar=k <1, cbar_ax=None if k else cbar_ax1, cbar_kws={'label': '$\mathregular{log_{10}}({V^2}/{Hz})$'})
                    print(k<1)
                if i == 1:
                    sns.heatmap(plot[1:ylim,:], cmap=sns.color_palette("coolwarm", n_colors=40, desat=0.8), vmin=vminSpec, vmax=vmaxSpec, ax = axes[i][j], yticklabels=10, center = 0.5,
                                cbar=k <1 , cbar_ax=None if k else cbar_ax2, cbar_kws={'label': '$\mathregular{log_{10}}({V^2}/{Hz})$'})
                    print(k < 1)
                old_ticks = axes[i][j].get_xticks()
                old_ticks_y = axes[i][j].get_yticks()
                new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), 5)
                new_ticks = new_ticks[range(1,4)]
                axes[i][j].set_xticks(new_ticks)
                axes[i][j].set_xticklabels(["25%", "50%", "75%"], rotation=0)
                if j == 0:
                    axes[i][j].set_xticks([new_ticks[1]])
                    axes[i][j].set_xticklabels(["~6hrs before"], rotation=0)
                    #axes[i][j].set_yticklabels([0, 10, 20], rotation=0)
                if j > 1:
                    axes[i][j].set_yticks([])
                    if j <4:
                        axes[i][j].axvline(x=np.shape(powerGMmean)[1], c = "black", linewidth=4, linestyle='--')
                k = k + 1
            axes[i][j].invert_yaxis()
            cbar_ax1.yaxis.set_ticks_position('left')
            cbar_ax1.yaxis.set_label_position('left')
            cbar_ax2.yaxis.set_ticks_position('left')
            cbar_ax2.yaxis.set_label_position('left')
    
    fig1.text(0.02, 0.7, 'Frequency (Hz)', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.02, 0.35, 'Frequency (Hz)', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.02, 0.121, 'SNR', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.48, 'Time (% seizure length)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.21, 'Distance from GM (mm)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.02, 'Time (% seizure length)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    
    fig1.text(0.145, 0.9, 'Interictal', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.31, 0.9, 'Preictal', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.515, 0.9, 'Ictal', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.72, 0.9, 'Postictal', ha='center', fontdict = {'fontsize': fontsize*0.7})
    subplot_x=0.94
    fig1.text(subplot_x, 0.8, 'Gray\nMatter', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.6, 'White\nMatter', ha='center', va='center',  fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.35, 'Distance\nvs\nPower', ha='center', va='center',  fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.121, 'SNR', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(0.5, 0.98, 'Univariate Analyses: Power and SNR', ha='center', va='top', fontdict = {'fontsize': fontsize*1.1})
    #fig1.text(subplot_x-0.05, 0.98, 'Patients: 5\nSeizures: 5\nGM electrodes: 434\nWM electrodes: 97', ha='left', va='top', fontdict = {'fontsize': 12})
    
    fig1.text(0.08, 0.121, '$\mathregular{SNR}=(\\frac{P_{segment}}{P_{interictal}})$', ha='left', fontdict = {'fontsize': fontsize*0.9})
    
    fig1.text(0.02, 0.9, 'A.', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(0.02, 0.47, 'B.', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(0.02, 0.195, 'C.', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    
    
    
    #PSDvsDistance
    cbar_ax3 = fig1.add_axes([0.85, 0.25, 0.02, 0.20])
    i = 2
    k = 0
    for j in range(0, 4):
        plot = np.log10(powerDistAvg[:ylim,:,j])
        #plot = np.delete(np.array(plot), range(30, np.shape(plot)[0]), axis=0)
        #plot = np.delete(plot, range(800, np.shape(plot)[1]), axis=1)
        sns.heatmap(plot,cmap=sns.color_palette("Spectral_r", n_colors=100, desat=0.6), vmin=-1,vmax=3.5,ax = axes[i][j],
                    cbar=k < 1, cbar_ax=None if k else cbar_ax3, cbar_kws={'label': '$\mathregular{log_{10}}({V^2}/{Hz})$'})
        old_ticks = axes[i][j].get_xticks()
        new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), 5)
        new_ticks = new_ticks[range(1,4)]
        axes[i][j].set_xticks([0,200,400,600,800])
        axes[i][j].set_xticklabels(["0", "2", "4", "6","8"], rotation=0)
        if j ==0:
            axes[i][j].set_yticks(np.arange(0,ylim, 10))
            axes[i][j].set_yticklabels(np.arange(0,ylim, 10), rotation=0)
        if j > 0:
            axes[i][j].set_yticks([])
        k = k + 1
        axes[i][j].invert_yaxis()
    cbar_ax3.yaxis.set_ticks_position('left')
    cbar_ax3.yaxis.set_label_position('left')
    #axes[i][0].add_patch(Ellipse((90, 7.1), width=160, height=12,edgecolor='#0000cccc',facecolor='none',linewidth=5))
    #axes[i][0].add_patch(Ellipse((700, 7.1), width=160, height=12,edgecolor='#0000cccc',facecolor='none',linewidth=5))
    
    #axes[i][2].add_patch(Ellipse((90, 7.1), width=160, height=12,edgecolor='#cc0000cc',facecolor='none',linewidth=5))
    #axes[i][2].add_patch(Ellipse((700, 7.1), width=160, height=12,edgecolor='#cc0000cc',facecolor='none',linewidth=5))
    
    
    #SNR
    distBounds = [0,3,6]
    SNRcategories = np.zeros(shape = (len(SNRAll[0]), len(distBounds)+1, len(SNRAll) ))
    
    #loop through all patients, find all electrode in distance and then average them
    for s in range(len(SNRAll)):
        SNRpatient = SNRAll[s]
        for b in range(len(distBounds)+1):
     
            if b == 0:
                SNRcategories[:,b,s] = np.nanmean(SNRpatient[:,np.where(distAll[s] <= distBounds[b] )[0]], axis = 1 )
            elif b >0 and b < len(distBounds):
                SNRcategories[:,b,s] = np.nanmean(SNRpatient[:,  np.where(    (distAll[s] > distBounds[b-1])  &  (distAll[s] <= distBounds[b])   )[0]  ], axis = 1 )
            else:
                SNRcategories[:,b,s] = np.nanmean(SNRpatient[:,np.where(distAll[s] > distBounds[b-1] )[0]], axis = 1 )
    SRNavg = np.nanmean(SNRcategories, axis = 2)    
    SRNavgdf = pd.DataFrame(SRNavg)
    SRNavgdf = SRNavgdf.reset_index()

    SRNavgdfLong = pd.melt(SRNavgdf, id_vars = "index")  
    SRNavgdfLong.columns = ["Time", "Distance (mm)", "SNR"]
    #sns.lineplot(data = SRNavgdfLong , x = "index", y = "SNR" , hue = "distance" )
    
    i = 3
    k = 0
    #line colors
    legend_labels = distBounds[:]
    names = []
    for f in range(len(legend_labels)):
        if f == 0:
            names.append("= {0}".format(legend_labels[f]))
        if f > 0:
            names.append("({0}, {1}]".format(legend_labels[f - 1], legend_labels[f]))
    names.append("≥ {0}]".format(legend_labels[len(legend_labels) - 1]))
    cols = ["Time"]
    for f in range(len(names)):
        cols.append(names[f])
    line_colors = sns.color_palette("Blues", n_colors=len(names)-2, desat=0.6)
    black = (0, 0, 0); red = (0.7, 0, 0); line_colors.append(red); line_colors.insert(0, black)
    for j in range(5):
        SNR_to_plot = SRNavgdfLong[(SRNavgdfLong["Time"] >= k*200) & (SRNavgdfLong["Time"] < 200*(k+1))  ]
        if j == 1:#create a blank space between interictal and preictal
            axes[i][j].remove()
        if j != 1:
            sns.lineplot(x='Time', y='SNR', hue='Distance (mm)', data=SNR_to_plot, palette=line_colors, ax = axes[i][j])
            
            
            axes[i][j].set_xlim(k*200,200*(k+1))
            axes[i][j].set_ylim(0, 5)
            axes[i][j].set(xlabel='', ylabel='')
            #axes[i][j].set( yscale="log")
            #axes[i][j].legend
            if j < 4:
                axes[i][j].get_legend().set_visible(False)
            if j == 4:
                handles, labels = axes[i][j].get_legend_handles_labels()
                axes[i][j].legend(handles=handles[0:], labels=names[0:], title="Distance (mm)",
                                  bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            old_ticks = axes[i][j].get_xticks()
            new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), 5)
            new_ticks = new_ticks[range(1, 4)]
            axes[i][j].set_xticks(new_ticks)
            axes[i][j].set_xticklabels(["25%", "50%", "75%"], rotation=0)
            if j == 0:
                axes[i][j].set_xticks([new_ticks[1]])
                axes[i][j].set_xticklabels(["~6hrs before"], rotation=0)
             #   axes[i][j].set_yticklabels([0, 10, 20, 30, 40, 50], rotation=0)
            if j > 1:
                axes[i][j].set_yticks([])
                if j < 4:
                    axes[i][j].axvline(x=np.shape(powerGMmean)[1], c="black", linewidth=4, linestyle='--')
            k = k + 1
            print(j)
#%%





























