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




def plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM):
    powerGMall = powerGM
    powerWMall = powerWM
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
    gs3 = fig1.add_gridspec(nrows=1, ncols=4, left=left, right=right, bottom = 0.30, top = 0.45 ,wspace=0.1, width_ratios=[1, 1, 1, 1])
    gs4 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.062, top = 0.23 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs5 = fig1.add_gridspec(nrows=1, ncols=1, left=0.165, right=0.235, bottom = 0.135, top = 0.225 ,wspace=0.00)#plots in corner of SNR
    gs6 = fig1.add_gridspec(nrows=1, ncols=1, left=0.3525, right=0.4225, bottom = 0.135, top = 0.225 ,wspace=0.00) #plots in corner of SNR
    gs7 = fig1.add_gridspec(nrows=1, ncols=1, left=0.55, right=0.62, bottom = 0.082, top = 0.158 ,wspace=0.00)
    gs8 = fig1.add_gridspec(nrows=1, ncols=1, left=0.725, right=0.795, bottom = 0.135, top = 0.225 ,wspace=0.00)
    gs9 = fig1.add_gridspec(nrows=1, ncols=1, left=0.67, right=0.795, bottom = 0.58, top = 0.685 ,wspace=0.00) #plots in corner of GM vs WM plot
    
    gs = [gs1, gs2, gs3, gs4, gs5, gs6, gs7, gs8, gs9]
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
                #if j == 0:
                #    axes[i][j].set_xticks([new_ticks[1]])
                #    axes[i][j].set_xticklabels(["~6hrs before"], rotation=0)
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
    fig1.text(0.02, 0.37, 'Frequency (Hz)', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.02, 0.151, 'SNR', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.48, 'Time (% seizure length)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.25, 'Distance from GM (mm)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.02, 'Time (% seizure length)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    
    fig1.text(0.145, 0.9, 'Interictal (ii)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.34, 0.9, 'Preictal (pi)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.525, 0.9, 'Ictal (ic)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.71, 0.9, 'Postictal (po)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    subplot_x=0.94
    fig1.text(subplot_x, 0.8, 'Gray\nMatter', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.6, 'White\nMatter', ha='center', va='center',  fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.35, 'Distance\nvs\nPower', ha='center', va='center',  fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.121, 'SNR', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(0.5, 0.98, 'Univariate Analyses: Power and SNR', ha='center', va='top', fontdict = {'fontsize': fontsize*1.1})
    #fig1.text(subplot_x-0.05, 0.98, 'Patients: 5\nSeizures: 5\nGM electrodes: 434\nWM electrodes: 97', ha='left', va='top', fontdict = {'fontsize': 12})
    
    fig1.text(0.11, 0.21, 'SNR =', ha='center', fontdict = {'fontsize': fontsize*0.5})
    fig1.text(0.11, 0.17, '$\mathregular{} \\frac{P_{segment}}{P_{interictal}}$', ha='center', fontdict = {'fontsize': fontsize*0.85})
    
    fig1.text(0.02, 0.9, 'a', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(0.02, 0.47, 'b', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(0.02, 0.245, 'c', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    
    
    
    #PSDvsDistance
    cbar_ax3 = fig1.add_axes([0.85, 0.25, 0.02, 0.20])
    i = 2
    k = 0
    for j in range(0, 4):
        plot = np.log10(powerDistAvg[:ylim,:,j])
        #plot = np.delete(np.array(plot), range(30, np.shape(plot)[0]), axis=0)
        #plot = np.delete(plot, range(800, np.shape(plot)[1]), axis=1)
        sns.heatmap(plot,cmap=sns.color_palette("Spectral_r", n_colors=100, desat=0.6), vmin=-2.2,vmax=3,ax = axes[i][j],
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
    
    patientsUnique = np.unique(paientList["patient"])
    SNRavgByPatients = np.zeros(shape = (SNRcategories.shape[0], SNRcategories.shape[1], len(np.unique(paientList["patient"])) ))
    for pt in range(len(patientsUnique)):
        ind = np.where(patientsUnique[pt] == paientList["patient"])[0]
        SNRavgByPatients[:,:,pt] = np.nanmean( SNRcategories[:,:, ind], axis = 2     )
    SRNavg = np.nanmean(SNRavgByPatients, axis = 2)    
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
            axes[i][j].set_ylim(0.00, 8)
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
            #if j == 0:
            #    axes[i][j].set_xticks([new_ticks[1]])
            #    axes[i][j].set_xticklabels(["~6hrs before"], rotation=0)
             #   axes[i][j].set_yticklabels([0, 10, 20, 30, 40, 50], rotation=0)
            linecolor = "#000000"
            axes[i][j].axvline(100+200*k, linestyle='dashed',  linewidth=3, c = linecolor)
            
            #lines connecting subplots of SNR
            if j > 1:
                axes[i][j].set_yticks([])
                if j < 4:
                    axes[i][j].axvline(x=np.shape(powerGMmean)[1], c="black", linewidth=4, linestyle='--')
                if j ==3:
                    axes[i][j].axvline(100+200*k, linestyle='dashed',  linewidth=3, c ="black")
                    lim = axes[i][j].get_ylim()
                    x1 = 100+200*k;    y1 = lim[0] + (lim[1] - lim[0])*0.33
                    x2 = 100+200*k+20;    y2 = lim[0] + (lim[1] - lim[0])*0.56; y3 = lim[0] + (lim[1] - lim[0])*0.13
                    axes[i][j].plot([x1, x2], [y1, y2], '-', lw=2, color="black")
                    axes[i][j].plot([x1, x2], [y1, y3], '-', lw=2, color="black")
            if j == 0 or j == 2 or j == 4:
                 lim = axes[i][j].get_ylim()
                 x1 = 100+200*k;    y1 = lim[0] + (lim[1] - lim[0])*0.73
                 x2 = 100+200*k+12;    y2 = lim[0] + (lim[1] - lim[0])*0.95; y3 = lim[0] + (lim[1] - lim[0])*0.455
                 axes[i][j].plot([x1, x2], [y1, y2], '-', lw=2, color=linecolor)
                 axes[i][j].plot([x1, x2], [y1, y3], '-', lw=2, color=linecolor)
            k = k + 1
    
    #plot boxplot of SNR at 50 % of the way thru seizure
    halfway = [110, 280,478,750]
    j = 4
    for h in range(len(halfway)):
        SNRavgByPatients50 = SNRavgByPatients[halfway[h],:,:]        
        SNRavgByPatients50df = pd.DataFrame(SNRavgByPatients50)
        SNRavgByPatients50df = SNRavgByPatients50df.reset_index()
        SNRavgByPatients50dfLong = pd.melt(SNRavgByPatients50df, "index")
        
        w1 = np.array(SNRavgByPatients50dfLong.loc[SNRavgByPatients50dfLong['index'] == 0, 'value'])
        w2 =  np.array(SNRavgByPatients50dfLong.loc[SNRavgByPatients50dfLong['index'] == 3, 'value'])
        stats.wilcoxon(  w1, w2 )[1] *4
        print( stats.wilcoxon(  w1, w2 )[1]  *4)
        if h >=0:
            sns.boxplot(data=SNRavgByPatients50dfLong, x="index", y="value", palette=line_colors, showfliers=False,
                        ax=axes[j][0], color=line_colors, order=[0, 1, 2, 3])
            sns.stripplot(x="index", y="value",  data=SNRavgByPatients50dfLong, palette=line_colors,
                          dodge=True, size=3, order=[0, 1, 2, 3],  ax=axes[j][0])
            axes[j][0].set(xlabel='', ylabel='')
            axes[j][0].set_yticks([])
            axes[j][0].set_xticks([])
            lw = 3
            if h == 0:
                axes[j][0].set_ylim(-0.1, 2.8)
                lim = axes[j][0].get_ylim()
                axes[j][0].text( (0+3)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = "p>0.05", va='top', ha = "center")
            if h == 1:
                axes[j][0].set_ylim(-0.15, 3)
                lim = axes[j][0].get_ylim()
                axes[j][0].text( (0+3)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = "p>0.05", va='top', ha = "center")
            if h == 2:
                axes[j][0].set_ylim(-2.7, 11.9)
                lim = axes[j][0].get_ylim()
                x1 = 0;    y1 = lim[0] + (lim[1] - lim[0])*0.15
                x2 = 3;    y2 = lim[0] + (lim[1] - lim[0])*0.15; y3 = lim[0] + (lim[1] - lim[0])*0.19
                axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color="black")
                axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*0.125 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = "p<0.00005", va='top', ha = "center")
            if h == 3:
                axes[j][0].set_ylim(-1.5, 5.6)
                lim = axes[j][0].get_ylim()
                x1 = 0;    y1 = lim[0] + (lim[1] - lim[0])*0.13
                x2 = 3;    y2 = lim[0] + (lim[1] - lim[0])*0.13; y3 = lim[0] + (lim[1] - lim[0])*0.16
                axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color="black")
                axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*0.11 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = "p<0.05", va='top', ha = "center")
            j = j+1
    
    
    """
    #making plot layout
    fig1 = plt.figure(constrained_layout=False)
    left = 0.07; right = 0.8
    gs1 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.72, top = 0.88 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs2 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.52, top = 0.69 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs3 = fig1.add_gridspec(nrows=1, ncols=4, left=left, right=right, bottom = 0.25, top = 0.45 ,wspace=0.1, width_ratios=[1, 1, 1, 1])
    gs4 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.062, top = 0.18 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs5 = fig1.add_gridspec(nrows=1, ncols=1, left=0.165, right=0.235, bottom = 0.115, top = 0.175 ,wspace=0.00)
    gs6 = fig1.add_gridspec(nrows=1, ncols=1, left=0.3525, right=0.4225, bottom = 0.115, top = 0.175 ,wspace=0.00) #plots in corner of SNR
    gs7 = fig1.add_gridspec(nrows=1, ncols=1, left=0.555, right=0.625, bottom = 0.068, top = 0.128 ,wspace=0.00)
    gs8 = fig1.add_gridspec(nrows=1, ncols=1, left=0.725, right=0.795, bottom = 0.115, top = 0.175 ,wspace=0.00)
    gs9 = fig1.add_gridspec(nrows=1, ncols=1, left=0.675, right=0.795, bottom = 0.59, top = 0.685 ,wspace=0.00)
    
    gs = [gs1, gs2, gs3, gs4, gs5, gs6, gs7, gs8, gs9]
    axes = [None] * len(gs)
    for a in range(len(gs)):
        axes[a] = [None] * gs[a]._ncols
    
    for a in range(len(axes)):
        for b in range(len(axes[a])):
            axes[a][b] = fig1.add_subplot(gs[a][:, b])
 
   """
    #plot boxplot of power GM vs WM 
    j=8    
    nseiuzres = powerGMall.shape[2]
    upperFreq = 60
    GMmeanII = np.nanmean(powerGMall[0:upperFreq, 0:200, :], axis=1)
    GMmeanPI = np.nanmean(powerGMall[0:upperFreq, 200:400, :], axis=1)
    GMmeanIC = np.nanmean(powerGMall[0:upperFreq, 400:600, :], axis=1)
    GMmeanPO = np.nanmean(powerGMall[0:upperFreq, 600:800, :], axis=1)
    
    WMmeanII = np.nanmean(powerWMall[0:upperFreq, 0:200, :], axis=1)
    WMmeanPI = np.nanmean(powerWMall[0:upperFreq, 200:400, :], axis=1)
    WMmeanIC = np.nanmean(powerWMall[0:upperFreq, 400:600, :], axis=1)
    WMmeanPO = np.nanmean(powerWMall[0:upperFreq, 600:800, :], axis=1)
    
    area = np.zeros(shape=(2, nseiuzres, 4))
    for i in range(nseiuzres):
        area[0, i, 0] = simps(GMmeanII[:, i], dx=1)
        area[1, i, 0] = simps(WMmeanII[:, i], dx=1)
        area[0, i, 1] = simps(GMmeanPI[:, i], dx=1)
        area[1, i, 1] = simps(WMmeanPI[:, i], dx=1)
        area[0, i, 2] = simps(GMmeanIC[:, i], dx=1)
        area[1, i, 2] = simps(WMmeanIC[:, i], dx=1)
        area[0, i, 3] = simps(GMmeanPO[:, i], dx=1)
        area[1, i, 3] = simps(WMmeanPO[:, i], dx=1)
    
    area = np.log10(area)
    
    index = pd.MultiIndex.from_product([range(s)for s in area.shape], names=[
                                       'tissue', 'seizure', "state"])
    
    dfArea = pd.DataFrame({'power': area.flatten()}, index=index)['power']
    dfArea.index.names = ['tissue', 'seizure', "state"]
    dfArea = dfArea.rename(index={0: "GM", 1: "WM"}, level=0)
    dfArea = dfArea.rename(
        index={0: "interictal", 1: "preictal", 2: "ictal", 3: "postictal"}, level=2)
    dfArea = pd.DataFrame(dfArea)
    dfArea.reset_index(inplace=True)
    dfArea["patient"] = np.nan
    for i in range(len(dfArea)):
        dfArea["patient"][i] = paientList["patient"][dfArea["seizure"][i]]
    
    colors1 = ["#c6b4a5", "#b6d4ee"]
    colors2 = ["#a08269", "#76afdf"]
    colors3 = ["#544335", "#1f5785"]
    
    #dfAreaPlot = dfArea.groupby(["tissue", "seizure", "state"]).mean() #plot by seizure
    dfAreaPlot = dfArea.groupby(["tissue", "patient", "state"]).mean() #plot by patient
    dfAreaPlot.reset_index(inplace=True)
   
    sns.boxplot(data=dfAreaPlot, x="state", y="power", hue="tissue", palette=colors2, showfliers=False,
                ax=axes[j][0], color=colors3, order=["interictal", "preictal", "ictal", "postictal"])
    sns.stripplot(x="state", y="power", hue="tissue",  data=dfAreaPlot, palette=colors3,ax=axes[j][0],
                  dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])
    # Set only one legend
    handles, labels = axes[j][0].get_legend_handles_labels()
    l =  axes[j][0].legend(handles[0:2], labels[0:2], loc=2, borderaxespad=0.5, framealpha = 0, edgecolor = "inherit",fontsize = 14, bbox_to_anchor=(-0.04, 1.06))
    #axes[j][0].set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
    axes[j][0].set(xlabel='', ylabel='')
    axes[j][0].set_yticks([])
    axes[j][0].set_xticks([])
    axes[j][0].set_ylim(0.7, 6.3)
    axes[j][0].set_xlim(-0.5, 3.45)
    lim = axes[j][0].get_ylim()
    #axes[j][0].legend([],[], frameon=False)
    axes[j][0].text( 0, y = lim[0] + (lim[1] - lim[0])*0.62 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
    axes[j][0].text( 1, y = lim[0] + (lim[1] - lim[0])*0.62 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
    axes[j][0].text( 2, y = lim[0] + (lim[1] - lim[0])*0.9 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
    axes[j][0].text( 3, y = lim[0] + (lim[1] - lim[0])*0.62 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
    axes[j][0].text( 0.72, y = lim[0] + (lim[1] - lim[0])*0.18 , fontsize = 16, s = "*", va='center', ha = "right", fontweight='bold', color=colors3[0])
    axes[j][0].text( 1.22, y = lim[0] + (lim[1] - lim[0])*0.05 , fontsize = 16, s = "*", va='center', ha = "right", fontweight='bold', color=colors3[1])
    
    lw = 3
    x1 = 0.75;    y1 = lim[0] + (lim[1] - lim[0])*0.23
    x2 = 1.75;    y2 = lim[0] + (lim[1] - lim[0])*0.23; y3 = lim[0] + (lim[1] - lim[0])*0.26
    axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color=colors2[0])
    axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color=colors2[0])
    axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color=colors2[0])
    x1 = 0.75;    y1 = lim[0] + (lim[1] - lim[0])*0.16
    x2 = 2.75;    y2 = lim[0] + (lim[1] - lim[0])*0.16; y3 = lim[0] + (lim[1] - lim[0])*0.19
    axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color=colors2[0])
    axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color=colors2[0])
    axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color=colors2[0])

    x1 = 1.25;    y1 = lim[0] + (lim[1] - lim[0])*0.09
    x2 = 2.25;    y2 = lim[0] + (lim[1] - lim[0])*0.09; y3 = lim[0] + (lim[1] - lim[0])*0.12
    axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color=colors2[1])
    axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color=colors2[1])
    axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color=colors2[1])
    x1 = 1.25;    y1 = lim[0] + (lim[1] - lim[0])*0.03
    x2 = 3.25;    y2 = lim[0] + (lim[1] - lim[0])*0.03; y3 = lim[0] + (lim[1] - lim[0])*0.06
    axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color=colors2[1])
    axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color=colors2[1])
    axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color=colors2[1])
    [s.set_visible(False) for s in axes[j][0].spines.values()]
    
    fs = 20
    tp = -0.05
    axes[j][0].text( 0, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fs, s = "ii", va='top', ha = "center")
    axes[j][0].text( 1, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fs, s = "pi", va='top', ha = "center")
    axes[j][0].text( 2, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fs, s = "ic", va='top', ha = "center")
    axes[j][0].text( 3, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fs, s = "po", va='top', ha = "center")




#%%


























