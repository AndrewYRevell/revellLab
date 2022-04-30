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

from revellLab.packages.utilities import utils





#%%
#POWER

def plot_power_vs_distance_and_SNR(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM):
    powerGMall = powerGM
    powerWMall = powerWM
    powerGM = [powerGMmean[:,0:200], powerGMmean[:,200:400], powerGMmean[:,400:600], powerGMmean[:,600:800]]
    powerWM = [powerWMmean[:,0:200], powerWMmean[:,200:400], powerWMmean[:,400:600], powerWMmean[:,600:800]]

    #general plotting parameters
    plt.rcParams['figure.figsize'] = (11, 11.3)
    plt.rcParams['font.family'] = "serif"
    sns.set_context("talk")
    fontsize = 30
    vminSpec = -1; vmaxSpec = 2.5
    #making plot layout
    fig1 = plt.figure(constrained_layout=False)
    left = 0.09; right = 1
    gs3 = fig1.add_gridspec(nrows=1, ncols=4, left=left, right=right, bottom = 0.58, top = 0.9 ,wspace=0.1, width_ratios=[1, 1, 1, 1])
    gs4 = fig1.add_gridspec(nrows=1, ncols=5, left=left, right=right, bottom = 0.08, top = 0.4 ,wspace=0.00, width_ratios=[1,0.1, 1, 1.2, 1])
    gs5 = fig1.add_gridspec(nrows=1, ncols=1, left=0.105, right=0.295, bottom = 0.3, top = 0.39 ,wspace=0.00)#plots in corner of SNR
    gs6 = fig1.add_gridspec(nrows=1, ncols=1, left=0.33, right=0.525, bottom = 0.3, top = 0.39 ,wspace=0.00) #plots in corner of SNR
    gs7 = fig1.add_gridspec(nrows=1, ncols=1, left=0.62, right=0.78, bottom = 0.085, top = 0.18 ,wspace=0.00)  #plots in corner of SNR
    gs8 = fig1.add_gridspec(nrows=1, ncols=1, left=0.80, right=0.98, bottom = 0.3, top = 0.39 ,wspace=0.00)  #plots in corner of SNR


    gs = [gs3, gs4, gs5, gs6, gs7, gs8]
    axes = [None] * len(gs)
    for a in range(len(gs)):
        axes[a] = [None] * gs[a]._ncols

    for a in range(len(axes)):
        for b in range(len(axes[a])):
            axes[a][b] = fig1.add_subplot(gs[a][:, b])


    fig1.text(0.5, 0.45, 'SNR', ha='center', va='top', fontdict = {'fontsize': fontsize*1.2})
    fig1.text(0.5, 0.95, 'Power vs Distance', ha='center', va='top', fontdict = {'fontsize': fontsize*1.2})


    fig1.text(0.02, 0.72, 'Frequency (Hz)', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize})
    fig1.text(0.02, 0.23, 'SNR', ha='center', va='center', rotation='vertical', fontdict = {'fontsize': fontsize})

    fig1.text(0.5, 0.515, 'Distance from GM (mm)', ha='center', va='center', fontdict = {'fontsize': fontsize*.8})
    #fig1.text(0.09, 0.41, 'Distance from GM (mm):', ha='left', va='center', fontdict = {'fontsize': fontsize*0.55})
    fig1.text(0.5, 0.00, 'Time (% seizure length)', ha='center', va='bottom', fontdict = {'fontsize': fontsize*.8})

    state_height = 0.85
    #fig1.text(0.17, state_height, 'ii', ha='center', fontdict = {'fontsize': fontsize})
    #fig1.text(0.35, state_height, 'pi', ha='center', fontdict = {'fontsize': fontsize})
    #fig1.text(0.54, state_height, 'ic', ha='center', fontdict = {'fontsize': fontsize})
    #fig1.text(0.71, state_height, 'po', ha='center', fontdict = {'fontsize': fontsize})
    subplot_x=0.94


    #fig1.text(0.11, 0.21, 'SNR =', ha='center', fontdict = {'fontsize': fontsize*0.5})
    #fig1.text(0.11, 0.17, '$\mathregular{} \\frac{P_{segment}}{P_{interictal}}$', ha='center', fontdict = {'fontsize': fontsize*0.85})

    ylim = 100


    #PSDvsDistance
    cbar_ax3 = fig1.add_axes([0.79, 0.91, 0.21, 0.015]) #fig1.add_axes([0.09, 0.49, 0.21, 0.03]) #fig1.add_axes([0.09, 0.42, 0.02, 0.11]) # fig1.add_axes([0.9, 0.55, 0.05, 0.33])
    i = 0
    k = 0
    for j in range(0, 4):
        plot = np.log10(powerDistAvg[:ylim,:,j])
        #plot = np.delete(np.array(plot), range(30, np.shape(plot)[0]), axis=0)
        #plot = np.delete(plot, range(800, np.shape(plot)[1]), axis=1)
        #sns.heatmap(plot, cmap=sns.color_palette("Spectral_r", n_colors=100, desat=0.6), vmin=-2.2,vmax=3,ax = axes[i][j],
        #            cbar=k < 1, cbar_ax=None if k else cbar_ax3, cbar_kws={'orientation':'vertical', 'label': '$\mathregular{log_{10}}({V^2}/{Hz})$'})
        sns.heatmap(plot, cmap=sns.color_palette("Spectral_r", n_colors=100, desat=0.6), vmin=-2.2,vmax=3,ax = axes[i][j],
                    cbar=k < 1, cbar_ax=None if k else cbar_ax3, cbar_kws={'orientation':'horizontal', 'label': ''})
        old_ticks = axes[i][j].get_xticks()
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
    cbar_ax3.xaxis.set_ticks_position('top')
    cbar_ax3.xaxis.set_label_position('top')
    cbar_ax3.tick_params(labelsize=fontsize*0.5)
    cbar_ax3.xaxis.label.set_size(fontsize*.7)
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




    i = 1
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
                leg = axes[i][j].legend()
                for line in leg.get_lines():
                    line.set_linewidth(10.0)
                handles, labels = axes[i][j].get_legend_handles_labels()
                axes[i][j].legend(handles=handles[0:], labels=names[0:], title="",
                                  bbox_to_anchor=(-0.0, 1.45), loc=2, ncol = 1, borderaxespad=0., frameon = False, prop={'size': 20}, markerscale = 2)
            old_ticks = axes[i][j].get_xticks()
            new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), 5)
            new_ticks = new_ticks[range(1, 4)]
            axes[i][j].set_xticks(new_ticks)
            axes[i][j].set_xticklabels(["25", "50", "75"], rotation=0)
            #if j == 0:
            #    axes[i][j].set_xticks([new_ticks[1]])
            #    axes[i][j].set_xticklabels(["~6hrs before"], rotation=0)
             #   axes[i][j].set_yticklabels([0, 10, 20, 30, 40, 50], rotation=0)
            linecolor = "#000000"
            #axes[i][j].axvline(100+200*k, linestyle='dashed',  linewidth=3, c = linecolor)
            if j !=3:
                axes[i][j].plot([100+200*k, 100+200*k], [1.2, 3.5], linestyle='dashed', lw=4, color="black")
            #lines connecting subplots of SNR
            if j > 1:
                axes[i][j].set_yticks([])
                if j < 4:
                    axes[i][j].axvline(x=np.shape(powerGMmean)[1], c="black", linewidth=4, linestyle='--')
                if j ==3:
                    #axes[i][j].axvline(100+200*k, linestyle='dashed',  linewidth=3, c ="black")
                    axes[i][j].plot([100+200*k, 100+200*k], [6, 3], linestyle='dashed', lw=4, color="black")
                    lim = axes[i][j].get_ylim()
                    x1 = 100+200*k;    y1 =lim[0] + (lim[1] - lim[0])*0.38
                    x2 = 100+200*k+65;    y2 = lim[0] + (lim[1] - lim[0])*0.33
                    x3 = 100+200*k-13
                    axes[i][j].plot([x1, x2], [y1, y2], '-', lw=4, color="black")
                    axes[i][j].plot([x1, x3], [y1, y2], '-', lw=4, color="black")
            if j == 0 or j == 2 or j == 4:
                 lim = axes[i][j].get_ylim()
                 x1 = 100+200*k;    y1 = lim[0] + (lim[1] - lim[0])*0.45
                 x2 = 100+200*k+86;    y2 = lim[0] + (lim[1] - lim[0])*0.62;
                 x3 = 100+200*k-86
                 axes[i][j].plot([x1, x2], [y1, y2], '-', lw=4, color=linecolor)
                 axes[i][j].plot([x1, x3], [y1, y2], '-', lw=4, color=linecolor)
            k = k + 1

    #plot boxplot of SNR at 50 % of the way thru seizure
    halfway = [50, 250,450,750]
    j = 2
    for h in range(len(halfway)):
        SNRavgByPatients50 = SNRavgByPatients[halfway[h],:,:]
        SNRavgByPatients50df = pd.DataFrame(SNRavgByPatients50)
        SNRavgByPatients50df = SNRavgByPatients50df.reset_index()
        SNRavgByPatients50dfLong = pd.melt(SNRavgByPatients50df, "index")

        w1 = np.array(SNRavgByPatients50dfLong.loc[SNRavgByPatients50dfLong['index'] == 0, 'value'])
        w2 =  np.array(SNRavgByPatients50dfLong.loc[SNRavgByPatients50dfLong['index'] == 3, 'value'])
        ppp = stats.wilcoxon(  w1, w2 )[1] *4
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
                axes[j][0].text( (0+3)/2, y = lim[0] + (lim[1] - lim[0])*-0.1 , fontsize = 25, s = f"p={np.round(ppp,2)}", va='top', ha = "center")
            if h == 1:
                axes[j][0].set_ylim(-0.15, 3)
                lim = axes[j][0].get_ylim()
                axes[j][0].text( (0+3)/2, y = lim[0] + (lim[1] - lim[0])*-0.1 , fontsize = 25, s = f"p={np.round(ppp,2)}", va='top', ha = "center")
            if h == 2:
                axes[j][0].set_ylim(-2.7, 11.9)
                lim = axes[j][0].get_ylim()
                x1 = 0;    y1 = lim[0] + (lim[1] - lim[0])*0.15
                x2 = 3;    y2 = lim[0] + (lim[1] - lim[0])*0.15; y3 = lim[0] + (lim[1] - lim[0])*0.19
                axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color="black")
                axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].text( (x1+x2)/1.3, y = lim[0] + (lim[1] - lim[0])*2.1 , fontsize = 50, s = "*", va='top', ha = "center", fontweight='bold')
                axes[j][0].text( (x1+x2)/1.4, y = lim[0] + (lim[1] - lim[0])*1.65 , fontsize = 25, s = f"p={np.round(ppp,5)}", va='top', ha = "center")
            if h == 3:
                axes[j][0].set_ylim(-1.5, 5.6)
                lim = axes[j][0].get_ylim()
                x1 = 0;    y1 = lim[0] + (lim[1] - lim[0])*0.13
                x2 = 3;    y2 = lim[0] + (lim[1] - lim[0])*0.13; y3 = lim[0] + (lim[1] - lim[0])*0.16
                axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color="black")
                axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.39 , fontsize = 50, s = "*", va='top', ha = "center", fontweight='bold')
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.1 , fontsize = 25, s = f"p={np.round(ppp,2)}", va='top', ha = "center")
            j = j+1

#%%
#POWER
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
    halfway = [50, 250, 450, 750]
    j = 4
    for h in range(len(halfway)):
        SNRavgByPatients50 = SNRavgByPatients[halfway[h],:,:]
        SNRavgByPatients50df = pd.DataFrame(SNRavgByPatients50)
        SNRavgByPatients50df = SNRavgByPatients50df.reset_index()
        SNRavgByPatients50dfLong = pd.melt(SNRavgByPatients50df, "index")

        w1 = np.array(SNRavgByPatients50dfLong.loc[SNRavgByPatients50dfLong['index'] == 0, 'value'])
        w2 =  np.array(SNRavgByPatients50dfLong.loc[SNRavgByPatients50dfLong['index'] == 3, 'value'])
        stats.wilcoxon(  w1, w2 )[1] *4
        ppp = stats.wilcoxon(  w1, w2 )[1] *4
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
                axes[j][0].text( (0+3)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = f"p={np.round(ppp,2)}", va='top', ha = "center")
            if h == 1:
                axes[j][0].set_ylim(-0.15, 3)
                lim = axes[j][0].get_ylim()
                axes[j][0].text( (0+3)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = f"p={np.round(ppp,2)}", va='top', ha = "center")
            if h == 2:
                axes[j][0].set_ylim(-2.7, 11.9)
                lim = axes[j][0].get_ylim()
                x1 = 0;    y1 = lim[0] + (lim[1] - lim[0])*0.15
                x2 = 3;    y2 = lim[0] + (lim[1] - lim[0])*0.15; y3 = lim[0] + (lim[1] - lim[0])*0.19
                axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color="black")
                axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*0.125 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = f"p={np.round(ppp,5)}", va='top', ha = "center")
            if h == 3:
                axes[j][0].set_ylim(-1.5, 5.6)
                lim = axes[j][0].get_ylim()
                x1 = 0;    y1 = lim[0] + (lim[1] - lim[0])*0.13
                x2 = 3;    y2 = lim[0] + (lim[1] - lim[0])*0.13; y3 = lim[0] + (lim[1] - lim[0])*0.16
                axes[j][0].plot([x1, x2], [y1, y2], '-', lw=lw, color="black")
                axes[j][0].plot([x1, x1], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].plot([x2, x2], [y1, y3], '-', lw=lw, color="black")
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*0.11 , fontsize = 16, s = "*", va='top', ha = "center", fontweight='bold')
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = f"p={np.round(ppp,2)}", va='top', ha = "center")
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
#POWER
def plotUnivariatePercent(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM):
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
    fig1.text(0.5, 0.25, 'WM Depth (%)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.5, 0.02, 'Time (% seizure length)', ha='center', va='center', fontdict = {'fontsize': fontsize*0.7})

    fig1.text(0.145, 0.9, 'Interictal (ii)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.34, 0.9, 'Preictal (pi)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.525, 0.9, 'Ictal (ic)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    fig1.text(0.71, 0.9, 'Postictal (po)', ha='center', fontdict = {'fontsize': fontsize*0.7})
    subplot_x=0.94
    fig1.text(subplot_x, 0.8, 'Gray\nMatter', ha='center', va='center', fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.6, 'White\nMatter', ha='center', va='center',  fontdict = {'fontsize': fontsize*1.1})
    fig1.text(subplot_x, 0.35, 'Depth\nvs\nPower', ha='center', va='center',  fontdict = {'fontsize': fontsize*1.1})
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
        axes[i][j].set_xticks([0,25,50,75,100])
        axes[i][j].set_xticklabels(["0", "25", "50", "75","100"], rotation=0)
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
    distBounds = [0.5,0.75, 0.9]
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
    SRNavgdfLong.columns = ["Time", "Purity (%)", "SNR"]



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
            sns.lineplot(x='Time', y='SNR', hue='Purity (%)', data=SNR_to_plot, palette=line_colors, ax = axes[i][j])


            axes[i][j].set_xlim(k*200,200*(k+1))
            axes[i][j].set_ylim(0.00, 8)
            axes[i][j].set(xlabel='', ylabel='')
            #axes[i][j].set( yscale="log")
            #axes[i][j].legend
            if j < 4:
                axes[i][j].get_legend().set_visible(False)
            if j == 4:
                handles, labels = axes[i][j].get_legend_handles_labels()
                axes[i][j].legend(handles=handles[0:], labels=names[0:], title="Depth (0-1)",
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
                axes[j][0].text( (x1+x2)/2, y = lim[0] + (lim[1] - lim[0])*-0.03 , fontsize = 16, s = "p<0.005", va='top', ha = "center")
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


def plot_boxplot_tissue_power_differences(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM, palette1, pallet2):
    power_analysis_nseiuzres = powerGM.shape[2]

    upperFreq = 60
    GMmeanII = np.nanmean(powerGM[0:upperFreq, 0:200, :], axis=1)
    GMmeanPI = np.nanmean(powerGM[0:upperFreq, 200:400, :], axis=1)
    GMmeanIC = np.nanmean(powerGM[0:upperFreq, 400:600, :], axis=1)
    GMmeanPO = np.nanmean(powerGM[0:upperFreq, 600:800, :], axis=1)

    WMmeanII = np.nanmean(powerWM[0:upperFreq, 0:200, :], axis=1)
    WMmeanPI = np.nanmean(powerWM[0:upperFreq, 200:400, :], axis=1)
    WMmeanIC = np.nanmean(powerWM[0:upperFreq, 400:600, :], axis=1)
    WMmeanPO = np.nanmean(powerWM[0:upperFreq, 600:800, :], axis=1)

    area = np.zeros(shape=(2, power_analysis_nseiuzres, 4))
    for i in range(power_analysis_nseiuzres):
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

    dfAreaPlot = dfArea.groupby(["tissue", "seizure", "state"]).mean()
    dfAreaPlot.reset_index(inplace=True)


    #Plot by Patient: each Patient is its own independent event
    colors1 = ["#c6b4a5", "#b6d4ee"]
    colors2 = ["#a08269", "#76afdf"]
    colors3 = ["#544335", "#1f5785"]


    fig, axes = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    sns.boxplot(data=dfAreaPlot, x="state", y="power", hue="tissue", palette=colors2, showfliers=False,
                ax=axes, color=colors3, order=["interictal", "preictal", "ictal", "postictal"])
    sns.stripplot(x="state", y="power", hue="tissue",  data=dfAreaPlot, palette=colors3,ax=axes,
                  dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])
    # Set only one legend
    handles, labels =axes.get_legend_handles_labels()
    l = axes.legend(handles[0:2], labels[0:2], loc=2, borderaxespad=0.05, framealpha = 0, edgecolor = "inherit",fontsize = 20, bbox_to_anchor=(0.0, 1.0))
    #axes[j][0].set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
    axes.set(xlabel='', ylabel='')
    axes.set_yticks([2,4,6])
    axes.set_xticks([])
    axes.set_ylim(0.7, 6.3)
    axes.set_xlim(-0.5, 3.45)
    lim =axes.get_ylim()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    #axes[j][0].legend([],[], frameon=False)
    fontsize = 35
    axes.text( 0, y = lim[0] + (lim[1] - lim[0])*0.64 , fontsize = fontsize, s = "*", va='top', ha = "center", fontweight='bold')
    axes.text( 1, y = lim[0] + (lim[1] - lim[0])*0.64 , fontsize = fontsize, s = "*", va='top', ha = "center", fontweight='bold')
    axes.text( 2, y = lim[0] + (lim[1] - lim[0])*0.9 , fontsize = fontsize, s = "*", va='top', ha = "center", fontweight='bold')
    axes.text( 3, y = lim[0] + (lim[1] - lim[0])*0.70 , fontsize = fontsize, s = "*", va='top', ha = "center", fontweight='bold')
    axes.text( 0.70, y = lim[0] + (lim[1] - lim[0])*0.08 , fontsize = fontsize, s = "*", va='center', ha = "right", fontweight='bold', color=colors3[0])
    axes.text( 1.20, y = lim[0] + (lim[1] - lim[0])*-0.02 , fontsize = fontsize, s = "*", va='center', ha = "right", fontweight='bold', color=colors3[1])

    lw = 5
    x1 = 0.75;    y1 = lim[0] + (lim[1] - lim[0])*0.11
    x2 = 1.75;    y2 = lim[0] + (lim[1] - lim[0])*0.11; y3 = lim[0] + (lim[1] - lim[0])*0.14
    axes.plot([x1, x2], [y1, y2], '-', lw=lw, color=colors2[0])
    axes.plot([x1, x1], [y1, y3], '-', lw=lw, color=colors2[0])
    axes.plot([x2, x2], [y1, y3], '-', lw=lw, color=colors2[0])


    x1 = 1.25;    y1 = lim[0] + (lim[1] - lim[0])*0.03
    x2 = 2.25;    y2 = lim[0] + (lim[1] - lim[0])*0.03; y3 = lim[0] + (lim[1] - lim[0])*0.06
    axes.plot([x1, x2], [y1, y2], '-', lw=lw, color=colors2[1])
    axes.plot([x1, x1], [y1, y3], '-', lw=lw, color=colors2[1])
    axes.plot([x2, x2], [y1, y3], '-', lw=lw, color=colors2[1])



    fontsize = 20
    tp = -0.05
    axes.text( 0, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fontsize, s = "ii", va='top', ha = "center")
    axes.text( 1, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fontsize, s = "pi", va='top', ha = "center")
    axes.text( 2, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fontsize, s = "ic", va='top', ha = "center")
    axes.text( 3, y = lim[0] + (lim[1] - lim[0])*tp , fontsize = fontsize, s = "po", va='top', ha = "center")

    axes.set(xlabel='', ylabel='power (log10)')

    """
    #Plot by seizure: each seizure is its own independent event
    fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
    sns.boxplot(data=dfAreaPlot, x="state", y="power", hue="tissue", palette=plot.COLORS_TISSUE_LIGHT_MED_DARK[1], showfliers=False,
                ax=axes, color=plot.COLORS_TISSUE_LIGHT_MED_DARK[2], order=["interictal", "preictal", "ictal", "postictal"])
    sns.stripplot(x="state", y="power", hue="tissue",  data=dfAreaPlot, palette=plot.COLORS_TISSUE_LIGHT_MED_DARK[2],
                  dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])

    # Set only one legend
    handles, labels = axes.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(
        0.0, 1), loc=2, borderaxespad=0.5)
    axes.set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
    """



#%%
#FUNCTIONAL CONNECTIVITY
# All FC and all Frequency Boxplot of FC values for ii, pi, ic, and po states
def plot_boxplot_all_FC_deltaT(summaryStatsLong, FREQUENCY_NAMES, FC_TYPES, palette1, palette2):
    g = sns.FacetGrid(summaryStatsLong, col="frequency", row = "FC_type",sharey=False, col_order=FREQUENCY_NAMES, row_order = FC_TYPES)
    #g.map(sns.boxplot, "state","FC_deltaT", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = palette1)

    g.map(sns.pointplot, "state","FC_deltaT", order=["interictal", "preictal", "ictal", "postictal"], join= False, showfliers=False, color = "black", dodge=0.4, errwidth = 3,capsize = 0.3, linestyles = "--", scale = 0.9)
    g.map(sns.stripplot, "state","FC_deltaT", order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = palette1)
    for x in range(g.axes.shape[0]):
        for y in range(g.axes.shape[1]):
            axes = g.axes[int(x)][int(y)]
            plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")
    ylims =[  [-0.03, 0.6], [-0.005, 0.5] ,[-0.055, 0.6] ] #[ [-0.03, 0.1], [-0.005, 0.04] ,[-0.025, 0.125] ]
    bonferroniFactor = len(FC_TYPES)*len(FREQUENCY_NAMES)
    for func in range(len(FC_TYPES)):
        for f in range(len(FREQUENCY_NAMES)):
            if not f == 1: g.axes[func][f].set_ylim(ylims[func])

            v1 = summaryStatsLong.loc[summaryStatsLong['state'] == "ictal"].loc[summaryStatsLong['FC_type'] == FC_TYPES[func]].loc[summaryStatsLong['frequency'] == FREQUENCY_NAMES[f]]["FC_deltaT"]
            v2 = summaryStatsLong.loc[summaryStatsLong['state'] == "preictal"].loc[summaryStatsLong['FC_type'] == FC_TYPES[func]].loc[summaryStatsLong['frequency'] == FREQUENCY_NAMES[f]]["FC_deltaT"]
            pval = stats.wilcoxon(v1, v2)[1] #* bonferroniFactor #change bonferroniFactor here by multiplying the stats.wilcoxon
            print(pval)
            if pval <=0.05:
                g.axes[func][f].text(1.25,ylims[func][0]+ (ylims[func][1]- ylims[func][0])*0.75,"*", size= 40, weight = 1000)

            g.axes[func][f].set(ylabel='', xlabel='', title="")
            if func == 0: g.axes[func][f].set(ylabel=FC_TYPES[func], xlabel='', title=FREQUENCY_NAMES[f])
            if func  > 0 and f == 0: g.axes[func][f].set(ylabel=FC_TYPES[func], xlabel='', title="")
            if func == len(FC_TYPES)-1: g.axes[func][f].set_xticklabels(["inter", "pre", "ictal", "post"])


def plot_boxplot_single_FC_deltaT(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func, freq, params_plot, ylim = None):
    df = summaryStatsLong_bootstrap.loc[summaryStatsLong_bootstrap['FC_type'] == FC_TYPES[func]].loc[summaryStatsLong_bootstrap['frequency'] == FREQUENCY_NAMES[freq]]
    fig, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
    sns.pointplot( data= df, x = "state", y = "FC_deltaT", order=["interictal", "preictal", "ictal", "postictal"],
                  color = "black", ax = axes, dodge=0.4, errwidth = 5,capsize = 0.2, linestyles = "--", scale = 0.9, join = False)
    plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")
    #sns.boxplot( data= df, x = "state", y = "FC_deltaT", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = params_plot.COLORS_STATE4[0], ax = axes)
    sns.stripplot( data= df, x = "state", y = "FC_deltaT",order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = params_plot.COLORS_STATE4[1], ax = axes,  jitter = 0.3)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if not ylim == None:
        axes.set_ylim(ylim);
    print("\ndeltaT for interictal is not zero:")
    print(stats.ttest_1samp(np.array(df.loc[df["state"] == "interictal"]["FC_deltaT"]), 0)[1])
    print("\ndeltaT for preictal is not zero:")
    print(stats.ttest_1samp(np.array(df.loc[df["state"] == "preictal"]["FC_deltaT"]), 0)[1])
    print("\ndeltaT for ictal is not zero:")
    print(stats.ttest_1samp(np.array(df.loc[df["state"] == "ictal"]["FC_deltaT"]), 0)[1])
    print("\ndeltaT for postictal is not zero:")
    print(stats.ttest_1samp(np.array(df.loc[df["state"] == "postictal"]["FC_deltaT"]), 0)[1])
    print("\ndifference between preictal and ictal")
    print(stats.wilcoxon( np.array(df.loc[df["state"] == "ictal"]["FC_deltaT"]) , np.array(df.loc[df["state"] == "preictal"]["FC_deltaT"])  )[1])
#%%

def plot_FC_example_patient_ADJ(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index,
                                                 dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state , func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, params_plot):
    #% Plot FC distributions for example patient
    FC_type = FC_TYPES[func]
    #plot heatmap, not ordered, ordered
    FCexample = FC[state][freq]
    FCexampleOrdered =  utils.reorderAdj(FCexample, dist_order["index"])
    vmin = -0.1
    vmax = 0.5
    center = 0.2
    utils.plot_adj_heatmap(FCexample, square=True, vmin = vmin, vmax = vmax, center = center, cmap = "mako")
    utils.plot_adj_heatmap(FCexampleOrdered, square=True, vmin = vmin, vmax = vmax, center = center, cmap = "mako")
    print(np.where(dist_order["distance"]> (TISSUE_DEFINITION_GM + TISSUE_DEFINITION_WM)/2)[0][0])


def plot_FC_example_patient_GMWMall(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index,
                                                 dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state , func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, params_plot,
                                                 xlim = [0,0.1]):
    FC_type = FC_TYPES[func]
    wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WM_index)   )
    gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GM_index)  )
    gmwm = np.concatenate([wm, gm])

    binwidth=0.025
    xlim = xlim

    #plot FC distributions, plot entire distribution (GM + WM)
    fig, axes = plt.subplots(figsize=(3, 4), dpi = 600)
    sns.histplot(gmwm, kde = True, color =(0.2, 0.2, 0.2) , ax = axes, binwidth =binwidth, line_kws=dict(lw=params_plot.LW_STD2));
    axes.set_xlim(xlim)
    axes.title.set_text(f"{FC_type}, {FREQUENCY_NAMES[freq]}, {sub}")
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_ylabel('Count')
    axes.set_xlabel(f'{FC_type} {FREQUENCY_NAMES[freq]}')
    #axes.set_yticks([200,400,600])

def plot_FC_example_patient_GMWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index,
                                                 dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state , func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, params_plot,
                                                 xlim = [0,0.1]):
    FC_type = FC_TYPES[func]
    wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WM_index)   )
    gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GM_index)  )
    gmwm =  utils.getAdjSubset(FC[state][freq], GM_index, WM_index).flatten()

    binwidth=0.025
    xlim = xlim

    #plot FC distributions, plot entire distribution (GM + WM)
    fig, axes = plt.subplots(figsize=(3, 4), dpi = 600)
    sns.histplot( wm, kde = True, color =(	0.463, 0.686, 0.875) , ax = axes, binwidth =binwidth, binrange = [-1,1], line_kws=dict(lw=params_plot.LW_STD2));
    sns.histplot(  gm, kde = True, color = (0.545, 0.439, 0.345) , ax = axes,  binwidth =binwidth, binrange = [-1,1], line_kws=dict(lw=params_plot.LW_STD2));
    sns.histplot(gmwm, kde = True, color =(0.608, 0.455, 0.816) , ax = axes, binwidth =binwidth, line_kws=dict(lw=params_plot.LW_STD2));
    axes.set_xlim(xlim)
    axes.title.set_text(f"{FC_type}, {FREQUENCY_NAMES[freq]}, {sub}")
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_ylabel('Count')
    axes.set_xlabel(f'{FC_type} {FREQUENCY_NAMES[freq]}')
    #axes.set_yticks([200,400,600])


def plot_FC_example_patient_GMvsWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index,
                                                 dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state , func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, params_plot,
                                                 xlim = [0,0.1]):
    #plot FC distributions, WM vs GM
    FC_type = FC_TYPES[func]
    wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WM_index)   )
    gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GM_index)  )


    binwidth=0.025
    xlim = xlim

    fig, axes = plt.subplots(figsize=(3, 4), dpi = 600)
    sns.histplot( wm, kde = True, color =(	0.463, 0.686, 0.875) , ax = axes, binwidth =binwidth, binrange = [-1,1], line_kws=dict(lw=params_plot.LW_STD2));
    sns.histplot(  gm, kde = True, color = (0.545, 0.439, 0.345) , ax = axes,  binwidth =binwidth, binrange = [-1,1], line_kws=dict(lw=params_plot.LW_STD2));
    axes.set_xlim(xlim); #axes.set_ylim([0, 200])
    axes.title.set_text(f"{FC_type}, {FREQUENCY_NAMES[freq]}, {sub}")
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_ylabel('Count')
    axes.set_xlabel(f'{FC_type} {FREQUENCY_NAMES[freq]}')
    #axes.set_yticks([100,200,300])

def plot_FC_example_patient_GMvsWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index,
                                                 dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state , func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, params_plot):
    #ECDF Plots GM vs WM single patient
    FC_type = FC_TYPES[func]
    wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WM_index)   )
    gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GM_index)  )

    print(stats.ks_2samp(gm,wm))
    xlim = [-0.0,0.6]
    fig, axes = plt.subplots(figsize=(5, 5), dpi = 600)
    sns.ecdfplot(data=wm, ax = axes, color = params_plot.COLORS_TISSUE_LIGHT_MED_DARK[1][1], lw = params_plot.LW_STD5)
    sns.ecdfplot(data= gm , ax = axes, color = params_plot.COLORS_TISSUE_LIGHT_MED_DARK[1][0], lw = params_plot.LW_STD5)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlabel(f'{FC_type} {FREQUENCY_NAMES[freq]}' )
    #axes.set_xlim([-0.4, 0.4]);
    #axes.set_xticks([-0.4,-0.2,0,0.2,0.4])
def plot_FC_example_patient_GMWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index,
                                                 dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state , func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, params_plot):
    #ECDF Plots GM vs WM single patient
    FC_type = FC_TYPES[func]
    wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WM_index)   )
    gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GM_index)  )
    gmwm =  utils.getAdjSubset(FC[state][freq], GM_index, WM_index).flatten()
    print(stats.ks_2samp(gm,wm))
    xlim = [-0.0,0.6]
    fig, axes = plt.subplots(figsize=(5, 5), dpi = 600)
    sns.ecdfplot(data=wm, ax = axes, color = params_plot.COLORS_TISSUE_LIGHT_MED_DARK[1][1], lw = params_plot.LW_STD5)
    sns.ecdfplot(data= gm , ax = axes, color = params_plot.COLORS_TISSUE_LIGHT_MED_DARK[1][0], lw = params_plot.LW_STD5)
    sns.ecdfplot(data= gmwm , ax = axes, color =(0.608, 0.455, 0.816), lw = params_plot.LW_STD5)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlabel(f'{FC_type} {FREQUENCY_NAMES[freq]}' )
    #axes.set_xlim([-0.4, 0.4]);
    #axes.set_xticks([-0.4,-0.2,0,0.2,0.4])

def plot_FC_all_patients_GMvsWM_ECDF(FCtissueAll_bootstrap_flatten, STATE_NUMBER , params_plot):
    xlim = [-0.0,0.6]
    fig, axes = plt.subplots(1,4,figsize=(8, 2.5), dpi = 600)
    for s in range(STATE_NUMBER):
        sns.ecdfplot(data = FCtissueAll_bootstrap_flatten[1][s], ax = axes[s], color = params_plot.COLORS_TISSUE_LIGHT_MED_DARK[1][0], ls = "-", lw = params_plot.LW_STD3)
        sns.ecdfplot(data = FCtissueAll_bootstrap_flatten[2][s] , ax = axes[s], color = params_plot.COLORS_TISSUE_LIGHT_MED_DARK[1][1], ls = "--", lw = params_plot.LW_STD3)
        axes[s].spines['top'].set_visible(False)
        axes[s].spines['right'].set_visible(False)
        axes[s].set_xlim(xlim);
        if s > 0: #dont fill in y axis
            axes[s].set_ylabel('')
            axes[s].get_yaxis().set_ticks([])


def plot_FC_all_patients_GMvsWM_ECDF_PVALUES(pvals, STATE_NUMBER , params_plot):
    fig, axes = plt.subplots(1,4,figsize=(10, 2.5), dpi = 600)
    xlim = [0,0.2]
    for s in range(STATE_NUMBER):
        sns.histplot(data = pvals[:,s], ax = axes[s], color = "#aaaaaa", kde = False, bins = np.arange(0,0.5,0.01))
        axes[s].spines['top'].set_visible(False)
        axes[s].spines['right'].set_visible(False)
        axes[s].set_xlim(xlim);
        if s == 0 or s == 1:
            axes[s].set_ylim([0,50])
        if s == 3:
            axes[s].set_ylim([0,200])
        print( len(np.where(pvals[:,s] <= 0.05)[0])/len(pvals))


def plot_FC_deltaT_PVALUES(pvals_deltaT, params_plot):
    fig, axes = plt.subplots(1,pvals_deltaT.shape[1],figsize=(2.5*pvals_deltaT.shape[1], 2.5), dpi = 600)
    xlim = [0,0.2]
    for s in range(pvals_deltaT.shape[1]):
        sns.histplot(data = pvals_deltaT[:,s], ax = axes[s], color = "#aaaaaa", kde = False, bins = np.arange(0,0.5,0.01))
        axes[s].spines['top'].set_visible(False)
        axes[s].spines['right'].set_visible(False)
        axes[s].set_xlim(xlim);
        if s == 0 or s== 1 or s == 3:
            axes[s].set_ylim([0,50])

        print( len(np.where(pvals_deltaT[:,s] <= 0.05)[0])/len(pvals_deltaT))







def plot_FC_vs_WM_cutoff_PVALUES(pvalues, binwidth, xlim, params_plot):
    fig, axes = utils.plot_make()
    sns.histplot( pvalues, color =(	0.463, 0.686, 0.875) , ax = axes, binwidth = binwidth, line_kws=dict(lw=params_plot.LW_STD5))
    axes.set_xlim(xlim); #axes.set_ylim([0, 200])
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_ylabel('Count')
    iterations = len(pvalues)
    print(len(np.where(pvalues < 0.05)[0])/iterations)


def plot_FC_vs_contact_distance(summaryStats_Wm_FC_bootstrap_func_freq_long_state, xlim = [3,100], ylim = [0,0.6]):
    fig, axes = utils.plot_make()
    x = summaryStats_Wm_FC_bootstrap_func_freq_long_state["WM_median_distance"]
    y = summaryStats_Wm_FC_bootstrap_func_freq_long_state["FC"]
    print(stats.spearmanr(x,y))
    sns.regplot(data=summaryStats_Wm_FC_bootstrap_func_freq_long_state, x = "WM_median_distance", y = "FC", scatter_kws = dict(s = 10), order = 1, logx = True)
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)


def plot_FC_vs_WM_cutoff(summaryStats_Wm_FC_bootstrap_func_freq_long_state):
    fig, axes = utils.plot_make()
    sns.lineplot(data=summaryStats_Wm_FC_bootstrap_func_freq_long_state, x = "WM_cutoff", y = "FC")
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
































