"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#%% 1/4 Imports: always run this
import sys
import os
import json
import copy
import time
import bct
import glob
import random
import pickle
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
from scipy import signal, stats
from itertools import repeat
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr
from os.path import join, splitext, basename
from revellLab.packages.dataClass import DataClassAtlases, DataClassCohortsBrainAtlas, DataClassJson
from revellLab.packages.utilities import utils
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.eeg.echobase import echobase
from revellLab.papers.seeg_GMvsWM import plot_GMvsWM
from revellLab.packages.imaging.tractography import tractography
#%% 2/4 Paths and File names: always run this

pathFile = "linuxAndy.json"
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
with open(join(revellLabPath, "paths", pathFile)) as f: paths = json.load(f)
with open(join(paths["iEEGdata"])) as f: jsonFile = json.load(f)
with open(paths["atlasfilesPath"]) as f: atlasfiles = json.load(f)
with open(paths["iEEGusernamePassword"]) as f: usernameAndpassword = json.load(f)
atlases = DataClassAtlases.atlases(atlasfiles)
metadata = DataClassJson.DataClassJson(jsonFile)
# Get iEEG.org username and password
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]


#%% 3/4 Paramters: always run this
saveFigures = False

SesImplant = "implant01"
acq = "3D"
ieegSpace = "T00"
montage = "bipolar"
fsds = 256 #sampleing frequency for down-sampled data
stateNum = 4 #number of states: interictal, preictal, ictal, postictal
WMdefinition = 1 #where is the binary classification of WM contacts in mm
WMdefinition2 = 0
WMdefinition3 = 4 #deeper WM regions
WMdefinitionPercent = 0.5
WMdefinitionPercent2 = 0.75
frequencyNames = ["Broadband", "delta", "theta", "alpha", "beta", "gammaLow", "gammaMid", "gammaHigh"]
FCtypes = ["pearson", "coherence", "crossCorrelation"]
#colors for plotting
colorsGMWM1 = ["#c6b4a5", "#b6d4ee"]
colorsGMWM2 = ["#a08269", "#76afdf"]
colorsGMWM3 = ["#544335", "#1f5785"]
colorsInterPreIctalPost = ["#6495ed", "#1f66e5", "#a42813", "#ef8978"]
colorsInterPreIctalPost2 = ["#417de9", "#1347a4", "#5f170b", "#e74c32"]
colorsInterPreIctalPost3 = ["#666666", "#666666", "#990000", "#666666"]
colorsInterPreIctalPost4 = ["#333333", "#333333", "#660000", "#333333"]
lw1 = 3
#%% 4/4 General Parameter calculation: always run this

patientsWithseizures = metadata.get_patientsWithSeizuresAndInterictal()
# calculate length of seizures
patientsWithseizures = pd.concat([patientsWithseizures, pd.DataFrame(
    patientsWithseizures["stop"] - patientsWithseizures["EEC"], columns=['length'])], axis=1)
N = len(patientsWithseizures)



#%% Graphing summary statistics of seizures and patient population

# how many seizures per patient
seizureCounts = patientsWithseizures.groupby(['subject']).count()
fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
sns.histplot(data=seizureCounts, x="idKey", binwidth=1, kde=True, ax=axes)
axes.set_xlim(1, None)
axes.set(xlabel='Number of Seizures', ylabel='Number of Patients',
         title="Distribution of Seizure Occurrences")
utils.savefig(f"{paths['figures']}/seizureSummaryStats/seizureCounts.pdf", saveFigures=saveFigures)

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

utils.savefig(f"{paths['figures']}/seizureSummaryStats/seizureLengthDistribution.pdf", saveFigures=saveFigures)


#%% Electrode and atlas localization

iEEGpatientList = np.unique(list(patientsWithseizures["subject"]))
iEEGpatientList = ["sub-" + s for s in iEEGpatientList]

atl.atlasLocalizationBIDSwrapper(iEEGpatientList,  paths['BIDS'], "PIER", SesImplant, ieegSpace, acq,  paths['freesurferReconAll'],  paths['atlasLocaliztion'],
                                 paths['atlases'], paths['atlasLabels'], paths['MNItemplate'], paths['MNItemplateBrain'], multiprocess=False, cores=12, rerun=False)



#%% EEG download and preprocessing of electrodes


#Download iEEG data

for i in range(0, 40):
    metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                        BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata.get_associatedInterictal(
        patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, username, password,
                          BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=False)


for i in range(40, 80):
    print(i)
    metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                        BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata.get_associatedInterictal(
        patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, username, password,
                          BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=False)


for i in range(80, len(patientsWithseizures)):
    metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                        BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata.get_associatedInterictal(
        patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, username, password,
                          BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=False)


#%% Power analysis calculation
paientList = pd.DataFrame(columns=["patient"])
for i in [3, 5, 16]: #range(3, N):
    paientList = paientList.append(
        dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
    # get data
    seizure, fs, ictalStartIndex, ictalStopIndex = metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                                                                       BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=True)
    interictal, fs = metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", patientsWithseizures["AssociatedInterictal"][i], username, password,
                                           BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=True)

    ###filtering and downsampling
    ictalStartIndexDS = int(ictalStartIndex * (fsds/fs))
    ictalStopIndexDS = int(ictalStopIndex * (fsds/fs))
    seizureLength = (ictalStopIndexDS-ictalStartIndexDS)/fsds
    _, _, _, seizureFilt, channels = echobase.preprocess(
        seizure, fs, fs, montage="bipolar", prewhiten=False)
    _, _, _, interictalFilt, _ = echobase.preprocess(
        interictal, fs, fs, montage="bipolar", prewhiten=False)
    seizureFiltDS = metadata.downsample(seizureFilt, fs, fsds)
    interictalFiltDS = metadata.downsample(interictalFilt, fs, fsds)

    nchan = seizureFiltDS.shape[1]
    # calculating power
    power = echobase.get_power(seizureFiltDS, fsds)
    powerII = echobase.get_power(interictalFiltDS, fsds)
    # interpolate power
    powerInterp = echobase.power_interpolate( power, powerII, 180, 180 + int(np.round(seizureLength)), length=200)
    # Get atlas localization and distances
    file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01",
                f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file):
        localization = pd.read_csv(file)
        localizationChannels = localization["channel"]
        localizationChannels = echobase.channel2std(
            np.array(localizationChannels))
    # getting distances
    dist = pd.DataFrame(channels, columns=["channels"])
    dist["distance"] = np.nan
    for ch in range(len(channels)):
        channelName = channels[ch]
        if any(channelName == localizationChannels):
            dist.iloc[ch, 1] = localization["distance_to_GM_millimeters"][np.where(
                channelName == localizationChannels)[0][0]]
        else:
            # if channel has no localization, then just assume GM.
            dist.iloc[ch, 1] = 0

    # definition of WM by distance
    GMindex = np.where(dist["distance"] <= WMdefinition)[0]
    WMindex = np.where(dist["distance"] > WMdefinition)[0]

    # power by GM vs WM
    if i == 3:  # initialize
        powerGM = np.nanmean(powerInterp[:, :, GMindex], axis=2)
        powerWM = np.nanmean(powerInterp[:, :, WMindex], axis=2)
        powerDistAll = []
        distAll = []
        SNRAll = []
    else:
        powerGM = np.dstack([powerGM,   np.nanmean(
            powerInterp[:, :, GMindex], axis=2)])
        powerWM = np.dstack([powerWM,   np.nanmean(
            powerInterp[:, :, WMindex], axis=2)])

    # power by distance
    powerII = echobase.get_power(interictalFiltDS, fsds, avg=True)
    powerPI = echobase.get_power(seizureFiltDS[0:ictalStartIndexDS], fsds, avg=True)
    powerIC = echobase.get_power(seizureFiltDS[ictalStartIndexDS:ictalStopIndexDS], fsds, avg=True)
    powerPO = echobase.get_power(seizureFiltDS[ictalStopIndexDS:], fsds, avg=True)
    power = [powerII, powerPI, powerIC, powerPO]
    # interpolate by distance
    distArr = np.array(dist["distance"])
    xNew = np.arange(distArr.min(), distArr.max(), step=0.01)
    powerDist = np.zeros(shape=(len(powerII), len(xNew), 4))
    for p in range(len(power)):
        for f in range(len(powerDist)):
            freqs = power[p][f, :]
            interp = interpolate.interp1d(distArr,  freqs, kind="slinear")
            powerDist[f, :, p] = interp(xNew)
                 
    powerDistAll.append(powerDist)

    
    #SNR
    noise = np.zeros(shape = (nchan))
    powerInterpIIAvg = np.nanmean(powerInterp[0:200, :,:], axis = 1)
    for ch in range(nchan):
        noise[ch] = simps(powerInterpIIAvg[:,ch], dx=1)/2

    SNR = np.zeros(shape = (powerInterp.shape[1], nchan))
    for ch in range(nchan):
        for s in range(powerInterp.shape[1]):
            SNR[s,ch] = simps(powerInterp[:,s, ch], dx=1)/noise[ch]
        
    SNRAll.append(SNR)
    distAll.append(distArr)
        
    # plotting for sanity checks
    if i > 3:
        nseiuzres = powerGM.shape[2]
        powerGMmean = np.nanmean(powerGM, axis=2)
        powerWMmean = np.nanmean(powerWM, axis=2)
        #padding power vs distance with NaNs
        powerDistAllSame = np.full( shape= ( powerDistAll[1].shape[0],   utils.findMaxDim(powerDistAll), 4, len(powerDistAll)) , fill_value= np.nan)
        for d in range(len(powerDistAll)):
            filler = powerDistAll[d]
            powerDistAllSame[:,:filler.shape[1],:, d] = filler
        powerDistAvg = np.nanmean(powerDistAllSame, axis=3)
        
        
        plot_GMvsWM.plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)

        print(f"\n\n\n\n\n\n\n\n\n{i}\n\n\n\n\n\n\n\n\n")
        plt.show()
        """
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gs = fig.add_gridspec(4, 4)
        axes = 4 * [None]
        for t in range(2,4):
            axes[t] = 4 * [None]
        axes[0] = fig.add_subplot(gs[0, :]); axes[1] = fig.add_subplot(gs[1, :])
        n = 3
        for t in range(4):
            axes[2][t] = fig.add_subplot(gs[2, t])
            axes[3][t] = fig.add_subplot(gs[3, t])

        sns.heatmap(np.log10(powerGMmean[0:110, :]), vmin=0, vmax=4, center=0, ax=axes[0],cbar = False); axes[0].invert_yaxis()
        sns.heatmap(np.log10(powerWMmean[0:110, :]), vmin=0, vmax=4, center=0,  ax=axes[1],cbar = False); axes[1].invert_yaxis()
        for t in range(4):
            sns.heatmap(np.log10(powerDist[0:40, :,t]),  ax=axes[2][t],cbar = False, vmin=-2, vmax=3.5)
            axes[2][t].invert_yaxis()

        for t in range(4):
            sns.heatmap(np.log10(powerDistAvg[0:40, :,t]),  ax=axes[3][t],cbar = False, vmin=-2, vmax=3.5)
            axes[3][t].invert_yaxis()
        
        """


#%% Power analysis plots

plot_GMvsWM.plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)
#plt.savefig(f"{paths['figures']}/GMvsWM/powerspectra.png")
#plt.savefig(f"{paths['figures']}/GMvsWM/powerspectra.pdf")

upperFreq = 60
GMmeanII = np.nanmean(powerGM[0:upperFreq, 0:200, :], axis=1)
GMmeanPI = np.nanmean(powerGM[0:upperFreq, 200:400, :], axis=1)
GMmeanIC = np.nanmean(powerGM[0:upperFreq, 400:600, :], axis=1)
GMmeanPO = np.nanmean(powerGM[0:upperFreq, 600:800, :], axis=1)

WMmeanII = np.nanmean(powerWM[0:upperFreq, 0:200, :], axis=1)
WMmeanPI = np.nanmean(powerWM[0:upperFreq, 200:400, :], axis=1)
WMmeanIC = np.nanmean(powerWM[0:upperFreq, 400:600, :], axis=1)
WMmeanPO = np.nanmean(powerWM[0:upperFreq, 600:800, :], axis=1)

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

dfAreaPlot = dfArea.groupby(["tissue", "seizure", "state"]).mean()
dfAreaPlot.reset_index(inplace=True)


fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
sns.boxplot(data=dfAreaPlot, x="state", y="power", hue="tissue", palette=colorsGMWM2, showfliers=False,
            ax=axes, color=colorsGMWM3, order=["interictal", "preictal", "ictal", "postictal"])
sns.stripplot(x="state", y="power", hue="tissue",  data=dfAreaPlot, palette=colorsGMWM3,
              dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])
# Set only one legend
handles, labels = axes.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(
    0.0, 1), loc=2, borderaxespad=0.5)
axes.set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
utils.savefig(f"{paths['figures']}/GMvsWM/boxplot_powerDifferences_bySeziure.pdf", saveFigures=saveFigures)


dfAreaPlot = dfArea.groupby(["tissue", "patient", "state"]).mean()
dfAreaPlot.reset_index(inplace=True)
fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
sns.boxplot(data=dfAreaPlot, x="state", y="power", hue="tissue", palette=colorsGMWM2, showfliers=False,
            ax=axes, color=colorsGMWM3, order=["interictal", "preictal", "ictal", "postictal"])
sns.stripplot(x="state", y="power", hue="tissue",  data=dfAreaPlot, palette=colorsGMWM3,
              dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])
# Set only one legend
handles, labels = axes.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(
    0.0, 1), loc=2, borderaxespad=0.5)
axes.set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
utils.savefig(f"{paths['figures']}/GMvsWM/boxplot_powerDifferences_byPatient.pdf", saveFigures=saveFigures)


#power analysis stats
dfArea = dfArea.groupby(["tissue", "patient", "state"]).mean()
dfArea.reset_index(inplace=True)
statsTable = pd.DataFrame(
    columns=["tissue_1", "state_1", "tissue_2", "state_2", "pvalue"])
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "ictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "ictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="WM", state_1="ictal",
                                    state_2="ictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "interictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "interictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="WM", state_1="interictal",
                                    state_2="interictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "preictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="WM", state_1="preictal",
                                    state_2="preictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "postictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "postictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="WM", state_1="postictal",
                                    state_2="postictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "ictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="GM", state_1="preictal",
                                    state_2="ictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "ictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="WM", tissue_2="WM", state_1="preictal",
                                    state_2="ictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "interictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="GM", state_1="preictal",
                                    state_2="interictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "interictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="WM", tissue_2="WM", state_1="preictal",
                                    state_2="interictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "GM"].loc[dfArea['state'] == "postictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="GM", tissue_2="GM", state_1="preictal",
                                    state_2="postictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "postictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="WM", tissue_2="WM", state_1="preictal",
                                    state_2="postictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
statsTable["corrected"] = statsTable["pvalue"] * len(statsTable)
statsTable["significance"] = statsTable["corrected"] <0.05
#statsTable.to_csv(f"{paths['figures']}/GMvsWM/powerDifferences.csv", index=False)



#%% Calculating functional connectivity


FCtype = "pearson"
paientList = pd.DataFrame(columns=["patient"])
for i in range(3, N): #[3, 5, 22]: #
    sub = patientsWithseizures["subject"][i]
    functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
    utils.checkPathAndMake(functionalConnectivityPath, functionalConnectivityPath)
    
    metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                        BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                        functionalConnectivityPath = functionalConnectivityPath, 
                                        secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                        fsds = 256, montage = montage, FCtype = FCtype)
    
    paientList = paientList.append(dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
   
#%%    
#Combine FC from the above saved calculation, and calculate differences, and network measures

paientList = pd.DataFrame(columns=["patient"])
summaryStats = pd.DataFrame( columns=["patient", "seizure_number", "FCtype", "frequency", "interictal", "preictal", "ictal", "postictal"] )
FCtissueAll = np.empty((len(FCtypes), stateNum, len(frequencyNames), 2),dtype=object)

for func in range(len(FCtypes)):
    for freq in range(len(frequencyNames)):
        for i in range(3, N):#[3, 5, 22]: #
            FCtype = FCtypes[func]
            sub = patientsWithseizures["subject"][i]
            print(f"{montage}, {FCtype}, {frequencyNames[freq]}, {sub}")
            paientList = paientList.append(dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
            functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
            utils.checkPathAndMake(functionalConnectivityPath, functionalConnectivityPath, printBOOL=False)
            
            channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                                BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                                functionalConnectivityPath = functionalConnectivityPath, 
                                                secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                                fsds = 256, montage = montage, FCtype = FCtype)
            #if cross correlation, take absolute value
            if func == 2 or func == 0:
                for per in range(len(FC)):
                    for f in range(len(FC[per])):
                        FC[per][f] = abs(FC[per][f])
            # Get atlas localization and distances
            file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
            if utils.checkIfFileExistsGlob(file, printBOOL=False):
                localization = pd.read_csv(file)
                localizationChannels = localization["channel"]
                localizationChannels = echobase.channel2std(
                    np.array(localizationChannels))
            # getting distances
            dist = pd.DataFrame(channels, columns=["channels"])
            dist["distance"] = np.nan
            for ch in range(len(channels)):# Getting distances of the channels
                channelName = channels[ch]
                if any(channelName == localizationChannels):
                    dist.iloc[ch, 1] = localization["percent_WM"][np.where(#percent_WM, distance_to_GM_millimeters
                        channelName == localizationChannels)[0][0]]
                else:
                    # if channel has no localization, then just assume GM.
                    dist.iloc[ch, 1] = 0
            
            GMindex = np.where(dist["distance"] <= WMdefinitionPercent)[0]
            WMindex = np.where(dist["distance"] > WMdefinitionPercent)[0]
            distOrderInd = np.array(np.argsort(dist["distance"]))
            distOrder = dist.iloc[distOrderInd].reset_index()
        
           
            #get FC values for just the GM-GM connections and WM-WM connections
            FCtissue = [None] *2
            for t in range(len(FCtissue)):
                FCtissue[t] = []
            for s in range(len(FC)):
                #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
                FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )   )
                FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )   )
                if i == 3:
                    FCtissueAll[func][s][freq][0] = utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)    )
                    FCtissueAll[func][s][freq][1] = utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)    )
                else:
                    FCtissueAll[func][s][freq][0] = np.concatenate([  FCtissueAll[func][s][freq][0] , utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )  ]   )
                    FCtissueAll[func][s][freq][1] = np.concatenate([  FCtissueAll[func][s][freq][1] , utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )  ]   )
            
            diff = np.array([np.nanmedian(k) for k in zip(FCtissue[1] )]) -  np.array([np.nanmedian(k) for k in zip(FCtissue[0] )])
            summaryStats = summaryStats.append(dict(patient = sub ,seizure_number = i, FCtype = FCtype, frequency = frequencyNames[freq], interictal = diff[0], preictal = diff[1], ictal = diff[2], postictal = diff[3]) , ignore_index=True   )
            """
            fig,ax = plt.subplots(2,4,figsize=(16,8), dpi = 300)
            for s in range(stateNum):
                sns.ecdfplot(data=FCtissue[0][s], ax = ax[0][s])
                sns.ecdfplot(data=FCtissue[1][s], ax = ax[0][s])
                
                sns.kdeplot(data=FCtissue[0][s], ax = ax[1][s])
                sns.kdeplot(data=FCtissue[1][s], ax = ax[1][s])
            
            fig.suptitle(f"{sub}, {FCtype}, {freq}")
            plt.show()
            time.sleep(0)
            """
summaryStatsLong = pd.melt(summaryStats, id_vars = ["patient", "frequency", "FCtype", "seizure_number"], var_name = "state", value_name = "FC")
summaryStatsLong = summaryStatsLong.groupby(['patient', "frequency", "FCtype", "state"]).mean()
summaryStatsLong.reset_index(inplace=True)


#%%

# All FC and all Frequency Boxplot of FC values for ii, pi, ic, and po states
g = sns.FacetGrid(summaryStatsLong, col="frequency", row = "FCtype",sharey=False, col_order=frequencyNames, row_order = FCtypes)
g.map(sns.boxplot, "state","FC", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = colorsInterPreIctalPost)
g.map(sns.stripplot, "state","FC", order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = colorsInterPreIctalPost2)
ylims = [ [-0.02, 0.04], [-0.005, 0.04] ,[-0.025, 0.125] ]
bonferroniFactor = len(FCtypes)*len(frequencyNames)
for func in range(len(FCtypes)):
    for f in range(len(frequencyNames)):
        if not f == 1: g.axes[func][f].set_ylim(ylims[func])
        
        v1 = summaryStatsLong.loc[summaryStatsLong['state'] == "ictal"].loc[summaryStatsLong['FCtype'] == FCtypes[func]].loc[summaryStatsLong['frequency'] == frequencyNames[f]]["FC"]
        v2 = summaryStatsLong.loc[summaryStatsLong['state'] == "preictal"].loc[summaryStatsLong['FCtype'] == FCtypes[func]].loc[summaryStatsLong['frequency'] == frequencyNames[f]]["FC"]
        pval = stats.wilcoxon(v1, v2)[1] #* bonferroniFactor #change bonferroniFactor here by multiplying the stats.wilcoxon
        print(pval)
        if pval <=0.05:
            g.axes[func][f].text(1.25,ylims[func][0]+ (ylims[func][1]- ylims[func][0])*0.75,"*", size= 40, weight = 1000)

        g.axes[func][f].set(ylabel='', xlabel='', title="")
        if func == 0: g.axes[func][f].set(ylabel=FCtypes[func], xlabel='', title=frequencyNames[f])
        if func  > 0 and f == 0: g.axes[func][f].set(ylabel=FCtypes[func], xlabel='', title="")
        if func == len(FCtypes)-1: g.axes[func][f].set_xticklabels(["inter", "pre", "ictal", "post"])
            
utils.savefig( f"{paths['figures']}/GMvsWM/boxplot_FCstateDifferences_bonferroniCorrected.pdf", saveFigures=False  )

#%%
#% Plot FC distributions for example patient

i=24 #22, 24 25 107 79  78, sub-RID0309: 39-51
sub = patientsWithseizures["subject"][i]
func = 2
FCtype = FCtypes[func]
freq = 7
print(f"{FCtype}, {frequencyNames[freq]}, {sub}") 
functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                    BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                    functionalConnectivityPath = functionalConnectivityPath, 
                                    secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                    fsds = 256, montage = "bipolar", FCtype = FCtype)
if func == 2 or func == 5:
    for per in range(len(FC)):
        for f in range(len(FC[per])):
            FC[per][f] = abs(FC[per][f])
# Get atlas localization and distances
file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
if utils.checkIfFileExistsGlob(file, printBOOL=False):
    localization = pd.read_csv(file)
    localizationChannels = localization["channel"]
    localizationChannels = echobase.channel2std(
        np.array(localizationChannels))
# getting distances
dist = pd.DataFrame(channels, columns=["channels"])
dist["distance"] = np.nan
for ch in range(len(channels)):# Getting distances of the channels
    channelName = channels[ch]
    if any(channelName == localizationChannels):
        dist.iloc[ch, 1] = localization["percent_WM"][np.where(  #percent_WM, distance_to_GM_millimeters
            channelName == localizationChannels)[0][0]]
    else:
        # if channel has no localization, then just assume GM.
        dist.iloc[ch, 1] = 0
# definition of WM by distance

GMindex = np.where(dist["distance"] <= WMdefinitionPercent)[0]
WMindex = np.where(dist["distance"] > WMdefinitionPercent)[0]
distOrderInd = np.array(np.argsort(dist["distance"]))
distOrder = dist.iloc[distOrderInd].reset_index()

   
#get FC values for just the GM-GM connections and WM-WM connections
FCtissue = [None] *2
FCall = []
for t in range(len(FCtissue)):
    FCtissue[t] = []
for s in range(len(FC)):
    #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
    FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )   )
    FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )   )
    FCall.append(  utils.getUpperTriangle(FC[s][freq]       )   )



state = 2


#plot heatmap, not ordered, ordered
FCexample = FC[state][freq] 
FCexampleOrdered =  utils.reorderAdj(FCexample, distOrder["index"])  
vmin = -0.1
vmax = 0.5
center = 0.2
utils.plot_adj_heatmap(FCexample, square=True, vmin = vmin, vmax = vmax, center = center, cmap = "mako")
utils.plot_adj_heatmap(FCexampleOrdered, square=True, vmin = vmin, vmax = vmax, center = center, cmap = "mako")
print(np.where(distOrder["distance"]>0)[0][0])
#print(np.where(distOrder["distance"]>1)[0][0])

wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WMindex)   )
gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GMindex)  )
gmwm = np.concatenate([wm, gm])
gmwmthresh = gmwm[np.where(gmwm > 0.01)]

binwidth=0.025
xlim = [-0.0,0.6]

#plot FC distributions, WM vs GM
plot1, axes1 = plt.subplots(figsize=(3, 3), dpi = 600)
sns.histplot( wm, kde = True, color =(	0.463, 0.686, 0.875) , ax = axes1, binwidth =binwidth, binrange = [-1,1]); 
sns.histplot(  gm, kde = True, color = (0.545, 0.439, 0.345) , ax = axes1,  binwidth =binwidth, binrange = [-1,1]); 
axes1.set_xlim(xlim); #axes1.set_ylim([0, 200])
axes1.title.set_text(f"{FCtype}, {frequencyNames[freq]}, {sub}")
axes1.spines['top'].set_visible(False)
axes1.spines['right'].set_visible(False)
axes1.set_ylabel('Count')
axes1.set_xlabel(f'{FCtype} {frequencyNames[freq]}')
utils.savefig(f"{paths['figures']}/GMvsWM/histplot_FC_GMvsWM.pdf", saveFigures=saveFigures)

#plot FC distributions, plot entire distribution (GM + WM)
plot2, axes2 = plt.subplots(figsize=(3, 3), dpi = 600)
sns.histplot(gmwm, kde = True, color =(0.2, 0.2, 0.2) , ax = axes2, binwidth =binwidth); 
axes2.set_xlim(xlim)
axes2.title.set_text(f"{FCtype}, {frequencyNames[freq]}, {sub}")
axes2.spines['top'].set_visible(False)
axes2.spines['right'].set_visible(False)
axes2.set_ylabel('Count')
axes2.set_xlabel(f'{FCtype} {frequencyNames[freq]}')
axes2.set_yticks([100,200,300,400])
utils.savefig(f"{paths['figures']}/GMvsWM/histplot_FC_allGMWM.pdf", saveFigures=saveFigures)

#ECDF Plots GM vs WM single patient
plot3, axes3 = plt.subplots(figsize=(7, 5), dpi = 600)  
sns.ecdfplot(data=wm, ax = axes3, color = colorsGMWM2[1], lw = lw1)
sns.ecdfplot(data= gm , ax = axes3, color = colorsGMWM2[0], lw = lw1)
axes3.spines['top'].set_visible(False)
axes3.spines['right'].set_visible(False)
axes3.set_xlabel(f'{FCtype} {frequencyNames[freq]}' )
#axes3.set_xlim([-0.4, 0.4]);
#axes3.set_xticks([-0.4,-0.2,0,0.2,0.4])
utils.savefig(f"{paths['figures']}/GMvsWM/ECDF_FC_GMvsWM_singlePatient.pdf", saveFigures=saveFigures)

#ECDF Plots GM vs WM all patients combined
#xlim = [0,0.5]
plot4, axes4 = plt.subplots(1,4,figsize=(8, 2.5), dpi = 600)
for s in range(stateNum):
    sns.ecdfplot(data = FCtissueAll[func][s][freq][0], ax = axes4[s], color = colorsGMWM2[0], ls = "-", lw = lw1)
    sns.ecdfplot(data = FCtissueAll[func][s][freq][1] , ax = axes4[s], color = colorsGMWM2[1], ls = "--", lw = lw1)
    axes4[s].spines['top'].set_visible(False)
    axes4[s].spines['right'].set_visible(False)
    axes4[s].set_xlim(xlim);
    if s > 0: #dont fill in y axis
        axes4[s].set_ylabel('')
        axes4[s].get_yaxis().set_ticks([])
    #if s ==2:      #fill in red between lines
    #    y0 = axes4[s].lines[0].get_xydata()[:,1]
    #    y1 = axes4[s].lines[1].get_xydata()[:,1]
    #    x0 = axes4[s].lines[0].get_xydata()[:,0]
    #    x1 = axes4[s].lines[1].get_xydata()[:,0]
    #    np.append(x0, x1[::-1]), np.append(y0, y1[::-1])
    #    axes4[s].fill(np.append(x0, x1[::-1]), np.append(y0, y1[::-1]),    color = colorsInterPreIctalPost3[2])
utils.savefig(f"{paths['figures']}/GMvsWM/ECDF_FC_GMvsWM_allPatient.pdf", saveFigures=saveFigures)

np.nanmedian(FCtissueAll[func][2][freq][1]) - np.nanmedian(FCtissueAll[func][2][freq][0])
np.nanmedian(FCtissueAll[func][1][freq][1]) - np.nanmedian(FCtissueAll[func][1][freq][0])
np.nanmedian(FCtissueAll[func][0][freq][1]) - np.nanmedian(FCtissueAll[func][0][freq][0])
np.nanmedian(FCtissueAll[func][3][freq][1]) - np.nanmedian(FCtissueAll[func][3][freq][0])

#colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#131e2a","#4a4aa2", "#cbcae7"])
#plt.figure(figsize=(5, 5), dpi = 600); sns.heatmap(distance_matrix, square=True, cmap = colormap, center = 30); plt.title("Distance Matrix \n{0} ".format(sub_RID ))

#All patinets boxplot delta T

df = summaryStatsLong.loc[summaryStatsLong['FCtype'] == FCtype].loc[summaryStatsLong['frequency'] == frequencyNames[freq]]
plot5, axes5 = plt.subplots(1,1,figsize=(7, 5), dpi = 600)
sns.boxplot( data= df, x = "state", y = "FC", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = colorsInterPreIctalPost3, ax = axes5)
sns.stripplot( data= df, x = "state", y = "FC",order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = colorsInterPreIctalPost4, ax = axes5)
axes5.spines['top'].set_visible(False)
axes5.spines['right'].set_visible(False)
axes5.set_ylim([-0.0125, 0.0425]);
utils.savefig(f"{paths['figures']}/GMvsWM/boxplot_FC_differences_deltaT_GMvsWM.pdf", saveFigures=saveFigures)

print(stats.ttest_1samp(np.array(df.loc[df["state"] == "ictal"]["FC"]), 0)[1])


#%%

summaryStatsLong = pd.melt(summaryStats, id_vars = ["patient", "seizure_number", "frequency", "FCtype"], var_name = "state", value_name = "FC")
#summaryStatsLong = summaryStatsLong.groupby(['patient', "frequency", "FCtype", "state"]).mean()
summaryStatsLong.reset_index(inplace=True)
df = summaryStatsLong.loc[summaryStatsLong['FCtype'] == FCtype].loc[summaryStatsLong['frequency'] == frequencyNames[freq]]
#Bootstrap, get unique
patient_unique = np.unique(df["patient"])

pval = []
pval2 = []
for sim in range(1000):
    ind = []
    for i in range(len(patient_unique)):
       seizures_unique = np.unique(df[df["patient"] == patient_unique[i]]["seizure_number"])
       if len(seizures_unique) > 2:
           ind.append(random.sample(list(seizures_unique.astype(int)), 2))
       elif len(seizures_unique) == 2:
           ind.append(random.sample(list(seizures_unique.astype(int)), 2))
       else:
           ind.append([seizures_unique[0]]) 
           
    ind = [item for sublist in ind for item in sublist]
    df_bootstrap = df[df['seizure_number'].isin(ind)]
    
    patinets_good_outcomes = ["RID0309", "RID0238", "RID0440", "RID0320", "RID0307", "RID0365", "RID0274", "RID0371"]
    patinets_poor_outcomes = ["RID0278", "RID0405", "RID0442", "RID0382"]
    df_good = df_bootstrap[df_bootstrap['patient'].isin(patinets_good_outcomes)]
    df_poor = df_bootstrap[df_bootstrap['patient'].isin(patinets_poor_outcomes)]
    
    tmp1 = df_good[df_good["state"] == "ictal"]["FC"]
    tmp2 = df_poor[df_poor["state"] == "ictal"]["FC"]
    
    stats.mannwhitneyu(tmp1,tmp2)[1]
    p = stats.ttest_ind(tmp1,tmp2)[1]
    print(p)
    pval.append(p)
    p2 = stats.ttest_1samp(np.array(df_bootstrap.loc[df_bootstrap["state"] == "ictal"]["FC"]), 0)[1]
    pval2.append(p2)
utils.plot_histplot(pval)
utils.plot_histplot(pval2)



df = summaryStatsLong.loc[summaryStatsLong['FCtype'] == FCtype].loc[summaryStatsLong['frequency'] == frequencyNames[freq]]
plot5, axes5 = plt.subplots(1,1,figsize=(7, 5), dpi = 600)
sns.boxplot( data= df_bootstrap, x = "state", y = "FC", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = colorsInterPreIctalPost3, ax = axes5)
sns.stripplot( data= df_bootstrap, x = "state", y = "FC",order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = colorsInterPreIctalPost4, ax = axes5)
axes5.spines['top'].set_visible(False)
axes5.spines['right'].set_visible(False)
axes5.set_ylim([-0.0125, 0.0425]);
utils.savefig(f"{paths['figures']}/GMvsWM/boxplot_FC_differences_deltaT_GMvsWM.pdf", saveFigures=saveFigures)


#%% Network analysis comparison


networkMeasures = pd.DataFrame( columns=["patient", "FCtype", "frequency", "state", "tissueType", "density", "characteristicPathLength", "transitivity", "degree", "clusteringCoefficient", "betweennessCentrality"] )

for i in range(3, N):#[3, 5, 22]: #
    sub = patientsWithseizures["subject"][i]
    # Get atlas localization and distances
    file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file, printBOOL=False):
        localization = pd.read_csv(file)
        localizationChannels = localization["channel"]
        localizationChannels = echobase.channel2std(
            np.array(localizationChannels))
    # getting distances
    dist = pd.DataFrame(channels, columns=["channels"])
    dist["distance"] = np.nan
    for ch in range(len(channels)):# Getting distances of the channels
        channelName = channels[ch]
        if any(channelName == localizationChannels):
            dist.iloc[ch, 1] = localization["distance_to_GM_millimeters"][np.where(
                channelName == localizationChannels)[0][0]]
        else:
            # if channel has no localization, then just assume GM.
            dist.iloc[ch, 1] = 0
    
    GMindex = np.where(dist["distance"] <= WMdefinition2)[0]
    WMindex = np.where(dist["distance"] > WMdefinition2)[0]
    distOrderInd = np.array(np.argsort(dist["distance"]))
    distOrder = dist.iloc[distOrderInd].reset_index()
    
    for func in range(len(FCtypes)):
        FCtype = FCtypes[func]
    
        functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
        channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                            BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                            functionalConnectivityPath = functionalConnectivityPath, 
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                            fsds = 256, montage = montage, FCtype = FCtype)
        
        for freq in range(len(frequencyNames)):
            print(f"{FCtype}, {frequencyNames[freq]}, {sub}")
            
            
            s=2
            #networks: 0 = all tissue, 1 = GM-GM, 2 = WM-WM, 3 = GM-WM
            networks = [ copy.deepcopy(FC[s][freq] ),  utils.reorderAdj(FC[s][freq], GMindex)  ,  utils.reorderAdj(FC[s][freq], WMindex) , utils.getAdjSubset(FC[s][freq], GMindex, WMindex)   ]
            networksThreshold = copy.deepcopy(networks)
            
            #thresholding
            thresholdPearson = 0.0
            for t in range(len(networksThreshold)):
                networksThreshold[t][np.where(networksThreshold[t] < thresholdPearson)] = 0
                

            binwidth = 0.01
            plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
            sns.histplot( utils.getUpperTriangle(networks[0]), color = (0.5, 0.5, 0.5), ax = axes  , binwidth =binwidth , binrange = [-1,1] , kde = True)
            sns.histplot( networks[3].flatten(), color = (0.9, 0.686, 0.875), ax = axes, binwidth =binwidth , binrange = [-1,1]  , kde = True )
            sns.histplot( utils.getUpperTriangle(networks[2]), color = (0.463, 0.686, 0.875), ax = axes , binwidth =binwidth , binrange = [-1,1] , kde = True )
            sns.histplot( utils.getUpperTriangle(networks[1]), color = (0.545, 0.439, 0.345), ax = axes , binwidth =binwidth  , binrange = [-1,1] , kde = True)
            axes.set_xlim([-0.6,0.6])

            sign = 0
            
            plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
            sns.kdeplot( bct.strengths_und_sign(networks[0])[sign], color = (0.5, 0.5, 0.5, 1) , ax = axes )
            sns.kdeplot( bct.strengths_und_sign(networks[1])[sign], color = (0.545, 0.439, 0.345), ax = axes  )
            sns.kdeplot( bct.strengths_und_sign(networks[2])[sign], color = (0.463, 0.686, 0.875) , ax = axes )
            plot.suptitle(f"{sub}, {FCtype}, {freq} Strengths, Sign: {sign}")
            
            plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
            sns.kdeplot( bct.clustering_coef_wu_sign(networks[0])[sign], color = (0.5, 0.5, 0.5, 1) , ax = axes )
            sns.kdeplot( bct.clustering_coef_wu_sign(networks[1])[sign], color = (0.545, 0.439, 0.345), ax = axes  )
            sns.kdeplot( bct.clustering_coef_wu_sign(networks[2])[sign], color = (0.463, 0.686, 0.875) , ax = axes )
            plot.suptitle(f"{sub}, {FCtype}, {freq} Clustering, Sign: {sign}")
            
            
            
            #modularity 
            
            modularity1 = bct.community_louvain(  networks[0] , B = "negative_sym" )
            bct.community_louvain(  networks[0] , B = "negative_sym" , ci = modularity1[0])
            utils.calculateModularity3(networks[0], modularity1[0] , B = "negative_sym")

            
            modularity2 = np.zeros(shape = (len(channels)))
            modularity2[GMindex] = 2
            modularity2[WMindex] = 1
            
            modularity2 = utils.calculateModularity3(networks[0], modularity2 , B = "negative_sym")
            modularity2
            
            modularity2[1] - modularity1[1] 
                 
            tmp = []
            it = 1000
            for k in range(it):
                tmp.append(  utils.calculateModularity3(networks[0], np.random.choice(2, size=len(channels), replace=True) , B = "negative_sym")[1]   )
                print(k)
          
            sns.histplot( np.array(tmp), color = (0.5, 0.5, 0.5)    , kde = True)
                           

            
            tmp = np.concatenate(  [ utils.getUpperTriangle(networks[2]), utils.getUpperTriangle(networks[1]) ,networks[3].flatten()  ]   )
            tmp2 = utils.getUpperTriangle(networks[0])
            binwidth = 0.01
            plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
            sns.histplot( tmp, color = (0.463, 0.686, 0.875), ax = axes  , binwidth =binwidth , fill = False )
            sns.histplot( tmp2, color = (0.5, 0.5, 0.5), ax = axes  , binwidth =binwidth , fill = False )










#%% Patient outcome




#Good 309, engel 1a   39-51  #51
#good 238, engel 1a   4-10  #8,9
#good 440, engel 1b   78-81  #78
#good 320, Engel 1B   52-54 #52

#fair 274, engel 2a, 20

#poor 278, engel 4c   21-24 #24
#poor 322, engel 4a   55-59 #57

#good 194, engel 1a   3
#poor RID0139 Engel 1d-2a-3a???

#poor 294: 3A    28
i=61


sub = patientsWithseizures["subject"][i]
func = 0
FCtype = FCtypes[func]
freq = 7
print(f"{FCtype}, {frequencyNames[freq]}, {sub}") 
functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                    BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                    functionalConnectivityPath = functionalConnectivityPath, 
                                    secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                    fsds = 256, montage = "bipolar", FCtype = FCtype)

# Get atlas localization and distances
file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
if utils.checkIfFileExistsGlob(file, printBOOL=False):
    localization = pd.read_csv(file)
    localizationChannels = localization["channel"]
    localizationChannels = echobase.channel2std(
        np.array(localizationChannels))
# getting distances
dist = pd.DataFrame(channels, columns=["channels"])
dist["distance"] = np.nan
for ch in range(len(channels)):# Getting distances of the channels
    channelName = channels[ch]
    if any(channelName == localizationChannels):
        dist.iloc[ch, 1] = localization["distance_to_GM_millimeters"][np.where(
            channelName == localizationChannels)[0][0]]
    else:
        # if channel has no localization, then just assume GM.
        dist.iloc[ch, 1] = 0
# definition of WM by distance

GMindex = np.where(dist["distance"] <= WMdefinition2)[0]
WMindex = np.where(dist["distance"] > WMdefinition2)[0]
distOrderInd = np.array(np.argsort(dist["distance"]))
distOrder = dist.iloc[distOrderInd].reset_index()

   
#get FC values for just the GM-GM connections and WM-WM connections
FCtissue = [None] *2
FCall = []
for t in range(len(FCtissue)):
    FCtissue[t] = []
for s in range(len(FC)):
    #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
    FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )   )
    FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )   )
    FCall.append(  utils.getUpperTriangle(FC[s][freq]       )   )



state = 2



adj = FC[state][freq]
manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
print(manual_resected_electrodes)


tmp = utils.reorderAdj(adj, manual_resected_electrodes_index)
sns.heatmap(tmp, vmin = -0.2, vmax = 0.4)
plt.show()
sns.heatmap(adj, vmin = -0.2, vmax = 0.4)
plt.show()
channels[manual_resected_electrodes_index]


tmp_wm = utils.getAdjSubset(adj, manual_resected_electrodes_index, WMindex)
sns.heatmap(tmp_wm, vmin = -0.2, vmax = 0.4)
plt.show()
tmp_gm = utils.getAdjSubset(adj, manual_resected_electrodes_index, GMindex)
sns.heatmap(tmp_gm, vmin = -0.2, vmax = 0.4)
plt.show()
np.nanmean(tmp_wm[1,:])
np.nanmean(tmp_wm)
np.nanmean(tmp_gm)
np.nanmedian(tmp_wm)
np.nanmedian(tmp_gm)

stats.mannwhitneyu(tmp_wm.flatten(), tmp_gm.flatten())
stats.ttest_ind(tmp_wm.flatten(), tmp_gm.flatten())

plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
sns.ecdfplot( tmp_wm.flatten(), color = (0.463, 0.686, 0.875) , ax = axes )
sns.ecdfplot( tmp_gm.flatten(), color = (0.545, 0.439, 0.345), ax = axes  )
plot.suptitle(f"{sub}, {FCtype}, {freq}")
axes.set_xlim([-0.2,0.4])



binwidth = 0.01
plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
sns.histplot(tmp_wm.flatten(), color = (0.463, 0.686, 0.875), ax = axes , binwidth =binwidth , binrange = [-1,1] , kde = True )
sns.histplot( tmp_gm.flatten(), color = (0.545, 0.439, 0.345), ax = axes , binwidth =binwidth  , binrange = [-1,1] , kde = True)
axes.set_xlim([-0.2,0.4])

 
plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
sns.kdeplot( tmp_wm.flatten(), color = (0.463, 0.686, 0.875) , ax = axes )
sns.kdeplot( tmp_gm.flatten(), color = (0.545, 0.439, 0.345), ax = axes  )
plot.suptitle(f"{sub}, {FCtype}, {freq}")
axes.set_xlim([-0.2,0.4])
 




tmp_gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex)


binwidth = 0.01
plot, axes = plt.subplots(1,1,figsize=(5, 5), dpi = 600)
sns.histplot(tmp_wm.flatten(), color = (0.463, 0.686, 0.875), ax = axes , binwidth =binwidth , binrange = [-1,1] , kde = True )
sns.histplot( tmp_gm.flatten(), color = (0.545, 0.439, 0.345), ax = axes , binwidth =binwidth  , binrange = [-1,1] , kde = True)
sns.histplot( tmp_gm_to_wm.flatten(), color = (0.5, 0.5, 0.5), ax = axes , binwidth =binwidth  , binrange = [-1,1] , kde = True)
axes.set_xlim([-0.2,0.4])


##


state = 2

adj = FC[state][freq]

wm = utils.getAdjSubset(adj, WMindex, WMindex)
wm_flat = utils.getUpperTriangle(wm)
gm = utils.getAdjSubset(adj, GMindex, GMindex)
gm_flat = utils.getUpperTriangle(gm)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
sns.heatmap(wm, vmin = -0.2, vmax = 0.4, ax = axes[0], square = True)
sns.heatmap(gm, vmin = -0.2, vmax = 0.4, ax = axes[1], square = True)
sns.heatmap(utils.reorderAdj(adj, distOrderInd), vmin = -0.2, vmax = 0.4, ax = axes[2], square = True)
plt.show()

#%%
patients_list = [3, 75 , 103, 104,105, 109, 110, 111, 112, 113, 115 ]
for i in patients_list:
    sub = patientsWithseizures["subject"][i]
    #01 DTI correction
    cmd = f"qsiprep-docker {join(paths['BIDS'], 'PIER')} {paths['qsiprep']} participant --output-resolution 1.5 --fs-license-file {paths['freesurferLicense']} -w {paths['qsiprep']} --participant_label {sub}"
    print(cmd)
    #os.system(cmd) cannot run this docker command inside python interactive shell. Must run the printed command in terminal using qsiprep environment in mad in environment directory
    
    ses="research3T*"
    pathDWI = join(paths['qsiprep'], "qsiprep", f"sub-{sub}", f"ses-{ses}", "dwi", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
    pathTracts = join(paths['tractography'], f"sub-{sub}", "tracts")
    utils.checkPathAndMake(pathTracts,pathTracts)
    utils.checkIfFileExistsGlob(  pathDWI)
    
    pathDWI = glob.glob(pathDWI)[0]
    
    trkName = f"{join(pathTracts, utils.baseSplitextNiiGz(pathDWI)[2])}.trk.gz"
    if False:
        tractography.getTracts(paths['BIDS'], paths['dsiStudioSingularity'], pathDWI, pathTracts)
    
    


#make electrode specific ROIs




#radii = [1, 2,5, 10, 15, 25, 40]



    
    # Get atlas localization and distances
    file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file, printBOOL=False):
        localization = pd.read_csv(file)
        localizationChannels = localization["channel"]
        localizationChannels = echobase.channel2std(
        np.array(localizationChannels))
    
    ses = "implant*"
    #t1_image = join(paths['BIDS'], "PIER", f"sub-{sub}", f"ses-{ses}", "anat", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
    t1_image = glob.glob(join(paths['atlasLocaliztion'], f"sub-{sub}", f"ses-{ses}", "tmp", "orig_nu_std.nii.gz" ))[0]
    img = nib.load(t1_image)
    utils.show_slices(img, data_type = "img")
    img_data = img.get_fdata() 
    affine = img.affine
    shape = img_data.shape
    
    coordinates = np.array(localization[["x", "y", "z"]])
      
    coordinates_voxels = utils.transform_coordinates_to_voxel(coordinates, affine)
    
    path_spheres = join(paths['tractography'], f"sub-{sub}", "electrodeContactSphereROIs")
    utils.checkPathAndMake(path_spheres,path_spheres)
    
    for e in range(len(localizationChannels)):
        
        img_data_sphere = copy.deepcopy(img_data)
        img_data_sphere[  np.where(img_data_sphere != 0)  ] = 0
        
        x = coordinates_voxels[e][0]
        y = coordinates_voxels[e][1]
        z = coordinates_voxels[e][2]
        img_data_sphere = utils.make_sphere_from_point(img_data_sphere, x, y, z, radius = 7) #radius = 7mm
        
        fname_ROIs_sub_ID = join(path_spheres, f"sub-{sub}_ses-implant01_desc-{localizationChannels[e]}.nii.gz")
        img_sphere = nib.Nifti1Image(img_data_sphere, img.affine)
        nib.save(img_sphere, fname_ROIs_sub_ID)
        
        #utils.show_slices(img_data_sphere, data_type = "data")
        utils.printProgressBar(e, len(localizationChannels))
    
    
#get structural connectivity

#%% Analyzing SFC 

patients_list = [3, 75 , 103, 104,105, 109, 110, 111, 112, 113, 115 ]
#194: 3
#278: 21-24
#309: 39-51
#320: 52-54
#365: 60-65
#420: 75
#440: 78-81
#502: 103
#508: 104
#536: 105-108
#572: 109
#583: 110
#595: 111
#596: 112
#648: 113
#652: 115-116
np.unique(np.array(summaryStats.loc[summaryStats["patient"] == "RID0" + "652"]["seizure_number"])).astype(int)
patients_with_SC = ["RID0194", "RID0278","RID0309","RID0320","RID0365","RID0420","RID0440","RID0502","RID0508","RID0536","RID0572","RID0583","RID0595","RID0596","RID0648","RID0652"]
#%%
#bootstrapping patients and seizures
bootstrap_ind = []
for i in range(len(patients_with_SC)):
    ind = np.unique(summaryStats.loc[summaryStats["patient"] == patients_with_SC[i]]["seizure_number"])
    if len(ind)> 2:
        bootstrap_ind.append(random.sample(list(ind), 2))
    if len(ind) > 1:
        bootstrap_ind.append(random.sample(list(ind), 1))
    else:
        bootstrap_ind.append(list(ind))
bootstrap_ind = [item for sublist in bootstrap_ind for item in sublist]     


all_corr = np.zeros((len(bootstrap_ind),4))
all_corr_gm = np.zeros((len(bootstrap_ind),4))
all_corr_wm = np.zeros((len(bootstrap_ind),4))
all_corr_gmwm = np.zeros((len(bootstrap_ind),4))
for b in range(len(bootstrap_ind)):
    i = bootstrap_ind[b]
    sub = patientsWithseizures["subject"][i]
    print(f"{sub}: {i}")
    path_SC = join(paths['tractography'], f"sub-{sub}", "connectivity", "electrodeContactAtlas")
    
    
    path_SC_contacts = glob.glob(join(path_SC, "*.txt"))[0]
    SC = utils.read_DSI_studio_Txt_files_SC(path_SC_contacts)
    
    SC_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(path_SC_contacts)
    
    
    SC = utils.log_normalize_adj(SC)
    
    #utils.plot_adj_heatmap(SC)
    
    
    ##%% Look at FC for SC
    
    sub = patientsWithseizures["subject"][i]
    func = 0
    FCtype = FCtypes[func]
    freq = 7
    print(f"{FCtype}, {frequencyNames[freq]}, {sub}") 
    functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
    channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                        BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                        functionalConnectivityPath = functionalConnectivityPath, 
                                        secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                        fsds = 256, montage = "bipolar", FCtype = FCtype)
    if func == 0 or func == 2:
        for per in range(len(FC)):
            for f in range(len(FC[per])):
                FC[per][f] = abs(FC[per][f])
    # Get atlas localization and distances
    file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file, printBOOL=False):
        localization = pd.read_csv(file)
        localizationChannels = localization["channel"]
        localizationChannels = echobase.channel2std(
            np.array(localizationChannels))
    # getting distances
    dist = pd.DataFrame(channels, columns=["channels"])
    dist["distance"] = np.nan
    for ch in range(len(channels)):# Getting distances of the channels
        channelName = channels[ch]
        if any(channelName == localizationChannels):
            dist.iloc[ch, 1] = localization["percent_WM"][np.where(#percent_WM, distance_to_GM_millimeters
                channelName == localizationChannels)[0][0]]
        else:
            # if channel has no localization, then just assume GM.
            dist.iloc[ch, 1] = 0
    # definition of WM by distance
    WMdef = WMdefinitionPercent2
    GMindex = np.where(dist["distance"] <= WMdef)[0]
    WMindex = np.where(dist["distance"] > WMdef)[0]
    
            
    manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
    manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
    manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
    

       
    #get FC values for just the GM-GM connections and WM-WM connections
    FCtissue = [None] *3
    FCall = []
    for t in range(len(FCtissue)):
        FCtissue[t] = []
    for s in range(len(FC)):
        #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
        FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )   )
        FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )   )
        FCtissue[2].append(   utils.getAdjSubset(FC[s][freq], GMindex, WMindex).flatten()   )
        
        FCall.append(  utils.getUpperTriangle(FC[s][freq]       )   )
    
    
    #fig, axes = utils.plot_make()
    corr = []
    corr_gm = []
    corr_wm = []
    corr_gmwm = []
    for s in range(len(FC)):
        state = s
        adj = copy.deepcopy(FC[state][freq])
        order = utils.get_intersect1d_original_order(channels, SC_names)
        SC_order = utils.reorderAdj(SC, order)
        missing_delete_in_FC = utils.find_missing(channels, SC_names).astype(int) 
        adj = np.delete(adj, missing_delete_in_FC, 0)
        adj = np.delete(adj, missing_delete_in_FC, 1) #making sure both SC and FC have the same rows and columns represented
        #utils.plot_adj_heatmap(SC_order)
        #utils.plot_adj_heatmap(adj)
    
        corr.append( spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj))[0])
        #sns.regplot(x = utils.getUpperTriangle(SC_order), y = utils.getUpperTriangle(adj), scatter_kws={'s':2}, ax = axes)
        #plt.title(f"SFC {sub}")
        #plt.xlabel("SC")
        #plt.ylabel(f"FC ({FCtype} {frequencyNames[freq]})")
        #fig.legend(labels = [ f"ii: {np.round(corr[0],3)}", f"pi: {np.round(corr[1],3)}", f"ic: {np.round(corr[2],3)}", f"po: {np.round(corr[3],3)}"])
        
        if False:
            adj = bct.null_model_und_sign(adj)[0] #utils.plot_adj_heatmap(adj); utils.plot_adj_heatmap(adj_rand)
            SC_order = bct.null_model_und_sign(SC_order)[0] #utils.plot_adj_heatmap(SC_order); utils.plot_adj_heatmap(adj)
        #SFC for tissue
        dist_new = dist.drop(missing_delete_in_FC) #need new index of GM and WM because deleted FC channels that were not in SC
        GMindex = np.where(dist_new["distance"] <= WMdef)[0]
        WMindex = np.where(dist_new["distance"] > WMdef)[0]
    
        adj_gm =  utils.reorderAdj(adj, GMindex)   
        SC_order_gm = utils.reorderAdj(SC_order, GMindex)   
        adj_wm =  utils.reorderAdj(adj, WMindex)   
        SC_order_wm = utils.reorderAdj(SC_order, WMindex)   
        adj_gmwm =  utils.getAdjSubset(adj, GMindex, WMindex)
        SC_order_gmwm = utils.getAdjSubset(SC_order, GMindex, WMindex)
        #fig, axes = utils.plot_make(c =3)
        #sns.regplot(x = utils.getUpperTriangle(SC_order_gm), y = utils.getUpperTriangle(adj_gm), scatter_kws={'s':2}, ax = axes[0])
        #sns.regplot(x = utils.getUpperTriangle(SC_order_wm), y = utils.getUpperTriangle(adj_wm), scatter_kws={'s':2}, ax = axes[1])
        #sns.regplot(x =SC_order_gmwm.flatten(), y = adj_gmwm.flatten(), scatter_kws={'s':2}, ax = axes[2])
        
        
        corr_gm.append(spearmanr(utils.getUpperTriangle(SC_order_gm), utils.getUpperTriangle(adj_gm))[0])
        corr_wm.append(spearmanr(utils.getUpperTriangle(SC_order_wm), utils.getUpperTriangle(adj_wm))[0])
        corr_gmwm.append(spearmanr(SC_order_gmwm.flatten(), adj_gmwm.flatten())[0])
    all_corr[b,:] = corr
    all_corr_gm[b,:] = corr_gm
    all_corr_wm[b,:] = corr_wm
    all_corr_gmwm[b,:] = corr_gmwm

print(stats.ttest_rel(all_corr[:,2] ,all_corr[:,1])[1]*4)
print(stats.ttest_rel(all_corr_gm[:,2] ,all_corr_gm[:,1])[1]*4)
print(stats.ttest_rel(all_corr_wm[:,2] ,all_corr_wm[:,1])[1]*4)
print(stats.ttest_rel(all_corr_gmwm[:,2] ,all_corr_gmwm[:,1])[1]*4)


#%%
all_corr[:,2]
all_corr[:,1]
xlim = [-0.2, 0.3]
fig, axes = utils.plot_make(r =4)
sns.boxplot(x = all_corr[:,2] - all_corr[:,1] ,ax = axes[0]   )    
sns.boxplot(x =all_corr_gm[:,2] - all_corr_gm[:,1] ,ax = axes[1]   )    
sns.boxplot(x =all_corr_wm[:,2] - all_corr_wm[:,1] ,ax = axes[2]   )    
sns.boxplot(x = all_corr_gmwm[:,2] - all_corr_gmwm[:,1] ,ax = axes[3]   )    
sns.swarmplot(x = all_corr[:,2] - all_corr[:,1], ax = axes[0], color = "red")
sns.swarmplot(x =all_corr_gm[:,2] - all_corr_gm[:,1] ,ax = axes[1] , color = "red"  )    
sns.swarmplot(x =all_corr_wm[:,2] - all_corr_wm[:,1] ,ax = axes[2]  , color = "red" )    
sns.swarmplot(x = all_corr_gmwm[:,2] - all_corr_gmwm[:,1] ,ax = axes[3]  , color = "red" )    
axes[0].set_xlim(xlim)
axes[1].set_xlim(xlim)
axes[2].set_xlim(xlim)
axes[3].set_xlim(xlim)


print(stats.ttest_1samp(all_corr[:,2] - all_corr[:,1], 0)[1] *4)
print(stats.ttest_1samp(all_corr_gm[:,2] - all_corr_gm[:,1], 0)[1] *4)
print(stats.ttest_1samp(all_corr_wm[:,2] - all_corr_wm[:,1], 0)[1] *4)
print(stats.ttest_1samp(all_corr_gmwm[:,2] - all_corr_gmwm[:,1], 0)[1] * 4)


#%%
all_corr_rand = copy.deepcopy(all_corr)
all_corr_gm_rand = copy.deepcopy(all_corr_gm)
all_corr_wm_rand = copy.deepcopy(all_corr_wm)
all_corr_gmwm_rand = copy.deepcopy(all_corr_gmwm)

stats.ttest_ind(all_corr[:,2] - all_corr[:,1], all_corr_rand[:,2] - all_corr_rand[:,1])[1]
stats.ttest_ind(all_corr_gm[:,2] - all_corr_gm[:,1], all_corr_gm_rand[:,2] - all_corr_gm_rand[:,1])[1]
stats.ttest_ind(all_corr_wm[:,2] - all_corr_wm[:,1], all_corr_wm_rand[:,2] - all_corr_wm_rand[:,1])[1]
stats.ttest_ind(all_corr_gmwm[:,2] - all_corr_gmwm[:,1], all_corr_gmwm_rand[:,2] - all_corr_gmwm_rand[:,1])[1]

#%%
#seeing if removing resected electrodes secreases the change in correlation

#bootstrapping patients and seizures
bootstrap_ind = []
for i in range(len(patients_with_SC)):
    ind = np.unique(summaryStats.loc[summaryStats["patient"] == patients_with_SC[i]]["seizure_number"])
    if len(ind)> 2:
        bootstrap_ind.append(random.sample(list(ind), 2))
    if len(ind) > 1:
        bootstrap_ind.append(random.sample(list(ind), 1))
    else:
        bootstrap_ind.append(list(ind))
bootstrap_ind = [item for sublist in bootstrap_ind for item in sublist]     


all_corr = np.zeros((len(bootstrap_ind),4))
all_corr_gm = np.zeros((len(bootstrap_ind),4))
all_corr_wm = np.zeros((len(bootstrap_ind),4))
all_corr_gmwm = np.zeros((len(bootstrap_ind),4))
all_corr_ablated = np.zeros((len(bootstrap_ind),4))
all_corr_gm_ablated = np.zeros((len(bootstrap_ind),4))
all_corr_wm_ablated = np.zeros((len(bootstrap_ind),4))
all_corr_gmwm_ablated = np.zeros((len(bootstrap_ind),4))
count = 0
for b in range(len(bootstrap_ind)):
    i = bootstrap_ind[b]
    sub = patientsWithseizures["subject"][i]
    print(f"{sub}: {i}")
    path_SC = join(paths['tractography'], f"sub-{sub}", "connectivity", "electrodeContactAtlas")
    
    
    path_SC_contacts = glob.glob(join(path_SC, "*.txt"))[0]
    SC = utils.read_DSI_studio_Txt_files_SC(path_SC_contacts)
    
    SC_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(path_SC_contacts)
    
    
    SC = utils.log_normalize_adj(SC)
    
    #utils.plot_adj_heatmap(SC)
    
    
    ##%% Look at FC for SC
    
    sub = patientsWithseizures["subject"][i]
    func = 2
    FCtype = FCtypes[func]
    freq = 5
    print(f"{FCtype}, {frequencyNames[freq]}, {sub}") 
    functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
    channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                        BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                        functionalConnectivityPath = functionalConnectivityPath, 
                                        secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                        fsds = 256, montage = "bipolar", FCtype = FCtype)
    if func == 0 or func == 2:
        for per in range(len(FC)):
            for f in range(len(FC[per])):
                FC[per][f] = abs(FC[per][f])
    # Get atlas localization and distances
    file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file, printBOOL=False):
        localization = pd.read_csv(file)
        localizationChannels = localization["channel"]
        localizationChannels = echobase.channel2std(
            np.array(localizationChannels))
    # getting distances
    dist = pd.DataFrame(channels, columns=["channels"])
    dist["distance"] = np.nan
    for ch in range(len(channels)):# Getting distances of the channels
        channelName = channels[ch]
        if any(channelName == localizationChannels):
            dist.iloc[ch, 1] = localization["percent_WM"][np.where(#percent_WM, distance_to_GM_millimeters
                channelName == localizationChannels)[0][0]]
        else:
            # if channel has no localization, then just assume GM.
            dist.iloc[ch, 1] = 0
    # definition of WM by distance
    WMdef = WMdefinitionPercent2
    GMindex = np.where(dist["distance"] <= WMdef)[0]
    WMindex = np.where(dist["distance"] > WMdef)[0]
    
            
    manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
    manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
    manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
    
    if len(manual_resected_electrodes)>0:
       

        
        #fig, axes = utils.plot_make()
        corr = []
        corr_gm = []
        corr_wm = []
        corr_gmwm = []
        corr_ablated = []
        corr_gm_ablated = []
        corr_wm_ablated = []
        corr_gmwm_ablated = []
        for s in range(len(FC)):
            state = s
            adj = copy.deepcopy(FC[state][freq])
            order = utils.get_intersect1d_original_order(channels, SC_names)
            SC_order = utils.reorderAdj(SC, order)
            missing_delete_in_FC = utils.find_missing(channels, SC_names).astype(int) 
            adj = np.delete(adj, missing_delete_in_FC, 0)
            adj = np.delete(adj, missing_delete_in_FC, 1) #making sure both SC and FC have the same rows and columns represented
            #utils.plot_adj_heatmap(SC_order)
            #utils.plot_adj_heatmap(adj)
        
            corr.append( spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj))[0])
            #sns.regplot(x = utils.getUpperTriangle(SC_order), y = utils.getUpperTriangle(adj), scatter_kws={'s':2}, ax = axes)
            #plt.title(f"SFC {sub}")
            #plt.xlabel("SC")
            #plt.ylabel(f"FC ({FCtype} {frequencyNames[freq]})")
            #fig.legend(labels = [ f"ii: {np.round(corr[0],3)}", f"pi: {np.round(corr[1],3)}", f"ic: {np.round(corr[2],3)}", f"po: {np.round(corr[3],3)}"])
            
            
            #SFC for tissue
            dist_new = dist.drop(missing_delete_in_FC).reset_index() #need new index of GM and WM because deleted FC channels that were not in SC
            GMindex = np.where(dist_new["distance"] <= WMdef)[0]
            WMindex = np.where(dist_new["distance"] > WMdef)[0]
        
            adj_gm =  utils.reorderAdj(adj, GMindex)   
            SC_order_gm = utils.reorderAdj(SC_order, GMindex)   
            adj_wm =  utils.reorderAdj(adj, WMindex)   
            SC_order_wm = utils.reorderAdj(SC_order, WMindex)   
            adj_gmwm =  utils.getAdjSubset(adj, GMindex, WMindex)
            SC_order_gmwm = utils.getAdjSubset(SC_order, GMindex, WMindex)
            #fig, axes = utils.plot_make(c =3)
            #sns.regplot(x = utils.getUpperTriangle(SC_order_gm), y = utils.getUpperTriangle(adj_gm), scatter_kws={'s':2}, ax = axes[0])
            #sns.regplot(x = utils.getUpperTriangle(SC_order_wm), y = utils.getUpperTriangle(adj_wm), scatter_kws={'s':2}, ax = axes[1])
            #sns.regplot(x =SC_order_gmwm.flatten(), y = adj_gmwm.flatten(), scatter_kws={'s':2}, ax = axes[2])
            
            
            corr_gm.append(spearmanr(utils.getUpperTriangle(SC_order_gm), utils.getUpperTriangle(adj_gm))[0])
            corr_wm.append(spearmanr(utils.getUpperTriangle(SC_order_wm), utils.getUpperTriangle(adj_wm))[0])
            corr_gmwm.append(spearmanr(SC_order_gmwm.flatten(), adj_gmwm.flatten())[0])
            
            ablated_ind = utils.get_intersect1d_original_order( manual_resected_electrodes,dist.channels)
            ablated_ind = random.sample(range(len(dist.channels)), len(manual_resected_electrodes))
            adj_ablated = np.delete(adj, ablated_ind, 0)
            adj_ablated = np.delete(adj, ablated_ind, 1) #resection
            SC_order_ablated = np.delete(SC_order, ablated_ind, 0)
            SC_order_ablated = np.delete(SC_order, ablated_ind, 1) # utils.plot_adj_heatmap(SC_order)
            corr_ablated.append( spearmanr(utils.getUpperTriangle(SC_order_ablated), utils.getUpperTriangle(adj_ablated))[0])
            dist_new_ablated = dist.drop(ablated_ind).reset_index() #need new index of GM and WM because deleted FC channels that were not in SC
            GMindex_ablated = np.where(dist_new_ablated["distance"] <= WMdef)[0]
            WMindex_ablated = np.where(dist_new_ablated["distance"] > WMdef)[0]
            adj_gm_ablated =  utils.reorderAdj(adj_ablated, GMindex_ablated)   
            SC_order_gm_ablated = utils.reorderAdj(SC_order_ablated, GMindex_ablated)   
            adj_wm_ablated =  utils.reorderAdj(adj_ablated, WMindex_ablated)   
            SC_order_wm_ablated = utils.reorderAdj(SC_order_ablated, WMindex_ablated)   
            adj_gmwm_ablated =  utils.getAdjSubset(adj_ablated, GMindex_ablated, WMindex_ablated)
            SC_order_gmwm_ablated = utils.getAdjSubset(SC_order_ablated, GMindex_ablated, WMindex_ablated)
            #fig, axes = utils.plot_make(c =3)
            #sns.regplot(x = utils.getUpperTriangle(SC_order_gm), y = utils.getUpperTriangle(adj_gm), scatter_kws={'s':2}, ax = axes[0])
            #sns.regplot(x = utils.getUpperTriangle(SC_order_wm), y = utils.getUpperTriangle(adj_wm), scatter_kws={'s':2}, ax = axes[1])
            #sns.regplot(x =SC_order_gmwm.flatten(), y = adj_gmwm.flatten(), scatter_kws={'s':2}, ax = axes[2])
            
            
            corr_gm_ablated.append(spearmanr(utils.getUpperTriangle(SC_order_gm_ablated), utils.getUpperTriangle(adj_gm_ablated))[0])
            corr_wm_ablated.append(spearmanr(utils.getUpperTriangle(SC_order_wm_ablated), utils.getUpperTriangle(adj_wm_ablated))[0])
            corr_gmwm_ablated.append(spearmanr(SC_order_gmwm_ablated.flatten(), adj_gmwm_ablated.flatten())[0])
        all_corr[count,:] = corr
        all_corr_gm[count,:] = corr_gm
        all_corr_wm[count,:] = corr_wm
        all_corr_gmwm[count,:] = corr_gmwm
        all_corr_ablated[count,:] = corr_ablated
        all_corr_gm_ablated[count,:] = corr_gm_ablated
        all_corr_wm_ablated[count,:] = corr_wm_ablated
        all_corr_gmwm_ablated[count,:] = corr_gmwm_ablated
        count = count +1

all_corr = np.delete(all_corr, range(count, len(bootstrap_ind)), axis = 0)
all_corr_gm = np.delete(all_corr_gm, range(count, len(bootstrap_ind)), axis = 0)
all_corr_wm = np.delete(all_corr_wm, range(count, len(bootstrap_ind)), axis = 0)
all_corr_gmwm = np.delete(all_corr_gmwm, range(count, len(bootstrap_ind)), axis = 0)
all_corr_ablated = np.delete(all_corr_ablated, range(count, len(bootstrap_ind)), axis = 0)
all_corr_gm_ablated = np.delete(all_corr_gm_ablated, range(count, len(bootstrap_ind)), axis = 0)
all_corr_wm_ablated = np.delete(all_corr_wm_ablated, range(count, len(bootstrap_ind)), axis = 0)
all_corr_gmwm_ablated = np.delete(all_corr_gmwm_ablated, range(count, len(bootstrap_ind)), axis = 0)


tmp1 = all_corr_gm[:,0] 
tmp2 = all_corr_gm_ablated[:,0] 

xlim = [-0.0, 0.5]
fig, axes = utils.plot_make(r =2)
sns.boxplot(x = tmp1 ,ax = axes[0]   )    
sns.boxplot(x =tmp2 ,ax = axes[1]   )   
sns.swarmplot(x = tmp1 ,ax = axes[0], color = "red"   )    
sns.swarmplot(x =tmp2 ,ax = axes[1] , color = "red"   )   
axes[0].set_xlim(xlim)
axes[1].set_xlim(xlim)
np.nanmean(tmp1)
np.nanmean(tmp2)

stats.ttest_rel(tmp1, tmp2)[1]*4
#%%


LF01_ind = np.where("LF01" == channels)[0]







LF01_wm_fc = utils.getAdjSubset(adj, LF01_ind, WMindex)


np.nanmean(LF01_wm_fc)




gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex)
gm_to_wm_average = np.nanmean(gm_to_wm, axis = 1)



sns.histplot(gm_to_wm_average, bins = 20)





state = 2
adj_ic = FC[state][freq]
state = 1
adj_pi = FC[state][freq]

adj_delta = adj_ic - adj_pi

utils.plot_adj_heatmap(adj_ic)
utils.plot_adj_heatmap(adj_delta)



sns.regplot(x = utils.getUpperTriangle(SC_order), y = utils.getUpperTriangle(adj_delta), scatter_kws={'s':2})
spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj_delta))
LF01_wm_fc_delta = utils.getAdjSubset(adj_delta, LF01_ind, WMindex)
np.nanmean(LF01_wm_fc_delta)
gm_to_wm_delta = utils.getAdjSubset(adj_delta, GMindex, WMindex)
gm_to_wm_average_delta = np.nanmean(gm_to_wm_delta, axis = 1)
sns.histplot(gm_to_wm_average_delta, bins = 12)







#%%


#analysis of determining how ablated contacts are connected to diff GM/WM regions

#Good 309, Engel 1a (1 year)        39-51  random.randrange(39,52)       #51 
#good 238, Engel 1a (9 months)      4-10   random.randrange(4,11)        #9
#good 440, Engel 1b (       )       78-81  random.randrange(78,82)       #78 
#good 320, Engel 1B (1 year)        52-54  random.randrange(52,55)       #52 
#good 307, Engel 1A (4 years)       37-38  random.randrange(37,39)       #37
#good 365, Engel 1B (2 years)       60-65  random.randrange(60,66)       #62

#fair 274, Engel 2A (1 year)        19-20  random.randrange(19,21)       #20
#fair 371, Engel 2A  (1 year)        66-69  random.randrange(66,70)       #68

#poor 278, Engel 4c (2 years)       21-24  random.randrange(21,25)       #24 
#poor 405, Engel 3  (2 years)       73-74  random.randrange(73,75)       #73
#poor 442, Engel 3  (2.5 years)     82-102 random.randrange(82,103)      #96
#poor 382, Engel 3  (3 years)       70-72  random.randrange(70,73)       #71   


#poor 322, engel 4a   55-59 #57
pvals = []
pvals2 = []
for kk in range(100):
    print(f"{kk}") 
    patient_inds = [51, 9, 78, 52, 37, 62, 20, 68, 24, 73, 96, 71]
    #bootstrapping patients and seizures
    patient_inds = [random.randrange(39,52), random.randrange(55,59) , random.randrange(78,82)  , 
                    random.randrange(52,55) , random.randrange(37,39) , random.randrange(60,66)  ,
                    random.randrange(19,21) , random.randrange(66,70), 
                    random.randrange(21,25) , random.randrange(73,75)  , random.randrange(82,103) , random.randrange(71,73)  ]
    patient_inds_good = random.sample(patient_inds[:6], 6)
    patient_inds_poor = random.sample(patient_inds[8:], 4)
    patient_inds_bootstrap = patient_inds_good + patient_inds_poor
    patient_zscores_ictal = []
    patient_zscores_preictal = []
    delta = []
    delta_zscore = []
    icccc = []
    piiii = []
    filler_zscore_all = []
    difference_in_gmwm_fc = []
    percent_top_gm_to_wm_all = []
    zscores_ablated_contact_significantl_correlated_to_wm_all = []
    for pt in range(len(patient_inds_bootstrap)):
        i = patient_inds_bootstrap[pt]
    
        sub = patientsWithseizures["subject"][i]
        func = 2
        FCtype = FCtypes[func]
        freq = 7
        print(f"{FCtype}, {frequencyNames[freq]}, {sub}") 
        functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
        channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password, 
                                            BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01", 
                                            functionalConnectivityPath = functionalConnectivityPath, 
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC", 
                                            fsds = 256, montage = "bipolar", FCtype = FCtype)
        
        #if cross correlation, take absolute value
        if func == 2 or func == 0:
            for per in range(len(FC)):
                for f in range(len(FC[per])):
                    FC[per][f] = abs(FC[per][f])
        # Get atlas localization and distances
        file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01", f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
        if utils.checkIfFileExistsGlob(file, printBOOL=False):
            localization = pd.read_csv(file)
            localizationChannels = localization["channel"]
            localizationChannels = echobase.channel2std(
                np.array(localizationChannels))
        # getting distances
        dist = pd.DataFrame(channels, columns=["channels"])
        dist["distance"] = np.nan
        for ch in range(len(channels)):# Getting distances of the channels
            channelName = channels[ch]
            if any(channelName == localizationChannels):
                dist.iloc[ch, 1] = localization["percent_WM"][np.where( #percent_WM, distance_to_GM_millimeters
                    channelName == localizationChannels)[0][0]]
            else:
                # if channel has no localization, then just assume GM.
                dist.iloc[ch, 1] = 0
        # definition of WM by distance
        
        GMindex = np.where(dist["distance"] <= WMdefinitionPercent2)[0]
        WMindex = np.where(dist["distance"] > WMdefinitionPercent2)[0]
        distOrderInd = np.array(np.argsort(dist["distance"]))
        distOrder = dist.iloc[distOrderInd].reset_index()
        
        manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
        manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
        manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
        
           
        #get FC values for just the GM-GM connections and WM-WM connections and GM-WM connections
        FCtissue = [None] *3
        FCall = []
        for t in range(len(FCtissue)):
            FCtissue[t] = []
        for s in range(len(FC)):
            #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
            FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )   )
            FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )   )
            FCtissue[2].append(   utils.getAdjSubset(FC[s][freq], GMindex, WMindex).flatten()   )
            FCall.append(  utils.getUpperTriangle(FC[s][freq]       )   )
        
        #sns.histplot(FCtissue[2][2]- FCtissue[2][0])

        difference_in_gmwm_fc.append(np.nanmedian(FCtissue[2][2] - FCtissue[2][0]))
        

        #utils.plot_adj_heatmap(FC[1][freq])
        #utils.getUpperTriangle(FC[2][freq])
        #utils.plot_histplot( utils.getUpperTriangle(FC[2][freq]), kde = True)
        #utils.plot_histplot( utils.getUpperTriangle(FC[1][freq]), kde = True)
        adj_ic = copy.deepcopy(FC[2][freq])
        adj_pi = copy.deepcopy(FC[1][freq])
        threshold_bin = 0.1
        adj_ic_bin = utils.threshold_and_binarize_adj(adj_ic, t=threshold_bin)
        adj_pi_bin = utils.threshold_and_binarize_adj(adj_pi, t=threshold_bin)
        #utils.plot_adj_heatmap(adj_ic_bin)
        #utils.plot_adj_heatmap(adj_pi_bin)
        
        gm_to_wm_ic = utils.getAdjSubset(adj_ic_bin, GMindex, WMindex) # gm_to_wm_ic = utils.getAdjSubset(adj_ic, GMindex, WMindex) #utils.plot_adj_heatmap(gm_to_wm_ic);
        gm_to_wm_pi = utils.getAdjSubset(adj_pi_bin, GMindex, WMindex)# gm_to_wm_pi = utils.getAdjSubset(adj_pi, GMindex, WMindex) #utils.plot_adj_heatmap(gm_to_wm_pi);
        gm_to_wm_delta = gm_to_wm_ic - gm_to_wm_pi
        
        gm_to_wm_delta_avg  = np.sum(gm_to_wm_delta, axis = 1) #gm_to_wm_delta_avg  = np.nanmedian(gm_to_wm_delta, axis = 1)
        
        gm_to_wm_delta_avg_top_ind = np.argsort(gm_to_wm_delta_avg)
        GMindex_top_correlated_to_wm = GMindex[gm_to_wm_delta_avg_top_ind]
   
        intersect = np.intersect1d(  manual_resected_electrodes_index, GMindex_top_correlated_to_wm, return_indices=True )
 
        print(f"\n{manual_resected_electrodes[intersect[1]]}")
        
        percent_top_gm_to_wm = intersect[2]/len(GMindex_top_correlated_to_wm)
        percent_top_gm_to_wm_all.append(percent_top_gm_to_wm)
        print(f"{percent_top_gm_to_wm}\n")
        iters = 100
        percent_top_gm_to_wm_rand = np.zeros((iters, len(percent_top_gm_to_wm)))
        for d in range(iters):
            adj_ic_rand = copy.deepcopy(adj_ic)
            adj_pi_rand = copy.deepcopy(adj_pi)
            if func == 5: #if pearson
                adj_ic_rand = bct.randmio_und_signed(adj_ic_rand, itr = 2)[0] #utils.plot_adj_heatmap(adj_ic_rand)
                adj_pi_rand = bct.randmio_und_signed(adj_pi_rand, itr = 2)[0]#utils.plot_adj_heatmap(adj_ic_rand)
            if func == 0 or func == 1 or func == 2: #if coherence
                adj_ic_rand = bct.null_model_und_sign(adj_ic_rand)[0] #utils.plot_adj_heatmap(adj_ic_rand); utils.plot_adj_heatmap(adj_ic)
                adj_pi_rand = bct.null_model_und_sign(adj_pi_rand)[0] #utils.plot_adj_heatmap(adj_pi_rand); utils.plot_adj_heatmap(adj_pi)
                
            adj_ic_rand_bin = utils.threshold_and_binarize_adj(adj_ic_rand, t=threshold_bin) #utils.plot_adj_heatmap(adj_ic_rand_bin);
            adj_pi_rand_bin = utils.threshold_and_binarize_adj(adj_pi_rand, t=threshold_bin) #utils.plot_adj_heatmap(adj_pi_rand_bin);
            
            gm_to_wm_ic_rand = utils.getAdjSubset(adj_ic_rand_bin, GMindex, WMindex) #bin
            gm_to_wm_pi_rand = utils.getAdjSubset(adj_pi_rand_bin, GMindex, WMindex) #bin
            gm_to_wm_delta_rand = gm_to_wm_ic_rand - gm_to_wm_pi_rand
            
            gm_to_wm_delta_avg_rand  = np.sum(gm_to_wm_delta_rand, axis = 1) #sum to meadian
            
            gm_to_wm_delta_avg_top_ind_rand = np.argsort(gm_to_wm_delta_avg_rand)
            GMindex_top_correlated_to_wm_rand = GMindex[gm_to_wm_delta_avg_top_ind_rand]
       
            intersect = np.intersect1d(  manual_resected_electrodes_index, GMindex_top_correlated_to_wm_rand, return_indices=True )
            percent_top_gm_to_wm_rand[d,:] = intersect[2]/len(GMindex_top_correlated_to_wm_rand)
            utils.printProgressBar(d, iters)
        
        zscores_ablated_contact_significantl_correlated_to_wm = []
        for ch in range(len(percent_top_gm_to_wm)):
            zz = stats.zscore( np.concatenate([[np.array(percent_top_gm_to_wm[ch])], percent_top_gm_to_wm_rand[:,ch] ]) )[0] #utils.plot_histplot( percent_top_gm_to_wm_rand[:,1]   )
            zscores_ablated_contact_significantl_correlated_to_wm.append(zz)
        zscores_ablated_contact_significantl_correlated_to_wm_all.append(  zscores_ablated_contact_significantl_correlated_to_wm )
        print("\n\n")
        print(zscores_ablated_contact_significantl_correlated_to_wm)
        print("\n\n")
   
        
    good_percent = percent_top_gm_to_wm_all[:len(patient_inds_good)]
    poor_percent = percent_top_gm_to_wm_all[len(patient_inds_good):]
    
    good_percent = [item for sublist in good_percent for item in sublist]
    poor_percent = [item for sublist in poor_percent for item in sublist]
    
    df_good = pd.DataFrame(dict(good = good_percent))
    df_poor = pd.DataFrame(dict(poor = poor_percent))
    df_patient = pd.concat([pd.melt(df_good), pd.melt(df_poor)]  )
    print(stats.mannwhitneyu(good_percent,poor_percent)[1]    )
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    sns.boxplot(data = df_patient, x= "variable", y = "value", ax = axes )
    sns.swarmplot(data = df_patient, x= "variable", y = "value", ax = axes)
    plt.show()    
    ##
        
    good_fcavg = difference_in_gmwm_fc[:len(patient_inds_good)]
    poor_fcavg = difference_in_gmwm_fc[len(patient_inds_good):]
    
    df_good = pd.DataFrame(dict(good = good_fcavg))
    df_poor = pd.DataFrame(dict(poor = poor_fcavg))
    
    stats.mannwhitneyu(good_fcavg,poor_fcavg)[1]
    stats.ttest_ind(good_fcavg,poor_fcavg)[1]
    df_patient = pd.concat([pd.melt(df_good), pd.melt(df_poor)]  )
    pvalues = stats.ttest_ind(good_fcavg,poor_fcavg)[1]
    pvals2.append(pvalues)
    print(f"{np.round(pvalues,2)}")
fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
sns.boxplot(data = df_patient, x= "variable", y = "value", ax = axes )
sns.swarmplot(data = df_patient, x= "variable", y = "value", ax = axes)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)    
sns.boxplot(np.array(pvals2), ax = axes[0])
sns.histplot(np.array(pvals2), bins = 50, ax =axes[1])
print(len(np.where(np.array(pvals2) < 0.05)[0])/len(pvals2))


    
        
        
        
        ictal = []
        preictal = []
        ictal_zscore = []
        preictal_zscore = []
        
        preictal_all_connectivity_per_channel = []
        ictal_all_connectivity_per_channel = []
        
        
        
        
        for s in [1,2]:
            state = s
            
            
            manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
            manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
            manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
            #print(manual_resected_electrodes)
            
            
            
            """
            adj = FC[state][freq]
            if func == 0: #if perason
                adj_scrambled = bct.randmio_und_signed(adj, itr = 2)[0]
            if func == 0 or func == 1 or func == 2: #if coherence
                adj_scrambled = bct.null_model_und_sign(adj)[0]
            utils.plot_adj_heatmap(adj, vmin = np.min(adj), vmax = np.max(adj))
            utils.plot_adj_heatmap(adj_scrambled, vmin = np.min(adj), vmax = np.max(adj))
            ch = 0
            ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
            print(manual_resected_electrodes[ch])
            dist[channels == manual_resected_electrodes[ch]]
            ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WMindex)[0]
            print(np.nanmedian(ablated_wm_fc))
            gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex)
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            utils.plot_adj_heatmap(gm_to_wm, vmin = np.min(adj), vmax = np.max(adj))
            sns.histplot(gm_to_wm[2,:], bins = 20)
            sns.histplot(ablated_wm_fc.flatten(), bins = 20)
            ablated_wm_fc = utils.getAdjSubset(adj_scrambled, ablated_ind, WMindex)
            print(np.nanmedian(ablated_wm_fc))
            gm_to_wm = utils.getAdjSubset(adj_scrambled, GMindex, WMindex)
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            sns.histplot(gm_to_wm_average, bins = 20)
            """
            
            
            
            adj = FC[state][freq]
            manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
            manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
            manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
            
            gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex)
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            #sns.histplot(gm_to_wm_average, bins = 20)
            
            for ch in range(len(manual_resected_electrodes)):
                #ch = 1
                #print(dist[channels == manual_resected_electrodes[ch]])
                ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
                ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WMindex)[0]
                gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex  )
                gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
                #sns.histplot(gm_to_wm_average, bins = 20)
                #plt.show()
                value = np.nanmedian(ablated_wm_fc)
                #print(np.nanmedian(ablated_wm_fc))
                zz = stats.zscore( np.concatenate([[np.array(value)], gm_to_wm_average]) )[0]
                
                all_wm_connectivity = utils.getAdjSubset(adj, np.array(range(len(adj))), WMindex  )
                all_wm_connectivity_average = np.nanmedian(all_wm_connectivity, axis = 1)
                if s != 2: 
                    preictal.append(np.nanmedian(ablated_wm_fc))
                    preictal_zscore.append(zz)
                    preictal_all_connectivity_per_channel = all_wm_connectivity_average
                    
                if s == 2: 
                    ictal.append(np.nanmedian(ablated_wm_fc))
                    ictal_zscore.append(zz)
                    ictal_all_connectivity_per_channel = all_wm_connectivity_average
            if s == 2:   
                dd = np.array(ictal) - np.array(preictal)
                delta.append(dd)
                delta_zscore.append(np.array(ictal_zscore) - np.array(preictal_zscore))
                icccc.append(ictal)
                piiii.append(preictal)
                    
                diff1 = ictal_all_connectivity_per_channel - preictal_all_connectivity_per_channel
                #sns.histplot(diff1, bins = 20)
                #sns.histplot(np.array(ictal) - np.array(preictal), bins = 20)
                filler_zscore = []
                for ablated_channel in range(len(dd)):
                    filler_zscore.append(stats.zscore( np.concatenate( [np.array([dd[ablated_channel]]), diff1]) )[0]    )
                filler_zscore_all.append(filler_zscore)
            """
            z_scores_ablated = []
            for ch in range(len(manual_resected_electrodes)):    
                ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
                ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WMindex)[0]
                value = np.nanmedian(ablated_wm_fc)
                distribution = [value]
                iters = 50
                for d in range(iters):
                    if func == 5: #if pearson
                        adj_scrambled = bct.randmio_und_signed(adj, itr = 2)[0]
                    if func == 0 or func == 1 or func == 2: #if coherence
                        adj_scrambled = bct.null_model_und_sign(adj)[0]
                    ablated_wm_fc = utils.getAdjSubset(adj_scrambled, ablated_ind, WMindex)
                    #gm_to_wm = utils.getAdjSubset(adj_scrambled, GMindex, WMindex)
                    #gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
                    #sns.histplot(gm_to_wm_average, bins = 20)
                    avg = np.nanmedian(ablated_wm_fc)
                    print(f"{d}: {sub}: {s}: {pt}/{len(patient_inds_bootstrap)}: {np.round(value,3)} ::: {np.round(avg,3)}")
                    distribution.append(avg)
                
                #sns.histplot(  distribution, bins = 20)
                #plt.show()
                z_scores = stats.zscore( distribution )
                z_scores_ablated.append(z_scores[0])
                print(z_scores_ablated)
            if s != 2:
                patient_zscores_preictal.append(z_scores_ablated)
            if s == 2:
                patient_zscores_ictal.append(z_scores_ablated)
                
                
      

    good_preictal = np.concatenate(patient_zscores_preictal[:len(patient_inds_good)])
    poor_preictal = np.concatenate(patient_zscores_preictal[len(patient_inds_good):])
    
    
    good_ictal = np.concatenate(patient_zscores_ictal[:len(patient_inds_good)])
    poor_ictal = np.concatenate(patient_zscores_ictal[len(patient_inds_good):])
    
    np.nanmean(good_ictal - good_preictal)
    np.nanmean(poor_ictal - poor_preictal)
    
    df_good_delta = pd.DataFrame(dict(good = good_ictal - good_preictal))
    df_poor_delta = pd.DataFrame(dict(poor = poor_ictal - poor_preictal))
    
    
    
    df_patient = pd.concat([pd.melt(df_good_delta), pd.melt(df_poor_delta)]  )
    """
    """
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    sns.boxplot(data = df_patient, x= "variable", y = "value", ax = axes )
    sns.swarmplot(data = df_patient, x= "variable", y = "value", ax = axes)
    
    stats.ttest_ind(good_ictal,poor_ictal)
    """
    
    
    
    

    #good = np.concatenate(delta[:7])
    #poor = np.concatenate(delta[8:])
    good = np.concatenate(delta[:len(patient_inds_good)])
    poor = np.concatenate(delta[len(patient_inds_good):])
    
    df_good = pd.DataFrame(dict(good = good))
    df_poor = pd.DataFrame(dict(poor = poor))
    
    
    
    df_patient = pd.concat([pd.melt(df_good), pd.melt(df_poor)]  )
    """
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    sns.boxplot(data = df_patient, x= "variable", y = "value", ax = axes )
    sns.swarmplot(data = df_patient, x= "variable", y = "value", ax = axes)
    plt.show()
    """
    stats.mannwhitneyu(good,poor)[1]
    stats.ttest_ind(good,poor)[1]
    pvals.append(stats.mannwhitneyu(good,poor)[1])
    
    
    
    
    
    #%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)    
sns.boxplot(np.array(pvals), ax = axes[0])
sns.histplot(np.array(pvals), bins = 50, ax =axes[1])
print(len(np.where(np.array(pvals) < 0.05)[0])/len(pvals))



#%%
file_to_store = open("patient_zscores.pickle", "wb")
pickle.dump(patient_zscores, file_to_store)
file_to_store.close()
abs((np.mean(distribution) - value ) / (np.std(distribution)))

stats.norm.sf(abs(z_scores[0]))*2


file_to_read = open("patient_zscores.pickle", "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()









[max(p) for p in loaded_object]








