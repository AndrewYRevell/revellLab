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
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
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
frequencyNames = ["Broadband", "delta", "theta", "alpha", "beta", "gammaLow", "gammaMid", "gammaHigh"]

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
FCtypes = ["pearson", "coherence", "crossCorrelation"]
paientList = pd.DataFrame(columns=["patient"])
summaryStats = pd.DataFrame( columns=["patient", "FCtype", "frequency", "interictal", "preictal", "ictal", "postictal"] )
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
            
            GMindex = np.where(dist["distance"] <= WMdefinition)[0]
            WMindex = np.where(dist["distance"] > WMdefinition)[0]
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
            summaryStats = summaryStats.append(dict(patient = sub ,FCtype = FCtype, frequency = frequencyNames[freq], interictal = diff[0], preictal = diff[1], ictal = diff[2], postictal = diff[3]) , ignore_index=True   )
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
summaryStatsLong = pd.melt(summaryStats, id_vars = ["patient", "frequency", "FCtype"], var_name = "state", value_name = "FC")
summaryStatsLong = summaryStatsLong.groupby(['patient', "frequency", "FCtype", "state"]).mean()
summaryStatsLong.reset_index(inplace=True)


#%%

# All FC and all Frequency Boxplot of FC values for ii, pi, ic, and po states
g = sns.FacetGrid(summaryStatsLong, col="frequency", row = "FCtype",sharey=False, col_order=frequencyNames, row_order = FCtypes)
g.map(sns.boxplot, "state","FC", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = colorsInterPreIctalPost)
g.map(sns.stripplot, "state","FC", order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = colorsInterPreIctalPost2)
ylims = [ [-0.02, 0.04], [-0.005, 0.04] ,[-0.2, 0.18] ]
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

#% Plot FC distributions for example patient


i=51 #22, 24 25 107 79  78, sub-RID0309: 39-51
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



state = 1


#plot heatmap, not ordered, ordered
FCexample = FC[state][freq] 
FCexampleOrdered =  utils.reorderAdj(FCexample, distOrder["index"])  
vmin = -0.1
vmax = 0.5
center = 0.2
sns.heatmap(FCexample, square=True, vmin = vmin, vmax = vmax, center = center, cmap = "mako"); 
plt.show()
sns.heatmap(FCexampleOrdered, square=True, vmin = vmin, vmax = vmax, center = center, cmap = "mako" ); 
plt.show()
print(np.where(distOrder["distance"]>0)[0][0])
print(np.where(distOrder["distance"]>1)[0][0])

wm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], WMindex)   )
gm = utils.getUpperTriangle(     utils.reorderAdj(FC[state][freq], GMindex)  )
gmwm = np.concatenate([wm, gm])
gmwmthresh = gmwm[np.where(gmwm > 0.01)]

binwidth=0.025
xlim = [-0.2,0.6]

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
axes3.set_xlim([-0.4, 0.4]);
axes3.set_xticks([-0.4,-0.2,0,0.2,0.4])
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


#poor 278, engel 4c   21-24 #24
#poor 322, engel 4a   55-59 #57

#good 194, engel 1a   3
#poor RID0139 Engel 1d-2a-3a???


i=51


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
print(manual_resected_electrodes)
manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )


tmp = utils.reorderAdj(adj, manual_resected_electrodes_index)
sns.heatmap(tmp, vmin = -0.2, vmax = 0.4)
plt.show()
sns.heatmap(adj, vmin = -0.2, vmax = 0.4)
plt.show()



tmp_wm = utils.getAdjSubset(adj, manual_resected_electrodes_index, WMindex)
sns.heatmap(tmp_wm, vmin = -0.2, vmax = 0.4)
plt.show()
tmp_gm = utils.getAdjSubset(adj, manual_resected_electrodes_index, GMindex)
sns.heatmap(tmp_gm, vmin = -0.2, vmax = 0.4)
plt.show()



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


state = 1

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
i=51
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

tractography.getTracts(paths['BIDS'], paths['dsiStudioSingularity'], pathDWI, pathTracts)


































