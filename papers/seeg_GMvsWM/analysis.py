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

# %% 01 Imports
import sys
import os
import json
import copy
import time
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import multiprocessing
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
# % 02 Paths and File names

figureDir = "/media/arevell/sharedSSD/linux/figures"
# BIDS directory
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
dataset = "PIER"
metadataDir = "/media/arevell/sharedSSD/linux/data/metadata"

dsiStudioSingularityPatah = "/home/arevell/singularity/dsistudio/dsistudio_latest.sif"
freesurferLicense = "$FREESURFER_HOME/license.txt"
cohortsPath = join(metadataDir, "cohortData_brainAtlas.json")
jsonFilePath = join(metadataDir, "iEEGdataRevell.json")
fnameiEEGusernamePassword = join(
    "/media/arevell/sharedSSD/linux/", "ieegorg.json")

# tools
revellLabPath = pkg_resources.resource_filename("revellLab", "/")
tools = pkg_resources.resource_filename("revellLab", "tools")
atlasPath = join(tools, "atlases", "atlases")
atlasLabelsPath = join(tools, "atlases", "atlasLabels")
atlasfilesPath = join(tools, "atlases", "atlasMetadata.json")
MNItemplatePath = join(tools, "templates", "MNI",
                       "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz")
MNItemplateBrainPath = join(tools, "templates", "MNI",
                            "mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz")
randomAtlasesPath = join(atlasPath, "randomAtlasesWholeBrainMNI")
atlasDirectory = join(tools, "atlases", "atlases")
atlasLabelDirectory = join(tools, "atlases", "atlasLabels")

atlasLocaliztionDir = join(BIDS, "derivatives", "atlasLocalization")
atlasLocalizationFunctionDirectory = join(
    revellLabPath, "packages", "atlasLocalization")

# BrainAtlas Project data analysis path
derivatives = join(BIDS, "derivatives")

freesurferReconAllDir = join(BIDS, "derivatives", "freesurferReconAll")
atlasLocalizationDir = join(derivatives, "atlasLocalization")


# % 03 Paramters and read metadata

SesImplant = "implant01"
acq = "3D"
ieegSpace = "T00"

# %

# Atlas metadata
with open(atlasfilesPath) as f:
    atlasfiles = json.load(f)
atlases = DataClassAtlases.atlases(atlasfiles)

# Study cohort data
with open(cohortsPath) as f:
    cohortJson = json.load(f)
cohort = DataClassCohortsBrainAtlas.brainAtlasCohort(cohortJson)
# Get patietns with DTI data
patientsDTI = cohort.getWithDTI()
# get patients with iEEG times:
iEEGTimes = cohort.getiEEGdataKeys()


# JSON metadata data
with open(jsonFilePath) as f:
    jsonFile = json.load(f)
metadata = DataClassJson.DataClassJson(jsonFile)


# Get iEEG.org username and password
with open(fnameiEEGusernamePassword) as f:
    usernameAndpassword = json.load(f)
username = usernameAndpassword["username"]
password = usernameAndpassword["password"]


# % 01

patientsWithseizures = metadata.get_patientsWithSeizuresAndInterictal()
# calculate length of seizures
patientsWithseizures = pd.concat([patientsWithseizures, pd.DataFrame(
    patientsWithseizures["stop"] - patientsWithseizures["EEC"], columns=['length'])], axis=1)

#% custom functions
def findMaxDim(l, init = 0): #find the maximum second dimension of a list of arrays.
    for i in range(len(l)):
        if l[i].shape[1] > init:
            init = l[i].shape[1]
    return init
# %%
# how many seizures per patient
seizureCounts = patientsWithseizures.groupby(['subject']).count()
fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
sns.histplot(data=seizureCounts, x="idKey", binwidth=1, kde=True, ax=axes)
axes.set_xlim(1, None)
axes.set(xlabel='Number of Seizures', ylabel='Number of Patients',
         title="Distribution of Seizure Occurrences")
plt.savefig(f"{figureDir}/common/seizureCounts.pdf")

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

plt.savefig(f"{figureDir}/common/seizureLengthDistribution.pdf")

# %% 07 Electrode and atlas localization

iEEGpatientList = np.unique(list(patientsWithseizures["subject"]))
iEEGpatientList = ["sub-" + s for s in iEEGpatientList]

atl.atlasLocalizationBIDSwrapper(iEEGpatientList,  BIDS, dataset, SesImplant, ieegSpace, acq, freesurferReconAllDir, atlasLocalizationDir,
                                 atlasDirectory, atlasLabelDirectory, MNItemplatePath, MNItemplateBrainPath, multiprocess=False, cores=12, rerun=False)


# %% 08 EEG download and preprocessing of electrodes


# Download iEEG data

for i in range(0, 40):
    metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                        BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata.get_associatedInterictal(
        patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, username, password,
                          BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=False)


for i in range(40, 80):
    print(i)
    metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                        BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata.get_associatedInterictal(
        patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, username, password,
                          BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=False)


for i in range(80, len(patientsWithseizures)):
    metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                        BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata.get_associatedInterictal(
        patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, username, password,
                          BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=False)


# %%
paientList = pd.DataFrame(columns=["patient"])
N = len(patientsWithseizures)
for i in range(3, N):# [3,5, 16, 22,26, 40 , 46, 60, 80]:
    paientList = paientList.append(
        dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
    # get data
    seizure, fs, ictalStartIndex, ictalStopIndex = metadata.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], username, password,
                                                                                       BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", secondsBefore=180, secondsAfter=180, load=True)
    interictal, fs = metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", patientsWithseizures["AssociatedInterictal"][i], username, password,
                                           BIDS=BIDS, dataset="derivatives/iEEGorgDownload", session="implant01", startKey="Start", load=True)

    ###filtering and downsampling
    fsds = 256
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
    file = join(atlasLocalizationDir, f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01",
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

    WMdefinition = 0  # definition of WM by distance
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
        noise[ch] = simps(powerInterpIIAvg[:,ch], dx=1)

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
        powerDistAllSame = np.full( shape= ( powerDistAll[1].shape[0],   findMaxDim(powerDistAll), 4, len(powerDistAll)) , fill_value= np.nan)
        for d in range(len(powerDistAll)):
            filler = powerDistAll[d]
            powerDistAllSame[:,:filler.shape[1],:, d] = filler
        powerDistAvg = np.nanmean(powerDistAllSame, axis=3)
        
        
        plot_GMvsWM.plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll)

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

#%% Power analysis
plot_GMvsWM.plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll)


nseiuzres = powerGM.shape[2]
powerGMmean = np.nanmean(powerGM, axis=2)
powerWMmean = np.nanmean(powerWM, axis=2)
#padding power vs distance with NaNs
powerDistAllSame = np.full( shape= ( powerDistAll[1].shape[0],   findMaxDim(powerDistAll), 4, len(powerDistAll)) , fill_value= np.nan)
for d in range(len(powerDistAll)):
    filler = powerDistAll[d]
    powerDistAllSame[:,:filler.shape[1],:, d] = filler
powerDistAvg = np.nanmean(powerDistAllSame, axis=3)











  
fig = plt.figure(figsize=(8, 8), dpi=300)
gs = fig.add_gridspec(4, 4)
axes = 4 * [None]
for t in range(2,4):
    axes[t] = 4 * [None]
axes[0] = fig.add_subplot(gs[0, :]); axes[1] = fig.add_subplot(gs[1, :])
for t in range(4):
    axes[2][t] = fig.add_subplot(gs[2, t])
    axes[3][t] = fig.add_subplot(gs[3, t])
sns.heatmap(np.log10(powerGMmean[0:110, :]), vmin=0, vmax=4, center=0, ax=axes[0],cbar = False); axes[0].invert_yaxis()
sns.heatmap(np.log10(powerWMmean[0:110, :]), vmin=0, vmax=4, center=0,  ax=axes[1],cbar = False); axes[1].invert_yaxis()
for t in range(4):
    sns.heatmap(np.log10(powerDistAvg[1:50, :,t]),  ax=axes[2][t],cbar = False, vmin=-2, vmax=5)
    axes[2][t].invert_yaxis()


print(f"\n\n\n\n\n\n\n\n\n{i}\n\n\n\n\n\n\n\n\n")
plt.show()



ax = sns.heatmap(np.log10(powerGM[0:110, :, 10]), vmin=0, vmax=4, center=0)
ax.invert_yaxis()

nseiuzres = powerGM.shape[2]
powerGMmean = np.nanmean(powerGM, axis=2)
powerWMmean = np.nanmean(powerWM, axis=2)

fig, axes = plt.subplots(2, 1, figsize=(8, 4), dpi=300)
sns.heatmap(np.log10(powerGMmean[0:110, :]),
            vmin=0, vmax=4, center=0, ax=axes[0])
axes[0].invert_yaxis()
sns.heatmap(np.log10(powerWMmean[0:110, :]),
            vmin=0, vmax=4, center=0,  ax=axes[1])
axes[1].invert_yaxis()
print(f"\n\n\n\n\n\n\n\n\n{i}\n\n\n\n\n\n\n\n\n")
plt.show()

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

dfArea = dfArea.groupby(["tissue", "patient", "state"]).mean()
dfArea.reset_index(inplace=True)

colors1 = ["#c6b4a5", "#b6d4ee"]
colors2 = ["#a08269", "#76afdf"]
colors3 = ["#544335", "#1f5785"]
fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
sns.boxplot(data=dfArea, x="state", y="power", hue="tissue", palette=colors2, showfliers=False,
            ax=axes, color=colors3, order=["interictal", "preictal", "ictal", "postictal"])
sns.stripplot(x="state", y="power", hue="tissue",  data=dfArea, palette=colors3,
              dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])
# Set only one legend
handles, labels = axes.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(
    0.0, 1), loc=2, borderaxespad=0.5)
axes.set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
plt.savefig(f"{figureDir}/GMvsWM/powerDifferences_bySeziure.pdf")

dfArea = dfArea.groupby(["tissue", "patient", "state"]).mean()
dfArea.reset_index(inplace=True)
fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
sns.boxplot(data=dfArea, x="state", y="power", hue="tissue", palette=colors2, showfliers=False,
            ax=axes, color=colors3, order=["interictal", "preictal", "ictal", "postictal"])
sns.stripplot(x="state", y="power", hue="tissue",  data=dfArea, palette=colors3,
              dodge=True, size=3, order=["interictal", "preictal", "ictal", "postictal"])
# Set only one legend
handles, labels = axes.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(
    0.0, 1), loc=2, borderaxespad=0.5)
axes.set(xlabel='', ylabel='power (log10)', title="Tissue Power Differences")
plt.savefig(f"{figureDir}/GMvsWM/powerDifferences_byPatient.pdf")


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
statsTable = statsTable.append(dict(tissue_1="WM", tissue_2="WM", state_1="preictal",
                                    state_2="postictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)
v1 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "preictal"]["power"]
v2 = dfArea.loc[dfArea['tissue'] ==
                "WM"].loc[dfArea['state'] == "postictal"]["power"]
statsTable = statsTable.append(dict(tissue_1="WM", tissue_2="WM", state_1="preictal",
                                    state_2="postictal", pvalue=stats.wilcoxon(v1, v2)[1]), ignore_index=True)

statsTable.to_csv(f"{figureDir}/GMvsWM/powerDifferences.csv", index=False)
