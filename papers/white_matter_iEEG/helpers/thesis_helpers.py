import sys
import os
import json
import copy
import time
import bct
import glob
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
from pathos.multiprocessing import ProcessingPool as Pool
#revellLab
#utilities, constants/parameters, and thesis helper functions
from revellLab.packages.utilities import utils

#package functions
from revellLab.packages.dataclass import dataclass_atlases, dataclass_iEEG_metadata
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.imaging.tractography import tractography

from revellLab.papers.white_matter_iEEG.plotting import plot_GMvsWM
#%% Power analysis

def power_analysis(patientsWithseizures, indexes, metadata_iEEG, USERNAME, PASSWORD, SESSION, FREQUENCY_DOWN_SAMPLE, MONTAGE, paths, TISSUE_DEFINITION_NAME , TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM):
    paientList = pd.DataFrame(columns=["patient"])
    count = 0
    for i in indexes:
        paientList = paientList.append(dict(patient=patientsWithseizures["subject"][i]), ignore_index = True)
        # get data
        seizure, fs, ictalStartIndex, ictalStopIndex = metadata_iEEG.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], USERNAME, PASSWORD,
                                                                                           BIDS=paths.BIDS, dataset= paths.BIDS_DERIVATIVES_WM_IEEG_IEEG, session=SESSION, secondsBefore=180, secondsAfter=180, load=True)
        interictal, fs = metadata_iEEG.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", patientsWithseizures["AssociatedInterictal"][i], USERNAME, PASSWORD,
                                               BIDS=paths.BIDS, dataset= paths.BIDS_DERIVATIVES_WM_IEEG_IEEG, session= SESSION, startKey="Start", load = True)

        ###filtering and downsampling
        ictalStartIndexDS = int(ictalStartIndex * (FREQUENCY_DOWN_SAMPLE/fs))
        ictalStopIndexDS = int(ictalStopIndex * (FREQUENCY_DOWN_SAMPLE/fs))
        seizureLength = (ictalStopIndexDS-ictalStartIndexDS)/FREQUENCY_DOWN_SAMPLE
        _, _, _, seizureFilt, channels = echobase.preprocess(seizure, fs, fs, montage=MONTAGE, prewhiten = False)
        _, _, _, interictalFilt, _ = echobase.preprocess(interictal, fs, fs, MONTAGE, prewhiten = False)
        seizureFiltDS = metadata_iEEG.downsample(seizureFilt, fs, FREQUENCY_DOWN_SAMPLE)
        interictalFiltDS = metadata_iEEG.downsample(interictalFilt, fs, FREQUENCY_DOWN_SAMPLE)

        nchan = seizureFiltDS.shape[1]
        # calculating power
        power = echobase.get_power(seizureFiltDS, FREQUENCY_DOWN_SAMPLE)
        powerII = echobase.get_power(interictalFiltDS, FREQUENCY_DOWN_SAMPLE)
        # interpolate power
        powerInterp = echobase.power_interpolate( power, powerII, 180, 180 + int(np.round(seizureLength)), length=200)
        # Get atlas localization and distances
        file = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{patientsWithseizures['subject'][i]}", f"ses-{SESSION}",f"sub-{patientsWithseizures['subject'][i]}_ses-{SESSION}_desc-atlasLocalization.csv")
        if utils.checkIfFileExistsGlob(file):
            localization = pd.read_csv(file)
            localizationChannels = localization["channel"]
            localizationChannels = utils.channel2std(
                np.array(localizationChannels))
        # getting distances
        dist = pd.DataFrame(channels, columns=["channels"])
        dist["distance"] = np.nan
        for ch in range(len(channels)):
            channelName = channels[ch]
            if any(channelName == localizationChannels):
                dist.iloc[ch, 1] = localization[TISSUE_DEFINITION_NAME][np.where(
                    channelName == localizationChannels)[0][0]]
            else:
                # if channel has no localization, then just assume GM.
                dist.iloc[ch, 1] = 0

        # definition of WM by distance
        GMindex = np.where(dist["distance"] <= TISSUE_DEFINITION_GM)[0]
        WMindex = np.where(dist["distance"] > TISSUE_DEFINITION_WM)[0]

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
        powerII = echobase.get_power(interictalFiltDS, FREQUENCY_DOWN_SAMPLE, avg=True)
        powerPI = echobase.get_power(seizureFiltDS[0:ictalStartIndexDS], FREQUENCY_DOWN_SAMPLE, avg=True)
        powerIC = echobase.get_power(seizureFiltDS[ictalStartIndexDS:ictalStopIndexDS], FREQUENCY_DOWN_SAMPLE, avg=True)
        powerPO = echobase.get_power(seizureFiltDS[ictalStopIndexDS:], FREQUENCY_DOWN_SAMPLE, avg=True)
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
        if count > 0:
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
        count = count + 1
    return powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM
    """
    paientList = pd.DataFrame(columns=["patient"])
    for i in range(3, N):#[3, 5, 16]: #range(3, N):
        paientList = paientList.append(
            dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
        # get data
        seizure, fs, ictalStartIndex, ictalStopIndex = metadata_iEEG.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], USERNAME, PASSWORD,
                                                                                           BIDS=paths.BIDS, dataset="derivatives/iEEGorgDownload", session=SESSION, secondsBefore=180, secondsAfter=180, load=True)
        interictal, fs = metadata.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", patientsWithseizures["AssociatedInterictal"][i], username, password,
                                               BIDS=paths['BIDS'], dataset="derivatives/iEEGorgDownload", session= param.SESSIONS[0], startKey="Start", load=True)

        ###filtering and downsampling
        ictalStartIndexDS = int(ictalStartIndex * (param.FREQUENCY_DOWN_SAMPLE/fs))
        ictalStopIndexDS = int(ictalStopIndex * (param.FREQUENCY_DOWN_SAMPLE/fs))
        seizureLength = (ictalStopIndexDS-ictalStartIndexDS)/param.FREQUENCY_DOWN_SAMPLE
        _, _, _, seizureFilt, channels = echobase.preprocess(
            seizure, fs, fs, montage=param.MONTAGE_BIPOLAR, prewhiten=False)
        _, _, _, interictalFilt, _ = echobase.preprocess(
            interictal, fs, fs, param.MONTAGE_BIPOLAR, prewhiten=False)
        seizureFiltDS = metadata.downsample(seizureFilt, fs, param.FREQUENCY_DOWN_SAMPLE)
        interictalFiltDS = metadata.downsample(interictalFilt, fs, param.FREQUENCY_DOWN_SAMPLE)

        nchan = seizureFiltDS.shape[1]
        # calculating power
        power = echobase.get_power(seizureFiltDS, param.FREQUENCY_DOWN_SAMPLE)
        powerII = echobase.get_power(interictalFiltDS, param.FREQUENCY_DOWN_SAMPLE)
        # interpolate power
        powerInterp = echobase.power_interpolate( power, powerII, 180, 180 + int(np.round(seizureLength)), length=200)
        # Get atlas localization and distances
        file = join(paths["atlasLocaliztion"], f"sub-{patientsWithseizures['subject'][i]}", "ses-implant01",
                    f"sub-{patientsWithseizures['subject'][i]}_ses-implant01_desc-atlasLocalization.csv")
        if utils.checkIfFileExistsGlob(file):
            localization = pd.read_csv(file)
            localizationChannels = localization["channel"]
            localizationChannels = utils.channel2std(
                np.array(localizationChannels))
        # getting distances
        dist = pd.DataFrame(channels, columns=["channels"])
        dist["distance"] = np.nan
        for ch in range(len(channels)):
            channelName = channels[ch]
            if any(channelName == localizationChannels):
                dist.iloc[ch, 1] = localization["percent_WM"][np.where(
                    channelName == localizationChannels)[0][0]]
            else:
                # if channel has no localization, then just assume GM.
                dist.iloc[ch, 1] = 0

        # definition of WM by distance
        GMindex = np.where(dist["distance"] <= WMdefinitionPercent3)[0]
        WMindex = np.where(dist["distance"] > WMdefinitionPercent3)[0]

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
        powerII = echobase.get_power(interictalFiltDS, param.FREQUENCY_DOWN_SAMPLE, avg=True)
        powerPI = echobase.get_power(seizureFiltDS[0:ictalStartIndexDS], param.FREQUENCY_DOWN_SAMPLE, avg=True)
        powerIC = echobase.get_power(seizureFiltDS[ictalStartIndexDS:ictalStopIndexDS], param.FREQUENCY_DOWN_SAMPLE, avg=True)
        powerPO = echobase.get_power(seizureFiltDS[ictalStopIndexDS:], param.FREQUENCY_DOWN_SAMPLE, avg=True)
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

def power_analysis_stats(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM):
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
    return statsTable


#%%

#take absolute value of FC

def functional_connectivity_absolute_value(FC_list, FC_type):
    """
    :param FC_list: functional connectivity list. state x freq
    :type FC_list: list
    :param FC_type: one of "pearson", "crossCorrelation", "coherence"
    :type FC_type: string
    :return: FC_list absolute valued
    :rtype: list
    """
    if FC_type == "pearson" or FC_type == "crossCorrelation":
        for state in range(len(FC_list)):
            for f in range(len(FC_list[state])):
                FC_list[state][f] = abs(FC_list[state][f])
    return FC_list




def get_channel_distances_from_WM(localization, localization_channels, channels, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM):
    """
    :param localization: csv file from atlas_localization pipeline
    :type localization: pandas dataframe
    :param localization_channels: channel names from localization file above, but with proper/standard channel names (LA01, and not LA1)
    :type localization_channels: array of strings
    :param channels: channel names from iEEG.org, specifically from metadata_iEEG.get_FunctionalConnectivity
    :type channels: array of strings
    :param TISSUE_DEFINITION_NAME: DESCRIPTION
    :type TISSUE_DEFINITION_NAME: TYPE
    :param TISSUE_DEFINITION_GM: DESCRIPTION
    :type TISSUE_DEFINITION_GM: TYPE
    :param TISSUE_DEFINITION_WM: DESCRIPTION
    :type TISSUE_DEFINITION_WM: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    dist = pd.DataFrame(channels, columns=["channels"])
    dist["distance"] = np.nan
    for ch in range(len(channels)):# Getting distances of the channels
        channel_name = channels[ch]
        if any(channel_name == localization_channels):
            dist.iloc[ch, 1] = localization[TISSUE_DEFINITION_NAME][np.where(
                channel_name == localization_channels)[0][0]]
        else:
            # if channel has no localization, then just assume GM.
            dist.iloc[ch, 1] = 0

    GM_index = np.where(dist["distance"] <= TISSUE_DEFINITION_GM)[0]
    WM_index = np.where(dist["distance"] > TISSUE_DEFINITION_WM)[0]
    dist_order_ind = np.array(np.argsort(dist["distance"]))
    dist_order = dist.iloc[dist_order_ind].reset_index()
    return dist, GM_index, WM_index, dist_order
def get_channel_xyz(localization, localization_channels, channels ):

    coordinates = pd.DataFrame(channels, columns=["channels"])
    coordinates["x"] = 0
    coordinates["y"] = 0
    coordinates["z"] = 0
    for ch in range(len(channels)):# Getting coordinatesof the channels
        channel_name = channels[ch]
        if any(channel_name == localization_channels):
            coordinates.iloc[ch, 1]  = localization["x"][np.where(
                        channel_name == localization_channels)[0][0]]
            coordinates.iloc[ch, 2]  = localization["y"][np.where(
                channel_name == localization_channels)[0][0]]
            coordinates.iloc[ch, 3]  = localization["z"][np.where(
                channel_name == localization_channels)[0][0]]
        else:
            # if channel has no localization, then just assume 0,0,0. Dont want to deal nans right now.
            coordinates.iloc[ch, 1] = 0

    return coordinates

def get_functional_connectivity_for_tissue_subnetworks(FC, freq, GM_index, WM_index):
    #get FC values for just the GM-GM connections and WM-WM connections
    FC_tissue = [None] *4 # All connections, GM-only connections, WM-only connection, GM-WM connections
    for t in range(len(FC_tissue)):
        FC_tissue[t] = []
    for s in range(len(FC)):
        #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
        FC_tissue[0].append(   FC[s][freq]       )
        FC_tissue[1].append(   utils.reorderAdj(FC[s][freq], GM_index)       )
        FC_tissue[2].append(   utils.reorderAdj(FC[s][freq], WM_index)  )
        FC_tissue[3].append(   utils.getAdjSubset(FC[s][freq], GM_index, WM_index) )
    return FC_tissue


def get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures, index, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                 FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM,
                                                 FC_TYPES_ind, freq_ind):
    sub = patientsWithseizures["subject"][index]
    FC_type = FC_TYPES[FC_TYPES_ind]

    channels, FC = metadata_iEEG.get_FunctionalConnectivity(sub, idKey = patientsWithseizures["idKey"][index], username = USERNAME, password = PASSWORD,
                                        BIDS = paths.BIDS, dataset ="derivatives/iEEGorgDownload", session = SESSION,
                                        functionalConnectivityPath = join(paths.BIDS_DERIVATIVES_FUNCTIONAL_CONNECTIVITY_IEEG, f"sub-{sub}"),
                                        secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                        fsds = FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_type)


    FC = functional_connectivity_absolute_value(FC, FC_type)
    # Get atlas localization and distances
    localization, localization_channels = atl.get_atlas_localization_file( sub, SESSION, paths)
    dist, GM_index, WM_index, dist_order = get_channel_distances_from_WM(localization, localization_channels, channels, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM)

    FC_tissue = get_functional_connectivity_for_tissue_subnetworks(FC, freq_ind, GM_index, WM_index)
    return FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue


def combine_functional_connectivity_from_all_patients_and_segments(patientsWithseizures, indexes, metadata_iEEG, MONTAGE, FC_TYPES,
                                                                   STATE_NUMBER, FREQUENCY_NAMES, USERNAME, PASSWORD, FREQUENCY_DOWN_SAMPLE, paths,
                                                                   SESSION, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM):
    paientList = pd.DataFrame(columns=["patient"])
    summaryStats = pd.DataFrame( columns=["patient", "seizure_number", "FC_type", "frequency", "interictal", "preictal", "ictal", "postictal"] )
    FCtissueAll = [None] *len(indexes)
    seizure_number = []
    #FCtissueAll: patient x FC type x freq x tissue type (all, GM-only, WM-only, GM-WM) x state
    count = 0
    for pt in range(len(indexes)):
        i = indexes[pt]
        paientList = paientList.append(dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
        count_func = 0
        for func in range(len(FC_TYPES)):
            if count_func == 0: FCtissueAll[pt] = [None] * len(FC_TYPES)
            count_freq = 0
            for freq in range(len(FREQUENCY_NAMES)):
                if count_freq == 0: FCtissueAll[pt][func] = [None] * len(FREQUENCY_NAMES)
                sub = patientsWithseizures["subject"][i]
                FC_type = FC_TYPES[func]
                FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order ,FC_tissue = get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures, i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                 FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM,
                                                 func, freq)
                FCtissueAll[pt][func][freq] = FC_tissue

                diff = np.array([np.nanmedian(k) for k in zip(FC_tissue[2] )]) -  np.array([np.nanmedian(k) for k in zip(FC_tissue[1] )])
                summaryStats = summaryStats.append(dict(patient = sub ,seizure_number = i, FC_type = FC_type, frequency = FREQUENCY_NAMES[freq], interictal = diff[0], preictal = diff[1], ictal = diff[2], postictal = diff[3]) , ignore_index=True   )
                count_freq = count_freq +1
            count_func = count_func + 1
        count =  count + 1
        utils.printProgressBar(pt+1, len(indexes))
        seizure_number.append(i)
    summaryStatsLong = pd.melt(summaryStats, id_vars = ["patient", "frequency", "FC_type", "seizure_number"], var_name = "state", value_name = "FC_deltaT")
    #summaryStatsLong = summaryStatsLong.groupby(['patient', "frequency", "FC_type", "state"]).mean()
    summaryStatsLong.reset_index(inplace=True)
    return summaryStatsLong, FCtissueAll, seizure_number




def summaryStatsLong_bootstrap(summaryStatsLong, ratio_patients = 0.8, max_seizures = 2):
    """

    :param summaryStatsLong: summaryStatsLong from above
    :type summaryStatsLong: TYPE
    :param ratio_patients: ratio of patients being randomly sampled from the full patient list, defaults to 0.8
    :type ratio_patients: TYPE, optional
    :param max_seizures: maximum number of seizures taken from each patient. If only 1 seizure, then will pick that one, defaults to 2
    :type max_seizures: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """

    patient_unique = np.unique(summaryStatsLong["patient"])
    N_bootstrap = int(np.rint((len(patient_unique) *ratio_patients)))
    patient_bootstrap = np.random.choice(list(patient_unique ), N_bootstrap )
    seizure_number_bootstrap = []
    for i in range(len(patient_bootstrap)):
       seizures_unique = np.unique(summaryStatsLong[summaryStatsLong["patient"] == patient_bootstrap[i]]["seizure_number"])
       if len(seizures_unique) > 2:
           seizure_number_bootstrap.append(np.random.choice(list(seizures_unique.astype(int)), len(seizures_unique)))
       elif len(seizures_unique) == 2:
           seizure_number_bootstrap.append(np.random.choice(list(seizures_unique.astype(int)), len(seizures_unique)))
       else:
           seizure_number_bootstrap.append([seizures_unique[0]])

    seizure_number_bootstrap = [item for sublist in seizure_number_bootstrap for item in sublist]
    summaryStatsLong_bootstrap = summaryStatsLong[summaryStatsLong['seizure_number'].isin(seizure_number_bootstrap)]
    return summaryStatsLong_bootstrap, seizure_number_bootstrap


def summaryStatsLong_bootstrap_GOOD_vs_POOR(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor, ratio_patients = 3, max_seizures = 1):
    """

    :param summaryStatsLong: summaryStatsLong from above
    :type summaryStatsLong: TYPE
    :param ratio_patients: ratio of patients being randomly sampled from the full patient list, defaults to 0.8
    :type ratio_patients: TYPE, optional
    :param max_seizures: maximum number of seizures taken from each patient. If only 1 seizure, then will pick that one, defaults to 2
    :type max_seizures: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    good_poor = [patient_outcomes_good, patient_outcomes_poor]
    seizure_number_bootstrap = []
    for gp in range(2):
        patient_outcomes = good_poor[gp]
        N_bootstrap_good = int(np.rint((len(patient_outcomes) *ratio_patients)))
        patient_bootstrap = np.random.choice(list(patient_outcomes ), N_bootstrap_good )
        seizure_number_bootstrap_gp = []
        for i in range(len(patient_bootstrap)):
           seizures_unique = np.unique(summaryStatsLong[summaryStatsLong["patient"] == patient_bootstrap[i]]["seizure_number"])
           if len(seizures_unique) > 2:
               seizure_number_bootstrap_gp.append(np.random.choice(list(seizures_unique.astype(int)), len(seizures_unique)))
           elif len(seizures_unique) == 2:
               seizure_number_bootstrap_gp.append(np.random.choice(list(seizures_unique.astype(int)), len(seizures_unique)))
           else:
               seizure_number_bootstrap_gp.append(np.random.choice(list(seizures_unique.astype(int)), len(seizures_unique)))
               #seizure_number_bootstrap_gp.append([seizures_unique[0]])
        seizure_number_bootstrap.append(seizure_number_bootstrap_gp)

    seizure_number_bootstrap_good = [item for sublist in seizure_number_bootstrap[0] for item in sublist]
    seizure_number_bootstrap_poor = [item for sublist in seizure_number_bootstrap[1] for item in sublist]
    summaryStatsLong_bootstrap_good = summaryStatsLong[summaryStatsLong['seizure_number'].isin(seizure_number_bootstrap_good)]
    summaryStatsLong_bootstrap_poor = summaryStatsLong[summaryStatsLong['seizure_number'].isin(seizure_number_bootstrap_poor)]
    return summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor








def bootstrap_patientsWithseizures(patientsWithseizures, indices_to_include, ratio_patients = 0.8, max_seizures = 2):
    df = patientsWithseizures.iloc[indices_to_include]
    patient_unique = np.unique(df["subject"])
    N_bootstrap = int(np.rint((len(patient_unique) *ratio_patients)))
    patient_bootstrap = np.random.choice(list(patient_unique ), N_bootstrap )
    index_number_bootstrap = []
    for i in range(len(patient_bootstrap)):
       index_unique = np.unique(patientsWithseizures.index[patientsWithseizures["subject"] == patient_bootstrap[i]])
       if len(index_unique) > 2:
           index_number_bootstrap.append(np.random.choice(list(index_unique.astype(int)), len(index_unique)))
       elif len(index_unique) == 2:
           index_number_bootstrap.append(np.random.choice(list(index_unique.astype(int)), len(index_unique)))
       else:
           index_number_bootstrap.append([index_unique[0]])

    index_number_bootstrap = [item for sublist in index_number_bootstrap for item in sublist]
    return index_number_bootstrap

def bootstrap_sfc_patient_list(patientsWithseizures, sfc_patient_list, ratio_patients = 0.8, max_seizures = 2):
    patient_unique = np.unique(sfc_patient_list)
    N_bootstrap = int(np.rint((len(patient_unique) *ratio_patients)))
    patient_bootstrap = np.random.choice(list(patient_unique ), N_bootstrap )
    index_number_bootstrap = []
    for i in range(len(patient_bootstrap)):
       index_unique = np.unique(patientsWithseizures.index[patientsWithseizures["subject"] == patient_bootstrap[i]])
       if len(index_unique) > 2:
           index_number_bootstrap.append(np.random.choice(list(index_unique.astype(int)), len(index_unique)))
       elif len(index_unique) == 2:
           index_number_bootstrap.append(np.random.choice(list(index_unique.astype(int)), len(index_unique)))
       else:
           index_number_bootstrap.append(np.random.choice(list(index_unique.astype(int)), len(index_unique)))
           #index_number_bootstrap.append([index_unique[0]])

    index_number_bootstrap = [item for sublist in index_number_bootstrap for item in sublist]
    return index_number_bootstrap



def FCtissueAll_bootstrap(FCtissueAll, seizure_number, seizure_number_bootstrap):
    """

    :param FCtissueAll: from combine_functional_connectivity_from_all_patients_and_segments
    :type FCtissueAll: TYPE
    :param seizure_number_bootstrap: from summaryStatsLong_bootstrap
    :type seizure_number_bootstrap: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    #finding the indices of FCtissueAll that corresponds to which seizure number was boostrapped
    sorted_array = np.sort(seizure_number_bootstrap)
    x,ind, y = np.intersect1d(  seizure_number,  sorted_array, return_indices=True)
    ind_array = []
    for a in range(len(sorted_array)):
        ind_array.append( ind[np.where(sorted_array[a] == x)[0]] )
    ind_array_propper = np.array([item for sublist in ind_array for item in sublist])
    FCtissueAll_bootstrap = [FCtissueAll[i] for i in ind_array_propper]
    return FCtissueAll_bootstrap


def FCtissueAll_flatten(FCtissueAll_bootstrap, STATE_NUMBER, func = 2 ,freq = 7, max_connections = 10):
    """

    :param FCtissueAll_bootstrap: DESCRIPTION
    :type FCtissueAll_bootstrap: TYPE
    :param STATE_NUMBER: DESCRIPTION
    :type STATE_NUMBER: TYPE
    :param func: DESCRIPTION, defaults to 2
    :type func: TYPE, optional
    :param freq: DESCRIPTION, defaults to 7
    :type freq: TYPE, optional
    :param max_connections: maximum number of connectivity values of upper triangle of adjacency matrix pulled for each adj matrix. This is so that regardless of adj size for each patient, an equal number is pulled, defaults to 1000
    :type max_connections: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    #FCtissueAll: patient x FC type x freq x tissue type (all, GM-only, WM-only, GM-WM) x state

    FCtissueAll_bootstrap_flatten= []
    for t in range(len(FCtissueAll_bootstrap[0][0][0])):
        FCtissueAll_bootstrap_flatten.append( [None] * STATE_NUMBER)

    #FCtissueAll_bootstrap_flatten: tissue (all, GM-only, WM-only, GM-WM), state
    for i in range(len(FCtissueAll_bootstrap)):
        for s in range(STATE_NUMBER):
            for t in range(len(FCtissueAll_bootstrap[0][0][0])): #number of tissue types
                if t == 0 or t ==1: #if GM or WM connections (not GM-WM connections)
                    connections = utils.getUpperTriangle(FCtissueAll_bootstrap[i][func][freq][t][s])
                else:
                    connections = FCtissueAll_bootstrap[i][func][freq][t][s].flatten()
                #bootstrapping
                if len(connections) > max_connections:
                    connections = np.array(  np.random.choice(connections, max_connections) )
                #initializing
                if i == 0:
                    FCtissueAll_bootstrap_flatten[t][s] = connections
                else:
                    FCtissueAll_bootstrap_flatten[t][s] = np.concatenate(  [FCtissueAll_bootstrap_flatten[t][s], connections ] )
    pvals = []
    for s in range(STATE_NUMBER):
        pvals.append(stats.ks_2samp( FCtissueAll_bootstrap_flatten[1][s], FCtissueAll_bootstrap_flatten[2][s] )[1] )
    #pvals are the diff between GM-on and WM-only
    return FCtissueAll_bootstrap_flatten, pvals


def deltaT_stats(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func , freq ):
    df = summaryStatsLong_bootstrap.loc[summaryStatsLong_bootstrap['FC_type'] == FC_TYPES[func]].loc[summaryStatsLong_bootstrap['frequency'] == FREQUENCY_NAMES[freq]]
    p1 = stats.ttest_1samp(np.array(df.loc[df["state"] == "interictal"]["FC_deltaT"]), 0)[1]
    p2 = stats.ttest_1samp(np.array(df.loc[df["state"] == "preictal"]["FC_deltaT"]), 0)[1]
    p3 = stats.ttest_1samp(np.array(df.loc[df["state"] == "ictal"]["FC_deltaT"]), 0)[1]
    p4 = stats.ttest_1samp(np.array(df.loc[df["state"] == "postictal"]["FC_deltaT"]), 0)[1]
    p5 = stats.wilcoxon( np.array(df.loc[df["state"] == "ictal"]["FC_deltaT"]) , np.array(df.loc[df["state"] == "preictal"]["FC_deltaT"])  )[1]
    return p1, p2, p3, p4, p5

def deltaT_bootstrap_means(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func , freq ):
    df = summaryStatsLong_bootstrap.loc[summaryStatsLong_bootstrap['FC_type'] == FC_TYPES[func]].loc[summaryStatsLong_bootstrap['frequency'] == FREQUENCY_NAMES[freq]]

    mean_1 = df.loc[df["state"] == "interictal"]["FC_deltaT"].mean()
    mean_2 = df.loc[df["state"] == "preictal"]["FC_deltaT"].mean()
    mean_3 = df.loc[df["state"] == "ictal"]["FC_deltaT"].mean()
    mean_4 = df.loc[df["state"] == "postictal"]["FC_deltaT"].mean()
    return mean_1, mean_2, mean_3, mean_4

#%% Calculate FC as a function of purity


def get_FC_vs_tissue_definition(save_directory, patientsWithseizures, indexes, MONTAGE, TISSUE_DEFINITION,
                                tissue_definition_flag, FC_TYPES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths, FREQUENCY_DOWN_SAMPLE, save_pickle = False, recalculate = False ):
    fname = join(f"{save_directory}",f"summaryStats_Wm_FC_{MONTAGE}_{TISSUE_DEFINITION[0]}.pickle")
    if utils.checkIfFileDoesNotExist(fname) or recalculate:
        if tissue_definition_flag == 0:
            WM_def_array = np.arange(0,8,0.25)
        if tissue_definition_flag == 1:
            WM_def_array = TISSUE_DEFINITION[3][:-2]#[ int(len(TISSUE_DEFINITION[3])/2):-1]

        summaryStats_Wm_FC = pd.DataFrame( columns=["patient", "seizure_number", "FC_type", "frequency", "WM_cutoff", "WM_median_distance", "interictal", "preictal", "ictal", "postictal"] )

        count = 0
        for pt in range(len(indexes)):#[3, 5, 22]: #
            i = indexes[pt]
            for func in range(len(FC_TYPES)):
                for freq in range(len(FREQUENCY_NAMES)):
                    FC_type = FC_TYPES[func]
                    sub = patientsWithseizures["subject"][i]

                    FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = get_functional_connectivity_and_tissue_subnetworks_for_single_patient(
                        patientsWithseizures, i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                        FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION[0],  TISSUE_DEFINITION[1],  TISSUE_DEFINITION[2], func, freq)

                    coordinates = get_channel_xyz(localization, localization_channels, channels )
                    coordinates_array = np.array(coordinates[["x","y","z"]])
                    distance_pairwise = utils.get_pariwise_distances(coordinates_array)
                    for wm in range(len(WM_def_array)):

                        GM_index = np.where(dist["distance"] <=  WM_def_array[wm])[0]
                        WM_index = np.where(dist["distance"] > WM_def_array[wm])[0]
                        #get FC values for just the GM-GM connections and WM-WM connections

                        FC_tissue = get_functional_connectivity_for_tissue_subnetworks(FC, freq, GM_index, WM_index)
                        for a in range(len(FC_tissue)):
                            for b in range(len(FC_tissue[a])):
                                FC_tissue[a][b] = utils.getUpperTriangle(FC_tissue[a][b])
                        wm_median_distance = np.nanmedian(utils.getUpperTriangle( utils.reorderAdj(distance_pairwise, WM_index)))
                        wmFC = np.array([np.nanmedian(k) for k in zip(FC_tissue[2] )])
                        summaryStats_Wm_FC = summaryStats_Wm_FC.append(dict(patient = sub ,seizure_number = i, FC_type = FC_type, frequency = FREQUENCY_NAMES[freq], WM_cutoff = WM_def_array[wm], WM_median_distance = wm_median_distance , interictal = wmFC[0], preictal = wmFC[1], ictal = wmFC[2], postictal = wmFC[3]) , ignore_index=True   )
            utils.printProgressBar(pt+1, len(indexes))
        if save_pickle:
            utils.save_pickle(summaryStats_Wm_FC, fname  )
    else:
        summaryStats_Wm_FC = utils.open_pickle(fname)
    return summaryStats_Wm_FC




def bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state, print_results = False):

    summaryStats_Wm_FC_bootstrap, seizure_number_bootstrap = summaryStatsLong_bootstrap(summaryStats_Wm_FC, ratio_patients = 0.8, max_seizures = 2)
    summaryStats_Wm_FC_bootstrap_func_freq = summaryStats_Wm_FC_bootstrap.loc[summaryStats_Wm_FC_bootstrap["FC_type"] == FC_TYPES[func]].loc[summaryStats_Wm_FC_bootstrap["frequency"] == FREQUENCY_NAMES[freq]]
    summaryStats_Wm_FC_bootstrap_func_freq = summaryStats_Wm_FC_bootstrap_func_freq.groupby(['patient', "WM_cutoff", "WM_median_distance"]).mean()
    summaryStats_Wm_FC_bootstrap_func_freq.reset_index(inplace=True)
    summaryStats_Wm_FC_bootstrap_func_freq_long = pd.melt(summaryStats_Wm_FC_bootstrap_func_freq, id_vars = ["patient", "WM_cutoff", "WM_median_distance"], var_name = "state", value_name = "FC")
    summaryStats_Wm_FC_bootstrap_func_freq_long_state = summaryStats_Wm_FC_bootstrap_func_freq_long.loc[summaryStats_Wm_FC_bootstrap_func_freq_long["state"] == STATE_NAMES[state]]

    model_lin = sm.OLS.from_formula("FC ~ WM_cutoff + WM_median_distance", data=summaryStats_Wm_FC_bootstrap_func_freq_long_state)
    result_lin = model_lin.fit()
    if print_results:
        print(result_lin.summary())
    return summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin





def bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC_PVALUES(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state):
    pvalues = np.zeros((iterations))
    for it in range(iterations):
        summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin = bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations,
                                                                                                                    summaryStats_Wm_FC, FC_TYPES,
                                                                                                                    FREQUENCY_NAMES, STATE_NAMES,
                                                                                                                    func, freq, state, print_results = False)
        pvalues[it] = result_lin.pvalues.loc["WM_cutoff"]
        utils.printProgressBar(it+1, iterations)
    return pvalues



#%%
#Structure-function analysis

def get_tissue_SFC(patientsWithseizures, sfc_patient_list, paths,
                   FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                   TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM,
                   ratio_patients = 5, max_seizures = 1,
                   func = 2, freq = 5, print_pvalues = True):
    index_number_bootstrap = bootstrap_sfc_patient_list(patientsWithseizures, sfc_patient_list, ratio_patients = ratio_patients, max_seizures = max_seizures)

    all_corr = np.zeros((len(index_number_bootstrap),4))
    all_corr_gm = np.zeros((len(index_number_bootstrap),4))
    all_corr_wm = np.zeros((len(index_number_bootstrap),4))
    all_corr_gmwm = np.zeros((len(index_number_bootstrap),4))
    delta_corr = np.zeros((len(index_number_bootstrap),4))
    for b in range(len(index_number_bootstrap)):
        i = index_number_bootstrap[b]
        sub = patientsWithseizures["subject"][i]
        #print(f"{sub}: {i}")
        path_SC = join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", "connectivity", "electrodeContactAtlas")

        path_SC_contacts = glob.glob(join(path_SC, "*.txt"))[0]
        SC = utils.read_DSI_studio_Txt_files_SC(path_SC_contacts)
        SC_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(path_SC_contacts)
        SC = utils.log_normalize_adj(SC)

        ##%% Look at FC for SC
        sub = patientsWithseizures["subject"][i]
        FCtype = FC_TYPES[func]

        FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures, i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                     FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                     TISSUE_DEFINITION_NAME,
                                                     TISSUE_DEFINITION_GM,
                                                     TISSUE_DEFINITION_WM,
                                                     func, freq)

        corr = []
        corr_gm = []
        corr_wm = []
        corr_gmwm = []
        for s in range(len(FC)):
            state = s
            adj = copy.deepcopy(FC[state][freq])
            order = utils.get_intersect1d_original_order(channels, SC_names)

            SC_names_in_FC = SC_names[order]
            SC_order = utils.reorderAdj(SC, order)
            missing_delete_in_FC = utils.find_missing(channels, SC_names).astype(int)
            channels_in_SC = copy.deepcopy(channels); channels_in_SC = np.delete(channels_in_SC,missing_delete_in_FC)
            if not all(SC_names_in_FC == channels_in_SC): #checking that the structural connectivity and functional connectivity rows and columns match up
                IOError(f"{sub}: {i}: SC and FC matrices do not match up " )
            adj = np.delete(adj, missing_delete_in_FC, 0)
            adj = np.delete(adj, missing_delete_in_FC, 1) #making sure both SC and FC have the same rows and columns represented
            #utils.plot_adj_heatmap(SC_order)
            #utils.plot_adj_heatmap(adj)


            corr.append( spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj))[0])
            #sns.regplot(x = utils.getUpperTriangle(SC_order), y = utils.getUpperTriangle(adj), scatter_kws={'s':2})

            #if False:
            #    adj = bct.null_model_und_sign(adj)[0] #utils.plot_adj_heatmap(adj); utils.plot_adj_heatmap(adj_rand)
            #    SC_order = bct.null_model_und_sign(SC_order)[0] #utils.plot_adj_heatmap(SC_order); utils.plot_adj_heatmap(adj)

            #SFC for tissue
            dist_new = dist.drop(missing_delete_in_FC) #need new index of GM and WM because deleted FC channels that were not in SC
            GM_index = np.where(dist_new["distance"] <= TISSUE_DEFINITION_GM)[0]
            WM_index = np.where(dist_new["distance"] > TISSUE_DEFINITION_WM)[0]

            adj_gm =  utils.reorderAdj(adj, GM_index)
            SC_order_gm = utils.reorderAdj(SC_order, GM_index)
            adj_wm =  utils.reorderAdj(adj, WM_index)
            SC_order_wm = utils.reorderAdj(SC_order, WM_index)
            adj_gmwm =  utils.getAdjSubset(adj, GM_index, WM_index)
            SC_order_gmwm = utils.getAdjSubset(SC_order, GM_index, WM_index)
            ind_zero = np.where(utils.getUpperTriangle(SC_order_gm) == 0)

            corr_gm.append( spearmanr(utils.getUpperTriangle(SC_order_gm), utils.getUpperTriangle(adj_gm))[0] )
            corr_wm.append( spearmanr(utils.getUpperTriangle(SC_order_wm), utils.getUpperTriangle(adj_wm))[0] )
            corr_gmwm.append( spearmanr(SC_order_gmwm.flatten(), adj_gmwm.flatten())[0] )
            if s == 1:
                pi_all = spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj))[0]
                pi_gm = spearmanr(utils.getUpperTriangle(SC_order_gm), utils.getUpperTriangle(adj_gm))[0]
                pi_wm = spearmanr(utils.getUpperTriangle(SC_order_wm), utils.getUpperTriangle(adj_wm))[0]
                pi_gmwm = spearmanr(SC_order_gmwm.flatten(), adj_gmwm.flatten())[0]
            if s ==2:
                delta_corr[b,:] = [spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj))[0] - pi_all,
                    spearmanr(utils.getUpperTriangle(SC_order_gm), utils.getUpperTriangle(adj_gm))[0] - pi_gm,
                    spearmanr(utils.getUpperTriangle(SC_order_wm), utils.getUpperTriangle(adj_wm))[0] - pi_wm,
                    spearmanr(SC_order_gmwm.flatten(), adj_gmwm.flatten())[0] - pi_gmwm ]

        all_corr[b,:] = corr
        all_corr_gm[b,:] = corr_gm
        all_corr_wm[b,:] = corr_wm
        all_corr_gmwm[b,:] = corr_gmwm
        if print_pvalues: utils.printProgressBar(b+1,len(index_number_bootstrap) )

    df_all_corr = pd.DataFrame(all_corr, columns = STATE_NAMES)
    df_all_corr["tissue"] = "Full Network"
    df_all_corr_gm = pd.DataFrame(all_corr_gm, columns = STATE_NAMES)
    df_all_corr_gm["tissue"] = "GM"
    df_all_corr_wm = pd.DataFrame(all_corr_wm, columns = STATE_NAMES)
    df_all_corr_wm["tissue"] = "WM"
    df_all_corr_gmwm = pd.DataFrame(all_corr_gmwm, columns = STATE_NAMES)
    df_all_corr_gmwm["tissue"] = "GM-WM"
    df = pd.concat([df_all_corr,df_all_corr_gm ,df_all_corr_gmwm ,df_all_corr_wm ])
    df_long = pd.melt(df,id_vars = ["tissue"], var_name = "state", value_name = "FC")

    if print_pvalues:
        print("\n\n\n")
        print( f"Full Network; Ictal vs Preictal    {stats.ttest_rel(all_corr[:,2] ,all_corr[:,1])[1]*4}"   )
        print(f"GM Network; Ictal vs Preictal      {stats.ttest_rel(all_corr_gm[:,2] ,all_corr_gm[:,1])[1]*4}")
        print(f"GM-WM Network; Ictal vs Preictal   {stats.ttest_rel(all_corr_gmwm[:,2] ,all_corr_gmwm[:,1])[1]*4}")
        print(f"WM Network; Ictal vs Preictal      {stats.ttest_rel(all_corr_wm[:,2] ,all_corr_wm[:,1])[1]*4}")


        print("\n")
        print(f" GM-GM  vs WM; delta Corr      {stats.ttest_rel(delta_corr[:,1] ,delta_corr[:,2])[1]}")
        print(f" GM-GM vs GM-WM; delta Corr    {stats.ttest_rel(delta_corr[:,1] ,delta_corr[:,3])[1]}")
        print(f" WM vs GM-WM; delta Corr       {stats.ttest_rel(delta_corr[:,2] ,delta_corr[:,3])[1]}")

        #delta_corr: patient x delta correlation preictal to ictal (All, gm, wm , gm-wm)
    return df_long, delta_corr




def get_SC_and_FC_adj(patientsWithseizures, index, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                 FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                 TISSUE_DEFINITION_NAME,
                                                 TISSUE_DEFINITION_GM,
                                                 TISSUE_DEFINITION_WM,
                                                 func = 2, freq = 5, state = 2 ):
    i = index
    sub = patientsWithseizures["subject"][i]
    print(f"{sub}: {i}")
    path_SC = join(paths.BIDS_DERIVATIVES_TRACTOGRAPHY, f"sub-{sub}", "connectivity", "electrodeContactAtlas")

    path_SC_contacts = glob.glob(join(path_SC, "*.txt"))[0]
    SC = utils.read_DSI_studio_Txt_files_SC(path_SC_contacts)
    SC_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(path_SC_contacts)
    SC = utils.log_normalize_adj(SC)

    ##%% Look at FC for SC
    sub = patientsWithseizures["subject"][i]
    FCtype = FC_TYPES[func]

    FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures, i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                 FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                 TISSUE_DEFINITION_NAME,
                                                 TISSUE_DEFINITION_GM,
                                                 TISSUE_DEFINITION_WM,
                                                 func, freq)

    adj = copy.deepcopy(FC[state][freq])
    order = utils.get_intersect1d_original_order(channels, SC_names)

    SC_names_in_FC = SC_names[order]
    SC_order = utils.reorderAdj(SC, order)
    missing_delete_in_FC = utils.find_missing(channels, SC_names).astype(int)
    channels_in_SC = copy.deepcopy(channels); channels_in_SC = np.delete(channels_in_SC,missing_delete_in_FC)
    if not all(SC_names_in_FC == channels_in_SC): #checking that the structural connectivity and functional connectivity rows and columns match up
        IOError(f"{sub}: {i}: SC and FC matrices do not match up " )
    adj = np.delete(adj, missing_delete_in_FC, 0)
    adj = np.delete(adj, missing_delete_in_FC, 1) #making sure both SC and FC have the same rows and columns represented

    #SFC for tissue
    dist_new = dist.drop(missing_delete_in_FC) #need new index of GM and WM because deleted FC channels that were not in SC
    GM_index = np.where(dist_new["distance"] <= TISSUE_DEFINITION_GM)[0]
    WM_index = np.where(dist_new["distance"] > TISSUE_DEFINITION_WM)[0]

    adj_gm =  utils.reorderAdj(adj, GM_index)
    SC_order_gm = utils.reorderAdj(SC_order, GM_index)
    adj_wm =  utils.reorderAdj(adj, WM_index)
    SC_order_wm = utils.reorderAdj(SC_order, WM_index)
    adj_gmwm =  utils.getAdjSubset(adj, GM_index, WM_index)
    SC_order_gmwm = utils.getAdjSubset(SC_order, GM_index, WM_index)
    ind_zero = np.where(utils.getUpperTriangle(SC_order_gm) == 0)
    return SC_order, SC_order_gm, SC_order_wm, SC_order_gmwm, adj, adj_gm, adj_wm, adj_gmwm
    """
    from sklearn import linear_model

    reg = linear_model.TweedieRegressor(power=1, alpha=0)
    #reg = linear_model.LinearRegression()
    fig, axes = utils.plot_make(c =4)
    x = utils.getUpperTriangle(SC_order); y = utils.getUpperTriangle(adj); reg.fit(x.reshape(-1, 1), y); y_predict = reg.predict(x.reshape(-1, 1))
    sns.lineplot(x = x, y = y_predict, ax= axes[0]); sns.scatterplot(x = x, y = y, ax= axes[0])
    x = utils.getUpperTriangle(SC_order_gm); y = utils.getUpperTriangle(adj_gm); reg.fit(x.reshape(-1, 1), y); y_predict = reg.predict(x.reshape(-1, 1))
    sns.lineplot(x = x, y = y_predict, ax= axes[1]); sns.scatterplot(x =x, y = y, ax= axes[1])
    x = utils.getUpperTriangle(SC_order_wm); y = utils.getUpperTriangle(adj_wm); reg.fit(x.reshape(-1, 1), y); y_predict = reg.predict(x.reshape(-1, 1))
    sns.lineplot(x = x, y = y_predict, ax= axes[2]); sns.scatterplot(x = x, y = y, ax= axes[2])
    x = SC_order_gmwm.flatten(); y = adj_gmwm.flatten(); reg.fit(x.reshape(-1, 1), y); y_predict = reg.predict(x.reshape(-1, 1))
    sns.lineplot(x = x, y = y_predict, ax= axes[3]); sns.scatterplot(x = x, y = y, ax= axes[3])
    fig.suptitle(STATE_NAMES[s])
    axes[0].title.set_text( np.round( spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(adj))[0],2)   )
    axes[1].title.set_text( np.round( spearmanr(utils.getUpperTriangle(SC_order_gm), utils.getUpperTriangle(adj_gm))[0],2)   )
    axes[2].title.set_text( np.round( spearmanr(utils.getUpperTriangle(SC_order_wm), utils.getUpperTriangle(adj_wm))[0],2)   )
    axes[3].title.set_text( np.round( spearmanr(SC_order_gmwm.flatten(), adj_gmwm.flatten())[0],2)   )
    plt.show()
    """


def multicore_sfc(it, TISSUE_TYPE_NAMES2,STATE_NUMBER, patientsWithseizures, sfc_patient_list, paths,
                       FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                       TISSUE_DEFINITION, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM,
                       ratio_patients = 5, max_seizures = 1,
                       func = 2, freq = 0, print_pvalues = False):
    np.random.seed()
    df_long, delta_corr = get_tissue_SFC(patientsWithseizures, sfc_patient_list, paths,
                       FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                       TISSUE_DEFINITION, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM,
                       ratio_patients = ratio_patients, max_seizures = max_seizures,
                       func = func, freq = freq, print_pvalues = print_pvalues)
    means = np.zeros((len(TISSUE_TYPE_NAMES2) , STATE_NUMBER))
    means_delta_corr = np.zeros((len(TISSUE_TYPE_NAMES2) ))
    for t in range(len(TISSUE_TYPE_NAMES2)):
        for s in range(STATE_NUMBER):
            means[t, s] = df_long.query(f'tissue == "{TISSUE_TYPE_NAMES2[t]}" and state == "{STATE_NAMES[s]}"')["FC"].mean()
        means_delta_corr[t] = delta_corr[:,t].mean()

    #utils.printProgressBar(it+1,iterations )
    p1 = stats.ttest_rel(df_long.query('tissue == "Full Network" and state == "ictal"')["FC"] ,df_long.query('tissue == "Full Network" and state == "preictal"')["FC"])[1]
    p2 = stats.ttest_rel(df_long.query('tissue == "GM" and state == "ictal"')["FC"] ,df_long.query('tissue == "GM" and state == "preictal"')["FC"])[1]
    p3 = stats.ttest_rel(df_long.query('tissue == "GM-WM" and state == "ictal"')["FC"] ,df_long.query('tissue == "GM-WM" and state == "preictal"')["FC"])[1]
    p4 = stats.ttest_rel(df_long.query('tissue == "WM" and state == "ictal"')["FC"] ,df_long.query('tissue == "WM" and state == "preictal"')["FC"])[1]
    pvalues_delta = [p1,p2,p3,p4]
    pvalues_delta_Tissue = [stats.ttest_rel(delta_corr[:,1] ,delta_corr[:,2])[1],
                                  stats.ttest_rel(delta_corr[:,1] ,delta_corr[:,3])[1],
                                  stats.ttest_rel(delta_corr[:,2] ,delta_corr[:,3])[1]]
    return means, means_delta_corr



def multicore_sfc_wrapper(cores,iterations, TISSUE_TYPE_NAMES2, STATE_NUMBER, patientsWithseizures, sfc_patient_list, paths,
                       FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                       TISSUE_DEFINITION, TISSUE_DEFINITION_GM,TISSUE_DEFINITION_WM,
                       ratio_patients = 5, max_seizures = 1,
                       func = 2, freq = 0, print_pvalues = False ):
    p3 = Pool(cores)
    p3.close(); p3.join(); p3.clear()
    simulation = p3.map(multicore_sfc, range(iterations),
                       repeat(TISSUE_TYPE_NAMES2),
                       repeat(STATE_NUMBER),
                       repeat(patientsWithseizures),
                       repeat(sfc_patient_list),
                       repeat(paths),
                       repeat(FC_TYPES),
                       repeat(STATE_NAMES),
                       repeat(FREQUENCY_NAMES),
                       repeat(metadata_iEEG),
                       repeat(SESSION),
                       repeat(USERNAME),
                       repeat(PASSWORD),
                       repeat(FREQUENCY_DOWN_SAMPLE),
                       repeat(MONTAGE),
                       repeat(TISSUE_DEFINITION),
                       repeat(TISSUE_DEFINITION_GM),
                       repeat(TISSUE_DEFINITION_WM),
                       repeat(ratio_patients),
                       repeat(max_seizures),
                       repeat(func),
                       repeat(freq),
                       repeat(print_pvalues)
                        )
    p3.close(); p3.join(); p3.clear()
    return simulation


#%%

#good vs poor

###########################

def wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                          patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                          FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = 1, max_seizures = 1, func = 2, freq = 7,
                          closest_wm_threshold = 40):
    #for permutation:

    summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor = summaryStatsLong_bootstrap_GOOD_vs_POOR(
         summaryStatsLong,
         patient_outcomes_good, patient_outcomes_poor,
         ratio_patients = ratio_patients, max_seizures = max_seizures)
    patient_inds_bootstrap = seizure_number_bootstrap_good + seizure_number_bootstrap_poor

    delta = []
    icccc = []
    piiii = []
    difference_in_gmwm_fc = []

    gm_to_wm_all = []
    gm_to_wm_all_ablated = []
    gm_to_wm_all_ablated_closest_wm = []
    gm_to_wm_all_ablated_closest_wm_gradient = []
    for pt in range(len(patient_inds_bootstrap)):
        i = patient_inds_bootstrap[pt]

        sub = patientsWithseizures["subject"][i]

        FCtype = FC_TYPES[func]

        FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures,
                                                     i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                     FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                     TISSUE_DEFINITION_NAME,
                                                     TISSUE_DEFINITION_GM,
                                                     TISSUE_DEFINITION_WM,
                                                     func, freq)
        coordinates = get_channel_xyz(localization, localization_channels, channels )

        coordinates_array = np.array(coordinates[["x","y","z"]])
        distance_pairwise = utils.get_pariwise_distances(coordinates_array)

        closest_wm_threshold = closest_wm_threshold #in mm



        manual_resected_electrodes = metadata_iEEG.get_manual_resected_electrodes(sub)
        manual_resected_electrodes = np.array(utils.channel2std(manual_resected_electrodes))
        manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
        #print(manual_resected_electrodes)
        FCtissue = get_functional_connectivity_for_tissue_subnetworks(FC, freq, GM_index, WM_index)


        ictal = []
        preictal = []

        preictal_all_connectivity_per_channel = []
        ictal_all_connectivity_per_channel = []

        gm_to_wm_per_patient = []
        gm_to_wm_per_patient_ablated = []
        gm_to_wm_per_patient_ablated_closest_wm = []
        gm_to_wm_per_patient_ablated_closest_wm_gradient = []
        for s in range(4): #getting average gm-to-wm connectivity and gm to gm
            state = s
            adj = FC[state][freq]
            gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index)
            gm_to_wm_median= np.nanmedian(utils.getUpperTriangle(gm_to_wm))
            gm_to_wm_per_patient.append(gm_to_wm.flatten())

        gm_to_wm_all.append(gm_to_wm_per_patient)
        for s in [1,2]:
            state = s


            adj = FC[state][freq]

            gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index)
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            #sns.histplot(gm_to_wm_average, bins = 20)
            gm_to_wm_per_patient_ablated_per_channel = []
            gm_to_wm_per_patient_ablated_per_channel_closest = []
            gm_to_wm_per_patient_ablated_per_channel_closest_gradient = []
            for ch in range(len(manual_resected_electrodes)):
                #ch = 1
                #print(dist[channels == manual_resected_electrodes[ch]])
                ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
                ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WM_index)[0]
                gm_to_wm_per_patient_ablated_per_channel.append(ablated_wm_fc.flatten())
                gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index  )
                gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
                #sns.histplot(gm_to_wm_average, bins = 20)
                #plt.show()
                value = np.nanmedian(ablated_wm_fc)
                #print(np.nanmedian(ablated_wm_fc))


                all_wm_connectivity = utils.getAdjSubset(adj, np.array(range(len(adj))), WM_index  )
                all_wm_connectivity_average = np.nanmedian(all_wm_connectivity, axis = 1)

                #getting FC of closest WM to ablated region
                closest_wm_index = WM_index[np.where(distance_pairwise[ablated_ind,:][0][WM_index] <= closest_wm_threshold)[0]]
                ablated_closest_wm_fc = utils.getAdjSubset(adj, ablated_ind, closest_wm_index)[0]
                gm_to_wm_per_patient_ablated_per_channel_closest.append(ablated_closest_wm_fc.flatten())
                #getting FC of closest WM to ablated region GRADIENT
                closest_wm_threshold_gradient = np.arange(15, 125,5 )  #in mm
                gradient = []
                for gr in range(len(closest_wm_threshold_gradient)):
                    closest_wm_index = WM_index[np.where((distance_pairwise[ablated_ind,:][0][WM_index] < closest_wm_threshold_gradient[gr] ) & (distance_pairwise[ablated_ind,:][0][WM_index] > closest_wm_threshold_gradient[gr] - 15))[0]]
                    ablated_closest_wm_fc = utils.getAdjSubset(adj, ablated_ind, closest_wm_index)[0]
                    gradient.append(ablated_closest_wm_fc.flatten())
                gm_to_wm_per_patient_ablated_per_channel_closest_gradient.append(gradient)
                ##
                if s != 2:
                    preictal.append(ablated_wm_fc)
                    #preictal.append(np.nanmedian(ablated_wm_fc))
                    preictal_all_connectivity_per_channel = all_wm_connectivity_average

                if s == 2:
                    ictal.append(ablated_wm_fc)
                    #ictal.append(np.nanmedian(ablated_wm_fc))
                    ictal_all_connectivity_per_channel = all_wm_connectivity_average
            gm_to_wm_per_patient_ablated.append([item for sublist in gm_to_wm_per_patient_ablated_per_channel for item in sublist])
            gm_to_wm_per_patient_ablated_closest_wm.append([item for sublist in gm_to_wm_per_patient_ablated_per_channel_closest for item in sublist])
            #getting FC of closest WM to ablated region GRADIENT
            gm_to_wm_per_patient_ablated_per_channel_closest_gradient_reorganized = []
            for gr in range(len(closest_wm_threshold_gradient)):
                gradient_tmp= []
                for ch in range(len(manual_resected_electrodes)):
                    gradient_tmp.append( gm_to_wm_per_patient_ablated_per_channel_closest_gradient[ch][gr])
                tmp =   [item for sublist in gradient_tmp for item in sublist]
                gm_to_wm_per_patient_ablated_per_channel_closest_gradient_reorganized.append(tmp)
            gm_to_wm_per_patient_ablated_closest_wm_gradient.append(gm_to_wm_per_patient_ablated_per_channel_closest_gradient_reorganized)

            if s == 2:
                dd = np.array(ictal) - np.array(preictal)
                delta.append(dd)
                icccc.append(ictal)
                piiii.append(preictal)

                diff1 = ictal_all_connectivity_per_channel - preictal_all_connectivity_per_channel





        gm_to_wm_all_ablated.append(gm_to_wm_per_patient_ablated)
        gm_to_wm_all_ablated_closest_wm.append(gm_to_wm_per_patient_ablated_closest_wm)
        gm_to_wm_all_ablated_closest_wm_gradient.append(gm_to_wm_per_patient_ablated_closest_wm_gradient)

    return summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient








def wm_vs_gm_good_vs_poor_iteration(iterations,
                              summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                              patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                              FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = 1, max_seizures = 1, func = 2, freq = 7,
                              permute = False, closest_wm_threshold = 85, avg = 0):

    all_patients = patient_outcomes_good + patient_outcomes_poor

    test_statistic_array_delta = np.zeros((iterations))
    test_statistic_array_ablated = np.zeros((iterations))
    for it in range(iterations):
        if permute:
            #assign good or poor
            outcomes = []
            outcomes = random.choices(["good", "poor"], weights=(len(patient_outcomes_good), len(patient_outcomes_poor)), k=len(all_patients))
            patient_outcomes_good_permute = np.array(all_patients)[np.where(np.array(outcomes) == "good")[0]].tolist()
            patient_outcomes_poor_permute = np.array(all_patients)[np.where(np.array(outcomes) == "poor")[0]].tolist()

            patient_outcomes_good_permute = random.choices(list(all_patients), k = len(patient_outcomes_good))
            patient_outcomes_poor_permute = random.choices(list(all_patients),k = len(patient_outcomes_poor))
            summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient = wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good_permute, patient_outcomes_poor_permute,
                              patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                              FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq,
                              closest_wm_threshold = closest_wm_threshold)
        else:
            summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient = wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                                  patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                  FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq,
                                  closest_wm_threshold = closest_wm_threshold)

        #delta is the change in FC of the ablated contacts to wm

        delta_mean = []
        delta_mean = np.zeros(len(delta))
        for x in range(len(delta)):
            #delta_mean.append(sum(x)/len(x))
            delta_mean[x] = delta[x].mean()



        #good = np.concatenate(delta[:7])
        #poor = np.concatenate(delta[8:])
        #good = np.concatenate(delta[:len(seizure_number_bootstrap_good)])
        #poor = np.concatenate(delta[len(seizure_number_bootstrap_good):])
        good = delta_mean[:len(seizure_number_bootstrap_good)]
        poor = delta_mean[len(seizure_number_bootstrap_good):]
        df_good = pd.DataFrame(dict(good = good))
        df_poor = pd.DataFrame(dict(poor = poor))

        test_statistic_array_delta[it] = np.array(df_poor.mean()) - np.array(df_good.mean())[0] #stats.ttest_ind(poor,good)[0]  #stats.mannwhitneyu(poor,good)[0]
        test_statistic_array_delta[it] = stats.ttest_ind(poor,good)[0]  #stats.mannwhitneyu(poor,good)[0]
        #test_statistic_array_delta[it] = stats.ttest_ind(poor,good, equal_var=False)[0]  #stats.mannwhitneyu(poor,good)[0]
        #utils.printProgressBar(it + 1, iterations, suffix = np.round(test_statistic_array_delta[it],2) )

        difference_gm_to_wm_ablated = []
        for l in range(len(gm_to_wm_all_ablated)):
            difference_gm_to_wm_ablated.append(np.array(gm_to_wm_all_ablated[l][1] )- np.array(gm_to_wm_all_ablated[l][0]))
            difference_gm_to_wm_ablated[l] = np.random.choice( difference_gm_to_wm_ablated[l] , 5)

        difference_gm_to_wm_good_ablated = difference_gm_to_wm_ablated[:len(seizure_number_bootstrap_good)]
        difference_gm_to_wm_poor_ablated = difference_gm_to_wm_ablated[len(seizure_number_bootstrap_good):]


        difference_gm_to_wm_good_ablated = [item for sublist in difference_gm_to_wm_good_ablated for item in sublist]
        difference_gm_to_wm_poor_ablated = [item for sublist in difference_gm_to_wm_poor_ablated for item in sublist]
        test_statistic_array_ablated[it] = np.array(difference_gm_to_wm_poor_ablated).mean() - np.array(difference_gm_to_wm_good_ablated).mean()
        #test_statistic_array_ablated[it] = stats.ttest_ind(difference_gm_to_wm_poor_ablated,difference_gm_to_wm_good_ablated, equal_var=False)[0] #np.array(difference_gm_to_wm_poor_ablated).mean() - np.array(difference_gm_to_wm_good_ablated).mean()
        #test_statistic_array_ablated[it] = stats.mannwhitneyu(difference_gm_to_wm_poor_ablated,difference_gm_to_wm_good_ablated)[0] #np.array(difference_gm_to_wm_poor_ablated).mean() - np.array(difference_gm_to_wm_good_ablated).mean()
        test_statistic_array_ablated[it] = stats.ttest_ind( difference_gm_to_wm_poor_ablated,difference_gm_to_wm_good_ablated )[0]

        if permute:
            pvalue = len( np.where(avg  <= test_statistic_array_delta[:it+1]  )[0]) / (it+1)
            print(f"\r{it}: {np.round(test_statistic_array_delta[it],3)}   mean: {np.around(test_statistic_array_delta[:it+1].mean(),3)}   pval: {np.round(pvalue,3)}", end="\r")
        else:
            pvalue = len( np.where(avg  >= test_statistic_array_delta[:it+1].mean() )[0]) / (iterations)
            print(f"\r{it}: {np.round(test_statistic_array_delta[it],3)}   mean: {np.around(test_statistic_array_delta[:it+1].mean(),3)}   pval: {np.round(pvalue,3)}", end="\r")
        """
        fig, axes = utils.plot_make()
        sns.histplot(difference_gm_to_wm_good_ablated, ax = axes, kde = True, color = "blue", legend = True, binwidth = 0.01, binrange = [-0.05,0.5])
        sns.histplot(difference_gm_to_wm_poor_ablated, ax = axes, kde = True, color = "orange", binwidth = 0.01, binrange = [-0.05,0.5])
        fig.legend(labels=['good','poor'])

        fig, axes = utils.plot_make()
        sns.ecdfplot(difference_gm_to_wm_good_ablated, ax = axes, color = "blue", legend = True)
        sns.ecdfplot(difference_gm_to_wm_poor_ablated, ax = axes, color = "orange")
        fig.legend(labels=['difference_gm_to_wm_good_ablated','difference_gm_to_wm_poor_ablated'])

        medians_good = [np.nanmean(item) for item in difference_gm_to_wm_good_ablated]
        medians_poor = [np.nanmean(item) for item in difference_gm_to_wm_poor_ablated]
        medians_good = [medians_good for medians_good in medians_good if str(medians_good) != 'nan']
        medians_poor = [medians_poor for medians_poor in medians_poor if str(medians_poor) != 'nan']
        stats.ttest_ind(medians_poor,medians_good)
        test_statistic_array_closest[it] = stats.ttest_ind(medians_poor,medians_good)[0]
        """



        if it+1 % 100== 0 and it >0:
            binwidth = 1
            binrange = [-20,20]
            fig, axes = utils.plot_make()
            sns.histplot( test_statistic_array_delta[:it+1], kde = True, ax = axes, color = "#bbbbbb", binwidth = None, binrange = None)


            axes.axvline(x=test_statistic_array_delta[:it+1].mean(), color='k', linestyle='--')
            #axes.axvline(x=avg, color='k', linestyle='--')
            #axes.set_xlim([-20,20])
            #axes.set_ylim([0,10])
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            plt.show()
    return test_statistic_array_delta, test_statistic_array_ablated


#%% Multi core

def multicore_wm_good_vs_poor(it ,summaryStatsLong,
                                         all_patients,
                                         patient_outcomes_good,
                                         patient_outcomes_poor,
                                         patientsWithseizures,
                                         metadata_iEEG,
                                         SESSION,
                                         USERNAME,
                                         PASSWORD,
                                         paths,
                                         FREQUENCY_DOWN_SAMPLE,
                                         MONTAGE,
                                         FC_TYPES,
                                         TISSUE_DEFINITION_NAME,
                                         TISSUE_DEFINITION_GM,
                                         TISSUE_DEFINITION_WM,
                                         ratio_patients,
                                         max_seizures,
                                         func,
                                         freq,
                                         closest_wm_threshold,
                                         avg):
    print(it)
    np.random.seed()


    summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient = wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                          patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                          FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq,
                          closest_wm_threshold = closest_wm_threshold)

    #delta is the change in FC of the ablated contacts to wm
    delta_mean = []
    delta_mean = np.zeros(len(delta))
    for x in range(len(delta)):
        delta_mean[x] = delta[x].mean()
    good = delta_mean[:len(seizure_number_bootstrap_good)]
    poor = delta_mean[len(seizure_number_bootstrap_good):]
    df_good = pd.DataFrame(dict(good = good))
    df_poor = pd.DataFrame(dict(poor = poor))

    test_statistic_array_delta = stats.ttest_ind(poor,good)[0]


    #permutation
    #assign good or poor
    patient_outcomes_good_permute = random.choices(list(all_patients), k = len(patient_outcomes_good))
    patient_outcomes_poor_permute = random.choices(list(all_patients),k = len(patient_outcomes_poor))
    summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient = wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good_permute, patient_outcomes_poor_permute,
                      patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                      FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq,
                      closest_wm_threshold = closest_wm_threshold)
    delta_mean = []
    delta_mean = np.zeros(len(delta))
    for x in range(len(delta)):
        delta_mean[x] = delta[x].mean()
    good = delta_mean[:len(seizure_number_bootstrap_good)]
    poor = delta_mean[len(seizure_number_bootstrap_good):]
    df_good = pd.DataFrame(dict(good = good))
    df_poor = pd.DataFrame(dict(poor = poor))
    test_statistic_array_delta_permute = stats.ttest_ind(poor,good)[0]

    return test_statistic_array_delta, test_statistic_array_delta_permute

#%%


def multicore_wm_good_vs_poor_wrapper(cores, iterations,
                              summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                              patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                              FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = 1, max_seizures = 1, func = 2, freq = 7,
                              permute = False, closest_wm_threshold = 85, avg = 0):

    all_patients = patient_outcomes_good + patient_outcomes_poor
    p = Pool(cores)
    simulation = p.map(multicore_wm_good_vs_poor, range(iterations),
                                         repeat(summaryStatsLong),
                                         repeat(all_patients),
                                         repeat(patient_outcomes_good),
                                         repeat(patient_outcomes_poor),
                                         repeat(patientsWithseizures),
                                         repeat(metadata_iEEG),
                                         repeat(SESSION),
                                         repeat(USERNAME),
                                         repeat(PASSWORD),
                                         repeat(paths),
                                         repeat(FREQUENCY_DOWN_SAMPLE),
                                         repeat(MONTAGE),
                                         repeat(FC_TYPES),
                                         repeat(TISSUE_DEFINITION_NAME),
                                         repeat(TISSUE_DEFINITION_GM),
                                         repeat(TISSUE_DEFINITION_WM),
                                         repeat(ratio_patients),
                                         repeat(max_seizures),
                                         repeat(func),
                                         repeat(freq),
                                         repeat(closest_wm_threshold),
                                         repeat(avg)

              )

    p.close()
    p.join()
    p.clear()

    return simulation










#%%


def permute_resampling_pvalues(summaryStatsLong, FCtissueAll, seizure_number, patient_outcomes_good, patient_outcomes_poor,FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, ratio_patients = 5, max_seizures = 1, func = 7, freq = 7):

    summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor = summaryStatsLong_bootstrap_GOOD_vs_POOR(
        summaryStatsLong,
        patient_outcomes_good, patient_outcomes_poor,
        ratio_patients = ratio_patients, max_seizures = max_seizures)

    FCtissueAll_bootstrap_good = FCtissueAll_bootstrap(FCtissueAll, seizure_number, seizure_number_bootstrap_good)
    FCtissueAll_bootstrap_poor = FCtissueAll_bootstrap(FCtissueAll, seizure_number, seizure_number_bootstrap_poor)
    FCtissueAll_bootstrap_outcomes = [FCtissueAll_bootstrap_good, FCtissueAll_bootstrap_poor]

    df = pd.DataFrame( columns=["outcome", "tissue", "state", "FC"] )

    OUTCOME_NAMES = ["good", "poor"]
    TISSUE_TYPE_NAMES = ["Full Network", "GM-only", "WM-only", "GM-WM"]

    for o in range(2):
        for t in range(4):
            for s in range(4):
                FCtissueAll_bootstrap_outcomes_single = FCtissueAll_bootstrap_outcomes[o]
                for i in range(len(FCtissueAll_bootstrap_outcomes_single)):
                    fc = np.nanmedian(  utils.getUpperTriangle(FCtissueAll_bootstrap_outcomes_single[i][func][freq][t][s]))
                    df = df.append( dict( outcome = OUTCOME_NAMES[o], tissue =  TISSUE_TYPE_NAMES[t], state = STATE_NAMES[s] , FC = fc  ), ignore_index=True )

    summaryStatsLong_bootstrap_good.insert(0, 'outcome', 'good')
    summaryStatsLong_bootstrap_poor.insert(0, 'outcome', 'poor')

    summaryStatsLong_bootstrap_outcome = pd.concat( [ summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor])
    tmp = summaryStatsLong_bootstrap_outcome[summaryStatsLong_bootstrap_outcome["frequency"] == FREQUENCY_NAMES[freq]]

    summaryStatsLong_bootstrap_outcome = tmp[tmp["FC_type"] == FC_TYPES[func]]

    s=2
    v1 = summaryStatsLong_bootstrap_outcome[(summaryStatsLong_bootstrap_outcome["state"]==STATE_NAMES[s])&(summaryStatsLong_bootstrap_outcome["outcome"]<="good")].dropna()["FC_deltaT"]
    v2 = summaryStatsLong_bootstrap_outcome[(summaryStatsLong_bootstrap_outcome["state"]==STATE_NAMES[s])&(summaryStatsLong_bootstrap_outcome["outcome"]<="poor")].dropna()["FC_deltaT"]
    test_statistic = stats.ttest_ind(v2, v1); test_statistic = test_statistic[0]

    return test_statistic, summaryStatsLong_bootstrap_outcome


def mutilcore_permute_resampling(it, summaryStatsLong, FCtissueAll, seizure_number, patient_outcomes_good, patient_outcomes_poor,FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, ratio_patients = 5, max_seizures = 1 , func = 7, freq = 7):
    print(it)
    #permute good vs bad:
    np.random.seed()
    all_patients = patient_outcomes_good + patient_outcomes_poor
    patient_outcomes_good_permute = random.choices(list(all_patients), k = len(patient_outcomes_good))
    patient_outcomes_poor_permute = random.choices(list(all_patients),k = len(patient_outcomes_poor))
    test_statistic_array, _ = permute_resampling_pvalues(summaryStatsLong, FCtissueAll, seizure_number, patient_outcomes_good_permute, patient_outcomes_poor_permute,FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq)
    original_array,_ = permute_resampling_pvalues(summaryStatsLong, FCtissueAll, seizure_number, patient_outcomes_good, patient_outcomes_poor,FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq)
    return original_array, test_statistic_array


def mutilcore_permute_wrapper(cores, iterations, summaryStatsLong, FCtissueAll, seizure_number, patient_outcomes_good, patient_outcomes_poor, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, ratio_patients = 5, max_seizures = 1, func = 7, freq = 7):


    p2 = Pool(cores)
    p2.close()
    p2.join()
    p2.clear()
    simulation = p2.map(mutilcore_permute_resampling, range(iterations),
                        repeat(summaryStatsLong),
                        repeat(FCtissueAll),
                        repeat(seizure_number),
                        repeat(patient_outcomes_good),
                        repeat(patient_outcomes_poor),
                        repeat(FC_TYPES),
                        repeat(FREQUENCY_NAMES),
                        repeat(STATE_NAMES),
                        repeat(ratio_patients),
                        repeat(max_seizures ) ,
                        repeat(func ),
                        repeat(freq )
                        )
    p2.close()
    p2.join()
    p2.clear()


    return simulation






def compute_T(df_DT, group  = True, state = "ictal", i = 0, equal_var=True, var = "FC_deltaT", alternative='two-sided'):
    if group:
        df_DT_patient= df_DT.groupby(by=["patient", "outcome"]).mean().reset_index()
    else:
        df_DT_patient = copy.deepcopy(df_DT)
    v1 = df_DT_patient.query(f"outcome == 'good' and state == '{state}'  ")[var]
    v2 = df_DT_patient.query(f"outcome == 'poor' and state == '{state}'  ")[var]
    T = stats.ttest_ind(v2, v1, equal_var=equal_var, alternative = alternative)[i]
    return T

def compute_T_no_state(df_DT, group  = True, i = 0, equal_var=True, var = "FC_deltaT", alternative='two-sided'):
    if group:
        df_DT_patient= df_DT.groupby(by=["patient", "outcome"]).mean().reset_index()
    else:
        df_DT_patient = copy.deepcopy(df_DT)
    v1 = df_DT_patient.query(f"outcome == 'good'   ")[var]
    v2 = df_DT_patient.query(f"outcome == 'poor'  ")[var]
    T = stats.ttest_ind(v2, v1, equal_var=equal_var, alternative = alternative)[i]
    return T

def permute_df_DT(summaryStatsLong_outcome):
    permute = copy.deepcopy(summaryStatsLong_outcome)
    patients = np.unique(permute["patient"])
    for pt in range(len(patients)):
        outcome = np.array(permute.query(f'patient == "{patients[pt]}"')["outcome"])[0]
        switch = stats.binom.rvs(1, 0.5, size=1)[0]#simulating probability of switching outcomes
        if switch == 1:
            if outcome == "good": outcome = "poor"
            else: outcome = "good"
        permute.loc[ permute["patient"] == patients[pt]     , "outcome"] = outcome
    return permute




def bootstrap_df_DT(summaryStatsLong_outcome_seizures_not_combined):
    np.random.seed()
    bootstrap = copy.deepcopy(summaryStatsLong_outcome_seizures_not_combined)
    patients = np.unique(bootstrap["patient"])
    patients_bootstrap = np.random.choice(patients, len(patients))
    boostrap_df = pd.DataFrame( columns = bootstrap.columns )
    for pt in range(len(patients_bootstrap)):
        seizure_number = np.array(bootstrap.query(f'patient == "{patients_bootstrap[pt]}" and state == "ictal"')["seizure_number"])
        bootstrap_seizure_number =  np.random.choice(list(seizure_number), len(seizure_number))
        for sz in range(len(bootstrap_seizure_number)):
            boostrap_df = boostrap_df.append( bootstrap.query(f'patient == "{patients_bootstrap[pt]}" and seizure_number == {bootstrap_seizure_number[sz]}')   )
    return boostrap_df
#bootstrap original data


def multicore_deltaT(it, summaryStatsLong_outcome_seizures_not_combined):
    means = pd.DataFrame(columns = ["iteration", "state", "outcome", "FC_deltaT"])
    summaryStatsLong_outcome_bootstrap = bootstrap_df_DT(summaryStatsLong_outcome_seizures_not_combined)
    means_df = summaryStatsLong_outcome_bootstrap.groupby(by=['state', "outcome"]).mean().reset_index()
    means_df["iteration"] = it
    means = means.append( means_df[["iteration", "state", "outcome", "FC_deltaT" ]])
    return means


def mutilcore_deltaT_wrapper(cores, iterations, summaryStatsLong_outcome_seizures_not_combined):
    p4 = Pool(cores)
    p4.close(); p4.join(); p4.clear()
    simulation = p4.map(multicore_deltaT, range(iterations),
                        repeat(summaryStatsLong_outcome_seizures_not_combined)
                        )
    p4.close(); p4.join(); p4.clear()
    return simulation

def add_outcomes_to_summaryStatsLong(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor):
    summaryStatsLong["outcome"] = "NA"

    for i in range(len(summaryStatsLong)):
        patient = summaryStatsLong["patient"][i]
        if patient in patient_outcomes_good:
            summaryStatsLong.loc[  summaryStatsLong["patient"] == patient  , "outcome"] = "good"
        if patient in patient_outcomes_poor:
            summaryStatsLong.loc[  summaryStatsLong["patient"] == patient  , "outcome"] = "poor"
        utils.printProgressBar(i+1,len(summaryStatsLong) )
    return summaryStatsLong




def multicore_permute_deltaT(it, summaryStatsLong_outcome, state_bool = True, group = False, var = "FC_deltaT"):
    np.random.seed()
    summaryStatsLong_outcome_permute = permute_df_DT(summaryStatsLong_outcome)
    if state_bool:
        T_star = compute_T(summaryStatsLong_outcome_permute, group  = False)
    else:
        T_star =compute_T_no_state(summaryStatsLong_outcome_permute, group  = group, var = "delta")
    return T_star


def mutilcore_permute_deltaT_wrapper(cores, iterations, summaryStatsLong_outcome, state_bool = True, group = False, var = "FC_deltaT"):
    p4 = Pool(cores)
    p4.close(); p4.join(); p4.clear()
    simulation = p4.map(multicore_permute_deltaT, range(iterations),
                        repeat(summaryStatsLong_outcome),
                        repeat(state_bool),
                        repeat(group),
                        repeat(var)
                        )
    p4.close(); p4.join(); p4.clear()
    return simulation


def deltaT_multicore(it, summaryStatsLong, FCtissueAll, STATE_NUMBER, seizure_number, FREQUENCY_NAMES, FC_TYPES, func  ,freq  , max_connections = 50):
    np.random.seed()
    summaryStatsLong_bootstrap_var, seizure_number_bootstrap = summaryStatsLong_bootstrap(summaryStatsLong, ratio_patients = 1, max_seizures = 1)
    FCtissueAll_bootstrap_var = FCtissueAll_bootstrap(FCtissueAll, seizure_number, seizure_number_bootstrap)
    FCtissueAll_bootstrap_flatten, pvals_iter = FCtissueAll_flatten(FCtissueAll_bootstrap_var, STATE_NUMBER, func  ,freq  , max_connections = 50)
    medians =  [ [np.nanmedian(item) for item in FCtissueAll_bootstrap_flatten[1]] ,[np.nanmedian(item) for item in FCtissueAll_bootstrap_flatten[2]] ]
    means_deltaT = deltaT_bootstrap_means(summaryStatsLong_bootstrap_var, FREQUENCY_NAMES, FC_TYPES, func , freq )
    return medians, means_deltaT

def deltaT_multicore_wrapper(cores, iterations, summaryStatsLong,
                                                  FCtissueAll, STATE_NUMBER,seizure_number,  FREQUENCY_NAMES,
                                                  FC_TYPES,func  ,freq  , max_connections = 50):
    p4 = Pool(cores)
    p4.close(); p4.join(); p4.clear()
    simulation = p4.map(deltaT_multicore, range(iterations),
                        repeat(summaryStatsLong),
                        repeat(FCtissueAll),
                        repeat(STATE_NUMBER),
                        repeat(seizure_number),
                        repeat( FREQUENCY_NAMES),
                        repeat(FC_TYPES),
                        repeat(func ) ,
                        repeat(freq)  ,
                        repeat(max_connections)
                        )
    p4.close(); p4.join(); p4.clear()
    return simulation





def bootstrap_delta_mean(delta_mean):
    np.random.seed()
    bootstrap = copy.deepcopy(delta_mean)
    patients = np.unique(bootstrap["patient"])
    patients_bootstrap = np.random.choice(patients, len(patients))
    boostrap_df = pd.DataFrame( columns = bootstrap.columns )
    for pt in range(len(patients_bootstrap)):
        seizure_number = np.array(bootstrap.query(f'patient == "{patients_bootstrap[pt]}"')["seizure_number"])
        bootstrap_seizure_number =  np.random.choice(list(seizure_number), len(seizure_number))
        for sz in range(len(bootstrap_seizure_number)):
            boostrap_df = boostrap_df.append( bootstrap.query(f'patient == "{patients_bootstrap[pt]}" and seizure_number == {bootstrap_seizure_number[sz]}')   )
    return boostrap_df
#bootstrap original data

def bootstrap_delta_means_patient(delta_means_patients):
    np.random.seed()
    bootstrap = copy.deepcopy(delta_means_patients)
    patients = np.unique(bootstrap["patient"])
    patients_bootstrap = np.random.choice(patients, len(patients))
    boostrap_df = pd.DataFrame( columns = bootstrap.columns )
    for pt in range(len(patients_bootstrap)):
        boostrap_df = boostrap_df.append( bootstrap.query(f'patient == "{patients_bootstrap[pt]}"')   )
    return boostrap_df
#bootstrap original data


def multicore_delta_mean(it, delta_mean, by_seizure = True):
    means = pd.DataFrame(columns = ["iteration", "outcome", "delta"])
    if by_seizure:
        bootstrap_delta_mean_bootstrap = bootstrap_delta_mean(delta_mean)
    else:
        bootstrap_delta_means_patient = bootstrap_delta_means_patient(delta_mean)
    means_df = bootstrap_delta_mean_bootstrap.groupby(by=[ "outcome"]).mean().reset_index()
    means_df["iteration"] = it
    means = means.append( means_df[["iteration", "outcome", "delta" ]])
    return means


def mutilcore_delta_mean_wrapper(cores, iterations, delta_mean):
    p4 = Pool(cores)
    p4.close(); p4.join(); p4.clear()
    simulation = p4.map(multicore_delta_mean, range(iterations),
                        repeat(delta_mean)
                        )
    p4.close(); p4.join(); p4.clear()
    return simulation

























#%%








def wm_vs_gm_good_vs_poor_redone(summaryStatsLong,
                          patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                          FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM , func = 2, freq = 7,
                          closest_wm_threshold = 40):
    #for permutation:
    inds = summaryStatsLong.query(f"outcome !='NA'").seizure_number.unique().astype(int)

    delta = []
    difference_in_gmwm_fc = []

    gm_to_wm_all = []
    gm_to_wm_all_ablated = []
    gm_to_wm_all_ablated_closest_wm = []
    gm_to_wm_all_ablated_closest_wm_gradient = []

    delta = pd.DataFrame(columns = ["patient", "outcome","seizure_number", "delta"])
    gm_to_wm_all = pd.DataFrame(columns = ["patient", "outcome","seizure_number", "interictal", "preictal", "ictal", "postictal"])
    gm_to_wm_all_ablated = pd.DataFrame(columns = ["patient", "outcome","seizure_number", "gm_to_wm_all_ablated"])
    gm_to_wm_all_ablated_closest_wm = pd.DataFrame(columns = ["patient", "outcome","seizure_number", "gm_to_wm_all_ablated_closest_wm"])
    gm_to_wm_all_ablated_closest_wm_gradient =pd.DataFrame(columns = ["patient", "outcome","seizure_number", "gm_to_wm_all_ablated_closest_wm_gradient"])


    for pt in range(len(inds)):
        i = inds[pt]

        sub = patientsWithseizures["subject"][i]
        outcome = summaryStatsLong.query(f"patient =='{sub}'").outcome.unique()[0]
        FCtype = FC_TYPES[func]

        FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures,
                                                     i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                     FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                     TISSUE_DEFINITION_NAME,
                                                     TISSUE_DEFINITION_GM,
                                                     TISSUE_DEFINITION_WM,
                                                     func, freq)
        coordinates = get_channel_xyz(localization, localization_channels, channels )

        coordinates_array = np.array(coordinates[["x","y","z"]])
        distance_pairwise = utils.get_pariwise_distances(coordinates_array)


        manual_resected_electrodes = metadata_iEEG.get_manual_resected_electrodes(sub)
        manual_resected_electrodes = np.array(utils.channel2std(manual_resected_electrodes))
        manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
        #print(manual_resected_electrodes)
        FCtissue = get_functional_connectivity_for_tissue_subnetworks(FC, freq, GM_index, WM_index)

        ictal = []
        preictal = []
        preictal_all_connectivity_per_channel = []
        ictal_all_connectivity_per_channel = []

        gm_to_wm_per_patient = []
        gm_to_wm_per_patient_ablated = []
        gm_to_wm_per_patient_ablated_closest_wm = []
        gm_to_wm_per_patient_ablated_closest_wm_gradient = []
        for s in range(4): #getting average gm-to-wm connectivity and gm to gm
            state = s
            adj = FC[state][freq]
            gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index)
            gm_to_wm_median= np.nanmedian(utils.getUpperTriangle(gm_to_wm))
            gm_to_wm_per_patient.append(gm_to_wm.flatten())

        gm_to_wm_all = gm_to_wm_all.append( dict( patient = sub, seizure_number = i, outcome = outcome, interictal = gm_to_wm_per_patient[0], preictal = gm_to_wm_per_patient[1], ictal = gm_to_wm_per_patient[2], postictal = gm_to_wm_per_patient[3]), ignore_index=True)
        for s in [1,2]:
            state = s
            adj = FC[state][freq]
            gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index)
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            #sns.histplot(gm_to_wm_average, bins = 20)
            gm_to_wm_per_patient_ablated_per_channel = []
            gm_to_wm_per_patient_ablated_per_channel_closest = []
            gm_to_wm_per_patient_ablated_per_channel_closest_gradient = []
            for ch in range(len(manual_resected_electrodes)):
                #ch = 1
                #print(dist[channels == manual_resected_electrodes[ch]])
                ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
                ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WM_index)[0]
                gm_to_wm_per_patient_ablated_per_channel.append(ablated_wm_fc.flatten())
                gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index  )
                gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
                #sns.histplot(gm_to_wm_average, bins = 20)
                #plt.show()
                value = np.nanmedian(ablated_wm_fc)
                #print(np.nanmedian(ablated_wm_fc))


                all_wm_connectivity = utils.getAdjSubset(adj, np.array(range(len(adj))), WM_index  )
                all_wm_connectivity_average = np.nanmedian(all_wm_connectivity, axis = 1)

                #getting FC of closest WM to ablated region
                closest_wm_index = WM_index[np.where(distance_pairwise[ablated_ind,:][0][WM_index] <= closest_wm_threshold)[0]]
                ablated_closest_wm_fc = utils.getAdjSubset(adj, ablated_ind, closest_wm_index)[0]
                gm_to_wm_per_patient_ablated_per_channel_closest.append(ablated_closest_wm_fc.flatten())
                #getting FC of closest WM to ablated region GRADIENT
                closest_wm_threshold_gradient = np.arange(15, 125,5 )  #in mm
                gradient = []
                for gr in range(len(closest_wm_threshold_gradient)):
                    closest_wm_index = WM_index[np.where((distance_pairwise[ablated_ind,:][0][WM_index] < closest_wm_threshold_gradient[gr] ) & (distance_pairwise[ablated_ind,:][0][WM_index] > closest_wm_threshold_gradient[gr] - 15))[0]]
                    ablated_closest_wm_fc = utils.getAdjSubset(adj, ablated_ind, closest_wm_index)[0]
                    gradient.append(ablated_closest_wm_fc.flatten())
                gm_to_wm_per_patient_ablated_per_channel_closest_gradient.append(gradient)
                ##
                if s != 2:
                    preictal.append(ablated_wm_fc)
                    #preictal.append(np.nanmedian(ablated_wm_fc))
                    preictal_all_connectivity_per_channel = all_wm_connectivity_average

                if s == 2:
                    ictal.append(ablated_wm_fc)
                    #ictal.append(np.nanmedian(ablated_wm_fc))
                    ictal_all_connectivity_per_channel = all_wm_connectivity_average
            gm_to_wm_per_patient_ablated.append([item for sublist in gm_to_wm_per_patient_ablated_per_channel for item in sublist])
            gm_to_wm_per_patient_ablated_closest_wm.append([item for sublist in gm_to_wm_per_patient_ablated_per_channel_closest for item in sublist])
            #getting FC of closest WM to ablated region GRADIENT
            gm_to_wm_per_patient_ablated_per_channel_closest_gradient_reorganized = []
            for gr in range(len(closest_wm_threshold_gradient)):
                gradient_tmp= []
                for ch in range(len(manual_resected_electrodes)):
                    gradient_tmp.append( gm_to_wm_per_patient_ablated_per_channel_closest_gradient[ch][gr])
                tmp =   [item for sublist in gradient_tmp for item in sublist]
                gm_to_wm_per_patient_ablated_per_channel_closest_gradient_reorganized.append(tmp)

            gm_to_wm_per_patient_ablated_closest_wm_gradient.append(gm_to_wm_per_patient_ablated_per_channel_closest_gradient_reorganized)

            if s == 2:
                dd = np.array(ictal) - np.array(preictal)
                delta = delta.append( dict( patient = sub, outcome = outcome, seizure_number = i, delta = dd), ignore_index=True)
                diff1 = ictal_all_connectivity_per_channel - preictal_all_connectivity_per_channel


        gm_to_wm_all_ablated = gm_to_wm_all_ablated.append( dict( patient = sub, outcome = outcome, seizure_number = i, gm_to_wm_all_ablated = gm_to_wm_per_patient_ablated), ignore_index=True)
        gm_to_wm_all_ablated_closest_wm = gm_to_wm_all_ablated_closest_wm.append( dict( patient = sub, outcome = outcome, seizure_number = i, gm_to_wm_all_ablated_closest_wm = gm_to_wm_per_patient_ablated_closest_wm), ignore_index=True)
        gm_to_wm_all_ablated_closest_wm_gradient = gm_to_wm_all_ablated_closest_wm_gradient.append( dict( patient = sub,  outcome = outcome, seizure_number = i, gm_to_wm_all_ablated_closest_wm_gradient = gm_to_wm_per_patient_ablated_closest_wm_gradient), ignore_index=True)
        utils.printProgressBar(pt+1, len(inds))
    return delta, gm_to_wm_all , gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient





























































