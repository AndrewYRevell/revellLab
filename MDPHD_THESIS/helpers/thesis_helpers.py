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

#revellLab
#utilities, constants/parameters, and thesis helper functions
from revellLab.packages.utilities import utils

#package functions
from revellLab.packages.dataclass import dataclass_atlases, dataclass_iEEG_metadata
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.imaging.tractography import tractography

from revellLab.MDPHD_THESIS.plotting import plot_GMvsWM
#%% Power analysis

def power_analysis(patientsWithseizures, indexes, metadata_iEEG, USERNAME, PASSWORD, SESSION, FREQUENCY_DOWN_SAMPLE, MONTAGE, paths, TISSUE_DEFINITION_NAME , TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM):
    paientList = pd.DataFrame(columns=["patient"])
    count = 0
    for i in indexes:
        paientList = paientList.append(dict(patient=patientsWithseizures["subject"][i]), ignore_index = True)
        # get data
        seizure, fs, ictalStartIndex, ictalStopIndex = metadata_iEEG.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], USERNAME, PASSWORD,
                                                                                           BIDS=paths.BIDS, dataset="derivatives/iEEGorgDownload", session=SESSION, secondsBefore=180, secondsAfter=180, load=True)
        interictal, fs = metadata_iEEG.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", patientsWithseizures["AssociatedInterictal"][i], USERNAME, PASSWORD,
                                               BIDS=paths.BIDS, dataset="derivatives/iEEGorgDownload", session= SESSION, startKey="Start", load = True)

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
            localizationChannels = echobase.channel2std(
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
            localizationChannels = echobase.channel2std(
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


def get_atlas_localization_file(sub, SESSION, paths):
    """
    :param sub: DESCRIPTION
    :type sub: string, example 'RID0194' (without the "sub-)
    :param SESSION: DESCRIPTION
    :type SESSION: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    file = join(paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION, f"sub-{sub}", f"ses-{SESSION}", f"sub-{sub}_ses-{SESSION}_desc-atlasLocalization.csv")
    if utils.checkIfFileExistsGlob(file, printBOOL=False):
        localization = pd.read_csv(file)
        localization_channels = localization["channel"]
        localization_channels = echobase.channel2std(np.array(localization_channels))
        return localization, localization_channels


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
    FC_tissue = [None] *3
    for t in range(len(FC_tissue)):
        FC_tissue[t] = []
    for s in range(len(FC)):
        #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
        FC_tissue[0].append(   utils.reorderAdj(FC[s][freq], GM_index)          )
        FC_tissue[1].append(   utils.reorderAdj(FC[s][freq], WM_index)       )
        FC_tissue[2].append(   utils.getAdjSubset(FC[s][freq], GM_index, WM_index) )
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
    localization, localization_channels = get_atlas_localization_file( sub, SESSION, paths)
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
    #FCtissueAll: patient x FC type x freq x tissue type (GM-only, WM-only, all) x state
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

                diff = np.array([np.nanmedian(k) for k in zip(FC_tissue[1] )]) -  np.array([np.nanmedian(k) for k in zip(FC_tissue[0] )])
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
    """
    fig,ax = plt.subplots(2,4,figsize=(16,8), dpi = 300)
    for s in range(STATE_NUMBER):
        sns.ecdfplot(data=FC_tissue[0][s], ax = ax[0][s])
        sns.ecdfplot(data=FC_tissue[1][s], ax = ax[0][s])

        sns.kdeplot(data=FC_tissue[0][s], ax = ax[1][s])
        sns.kdeplot(data=FC_tissue[1][s], ax = ax[1][s])

    fig.suptitle(f"{sub}, {FC_type}, {freq}")
    plt.show()
    time.sleep(0)
    """



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
           seizure_number_bootstrap.append(np.random.choice(list(seizures_unique.astype(int)), max_seizures))
       elif len(seizures_unique) == 2:
           seizure_number_bootstrap.append(np.random.choice(list(seizures_unique.astype(int)), max_seizures))
       else:
           seizure_number_bootstrap.append([seizures_unique[0]])

    seizure_number_bootstrap = [item for sublist in seizure_number_bootstrap for item in sublist]
    summaryStatsLong_bootstrap = summaryStatsLong[summaryStatsLong['seizure_number'].isin(seizure_number_bootstrap)]
    return summaryStatsLong_bootstrap, seizure_number_bootstrap


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
    _,ind, _ = np.intersect1d(  seizure_number,  np.sort(seizure_number_bootstrap), return_indices=True)
    FCtissueAll_bootstrap = [FCtissueAll[i] for i in ind]
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
    #FCtissueAll: patient x FC type x freq x tissue type (GM-only, WM-only, all) x state
    FCtissueAll_bootstrap_flatten = [[None] * STATE_NUMBER, [None] * STATE_NUMBER, [None] * STATE_NUMBER]


    for i in range(len(FCtissueAll_bootstrap)):
        for s in range(STATE_NUMBER):
            for t in range(3): #number of tissue types
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
        pvals.append(stats.ks_2samp( FCtissueAll_bootstrap_flatten[0][s], FCtissueAll_bootstrap_flatten[1][s] )[1] )

    return FCtissueAll_bootstrap_flatten, pvals


def deltaT_stats(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func , freq ):
    df = summaryStatsLong_bootstrap.loc[summaryStatsLong_bootstrap['FC_type'] == FC_TYPES[func]].loc[summaryStatsLong_bootstrap['frequency'] == FREQUENCY_NAMES[freq]]
    p1 = stats.ttest_1samp(np.array(df.loc[df["state"] == "interictal"]["FC_deltaT"]), 0)[1]
    p2 = stats.ttest_1samp(np.array(df.loc[df["state"] == "preictal"]["FC_deltaT"]), 0)[1]
    p3 = stats.ttest_1samp(np.array(df.loc[df["state"] == "ictal"]["FC_deltaT"]), 0)[1]
    p4 = stats.ttest_1samp(np.array(df.loc[df["state"] == "postictal"]["FC_deltaT"]), 0)[1]
    p5 = stats.wilcoxon( np.array(df.loc[df["state"] == "ictal"]["FC_deltaT"]) , np.array(df.loc[df["state"] == "preictal"]["FC_deltaT"])  )[1]
    return p1, p2, p3, p4, p5





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
                        wmFC = np.array([np.nanmedian(k) for k in zip(FC_tissue[1] )])
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















