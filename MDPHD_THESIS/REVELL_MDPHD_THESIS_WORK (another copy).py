"""
Andy Revell's MD/PHD thesis work
2021
"""
#%% 1/4 Imports
import sys
import os
import json
import copy
import time
import bct
import glob
import math
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
from revellLab.MDPHD_THESIS import constants_parameters as params
from revellLab.MDPHD_THESIS import constants_plotting as plot
from revellLab.paths import constants_paths as paths
from revellLab.MDPHD_THESIS.helpers import thesis_helpers as helper

#package functions
from revellLab.packages.dataclass import dataclass_atlases, dataclass_iEEG_metadata
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.imaging.tractography import tractography
from revellLab.packages.imaging.makeSphericalRegions import make_spherical_regions

#plotting
from revellLab.MDPHD_THESIS.plotting import plot_GMvsWM
from revellLab.MDPHD_THESIS.plotting import plot_seizure_distributions
#% 2/4 Paths and File names


with open(paths.METADATA_IEEG_DATA) as f: JSON_iEEG_metadata = json.load(f)
with open(paths.ATLAS_FILES_PATH) as f: JSON_atlas_files = json.load(f)
with open(paths.IEEG_USERNAME_PASSWORD) as f: IEEG_USERNAME_PASSWORD = json.load(f)

#data classes
atlases = dataclass_atlases.dataclass_atlases(JSON_atlas_files)
metadata_iEEG = dataclass_iEEG_metadata.dataclass_iEEG_metadata(JSON_iEEG_metadata)


#% 3/4 Paramters

#ieeg.org username and password
USERNAME = IEEG_USERNAME_PASSWORD["username"]
PASSWORD = IEEG_USERNAME_PASSWORD["password"]

#montaging
MONTAGE = params.MONTAGE_BIPOLAR
SAVE_FIGURES = plot.SAVE_FIGURES[1]

#Frequencies
FREQUENCY_NAMES = params.FREQUENCY_NAMES
FREQUENCY_DOWN_SAMPLE = params.FREQUENCY_DOWN_SAMPLE

#Functional Connectivity
FC_TYPES = params.FC_TYPES

#States
STATE_NAMES = params.STATE_NAMES
STATE_NUMBER = params.STATE_NUMBER_TOTAL

#Imaging
SESSION = params.SESSION_IMPLANT
SESSION_RESEARCH3T = params.SESSION_RESEARCH3T
ACQ = params.ACQUISITION_RESEARCH3T_T1_MPRAGE
IEEG_SPACE = params.IEEG_SPACE

#Tissue definitions
TISSUE_DEFINITION = params.TISSUE_DEFINITION_PERCENT
TISSUE_DEFINITION_NAME = TISSUE_DEFINITION[0]
TISSUE_DEFINITION_GM = TISSUE_DEFINITION[1]
TISSUE_DEFINITION_WM = TISSUE_DEFINITION[2]
WM_DEFINITION_SEQUENCE = TISSUE_DEFINITION[3]
WM_DEFINITION_SEQUENCE_IND = TISSUE_DEFINITION[4]




#% 4/4 General Parameter calculation
# get all the patients with annotated seizures
patientsWithseizures = metadata_iEEG.get_patientsWithSeizuresAndInterictal()
N = len(patientsWithseizures)

iEEGpatientList = np.unique(list(patientsWithseizures["subject"]))
iEEGpatientList = ["sub-" + s for s in iEEGpatientList]


#%% Graphing summary statistics of seizures and patient population
#plot distribution of seizures per patient
plot_seizure_distributions.plot_distribution_seizures_per_patient(patientsWithseizures)
utils.save_figure(f"{paths.FIGURES}/seizureSummaryStats/seizureCounts.pdf", save_figure = SAVE_FIGURES)

#plot distribution of seizure lengths
plot_seizure_distributions.plot_distribution_seizure_length(patientsWithseizures)
utils.save_figure(f"{paths.FIGURES}/seizureSummaryStats/seizureLengthDistribution.pdf", save_figure = SAVE_FIGURES)

#%% Electrode and atlas localization
atl.atlasLocalizationBIDSwrapper(iEEGpatientList,  paths.BIDS, "PIER", SESSION, IEEG_SPACE, ACQ,  paths.BIDS_DERIVATIVES_RECONALL,  paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION,
                                 paths.ATLASES, paths.ATLAS_LABELS, paths.MNI_TEMPLATE, paths.MNI_TEMPLATE_BRAIN, multiprocess=False, cores=12, rerun=False)


#%% EEG download and preprocessing of electrodes
for i in range(len(patientsWithseizures)):
    metadata_iEEG.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], USERNAME, PASSWORD,
                                        BIDS=paths.BIDS, dataset="derivatives/iEEGorgDownload", session = SESSION, secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata_iEEG.get_associatedInterictal(patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata_iEEG.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, USERNAME, PASSWORD,
                          BIDS=paths.BIDS, dataset="derivatives/iEEGorgDownload", session= SESSION, startKey="Start", load=False)


#%% Power analysis

###################################
#WM power as a function of distance
fname = join(paths.DATA, "GMvsWM",f"power_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.pickle")
if utils.checkIfFileDoesNotExist(fname): #if power analysis already computed, then don't run
    powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = helper.power_analysis(patientsWithseizures, np.array(range(3, N)), metadata_iEEG, USERNAME, PASSWORD, SESSION, FREQUENCY_DOWN_SAMPLE, MONTAGE, paths, params.TISSUE_DEFINITION_DISTANCE[0] , params.TISSUE_DEFINITION_DISTANCE[1], params.TISSUE_DEFINITION_DISTANCE[2] )
    utils.save_pickle( [powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM], fname)
powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = utils.open_pickle(fname)
#Plots
#show figure for paper: Power vs Distance and SNR
plot_GMvsWM.plot_power_vs_distance_and_SNR(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"heatmap_pwr_vs_Distance_and_SNR_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.png"), save_figure= SAVE_FIGURES)
#Show summary figure
plot_GMvsWM.plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)
#boxplot comparing GM vs WM for the different seizure states (interictal, preictal, ictal, postictal)
plot_GMvsWM.plot_boxplot_tissue_power_differences(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM, plot.COLORS_TISSUE_LIGHT_MED_DARK[1], plot.COLORS_TISSUE_LIGHT_MED_DARK[2])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"boxplot_GM_vs_WM_seizure_state_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.pdf"), save_figure= SAVE_FIGURES)
#statistics
helper.power_analysis_stats(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)


####################################
#WM power as a function of WM percent
fname = join(paths.DATA, "GMvsWM",f"power_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pickle")
if utils.checkIfFileDoesNotExist(fname): #if power analysis already computed, then don't run
    powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = helper.power_analysis(patientsWithseizures, np.array(range(3, N)), metadata_iEEG, USERNAME, PASSWORD, SESSION, FREQUENCY_DOWN_SAMPLE, MONTAGE, paths, params.TISSUE_DEFINITION_PERCENT[0] , params.TISSUE_DEFINITION_PERCENT[1],params.TISSUE_DEFINITION_PERCENT[2] )
    utils.save_pickle( [powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM], fname)
powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = utils.open_pickle(fname)
#Plots
#Show summary figure
plot_GMvsWM.plotUnivariatePercent(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)
#boxplot comparing GM vs WM for the different seizure states (interictal, preictal, ictal, postictal)
plot_GMvsWM.plot_boxplot_tissue_power_differences(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM, plot.COLORS_TISSUE_LIGHT_MED_DARK[1], plot.COLORS_TISSUE_LIGHT_MED_DARK[2])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"boxplot_GM_vs_WM_seizure_state_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"), save_figure= SAVE_FIGURES)
#statistics
helper.power_analysis_stats(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)


#%% Calculating functional connectivity for whole seizure segment
for fc in range(len(FC_TYPES)):
    for i in range(3, N):
        sub = patientsWithseizures["subject"][i]
        functionalConnectivityPath = join(paths.BIDS_DERIVATIVES_FUNCTIONAL_CONNECTIVITY_IEEG, f"sub-{sub}")
        utils.checkPathAndMake(functionalConnectivityPath, functionalConnectivityPath)
        metadata_iEEG.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = USERNAME, password = PASSWORD,
                                            BIDS =paths.BIDS, dataset ="derivatives/iEEGorgDownload", session = SESSION,
                                            functionalConnectivityPath = functionalConnectivityPath,
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                            fsds = FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPES[fc])


#%%
#Combine FC from the above saved calculation, and calculate differences
summaryStatsLong, FCtissueAll, seizure_number = helper.combine_functional_connectivity_from_all_patients_and_segments(patientsWithseizures, np.array(range(3, N)), metadata_iEEG, MONTAGE, FC_TYPES,
                                                                   STATE_NUMBER, FREQUENCY_NAMES, USERNAME, PASSWORD, FREQUENCY_DOWN_SAMPLE,
                                                                   paths, SESSION,  params.TISSUE_DEFINITION_PERCENT[0], params.TISSUE_DEFINITION_PERCENT[1], params.TISSUE_DEFINITION_PERCENT[2])
#%%
#bootstrap
iterations = 1000
pvals = np.zeros((iterations, STATE_NUMBER))
pvals_deltaT = np.zeros((iterations, STATE_NUMBER +1))
medians = np.zeros((iterations,2, STATE_NUMBER)) #the medians of the bootstraps x GM and WM (index 0 and index 1 repspectively) x state
means_deltaT = np.zeros((iterations, STATE_NUMBER))
for it in range(iterations):
    func = 2
    freq = 7
    summaryStatsLong_bootstrap, seizure_number_bootstrap = helper.summaryStatsLong_bootstrap(summaryStatsLong, ratio_patients = 1, max_seizures = 1)

    FCtissueAll_bootstrap = helper.FCtissueAll_bootstrap(FCtissueAll, seizure_number, seizure_number_bootstrap)
    FCtissueAll_bootstrap_flatten, pvals_iter = helper.FCtissueAll_flatten(FCtissueAll_bootstrap, STATE_NUMBER, func  ,freq  , max_connections = 50)
    #FCtissueAll_bootstrap_flatten: tissue (all, GM-only, WM-only, GM-WM), state
    pvals[it,:] = pvals_iter
    tmp = FCtissueAll_bootstrap_flatten[1]
    medians[it,:,:] =  [ [np.nanmedian(item) for item in FCtissueAll_bootstrap_flatten[1]] ,[np.nanmedian(item) for item in FCtissueAll_bootstrap_flatten[2]] ]
    pvals_deltaT[it,:] = helper.deltaT_stats(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func , freq )
    means_deltaT[it,:] = helper.deltaT_bootstrap_means(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func , freq )

    utils.printProgressBar(it +1, iterations)



fig, axes = utils.plot_make(c = STATE_NUMBER, size_length = 12, sharey = True)
for x in range(STATE_NUMBER):

    data = pd.DataFrame(  dict( gm = medians[:, 0, x], wm = medians[:, 1, x]))
    xlim = [math.floor(data.to_numpy().min() * 1000)/1000 , math.ceil(data.to_numpy().max() * 1000)/1000]
    sns.histplot(data =data, palette = plot.COLORS_TISSUE_LIGHT_MED_DARK[1], kde = True, ax = axes[x], bins = 10, line_kws=dict(linewidth = 8), edgecolor = None)
    #sns.kdeplot(data =data, palette = plot.COLORS_TISSUE_LIGHT_MED_DARK[1], ax = axes[x], linewidth = 10)
    axes[x].set_xlim(xlim)
    #axes[x].set_ylim([0,350])
    print(stats.wilcoxon(data["gm"], data["wm"])[1])
    axes[x].spines['top'].set_visible(False)
    axes[x].spines['right'].set_visible(False)
    pvalue = stats.wilcoxon(data["gm"], data["wm"])[1]
    axes[x].set_title(pvalue)
    #print(stats.mannwhitneyu(data["gm"], data["wm"])[1])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"PVALUES_all_patients_GMvsWM_ECDF2_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=True)


data = pd.DataFrame(means_deltaT, columns = STATE_NAMES)
fig, axes = utils.plot_make()
sns.histplot(data =data, palette = plot.COLORS_STATE4[1], kde = True, ax = axes, binwidth = 0.001, binrange = [-0.002,0.025],
             line_kws = dict(lw = 2), kde_kws = dict(bw_method = 1), edgecolor = None)
#sns.kdeplot(data =data, palette = plot.COLORS_STATE4[1], ax = axes, lw = 5)
data.mean()
axes.set_xlim([-0.002,0.022])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.set_title(f"{stats.wilcoxon(data['preictal'], data['ictal'])[1]} {stats.ttest_1samp(data['ictal'], 0)[1]}")
for k in range(len(data.mean())):
    axes.axvline(x=data.mean()[k], color = plot.COLORS_STATE4[1][k], linestyle='--')
axes.legend([],[], frameon=False)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"BOOTSTRAP_all_patients_GMvsWM_ECDF2_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=True)


plot_GMvsWM.plot_FC_all_patients_GMvsWM_ECDF(FCtissueAll_bootstrap_flatten, STATE_NUMBER , plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"ECDF_all_patients_GMvsWM_ECDF2_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=SAVE_FIGURES)

plot_GMvsWM.plot_FC_all_patients_GMvsWM_ECDF_PVALUES(pvals, STATE_NUMBER , plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"ECDF_histplot_pvalues_all_patients_GMvsWM_ECDF_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=SAVE_FIGURES)


plot_GMvsWM.plot_boxplot_single_FC_deltaT(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func, freq, plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"boxplot_single_FC_deltaT_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_deltaT_PVALUES(pvals_deltaT, plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"boxplot_FC_deltaT_PVALUES_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=SAVE_FIGURES)

plot_GMvsWM.plot_boxplot_all_FC_deltaT(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, plot.COLORS_STATE4[0], plot.COLORS_STATE4[1])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM",
                  f"boxplot_all_FC_deltaT_Supplement_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=SAVE_FIGURES)


#%%
#% Plot FC distributions for example patient
#for i in range(3,N):
i=51
func = 2
freq = 7
state = 2
sub = patientsWithseizures["subject"][i]
FC_type = FC_TYPES[func]

FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = helper.get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures,
                                                 i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                 FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                 params.TISSUE_DEFINITION_PERCENT[0], params.TISSUE_DEFINITION_PERCENT[1], params.TISSUE_DEFINITION_PERCENT[2],
                                                 func, freq)

plot_GMvsWM.plot_FC_example_patient_ADJ(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)


plot_GMvsWM.plot_FC_example_patient_GMWMall(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order,
                                         FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot,
                                         xlim = [0,0.6])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"hist_GM_vs_WM_distribution_of_FC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_example_patient_GMvsWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order,
                                           FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot,
                                           xlim = [0,0.6])

utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"hist_GM_vs_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)

plot_GMvsWM.plot_FC_example_patient_GMvsWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"ECDF_GM_vs_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)

#GM-to-WM connections
plot_GMvsWM.plot_FC_example_patient_GMWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order,
                                         FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot,
                                         xlim = [0,0.6])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"hist_GM_to_WM_distribution_of_FC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_example_patient_GMWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"ECDF_GM_to_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)


#%% Calculate FC as a function of purity

save_directory = join(paths.DATA, "GMvsWM")
func = 2; freq = 0; state = 2

## WM definition =  distance
summaryStats_Wm_FC = helper.get_FC_vs_tissue_definition(save_directory, patientsWithseizures, range(3,5), MONTAGE,  params.TISSUE_DEFINITION_DISTANCE,
                                 0, FC_TYPES, FREQUENCY_NAMES,  metadata_iEEG, SESSION, USERNAME, PASSWORD, paths, FREQUENCY_DOWN_SAMPLE, save_pickle = False , recalculate = False)

summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state, print_results = True)
plot_GMvsWM.plot_FC_vs_contact_distance(summaryStats_Wm_FC_bootstrap_func_freq_long_state)
plot_GMvsWM.plot_FC_vs_WM_cutoff(summaryStats_Wm_FC_bootstrap_func_freq_long_state)

pvalues = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC_PVALUES(25, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state)
plot_GMvsWM.plot_FC_vs_WM_cutoff_PVALUES(pvalues, 0.00005, [-0.0,0.001], plot)


## WM definition = percent
summaryStats_Wm_FC = helper.get_FC_vs_tissue_definition(save_directory, patientsWithseizures, range(3,5), MONTAGE,  params.TISSUE_DEFINITION_PERCENT,
                                 1, FC_TYPES, FREQUENCY_NAMES,  metadata_iEEG, SESSION, USERNAME, PASSWORD, paths, FREQUENCY_DOWN_SAMPLE, save_pickle = False , recalculate = False)
summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state, print_results = True)
plot_GMvsWM.plot_FC_vs_contact_distance(summaryStats_Wm_FC_bootstrap_func_freq_long_state,xlim = [15,80] ,ylim = [0.02,0.2])
plot_GMvsWM.plot_FC_vs_WM_cutoff(summaryStats_Wm_FC_bootstrap_func_freq_long_state)

pvalues = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC_PVALUES(25, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state)
plot_GMvsWM.plot_FC_vs_WM_cutoff_PVALUES(pvalues, 0.005, [-0.0,0.1], plot)


#%%

##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

#Structure-function Analysis




#%% DWI correction and tractography


#get patients in patientsWithseizures that have dti
sfc_patient_list = tractography.get_patients_with_dwi(np.unique(patientsWithseizures["subject"]), paths, dataset = "PIER", SESSION_RESEARCH3T = SESSION_RESEARCH3T)
cmd = tractography.print_dwi_image_correction_QSIprep(sfc_patient_list, paths, dataset = "PIER")
tractography.get_tracts_loop_through_patient_list(sfc_patient_list, paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)


make_spherical_regions.make_spherical_regions(sfc_patient_list, SESSION, paths, radius = 7, rerun = False, show_slices = False)
#%%

#Analyzing SFC

iterations = 1000
pvalues_delta = np.zeros((iterations, 4))
pvalues_delta_Tissue = np.zeros((iterations, 3))
means = np.zeros((iterations, len(params.TISSUE_TYPE_NAMES2) , STATE_NUMBER))
means_delta_corr = np.zeros((iterations, len(params.TISSUE_TYPE_NAMES2) ))

for it in range(iterations):
    df_long, delta_corr = helper.get_tissue_SFC(patientsWithseizures, sfc_patient_list, paths,
                       FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                       params.TISSUE_DEFINITION_PERCENT[0], params.TISSUE_DEFINITION_PERCENT[1], params.TISSUE_DEFINITION_PERCENT[2],
                       ratio_patients = 5, max_seizures = 1,
                       func = 2, freq = 0, print_pvalues = False)

    for t in range(len(params.TISSUE_TYPE_NAMES)):
        for s in range(STATE_NUMBER):
            means[it, t, s] = df_long.query(f'tissue == "{params.TISSUE_TYPE_NAMES2[t]}" and state == "{STATE_NAMES[s]}"')["FC"].mean()
        means_delta_corr[it,t] = delta_corr[:,t].mean()

    utils.printProgressBar(it+1,iterations )


    p1 = stats.ttest_rel(df_long.query('tissue == "Full Network" and state == "ictal"')["FC"] ,df_long.query('tissue == "Full Network" and state == "preictal"')["FC"])[1]
    p2 = stats.ttest_rel(df_long.query('tissue == "GM" and state == "ictal"')["FC"] ,df_long.query('tissue == "GM" and state == "preictal"')["FC"])[1]
    p3 = stats.ttest_rel(df_long.query('tissue == "GM-WM" and state == "ictal"')["FC"] ,df_long.query('tissue == "GM-WM" and state == "preictal"')["FC"])[1]
    p4 = stats.ttest_rel(df_long.query('tissue == "WM" and state == "ictal"')["FC"] ,df_long.query('tissue == "WM" and state == "preictal"')["FC"])[1]
    pvalues_delta[it,:] = [p1,p2,p3,p4]
    pvalues_delta_Tissue[it,:] = [stats.ttest_rel(delta_corr[:,1] ,delta_corr[:,2])[1],
                                  stats.ttest_rel(delta_corr[:,1] ,delta_corr[:,3])[1],
                                  stats.ttest_rel(delta_corr[:,2] ,delta_corr[:,3])[1]]



cols = pd.MultiIndex.from_product([ params.TISSUE_TYPE_NAMES2, STATE_NAMES])

means_df = pd.DataFrame(columns = ["tissue", "state", "SFC"])
for t in range(len(params.TISSUE_TYPE_NAMES)):
    df_tissue = pd.DataFrame(means[:,t,:], columns = STATE_NAMES)
    df_tissue = pd.melt(df_tissue, var_name = ["state"], value_name = "SFC")
    df_tissue["tissue"] = params.TISSUE_TYPE_NAMES2[t]
    means_df = pd.concat([means_df, df_tissue])

np.mean(means, axis = 0)


wm_preictal = means_df.query('tissue == "WM" and state == "preictal"')["SFC"]
wm_ictal = means_df.query('tissue == "WM" and state == "ictal"')["SFC"]
stats.ttest_rel(means_df.query('tissue == "WM" and state == "ictal"')["SFC"] ,means_df.query('tissue == "WM" and state == "preictal"')["SFC"])[1]
print(stats.wilcoxon(wm_preictal, wm_ictal)[1])
print(stats.mannwhitneyu(wm_preictal, wm_ictal)[1])




palette_long = ["#808080", "#808080", "#282828", "#808080"] + ["#a08269", "#a08269", "#675241", "#a08269"] + ["#8a6ca1", "#8a6ca1", "#511e79", "#8a6ca1"]+ ["#76afdf", "#76afdf", "#1f5785", "#76afdf"]
palette = ["#bbbbbb", "#bbbbbb", "#282828", "#bbbbbb"] + ["#cebeb1", "#cebeb1", "#675241", "#cebeb1"] + ["#cdc0d7", "#cdc0d7", "#511e79", "#cdc0d7"] + ["#b6d4ee", "#b6d4ee", "#1f5785", "#b6d4ee"]

fig, axes = utils.plot_make()
sns.boxplot(data = means_df , x = "tissue", y = "SFC", hue = "state", ax = axes, showfliers=False, order = ["Full Network", "GM", "GM-WM", "WM"])
sns.stripplot(data= means_df, x = "tissue", y = "SFC", hue = "state", dodge=True, color = "#444444", ax = axes,s = 0.5, order = ["Full Network", "GM", "GM-WM", "WM"])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)


for a in range(len( axes.artists)):
    mybox = axes.artists[a]

    # Change the appearance of that box
    mybox.set_facecolor(palette[a])
    mybox.set_edgecolor(palette[a])
    #mybox.set_linewidth(3)

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"SFC_bootstrap_1000.pdf"), save_figure=True)

confidence_intervals = pd.DataFrame(columns = ["tissue", "state", "ci_lower", "ci_upper"])
for t in range(len(params.TISSUE_TYPE_NAMES)):
    for s in range(len(STATE_NAMES)):
        df_ci =  means_df.query(f'tissue == "{params.TISSUE_TYPE_NAMES2[t]}" and state == "{STATE_NAMES[s]}"')["SFC"]
        ci = stats.t.interval(alpha=0.95, df=len(df_ci)-1, loc=np.mean(df_ci), scale=stats.sem(df_ci))
        confidence_intervals = confidence_intervals.append(dict(tissue = params.TISSUE_TYPE_NAMES[t], state = STATE_NAMES[s], ci_lower = ci[0], ci_upper = ci[1]  ),ignore_index=True)



palette2 = ["#bbbbbb" , "#cebeb1"  , "#b6d4ee", "#cdc0d7"]
palette3 = ["#808080" , "#a08269"  , "#76afdf", "#8a6ca1"]

#reorder for plotting
df = pd.DataFrame(means_delta_corr, columns =params.TISSUE_TYPE_NAMES )
order =  [params.TISSUE_TYPE_NAMES[i] for i in [3,1,2,0]]
df = df[order]

palette2_reorder = [palette2[i] for i in [3,1,2,0]]
palette3_reorder = [palette3[i] for i in [0,2,1,3]] #idk why but seaborn and python are so stupid in their plotting. Makes no sense.

fig, axes = utils.plot_make()
sns.histplot(data = df, palette = palette2_reorder, ax = axes, binwidth = 0.01 , line_kws = dict(lw = 5), alpha=1 , edgecolor=None, kde = True)
#sns.kdeplot(data = df, palette =  palette3_reorder, ax = axes , lw = 5, bw_method = 1)
axes.set_xlim([-0.025, 0.14])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)
for l in range(len(axes.lines)):
    axes.lines[l].set_color(palette3_reorder[l])
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"SFC_bootstrap_delta_histogram_1000bootstrap.pdf"), save_figure=True)

stats.t.interval(alpha=0.95, df=len(df)-1, loc=np.mean(df), scale=stats.sem(df))


#%%
fig, axes = utils.plot_make(c = 3)


sns.histplot( pvalues_delta[:,2], ax = axes[0] , binrange = [0,0.1] , binwidth = 0.005,color = "gray" )
axes[0].set_xlim([0,0.08])
sns.histplot( pvalues_delta[:,3], ax = axes[1] , binrange = [0,1] , binwidth = 0.05 , color = "gray" )
axes[1].set_xlim([0,1])
sns.histplot( pvalues_delta_Tissue[:,2], ax = axes[2] , binrange = [0,1] , binwidth = 0.025, color = "gray" ) # WM vs GM-WM; delta Corr
axes[2].set_xlim([0,0.5])

for a in range(len(axes)):
    axes[a].spines['top'].set_visible(False)
    axes[a].spines['right'].set_visible(False)

utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"SFC_PVALUES_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
#%%
df_long, delta_corr = helper.get_tissue_SFC(patientsWithseizures, sfc_patient_list, paths,
                       FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                       params.TISSUE_DEFINITION_PERCENT[0], params.TISSUE_DEFINITION_PERCENT[1], params.TISSUE_DEFINITION_PERCENT[2],
                       ratio_patients = 5, max_seizures = 1,
                       func = 2, freq = 0, print_pvalues = True)


palette = ["#bbbbbb", "#808080", "#282828", "#808080"] + ["#cebeb1", "#a08269", "#675241", "#a08269"] +  ["#cdc0d7", "#8a6ca1", "#511e79", "#8a6ca1"] + ["#b6d4ee", "#76afdf", "#1f5785", "#76afdf"]
palette = ["#bbbbbb", "#bbbbbb", "#282828", "#bbbbbb"] + ["#cebeb1", "#cebeb1", "#675241", "#cebeb1"] + ["#cdc0d7", "#cdc0d7", "#511e79", "#cdc0d7"] + ["#b6d4ee", "#b6d4ee", "#1f5785", "#b6d4ee"]
palette2 = ["#808080", "#808080", "#282828", "#808080"] + ["#a08269", "#a08269", "#675241", "#a08269"] + ["#8a6ca1", "#8a6ca1", "#511e79", "#8a6ca1"]+ ["#76afdf", "#76afdf", "#1f5785", "#76afdf"]

#%%
fig, axes = utils.plot_make(size_length = 5, size_height = 4)

meanprops={"marker":"o", "markerfacecolor":"white",  "markeredgecolor":"black", "markersize":"10"}

sns.boxplot(data = df_long, x = "tissue", y = "FC", hue = "state", ax = axes , whis = 1,showmeans=True , meanline=True)
axes.legend([],[], frameon=False)

for a in range(len( axes.artists)):
    mybox = axes.artists[a]

    # Change the appearance of that box
    mybox.set_facecolor(palette[a])
    mybox.set_edgecolor(palette2[a])
    #mybox.set_linewidth(3)

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

count = 0
a = 0
for line in axes.get_lines():
    line.set_color(palette2[a])
    count = count + 1
    if count % 7 ==0:
        a = a +1
    if count % 7 ==5: #set median line
        line.set_color("white")
        line.set_ls("--")
        line.set_lw(1)
    if count % 7 ==6: #set mean line
        line.set_color("#970707")
        line.set_ls("-")
        line.set_lw(2.5)

utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"SFC_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
#%%


fig, axes = utils.plot_make()
binrange = [-0.3,0.3]
binwidth = 0.01
sns.histplot( delta_corr[:,1], ax = axes , binrange = binrange, binwidth = binwidth, kde = True, color = "red" )
sns.histplot( delta_corr[:,2], ax = axes ,  binrange = binrange, binwidth = binwidth , kde = True, color = "blue" )
sns.histplot( delta_corr[:,3], ax = axes ,  binrange = binrange, binwidth = binwidth , kde = True, color = "purple" )



func = 2
freq = 0
i=109 #3, 21, 51, 54, 63, 75, 78, 103, 104, 105, 109, 110, 111, 112, 113, 116
for s in range(STATE_NUMBER):
    SC_order, SC_order_gm, SC_order_wm, SC_order_gmwm, adj, adj_gm, adj_wm, adj_gmwm = helper.get_SC_and_FC_adj(patientsWithseizures,
                                                     i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                     FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                     TISSUE_DEFINITION_NAME,
                                                     TISSUE_DEFINITION_GM,
                                                     0.5,
                                                     func = func, freq = freq, state = s )
    if s == 0:
        interictal = [SC_order, SC_order_gm, SC_order_wm, SC_order_gmwm, adj, adj_gm, adj_wm, adj_gmwm]
    if s == 1:
        preictal = [SC_order, SC_order_gm, SC_order_wm, SC_order_gmwm, adj, adj_gm, adj_wm, adj_gmwm]
    if s == 2:
        ictal = [SC_order, SC_order_gm, SC_order_wm, SC_order_gmwm, adj, adj_gm, adj_wm, adj_gmwm]
    if s == 3:
        postictal = [SC_order, SC_order_gm, SC_order_wm, SC_order_gmwm, adj, adj_gm, adj_wm, adj_gmwm]



#109      116, 111   104?    78
def plot_adj_heatmap(adj, vmin = 0, vmax = 1, center = 0.5, cmap = "mako" ):
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    sns.heatmap(adj, vmin = vmin, vmax =vmax, center = center, cmap = cmap , ax = axes,  square=True , cbar = False, xticklabels = False, yticklabels = False )


cmap_structural = sns.cubehelix_palette(start=2.8, rot=-0.1, dark=.2, light=0.95, hue = 1, gamma = 4, reverse=True, as_cmap=True)


plot_adj_heatmap(SC_order, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_SC_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
plot_adj_heatmap(SC_order_gm, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_SC_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
plot_adj_heatmap(SC_order_wm, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_SC_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
plot_adj_heatmap(SC_order_gmwm, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_SC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)

pad = 0.015
cmap_functional = sns.cubehelix_palette(start=0.7, rot=-0.1, dark=0, light=0.95, hue = 0.8, gamma = 0.8, reverse=True, as_cmap=True)
t = 4
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_0_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_1_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_2_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_3_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)

t = 5
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_0_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_1_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_2_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_3_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)


t = 6
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_0_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_1_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_2_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_3_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)



t = 7
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_0_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_1_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_2_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"adj_FC_3_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)



from sklearn import linear_model

reg = linear_model.TweedieRegressor(power=1, alpha=0)
#reg = linear_model.LinearRegression()

states_SFC = [interictal, preictal, ictal, postictal]

fig, axes = utils.plot_make(c =STATE_NUMBER, size_length = 16, size_height = 3, sharey = True)
for s in range(STATE_NUMBER):
    matrix = states_SFC[s]
    x = utils.getUpperTriangle(SC_order)
    y = utils.getUpperTriangle(matrix[4])
    reg.fit(x.reshape(-1, 1), y)
    y_predict = reg.predict(x.reshape(-1, 1))
    sns.lineplot(x = x[1::10], y = y_predict[1::10], ax= axes[s], color = plot.COLORS_STATE4[1][s], lw = 5, alpha = 0.7);
    sns.scatterplot(x = x[1::2], y = y[1::2], ax= axes[s], s = 5, color = plot.COLORS_STATE4[1][s], linewidth=0, alpha = 0.3)

    axes[s].title.set_text( np.round( spearmanr(utils.getUpperTriangle(SC_order), utils.getUpperTriangle(matrix[4]))[0],2)   )
    axes[s].set_ylim([0,1])

    axes[s].spines['top'].set_visible(False)
    axes[s].spines['right'].set_visible(False)


utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"GLM_SFC_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)





















#%%


#analysis of determining how good vs poor outcome have diff GM-GM activity

patient_outcomes_good = ["RID0309", "RID0238", "RID0440", "RID0320", "RID0307", "RID0365", "RID0274", "RID0371", "RID0194"]
patient_outcomes_poor = ["RID0278", "RID0405", "RID0442", "RID0382", "RID0238"]



patient_outcomes_good = ["RID0309", "RID0238", "RID0440", "RID0320", "RID0307", "RID0365", "RID0274", "RID0371",
                         "RID0267", "RID0279", "RID0294", "RID0307", "RID0320", "RID0322"]
patient_outcomes_poor = ["RID0278", "RID0405", "RID0442", "RID0382", "RID0238", "RID0442"]






patient_outcomes_good = ["RID0309", "RID0238", "RID0440", "RID0320", "RID0307", "RID0365",
                         "RID0267", "RID0279", "RID0294", "RID0307", "RID0320", "RID0322"]
patient_outcomes_poor = ["RID0278", "RID0405", "RID0442", "RID0382", "RID0238", "RID0442",
                         "RID0274", "RID0371"]


patient_outcomes_good = ["RID0238", "RID0267", "RID0279", "RID0294", "RID0307", "RID0309", "RID0320","RID0322", "RID0365", "RID0440"]
patient_outcomes_poor = ["RID0274", "RID0278", "RID0382", "RID0405", "RID0442", "RID0371"]


patient_outcomes_good = ["RID0238", "RID0267", "RID0279", "RID0294", "RID0307", "RID0309", "RID0320", "RID0365", "RID0440", "RID0424"]
patient_outcomes_poor = ["RID0274", "RID0278", "RID0371", "RID0382", "RID0405", "RID0442", "RID0322"]



original_array_list = []
test_statistic_array_list = []

ratio_patients = 5
cores = 12
iterations = 12
total = 200
max_seizures= 2
func = 2
freq = 7
for i in range(total):
    simulation = helper.mutilcore_permute_wrapper(cores, iterations, summaryStatsLong, FCtissueAll, seizure_number,
                                                  patient_outcomes_good, patient_outcomes_poor,
                                                  FC_TYPES, FREQUENCY_NAMES, STATE_NAMES,
                                                  ratio_patients = ratio_patients, max_seizures = max_seizures, func = func, freq = freq)

    original_array_list.append([a_tuple[0] for a_tuple in simulation])
    original_array = [item for sublist in original_array_list for item in sublist]
    test_statistic_array_list.append([a_tuple[1] for a_tuple in simulation])
    test_statistic_array = [item for sublist in test_statistic_array_list for item in sublist]

    Tstat = np.array(original_array)
    permute  = np.array(test_statistic_array)
    pvalue = len( np.where(permute >= Tstat.mean()) [0]) / len(Tstat)
    print(f"{i}/{total}: {pvalue}")




is_abs = 0
Tstat = np.array(original_array)
permute  = np.array(test_statistic_array)
binrange = [  np.floor(np.min([Tstat, permute]) ),   np.ceil(np.max([Tstat, permute]) ) ]
if is_abs == 1:
    Tstat = abs(Tstat)
    permute = abs(permute)
    binrange = [0, 3]
binwidth = 0.1

fig, axes = utils.plot_make()
sns.histplot(Tstat, kde = True, ax = axes, color = "#222222", binwidth = binwidth, binrange = binrange, edgecolor = None)
sns.histplot(permute, kde = True, ax = axes, color = "#bbbbbb", binwidth = binwidth, binrange = binrange ,edgecolor = None)
axes.axvline(x=abs(Tstat).mean(), color='k', linestyle='--')
axes.set_xlim(binrange)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
pvalue = len( np.where(permute >= Tstat.mean()) [0]) / len(Tstat)
axes.set_title(f"{Tstat.mean()} {pvalue}" )



utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_FC_deltaT_PVALUES_PERMUTATION_morePatients3.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.1)
###############################################################
###############################################################
###############################################################
###############################################################
original,summaryStatsLong_bootstrap_outcome = permute_resampling_pvalues(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor, ratio_patients = ratio_patients, max_seizures = max_seizures)

fig, axes = utils.plot_make(size_length = 4, size_height = 4)

sns.boxplot(data = summaryStatsLong_bootstrap_outcome, x = "state", y = "FC_deltaT", hue = "outcome", ax = axes,showfliers=False , palette = plot.COLORS_GOOD_VS_POOR)
sns.stripplot(data = summaryStatsLong_bootstrap_outcome, x = "state", y = "FC_deltaT", hue = "outcome", ax = axes, palette = plot.COLORS_GOOD_VS_POOR,
                          dodge=True, size=3)
axes.legend([],[], frameon=False)

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
s = 2
v1 = summaryStatsLong_bootstrap_outcome[(summaryStatsLong_bootstrap_outcome["state"]==STATE_NAMES[s])&(summaryStatsLong_bootstrap_outcome["outcome"]<="good")].dropna()["FC_deltaT"]
v2 = summaryStatsLong_bootstrap_outcome[(summaryStatsLong_bootstrap_outcome["state"]==STATE_NAMES[s])&(summaryStatsLong_bootstrap_outcome["outcome"]<="poor")].dropna()["FC_deltaT"]
stats.mannwhitneyu(v1, v2)[1]
axes.set_title(f" {stats.mannwhitneyu(v1, v2)[1] }" )
#axes.set_title(f" {stats.ttest_ind(v1, v2, equal_var=False)[1] }" )
print(f" {stats.ttest_ind(v1, v2, equal_var=True)[1] }" )

utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_FC_deltaT_morePatient2.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.1)

######################################################
######################################################
######################################################
FCtissueAll_bootstrap_outcomes = [FCtissueAll_bootstrap_good, FCtissueAll_bootstrap_poor]

tissue_distribution = [None] * 2
for o in range(2):
    tissue_distribution[o] =  [None] * 4
    for t in range(4):
        tissue_distribution[o][t]  = [None] * 4


OUTCOME_NAMES = ["good", "poor"]
TISSUE_TYPE_NAMES = ["Full Network", "GM-only", "WM-only", "GM-WM"]


for o in range(2):
    for t in range(4):
        for s in range(4):
            FCtissueAll_bootstrap_outcomes_single = FCtissueAll_bootstrap_outcomes[o]
            fc_patient = []
            for i in range(len(FCtissueAll_bootstrap_outcomes_single)):
                fc =  utils.getUpperTriangle(FCtissueAll_bootstrap_outcomes_single[i][func][freq][t][s])
                fc_patient.append(fc)
            tissue_distribution[o][t][s] = np.array([item for sublist in fc_patient for item in sublist])


######################################################
######################################################
######################################################
for s in [1,2]:
    fig, axes = utils.plot_make(c = 2, r = 2, size_height = 5)
    axes = axes.flatten()
    for t in range(4):
        sns.ecdfplot(data = tissue_distribution[0][t][s], ax = axes[t], color = plot.COLORS_GOOD_VS_POOR[0], lw = 6)
        sns.ecdfplot(data = tissue_distribution[1][t][s], ax = axes[t], color =  plot.COLORS_GOOD_VS_POOR[1], lw = 6, ls = "-")
        axes[t].set_title(f"{TISSUE_TYPE_NAMES[t]}, {STATE_NAMES[s]}   {stats.ks_2samp( tissue_distribution[0][t][s],  tissue_distribution[1][t][s] )[1]*16  }" )
        axes[t].spines['top'].set_visible(False)
        axes[t].spines['right'].set_visible(False)
    utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_{STATE_NAMES[s]}.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.0)


#good vs poor -- delta
fig, axes = utils.plot_make()
sns.ecdfplot(data = tissue_distribution[0][t][2] - tissue_distribution[0][t][1], ax = axes, color = "blue")
sns.ecdfplot(data = tissue_distribution[1][t][2] - tissue_distribution[1][t][1], ax = axes, color = "red")


#good vs poor -- ictal
stats.ks_2samp( tissue_distribution[0][t][2], tissue_distribution[1][t][2])

#good vs poor -- preictal
stats.ks_2samp( tissue_distribution[0][t][1], tissue_distribution[0][t][1])

#good vs poor -- delta
stats.ks_2samp(  tissue_distribution[0][t][2] - tissue_distribution[0][t][1], tissue_distribution[1][t][2] - tissue_distribution[1][t][1]   )













t = 1
outcomes_good_tissue_preictal = []
outcomes_good_tissue_ictal = []
for i in range(len(FCtissueAll_bootstrap_good)):
    outcomes_good_tissue_preictal.append(utils.getUpperTriangle(FCtissueAll_bootstrap_good[i][func][freq][t][1]))
    outcomes_good_tissue_ictal.append( utils.getUpperTriangle(FCtissueAll_bootstrap_good[i][func][freq][t][2]))

outcomes_good_tissue_preictal = np.array([item for sublist in outcomes_good_tissue_preictal for item in sublist])
outcomes_good_tissue_ictal = np.array([item for sublist in outcomes_good_tissue_ictal for item in sublist])

outcomes_poor_tissue_preictal = []
outcomes_poor_tissue_ictal = []
for i in range(len(FCtissueAll_bootstrap_poor)):
    outcomes_poor_tissue_preictal.append( utils.getUpperTriangle(FCtissueAll_bootstrap_poor[i][func][freq][t][1]))
    outcomes_poor_tissue_ictal.append( utils.getUpperTriangle(FCtissueAll_bootstrap_poor[i][func][freq][t][2]))

outcomes_poor_tissue_preictal = np.array([item for sublist in outcomes_poor_tissue_preictal for item in sublist])
outcomes_poor_tissue_ictal = np.array([item for sublist in outcomes_poor_tissue_ictal for item in sublist])



fig, axes = utils.plot_make()
sns.ecdfplot(data = outcomes_good_tissue_ictal, ax = axes, color = "blue")
sns.ecdfplot(data = outcomes_poor_tissue_ictal, ax = axes, color = "red")

fig, axes = utils.plot_make()
sns.ecdfplot(data = outcomes_good_tissue_ictal - outcomes_good_tissue_preictal, ax = axes, color = "blue")
sns.ecdfplot(data = outcomes_poor_tissue_ictal - outcomes_poor_tissue_preictal, ax = axes, color = "red")


stats.ks_2samp( outcomes_good_tissue_ictal, outcomes_poor_tissue_ictal)
stats.ks_2samp( outcomes_good_tissue_preictal, outcomes_poor_tissue_preictal)

stats.ks_2samp( outcomes_good_tissue_ictal - outcomes_good_tissue_preictal, outcomes_poor_tissue_ictal - outcomes_poor_tissue_preictal)






fig, axes = utils.plot_make()
sns.ecdfplot(data = outcomes_good_tissue_preictal, ax = axes, color = "blue")
sns.ecdfplot(data = outcomes_poor_tissue_preictal, ax = axes, color = "red")

fig, axes = utils.plot_make()
sns.ecdfplot(data = outcomes_good_tissue_preictal, ax = axes, color = "blue")
sns.ecdfplot(data = outcomes_good_tissue_ictal, ax = axes, color = "red")















#%%
#patient outcomes good vs poor

#patient_outcomes_good = ["RID0309", "RID0238", "RID0440", "RID0320", "RID0307", "RID0365", "RID0274", "RID0371",
#                         "RID0267", "RID0279", "RID0294", "RID0307", "RID0320", "RID0442"]
#patient_outcomes_poor = ["RID0278", "RID0405", "RID0442", "RID0382", "RID0238"]




max_seizures=1
func = 2
freq = 7


test_statistic_array_delta_list = []
test_statistic_array_delta_permutation_list = []

ratio_patients = 5
cores = 24
iterations = 24
total = 200
for i in range(total):
    simulation = helper.multicore_wm_good_vs_poor_wrapper(cores, iterations,
                                  summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                                  patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                  FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = ratio_patients, max_seizures = 1, func = 2, freq = 7,
                                  permute = False, closest_wm_threshold = 85, avg = 0)

    test_statistic_array_delta_list.append([a_tuple[0] for a_tuple in simulation])
    test_statistic_array_delta = [item for sublist in test_statistic_array_delta_list for item in sublist]
    test_statistic_array_delta_permutation_list.append([a_tuple[1] for a_tuple in simulation])
    test_statistic_array_delta_permutation = [item for sublist in test_statistic_array_delta_permutation_list for item in sublist]

    Tstat = np.array(test_statistic_array_delta)
    permute  = np.array(test_statistic_array_delta_permutation)
    pvalue = len( np.where(permute >= Tstat.mean()) [0]) / len(Tstat)
    print(f"{i}/{total}: {pvalue}")


is_abs = 0
Tstat = np.array(test_statistic_array_delta)
permute  = np.array(test_statistic_array_delta_permutation)
binrange = [  np.floor(np.min([Tstat, permute]) ),   np.ceil(np.max([Tstat, permute]) ) ]
if is_abs == 1:
    Tstat = abs(Tstat)
    permute = abs(permute)
    binrange = [0, 15]
binwidth = 1#0.05

fig, axes = utils.plot_make()
sns.histplot(Tstat, kde = True, ax = axes, color = "#222222", binwidth = binwidth, binrange = binrange, edgecolor = None)
sns.histplot(permute, kde = True, ax = axes, color = "#bbbbbb", binwidth = binwidth, binrange = binrange ,edgecolor = None)
axes.axvline(x=abs(Tstat).mean(), color='k', linestyle='--')
axes.set_xlim(binrange)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
pvalue = len( np.where(permute >= Tstat.mean()) [0]) / len(Tstat)
axes.set_title(f"{Tstat.mean()} {pvalue}" )

utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_PERMUTE_delta_morePatients5_19.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.0)



binwidth = 0.005
binrange = [-0.5,0.5]
fig, axes = utils.plot_make()
sns.histplot(abs(test_statistic_array_ablated), kde = True, ax = axes, color = "#222222", binwidth = binwidth, binrange = binrange)
sns.histplot(abs(test_statistic_array_ablated_permutation), kde = True, ax = axes, color = "#bbbbbb", binwidth = binwidth, binrange = binrange)
axes.axvline(x=abs(test_statistic_array_ablated.mean()), color='k', linestyle='--')
axes.set_xlim([-0,0.3])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
pvalue = len( np.where(test_statistic_array_ablated_permutation  <= np.mean(test_statistic_array_ablated ) )[0]) / iterations
pvalue = len( np.where( abs(test_statistic_array_ablated_permutation ) >= np.mean(abs(test_statistic_array_ablated )) )[0]) / iterations
axes.set_title(f" {np.mean(abs(test_statistic_array_ablated) )}, {pvalue}" )
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_FC_ABLATION_PVALUES_PERMUTATION2.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.1)



#%%

summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient = helper.wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                                  patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                  FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM,
                                  ratio_patients = 2, max_seizures = 2, func = func, freq = freq,
                                  closest_wm_threshold = 85)

#delta is the change in FC of the ablated contacts to wm

delta_mean = []
delta_mean = np.zeros(len(delta))
for x in range(len(delta)):
    #delta_mean.append(sum(x)/len(x))
    #delta_mean[x] = np.nanmedian(delta[x])
    delta_mean[x] = np.nanmean(delta[x])



#good = np.concatenate(delta[:7])
#poor = np.concatenate(delta[8:])
good = delta[:len(seizure_number_bootstrap_good)]
poor = delta[len(seizure_number_bootstrap_good):]
#good = delta_mean[:len(patient_inds_good)]
#poor = delta_mean[len(patient_inds_good):]
#df_good = pd.DataFrame(dict(good = good))
#df_poor = pd.DataFrame(dict(poor = poor))

good = delta_mean[:len(seizure_number_bootstrap_good)]
poor = delta_mean[len(seizure_number_bootstrap_good):]
df_good = pd.DataFrame(dict(good = good))
df_poor = pd.DataFrame(dict(poor = poor))

#test_statistic_array2[it] = stats.ttest_ind(good,poor)[0]

##################################
##################################
df_patient = pd.concat([pd.melt(df_good), pd.melt(df_poor)]  )

fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
sns.boxplot(data = df_patient, x= "variable", y = "value", ax = axes, palette = plot.COLORS_GOOD_VS_POOR , showfliers = False)
sns.swarmplot(data = df_patient, x= "variable", y = "value", ax = axes, palette = plot.COLORS_GOOD_VS_POOR,s = 3)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_delta_morePatients3.pdf"), save_figure=True,
              bbox_inches = "tight", pad_inches = 0.0)

print(stats.mannwhitneyu(poor,good))
print(stats.ttest_ind(poor,good))
stats.ttest_ind(good,poor)




#shows the distributions of all the ablated channels and their connectivity to WM
difference_gm_to_wm_ablated = []
for l in range(len(gm_to_wm_all_ablated)):
    difference_gm_to_wm_ablated.append(np.array(gm_to_wm_all_ablated[l][1] )- np.array(gm_to_wm_all_ablated[l][0]))

difference_gm_to_wm_good_ablated = difference_gm_to_wm_ablated[:len(seizure_number_bootstrap_good)]
difference_gm_to_wm_poor_ablated = difference_gm_to_wm_ablated[len(seizure_number_bootstrap_good):]
difference_gm_to_wm_good_ablated = [item for sublist in difference_gm_to_wm_good_ablated for item in sublist]
difference_gm_to_wm_poor_ablated = [item for sublist in difference_gm_to_wm_poor_ablated for item in sublist]


fig, axes = utils.plot_make()
sns.ecdfplot(difference_gm_to_wm_good_ablated, ax = axes, color = plot.COLORS_GOOD_VS_POOR[0], lw = 5)
sns.ecdfplot(difference_gm_to_wm_poor_ablated, ax = axes, color = plot.COLORS_GOOD_VS_POOR[1], lw = 5)
axes.set_xlim([-0.1,0.5])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_difference_all_connecions_morePatients3.pdf"), save_figure=True,
                  bbox_inches = "tight", pad_inches = 0)

#Closest gradient
closest_wm_threshold_gradient = np.arange(15, 125,5 )
difference_gm_to_wm_ablated_closest_gradient = []
for l in range(len(gm_to_wm_all_ablated_closest_wm_gradient)):
    grad_tmp = []
    for gr in range(len(closest_wm_threshold_gradient)):
        grad_tmp.append(np.array(gm_to_wm_all_ablated_closest_wm_gradient[l][1][gr] )- np.array(gm_to_wm_all_ablated_closest_wm_gradient[l][0][gr]))
    difference_gm_to_wm_ablated_closest_gradient.append(grad_tmp)

difference_gm_to_wm_good_ablated_closest_gradient = difference_gm_to_wm_ablated_closest_gradient[:len(seizure_number_bootstrap_good)]
difference_gm_to_wm_poor_ablated_closest_gradient = difference_gm_to_wm_ablated_closest_gradient[len(seizure_number_bootstrap_good):]
gradient_good = []
for gr in range(len(closest_wm_threshold_gradient)):
    grad_tmp = []
    for l in range(len(difference_gm_to_wm_good_ablated_closest_gradient)):
        grad_tmp.append(difference_gm_to_wm_good_ablated_closest_gradient[l][gr])
    gradient_good.append( [item for sublist in grad_tmp for item in sublist])

gradient_poor = []
for gr in range(len(closest_wm_threshold_gradient)):
    grad_tmp = []
    for l in range(len(difference_gm_to_wm_poor_ablated_closest_gradient)):
        grad_tmp.append(difference_gm_to_wm_poor_ablated_closest_gradient[l][gr])
    gradient_poor.append( [item for sublist in grad_tmp for item in sublist])


#[np.nanmean(i) for i in gradient_good]
#[np.nanmean(i) for i in gradient_poor]

df_good_gradient = pd.DataFrame(gradient_good).transpose()
df_poor_gradient = pd.DataFrame(gradient_poor).transpose()
df_good_gradient.columns = closest_wm_threshold_gradient
df_poor_gradient.columns = closest_wm_threshold_gradient

df_good_gradient_long = pd.melt(df_good_gradient, var_name = "distance", value_name = "FC")
df_poor_gradient_long = pd.melt(df_poor_gradient, var_name = "distance", value_name = "FC")

#fig, axes = utils.plot_make()
#sns.regplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = "blue", ci=95,x_estimator=np.mean, scatter_kws={"s": 10})
#sns.regplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = "red", ci=95,x_estimator=np.mean, scatter_kws={"s": 10})

fig, axes = utils.plot_make(size_length = 10)
sns.lineplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[0], ci=None, err_style="bars")
sns.lineplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[1], ci=None, err_style="bars")

sns.lineplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[0], ci=95)
sns.lineplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[1], ci=95)


axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_gradient_morePatient18.pdf"), save_figure=True,
                  bbox_inches = "tight", pad_inches = 0)



#%%


pvals = []
pvals2 = []
pvals_gm_to_wm = []
pvals_gm_to_wm_closest = []
iterations = 1000
test_statistic_array2 = np.zeros((iterations))
for it in range(278,iterations):



    summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor, delta, gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm_gradient = helper.wm_vs_gm_good_vs_poor(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor,
                          patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                          FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM, ratio_patients = 1, max_seizures = 1, func = 2, freq = 7)

    #%%
    #patient_inds = [51, 9, 78, 52, 37, 62, 20, 68, 24, 73, 96, 71]
    #bootstrapping patients and seizures

    ###########################
    #for permutation:
    all_patients = patient_outcomes_good + patient_outcomes_poor
    patient_outcomes_good_permute = np.random.choice(list(all_patients), len(patient_outcomes_good))
    patient_outcomes_poor_permute = np.random.choice(list(all_patients), len(patient_outcomes_poor))
    #######
    #######
    #######
    summaryStatsLong_bootstrap_good, summaryStatsLong_bootstrap_poor, seizure_number_bootstrap_good, seizure_number_bootstrap_poor = helper.summaryStatsLong_bootstrap_GOOD_vs_POOR(
         summaryStatsLong,
         patient_outcomes_good, patient_outcomes_poor,
         ratio_patients = 1, max_seizures = 1)
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
        func = 2
        freq = 7


        FCtype = FC_TYPES[func]

        FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = helper.get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures,
                                                     i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                     FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                     TISSUE_DEFINITION_NAME,
                                                     TISSUE_DEFINITION_GM,
                                                     TISSUE_DEFINITION_WM,
                                                     func, freq)
        coordinates = helper.get_channel_xyz(localization, localization_channels, channels )

        coordinates_array = np.array(coordinates[["x","y","z"]])
        distance_pairwise = utils.get_pariwise_distances(coordinates_array)

        closest_wm_threshold = 20 #in mm



        manual_resected_electrodes = metadata_iEEG.get_manual_resected_electrodes(sub)
        manual_resected_electrodes = np.array(utils.channel2std(manual_resected_electrodes))
        manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )
        #print(manual_resected_electrodes)
        FCtissue = helper.get_functional_connectivity_for_tissue_subnetworks(FC, freq, GM_index, WM_index)


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
                closest_wm_index = WM_index[np.where(distance_pairwise[ablated_ind,:][0][WM_index] < closest_wm_threshold)[0]]
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
                    preictal.append(np.nanmedian(ablated_wm_fc))
                    preictal_all_connectivity_per_channel = all_wm_connectivity_average

                if s == 2:
                    ictal.append(np.nanmedian(ablated_wm_fc))
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

    #%

    ##################################
    ##################################
    ##################################
    #delta is the change in FC of the ablated contacts to wm

    delta_mean = []
    for x in delta:
        delta_mean.append(sum(x)/len(x))

    #good = np.concatenate(delta[:7])
    #poor = np.concatenate(delta[8:])
    good = np.concatenate(delta[:len(seizure_number_bootstrap_good)])
    poor = np.concatenate(delta[len(seizure_number_bootstrap_good):])
    #good = delta_mean[:len(patient_inds_good)]
    #poor = delta_mean[len(patient_inds_good):]
    df_good = pd.DataFrame(dict(good = good))
    df_poor = pd.DataFrame(dict(poor = poor))

    print(f"{it}: {stats.ttest_ind(good,poor)[0]}  {len( np.where(abs(test_statistic_array2[:it])  >= abs(original2) )[0]) / it}")

    #test_statistic_array2[it] = stats.ttest_ind(good,poor)[0]

    ##################################
    ##################################
    df_patient = pd.concat([pd.melt(df_good), pd.melt(df_poor)]  )

    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    sns.boxplot(data = df_patient, x= "variable", y = "value", ax = axes, palette = plot.COLORS_GOOD_VS_POOR , showfliers = False)
    sns.swarmplot(data = df_patient, x= "variable", y = "value", ax = axes, palette = plot.COLORS_GOOD_VS_POOR)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_delta.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.0)

    print(stats.mannwhitneyu(poor,good))
    stats.ttest_ind(good,poor)

    ####################original2 = stats.ttest_ind(good,poor)[0]
    original2 = stats.ttest_ind(good,poor)[0]
    fig, axes = utils.plot_make()
    sns.histplot(abs(test_statistic_array2), ax = axes, bins = int(10), color = "#888888")
    axes.axvline(x=abs(original2), color='k', linestyle='--')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    pvalue = len( np.where(abs(test_statistic_array2)  >= abs(original2) )[0]) / iterations
    axes.set_title(f"{pvalue}, {stats.ttest_ind(good,poor)[0]}  {stats.ttest_ind(good,poor)[1]}" )
    print(pvalue)
    utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_delta_RESAMPLING.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.0)
    #pvals.append(stats.mannwhitneyu(good,poor)[1])
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    """
    #gm_to_wm on a node basis
    difference_gm_to_wm = []
    for l in range(len(gm_to_wm_all)):
        difference_gm_to_wm.append(gm_to_wm_all[l][2] - gm_to_wm_all[l][1])
    difference_gm_to_wm_good = difference_gm_to_wm[:len(seizure_number_bootstrap_good)]
    difference_gm_to_wm_poor = difference_gm_to_wm[len(seizure_number_bootstrap_good):]
    difference_gm_to_wm_good = [item for sublist in difference_gm_to_wm_good for item in sublist]
    difference_gm_to_wm_poor = [item for sublist in difference_gm_to_wm_poor for item in sublist]

    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good, ax = axes, kde = True, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_poor, ax = axes, kde = True, color = "orange")
    fig.legend(labels=['good','poor'])
    """



    ##################################
    ##################################
    ##################################
    ##################################

    #shows the distributions of all the ablated channels and their connectivity to WM
    difference_gm_to_wm_ablated = []
    for l in range(len(gm_to_wm_all_ablated)):
        difference_gm_to_wm_ablated.append(np.array(gm_to_wm_all_ablated[l][1] )- np.array(gm_to_wm_all_ablated[l][0]))

    difference_gm_to_wm_good_ablated = difference_gm_to_wm_ablated[:len(seizure_number_bootstrap_good)]
    difference_gm_to_wm_poor_ablated = difference_gm_to_wm_ablated[len(seizure_number_bootstrap_good):]
    difference_gm_to_wm_good_ablated = [item for sublist in difference_gm_to_wm_good_ablated for item in sublist]
    difference_gm_to_wm_poor_ablated = [item for sublist in difference_gm_to_wm_poor_ablated for item in sublist]


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_good_ablated, ax = axes, color = plot.COLORS_GOOD_VS_POOR[0])
    sns.ecdfplot(difference_gm_to_wm_poor_ablated, ax = axes, color = plot.COLORS_GOOD_VS_POOR[1])

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_difference_all_connecions.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0)

    #pvals_gm_to_wm.append(stats.ks_2samp(difference_gm_to_wm_good_ablated,difference_gm_to_wm_poor_ablated)[1])

    """
    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good, ax = axes, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_good_ablated, ax = axes, color = "purple")
    fig.legend(labels=['good_all_GM_to_WM_connections','good_ablated_to_WM'])


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_poor, ax = axes, color = "orange", legend = True)
    sns.ecdfplot(difference_gm_to_wm_poor_ablated, ax = axes, color = "red")
    fig.legend(labels=['good_all_GM_to_WM_connections','good_ablated_to_WM'])
    """
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    """
    #closest
    difference_gm_to_wm_ablated_closest = []
    for l in range(len(gm_to_wm_all_ablated_closest_wm)):
        difference_gm_to_wm_ablated_closest.append(np.array(gm_to_wm_all_ablated_closest_wm[l][1] )- np.array(gm_to_wm_all_ablated_closest_wm[l][0]))

    difference_gm_to_wm_good_ablated_closest = difference_gm_to_wm_ablated_closest[:len(seizure_number_bootstrap_good)]
    difference_gm_to_wm_poor_ablated_closest = difference_gm_to_wm_ablated_closest[len(seizure_number_bootstrap_good):]
    difference_gm_to_wm_good_ablated_closest = [item for sublist in difference_gm_to_wm_good_ablated_closest for item in sublist]
    difference_gm_to_wm_poor_ablated_closest = [item for sublist in difference_gm_to_wm_poor_ablated_closest for item in sublist]
    """
    """
    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good_ablated_closest, ax = axes, kde = True, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_poor_ablated_closest, ax = axes, kde = True, color = "orange")
    fig.legend(labels=['good','poor'])

    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_good_ablated_closest, ax = axes, color = "blue", legend = True)
    sns.ecdfplot(difference_gm_to_wm_poor_ablated_closest, ax = axes, color = "orange")
    fig.legend(labels=['difference_gm_to_wm_good_ablated_closest','difference_gm_to_wm_poor_ablated_closest'])


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_good, ax = axes, color = "blue", legend = True)
    sns.ecdfplot(difference_gm_to_wm_good_ablated_closest, ax = axes, color = "purple")
    fig.legend(labels=['good_all_GM_to_WM_connections','good_ablated_closest_to_WM'])


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_poor, ax = axes, color = "orange", legend = True)
    sns.ecdfplot(difference_gm_to_wm_poor_ablated_closest, ax = axes, color = "red")
    fig.legend(labels=['poor_all_GM_to_WM_connections','poor_ablated_closest_to_WM'])
    pvals_gm_to_wm_closest.append(stats.ks_2samp(difference_gm_to_wm_good_ablated_closest,difference_gm_to_wm_poor_ablated_closest)[1])
    """





    ##################################
    ##################################
    ##################################
    ##################################
    #Closest gradient
    difference_gm_to_wm_ablated_closest_gradient = []
    for l in range(len(gm_to_wm_all_ablated_closest_wm_gradient)):
        grad_tmp = []
        for gr in range(len(closest_wm_threshold_gradient)):
            grad_tmp.append(np.array(gm_to_wm_all_ablated_closest_wm_gradient[l][1][gr] )- np.array(gm_to_wm_all_ablated_closest_wm_gradient[l][0][gr]))
        difference_gm_to_wm_ablated_closest_gradient.append(grad_tmp)

    difference_gm_to_wm_good_ablated_closest_gradient = difference_gm_to_wm_ablated_closest_gradient[:len(seizure_number_bootstrap_good)]
    difference_gm_to_wm_poor_ablated_closest_gradient = difference_gm_to_wm_ablated_closest_gradient[len(seizure_number_bootstrap_good):]
    gradient_good = []
    for gr in range(len(closest_wm_threshold_gradient)):
        grad_tmp = []
        for l in range(len(difference_gm_to_wm_good_ablated_closest_gradient)):
            grad_tmp.append(difference_gm_to_wm_good_ablated_closest_gradient[l][gr])
        gradient_good.append( [item for sublist in grad_tmp for item in sublist])

    gradient_poor = []
    for gr in range(len(closest_wm_threshold_gradient)):
        grad_tmp = []
        for l in range(len(difference_gm_to_wm_poor_ablated_closest_gradient)):
            grad_tmp.append(difference_gm_to_wm_poor_ablated_closest_gradient[l][gr])
        gradient_poor.append( [item for sublist in grad_tmp for item in sublist])


    [np.nanmean(i) for i in gradient_good]
    [np.nanmean(i) for i in gradient_poor]

    df_good_gradient = pd.DataFrame(gradient_good).transpose()
    df_poor_gradient = pd.DataFrame(gradient_poor).transpose()
    df_good_gradient.columns = closest_wm_threshold_gradient
    df_poor_gradient.columns = closest_wm_threshold_gradient

    df_good_gradient_long = pd.melt(df_good_gradient, var_name = "distance", value_name = "FC")
    df_poor_gradient_long = pd.melt(df_poor_gradient, var_name = "distance", value_name = "FC")

    #fig, axes = utils.plot_make()
    #sns.regplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = "blue", ci=95,x_estimator=np.mean, scatter_kws={"s": 10})
    #sns.regplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = "red", ci=95,x_estimator=np.mean, scatter_kws={"s": 10})

    fig, axes = utils.plot_make(size_length = 10)
    sns.lineplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[0], ci=None, err_style="bars")
    sns.lineplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[1], ci=None, err_style="bars")

    sns.lineplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[0], ci=95)
    sns.lineplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = plot.COLORS_GOOD_VS_POOR[1], ci=95)


    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"good_vs_poor_ABLATION_gradient.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0)
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################





































































#%%
#WM-to-WM connectivity increases more during seizures in poor outcome patients than in good outcome patients.
summaryStatsLong = pd.melt(summaryStats, id_vars = ["patient", "seizure_number", "frequency", "FC_type"], var_name = "state", value_name = "FC")
#summaryStatsLong = summaryStatsLong.groupby(['patient', "frequency", "FC_type", "state"]).mean()
summaryStatsLong.reset_index(inplace=True)
df = summaryStatsLong.loc[summaryStatsLong['FC_type'] == FC_type].loc[summaryStatsLong['frequency'] == FREQUENCY_NAMES[freq]]
#Bootstrap, get unique
patient_unique = np.unique(df["patient"])

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

pval = []
pval2 = []
iterations = 10000
for sim in range(iterations):
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

    df_bootstrap["outcome"] = np.nan
    df_bootstrap.loc[df_bootstrap['patient'].isin(patinets_good_outcomes),"outcome"] = "good"
    df_bootstrap.loc[df_bootstrap['patient'].isin(patinets_poor_outcomes),"outcome"] = "poor"
    """
    plot5, axes5 = plt.subplots(1,1,figsize=(7, 5), dpi = 600)
    sns.boxplot(data = df_bootstrap, x = "state", y = "FC", hue = "outcome", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette ="Set2", ax = axes5)
    sns.stripplot(data = df_bootstrap, x = "state", y = "FC", hue = "outcome",order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = colorsInterPreIctalPost4, ax = axes5)
    axes5.spines['top'].set_visible(False)
    axes5.spines['right'].set_visible(False)
    axes5.set_ylim([-0.005, 0.05]);
    legend_without_duplicate_labels(axes5)
    utils.savefig(f"{paths.FIGURES}/GMvsWM/boxplot_FC_differences_deltaT_GMvsWM.pdf", saveFigures=saveFigures)
    """


    tmp1 = df_good[df_good["state"] == "ictal"]["FC"]
    tmp2 = df_poor[df_poor["state"] == "ictal"]["FC"]

    stats.mannwhitneyu(tmp1,tmp2)[1]
    p = stats.ttest_ind(tmp1,tmp2)[1]
    #print(p)
    utils.printProgressBar(sim, iterations)
    pval.append(p)
    p2 = stats.ttest_1samp(np.array(df_bootstrap.loc[df_bootstrap["state"] == "ictal"]["FC"]), 0)[1]
    pval2.append(p2)
utils.plot_histplot(pval)
utils.plot_histplot(pval2)

len(np.where(   np.array(pval) < 0.05   )[0])/iterations

df = summaryStatsLong.loc[summaryStatsLong['FC_type'] == FC_type].loc[summaryStatsLong['frequency'] == FREQUENCY_NAMES[freq]]
plot5, axes5 = plt.subplots(1,1,figsize=(7, 5), dpi = 600)
sns.boxplot( data= df_bootstrap, x = "state", y = "FC", order=["interictal", "preictal", "ictal", "postictal"], showfliers=False, palette = colorsInterPreIctalPost3, ax = axes5)
sns.stripplot( data= df_bootstrap, x = "state", y = "FC",order=["interictal", "preictal", "ictal", "postictal"], dodge=True, palette = colorsInterPreIctalPost4, ax = axes5)
axes5.spines['top'].set_visible(False)
axes5.spines['right'].set_visible(False)
axes5.set_ylim([-0.0125, 0.0425]);
utils.savefig(f"{paths.FIGURES}/GMvsWM/boxplot_FC_differences_deltaT_GMvsWM.pdf", saveFigures=saveFigures)

#%% Calculate FC as a function of purity

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

    GM_index = np.where(dist["distance"] <= WMdefinition2)[0]
    WM_index = np.where(dist["distance"] > WMdefinition2)[0]
    distOrderInd = np.array(np.argsort(dist["distance"]))
    distOrder = dist.iloc[distOrderInd].reset_index()

    for func in range(len(FCtypes)):
        FCtype = FC_TYPEs[func]

        functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
        channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password,
                                            BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01",
                                            functionalConnectivityPath = functionalConnectivityPath,
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                            fsds = param.FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPE)

        for freq in range(len(FREQUENCY_NAMES)):
            print(f"{FCtype}, {FREQUENCY_NAMES[freq]}, {sub}")


            s=2
            #networks: 0 = all tissue, 1 = GM-GM, 2 = WM-WM, 3 = GM-WM
            networks = [ copy.deepcopy(FC[s][freq] ),  utils.reorderAdj(FC[s][freq], GM_index)  ,  utils.reorderAdj(FC[s][freq], WM_index) , utils.getAdjSubset(FC[s][freq], GM_index, WM_index)   ]
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
            modularity2[GM_index] = 2
            modularity2[WM_index] = 1

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






#%%




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
pvals_gm_to_wm = []
pvals_gm_to_wm_closest = []
for kk in range(50):
    print(f"{kk}")
    #patient_inds = [51, 9, 78, 52, 37, 62, 20, 68, 24, 73, 96, 71]
    #bootstrapping patients and seizures
    patient_inds = [random.randrange(39,52), random.randrange(55,59) , random.randrange(78,82)  ,
                    random.randrange(52,55) , random.randrange(37,39) , random.randrange(60,66)  ,
                    random.randrange(19,21) , random.randrange(66,70),
                    random.randrange(21,25) , random.randrange(73,75)  , random.randrange(82,103) , random.randrange(71,73)  ]
    patient_inds_good = random.sample(patient_inds[:6], 3)
    patient_inds_poor = random.sample(patient_inds[8:], 3)
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

    gm_to_wm_all = []
    gm_to_wm_all_ablated = []
    gm_to_wm_all_ablated_closest_wm = []
    gm_to_wm_all_ablated_closest_wm_gradient = []
    for pt in range(len(patient_inds_bootstrap)):
        i = patient_inds_bootstrap[pt]

        sub = patientsWithseizures["subject"][i]
        func = 2
        freq = 0


        FCtype = FC_TYPES[func]

        FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = helper.get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures,
                                                     i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                     FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES,
                                                     TISSUE_DEFINITION_NAME,
                                                     TISSUE_DEFINITION_GM,
                                                     TISSUE_DEFINITION_WM,
                                                     func, freq)
        coordinates = helper.get_channel_xyz(localization, localization_channels, channels )

        coordinates_array = np.array(coordinates[["x","y","z"]])
        distance_pairwise = utils.get_pariwise_distances(coordinates_array)

        closest_wm_threshold = 20 #in mm



        manual_resected_electrodes = metadata_iEEG.get_manual_resected_electrodes(sub)
        manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
        manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )

        FCtissue = helper.get_functional_connectivity_for_tissue_subnetworks(FC, freq, GM_index, WM_index)


        ictal = []
        preictal = []
        ictal_zscore = []
        preictal_zscore = []

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
            #sns.histplot(utils.getUpperTriangle(gm_to_wm), bins = 20)
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
                zz = stats.zscore( np.concatenate([[np.array(value)], gm_to_wm_average]) )[0]

                all_wm_connectivity = utils.getAdjSubset(adj, np.array(range(len(adj))), WM_index  )
                all_wm_connectivity_average = np.nanmedian(all_wm_connectivity, axis = 1)

                #getting FC of closest WM to ablated region
                closest_wm_index = WM_index[np.where(distance_pairwise[ablated_ind,:][0][WM_index] < closest_wm_threshold)[0]]
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
                    preictal.append(np.nanmedian(ablated_wm_fc))
                    preictal_zscore.append(zz)
                    preictal_all_connectivity_per_channel = all_wm_connectivity_average

                if s == 2:
                    ictal.append(np.nanmedian(ablated_wm_fc))
                    ictal_zscore.append(zz)
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

        gm_to_wm_all_ablated.append(gm_to_wm_per_patient_ablated)
        gm_to_wm_all_ablated_closest_wm.append(gm_to_wm_per_patient_ablated_closest_wm)
        gm_to_wm_all_ablated_closest_wm_gradient.append(gm_to_wm_per_patient_ablated_closest_wm_gradient)
        """
            z_scores_ablated = []
            for ch in range(len(manual_resected_electrodes)):
                ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
                ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WM_index)[0]
                value = np.nanmedian(ablated_wm_fc)
                distribution = [value]
                iters = 50
                for d in range(iters):
                    if func == 5: #if pearson
                        adj_scrambled = bct.randmio_und_signed(adj, itr = 2)[0]
                    if func == 0 or func == 1 or func == 2: #if coherence
                        adj_scrambled = bct.null_model_und_sign(adj)[0]
                    ablated_wm_fc = utils.getAdjSubset(adj_scrambled, ablated_ind, WM_index)
                    #gm_to_wm = utils.getAdjSubset(adj_scrambled, GM_index, WM_index)
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

    #%

    delta_mean = []
    for x in delta:
        delta_mean.append(sum(x)/len(x))

    #good = np.concatenate(delta[:7])
    #poor = np.concatenate(delta[8:])
    good = np.concatenate(delta[:len(patient_inds_good)])
    poor = np.concatenate(delta[len(patient_inds_good):])
    #good = delta_mean[:len(patient_inds_good)]
    #poor = delta_mean[len(patient_inds_good):]
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


    #gm_to_wm on a node basis
    difference_gm_to_wm = []
    for l in range(len(gm_to_wm_all)):
        difference_gm_to_wm.append(gm_to_wm_all[l][2] - gm_to_wm_all[l][1])
    difference_gm_to_wm_good = difference_gm_to_wm[:len(patient_inds_good)]
    difference_gm_to_wm_poor = difference_gm_to_wm[len(patient_inds_good):]
    difference_gm_to_wm_good = [item for sublist in difference_gm_to_wm_good for item in sublist]
    difference_gm_to_wm_poor = [item for sublist in difference_gm_to_wm_poor for item in sublist]
    """
    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good, ax = axes, kde = True, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_poor, ax = axes, kde = True, color = "orange")
    fig.legend(labels=['good','poor'])
    """
    difference_gm_to_wm_ablated = []
    for l in range(len(gm_to_wm_all_ablated)):
        difference_gm_to_wm_ablated.append(np.array(gm_to_wm_all_ablated[l][1] )- np.array(gm_to_wm_all_ablated[l][0]))

    difference_gm_to_wm_good_ablated = difference_gm_to_wm_ablated[:len(patient_inds_good)]
    difference_gm_to_wm_poor_ablated = difference_gm_to_wm_ablated[len(patient_inds_good):]
    difference_gm_to_wm_good_ablated = [item for sublist in difference_gm_to_wm_good_ablated for item in sublist]
    difference_gm_to_wm_poor_ablated = [item for sublist in difference_gm_to_wm_poor_ablated for item in sublist]

    pvals_gm_to_wm.append(stats.ks_2samp(difference_gm_to_wm_good_ablated,difference_gm_to_wm_poor_ablated)[1])

    """
    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good_ablated, ax = axes, kde = True, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_poor_ablated, ax = axes, kde = True, color = "orange")
    fig.legend(labels=['good','poor'])

    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good, ax = axes, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_good_ablated, ax = axes, color = "purple")
    fig.legend(labels=['good_all_GM_to_WM_connections','good_ablated_to_WM'])


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_poor, ax = axes, color = "orange", legend = True)
    sns.ecdfplot(difference_gm_to_wm_poor_ablated, ax = axes, color = "red")
    fig.legend(labels=['good_all_GM_to_WM_connections','good_ablated_to_WM'])
    """


    #closest
    difference_gm_to_wm_ablated_closest = []
    for l in range(len(gm_to_wm_all_ablated_closest_wm)):
        difference_gm_to_wm_ablated_closest.append(np.array(gm_to_wm_all_ablated_closest_wm[l][1] )- np.array(gm_to_wm_all_ablated_closest_wm[l][0]))

    difference_gm_to_wm_good_ablated_closest = difference_gm_to_wm_ablated_closest[:len(patient_inds_good)]
    difference_gm_to_wm_poor_ablated_closest = difference_gm_to_wm_ablated_closest[len(patient_inds_good):]
    difference_gm_to_wm_good_ablated_closest = [item for sublist in difference_gm_to_wm_good_ablated_closest for item in sublist]
    difference_gm_to_wm_poor_ablated_closest = [item for sublist in difference_gm_to_wm_poor_ablated_closest for item in sublist]
    """
    fig, axes = utils.plot_make()
    sns.histplot(difference_gm_to_wm_good_ablated_closest, ax = axes, kde = True, color = "blue", legend = True)
    sns.histplot(difference_gm_to_wm_poor_ablated_closest, ax = axes, kde = True, color = "orange")
    fig.legend(labels=['good','poor'])

    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_good_ablated_closest, ax = axes, color = "blue", legend = True)
    sns.ecdfplot(difference_gm_to_wm_poor_ablated_closest, ax = axes, color = "orange")
    fig.legend(labels=['difference_gm_to_wm_good_ablated_closest','difference_gm_to_wm_poor_ablated_closest'])


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_good, ax = axes, color = "blue", legend = True)
    sns.ecdfplot(difference_gm_to_wm_good_ablated_closest, ax = axes, color = "purple")
    fig.legend(labels=['good_all_GM_to_WM_connections','good_ablated_closest_to_WM'])


    fig, axes = utils.plot_make()
    sns.ecdfplot(difference_gm_to_wm_poor, ax = axes, color = "orange", legend = True)
    sns.ecdfplot(difference_gm_to_wm_poor_ablated_closest, ax = axes, color = "red")
    fig.legend(labels=['poor_all_GM_to_WM_connections','poor_ablated_closest_to_WM'])
    """
    pvals_gm_to_wm_closest.append(stats.ks_2samp(difference_gm_to_wm_good_ablated_closest,difference_gm_to_wm_poor_ablated_closest)[1])


    #Closest gradient

    difference_gm_to_wm_ablated_closest_gradient = []
    for l in range(len(gm_to_wm_all_ablated_closest_wm_gradient)):
        grad_tmp = []
        for gr in range(len(closest_wm_threshold_gradient)):
            grad_tmp.append(np.array(gm_to_wm_all_ablated_closest_wm_gradient[l][1][gr] )- np.array(gm_to_wm_all_ablated_closest_wm_gradient[l][0][gr]))
        difference_gm_to_wm_ablated_closest_gradient.append(grad_tmp)

    difference_gm_to_wm_good_ablated_closest_gradient = difference_gm_to_wm_ablated_closest_gradient[:len(patient_inds_good)]
    difference_gm_to_wm_poor_ablated_closest_gradient = difference_gm_to_wm_ablated_closest_gradient[len(patient_inds_good):]
    gradient_good = []
    for gr in range(len(closest_wm_threshold_gradient)):
        grad_tmp = []
        for l in range(len(difference_gm_to_wm_good_ablated_closest_gradient)):
            grad_tmp.append(difference_gm_to_wm_good_ablated_closest_gradient[l][gr])
        gradient_good.append( [item for sublist in grad_tmp for item in sublist])

    gradient_poor = []
    for gr in range(len(closest_wm_threshold_gradient)):
        grad_tmp = []
        for l in range(len(difference_gm_to_wm_poor_ablated_closest_gradient)):
            grad_tmp.append(difference_gm_to_wm_poor_ablated_closest_gradient[l][gr])
        gradient_poor.append( [item for sublist in grad_tmp for item in sublist])


    [np.nanmean(i) for i in gradient_good]
    [np.nanmean(i) for i in gradient_poor]

    df_good_gradient = pd.DataFrame(gradient_good).transpose()
    df_poor_gradient = pd.DataFrame(gradient_poor).transpose()
    df_good_gradient.columns = closest_wm_threshold_gradient
    df_poor_gradient.columns = closest_wm_threshold_gradient

    df_good_gradient_long = pd.melt(df_good_gradient, var_name = "distance", value_name = "FC")
    df_poor_gradient_long = pd.melt(df_poor_gradient, var_name = "distance", value_name = "FC")

    #fig, axes = utils.plot_make()
    #sns.regplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = "blue", ci=95,x_estimator=np.mean, scatter_kws={"s": 10})
    #sns.regplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = "red", ci=95,x_estimator=np.mean, scatter_kws={"s": 10})

    fig, axes = utils.plot_make()
    sns.lineplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = "blue", ci=95, err_style="bars")
    sns.lineplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = "red", ci=95, err_style="bars")

    sns.lineplot(data = df_good_gradient_long, x= "distance", y = "FC", ax = axes, color = "blue", ci=95)
    sns.lineplot(data = df_poor_gradient_long, x= "distance", y = "FC", ax = axes, color = "red", ci=95)

    #gm_to_wm
    df_gm_to_wm = pd.DataFrame(gm_to_wm_all)
    df_gm_to_wm_delta = df_gm_to_wm.loc[:,2] - df_gm_to_wm.loc[:,1]
    df_gm_to_wm.columns = state_names
    df_gm_to_wm["outcome"] = np.concatenate([np.repeat(["good"], len(patient_inds_good) ), np.repeat(["poor"], len(patient_inds_poor) )] )
    df_gm_to_wm = df_gm_to_wm.reset_index()
    df_gm_to_wm_delta = df_gm_to_wm_delta.reset_index()

    df_gm_to_wm_delta.columns = ["index", "delta_gm_to_wm"]
    df_gm_to_wm_delta["outcome"] = np.concatenate([np.repeat(["good"], len(patient_inds_good) ), np.repeat(["poor"], len(patient_inds_poor) )] )

    good = np.array(df_gm_to_wm_delta.loc[df_gm_to_wm_delta["outcome"] == "good"]["delta_gm_to_wm"])
    poor = np.array(df_gm_to_wm_delta.loc[df_gm_to_wm_delta["outcome"] == "poor"]["delta_gm_to_wm"])

    #pvals_gm_to_wm.append(stats.mannwhitneyu(good,poor)[1])




    """
    fig, axes = utils.plot_make()
    sns.boxplot(data = df_gm_to_wm, x= "outcome", y = "ictal", ax = axes )
    sns.swarmplot(data = df_gm_to_wm, x= "outcome", y = "ictal", ax = axes)
    plt.show()
    """
    """
    fig, axes = utils.plot_make()
    sns.boxplot(data = df_gm_to_wm_delta, x= "outcome", y = "delta_gm_to_wm", ax = axes )
    sns.swarmplot(data = df_gm_to_wm_delta, x= "outcome", y = "delta_gm_to_wm", ax = axes)
    plt.show()
    """


    #%%


fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
sns.boxplot(np.array(pvals), ax = axes[0])
sns.histplot(np.array(pvals), bins = 50, ax =axes[1], kde = True)
print(len(np.where(np.array(pvals) < 0.05)[0])/len(pvals))

fig, axes = utils.plot_make(c = 2)
sns.boxplot(np.array(pvals_gm_to_wm), ax = axes[0])
sns.histplot(np.array(pvals_gm_to_wm), bins = 50, ax =axes[1], kde = True)
print(len(np.where(np.array(pvals) < 0.05)[0])/len(pvals))


fig, axes = utils.plot_make(c = 2)
sns.boxplot(np.array(pvals_gm_to_wm_closest), ax = axes[0])
sns.histplot(np.array(pvals_gm_to_wm_closest), bins = 50, ax =axes[1], kde = True)
print(len(np.where(np.array(pvals) < 0.05)[0])/len(pvals))
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
    #patient_inds = [51, 9, 78, 52, 37, 62, 20, 68, 24, 73, 96, 71]
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
        FCtype = FC_TYPEs[func]
        freq = 7
        print(f"{FCtype}, {FREQUENCY_NAMES[freq]}, {sub}")
        functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
        channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password,
                                            BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01",
                                            functionalConnectivityPath = functionalConnectivityPath,
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                            fsds = param.FREQUENCY_DOWN_SAMPLE , montage = "bipolar", FCtype = FC_TYPE)

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

        GM_index = np.where(dist["distance"] <= WMdefinitionPercent2)[0]
        WM_index = np.where(dist["distance"] > WMdefinitionPercent2)[0]
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
            FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GM_index)       )   )
            FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WM_index)       )   )
            FCtissue[2].append(   utils.getAdjSubset(FC[s][freq], GM_index, WM_index).flatten()   )
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

        gm_to_wm_ic = utils.getAdjSubset(adj_ic_bin, GM_index, WM_index) # gm_to_wm_ic = utils.getAdjSubset(adj_ic, GM_index, WM_index) #utils.plot_adj_heatmap(gm_to_wm_ic);
        gm_to_wm_pi = utils.getAdjSubset(adj_pi_bin, GM_index, WM_index)# gm_to_wm_pi = utils.getAdjSubset(adj_pi, GM_index, WM_index) #utils.plot_adj_heatmap(gm_to_wm_pi);
        gm_to_wm_delta = gm_to_wm_ic - gm_to_wm_pi

        gm_to_wm_delta_avg  = np.sum(gm_to_wm_delta, axis = 1) #gm_to_wm_delta_avg  = np.nanmedian(gm_to_wm_delta, axis = 1)

        gm_to_wm_delta_avg_top_ind = np.argsort(gm_to_wm_delta_avg)
        GM_index_top_correlated_to_wm = GM_index[gm_to_wm_delta_avg_top_ind]

        intersect = np.intersect1d(  manual_resected_electrodes_index, GM_index_top_correlated_to_wm, return_indices=True )

        print(f"\n{manual_resected_electrodes[intersect[1]]}")

        percent_top_gm_to_wm = intersect[2]/len(GM_index_top_correlated_to_wm)
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

            gm_to_wm_ic_rand = utils.getAdjSubset(adj_ic_rand_bin, GM_index, WM_index) #bin
            gm_to_wm_pi_rand = utils.getAdjSubset(adj_pi_rand_bin, GM_index, WM_index) #bin
            gm_to_wm_delta_rand = gm_to_wm_ic_rand - gm_to_wm_pi_rand

            gm_to_wm_delta_avg_rand  = np.sum(gm_to_wm_delta_rand, axis = 1) #sum to meadian

            gm_to_wm_delta_avg_top_ind_rand = np.argsort(gm_to_wm_delta_avg_rand)
            GM_index_top_correlated_to_wm_rand = GM_index[gm_to_wm_delta_avg_top_ind_rand]

            intersect = np.intersect1d(  manual_resected_electrodes_index, GM_index_top_correlated_to_wm_rand, return_indices=True )
            percent_top_gm_to_wm_rand[d,:] = intersect[2]/len(GM_index_top_correlated_to_wm_rand)
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
        ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WM_index)[0]
        print(np.nanmedian(ablated_wm_fc))
        gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index)
        gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
        utils.plot_adj_heatmap(gm_to_wm, vmin = np.min(adj), vmax = np.max(adj))
        sns.histplot(gm_to_wm[2,:], bins = 20)
        sns.histplot(ablated_wm_fc.flatten(), bins = 20)
        ablated_wm_fc = utils.getAdjSubset(adj_scrambled, ablated_ind, WM_index)
        print(np.nanmedian(ablated_wm_fc))
        gm_to_wm = utils.getAdjSubset(adj_scrambled, GM_index, WM_index)
        gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
        sns.histplot(gm_to_wm_average, bins = 20)
        """



        adj = FC[state][freq]
        manual_resected_electrodes = metadata.get_manual_resected_electrodes(sub)
        manual_resected_electrodes = np.array(echobase.channel2std(manual_resected_electrodes))
        manual_resected_electrodes, _, manual_resected_electrodes_index = np.intersect1d(  manual_resected_electrodes, np.array(dist["channels"]), return_indices=True )

        gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index)
        gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
        #sns.histplot(gm_to_wm_average, bins = 20)

        for ch in range(len(manual_resected_electrodes)):
            #ch = 1
            #print(dist[channels == manual_resected_electrodes[ch]])
            ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
            ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WM_index)[0]
            gm_to_wm = utils.getAdjSubset(adj, GM_index, WM_index  )
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            #sns.histplot(gm_to_wm_average, bins = 20)
            #plt.show()
            value = np.nanmedian(ablated_wm_fc)
            #print(np.nanmedian(ablated_wm_fc))
            zz = stats.zscore( np.concatenate([[np.array(value)], gm_to_wm_average]) )[0]

            all_wm_connectivity = utils.getAdjSubset(adj, np.array(range(len(adj))), WM_index  )
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
            ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WM_index)[0]
            value = np.nanmedian(ablated_wm_fc)
            distribution = [value]
            iters = 50
            for d in range(iters):
                if func == 5: #if pearson
                    adj_scrambled = bct.randmio_und_signed(adj, itr = 2)[0]
                if func == 0 or func == 1 or func == 2: #if coherence
                    adj_scrambled = bct.null_model_und_sign(adj)[0]
                ablated_wm_fc = utils.getAdjSubset(adj_scrambled, ablated_ind, WM_index)
                #gm_to_wm = utils.getAdjSubset(adj_scrambled, GM_index, WM_index)
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
"""
file_to_store = open("patient_zscores.pickle", "wb")
pickle.dump(patient_zscores, file_to_store)
file_to_store.close()
abs((np.mean(distribution) - value ) / (np.std(distribution)))

stats.norm.sf(abs(z_scores[0]))*2


file_to_read = open("patient_zscores.pickle", "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()









[max(p) for p in loaded_object]
"""







