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
from pathos.multiprocessing import ProcessingPool as Pool

#revellLab
#utilities, constants/parameters, and thesis helper functions
from revellLab.packages.utilities import utils
from revellLab.papers.white_matter_iEEG import constants_parameters as params
from revellLab.papers.white_matter_iEEG import constants_plotting as plot
from revellLab.paths import constants_paths as paths
from revellLab.papers.white_matter_iEEG.helpers import thesis_helpers as helper

#package functions
from revellLab.packages.dataclass import dataclass_atlases, dataclass_iEEG_metadata
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg
from revellLab.packages.atlasLocalization import atlasLocalizationFunctions as atl
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.imaging.tractography import tractography
from revellLab.packages.imaging.makeSphericalRegions import make_spherical_regions

#plotting
from revellLab.papers.white_matter_iEEG.plotting import plot_GMvsWM
from revellLab.papers.white_matter_iEEG.plotting import plot_seizure_distributions
#% 2/4 Paths and File names


with open(paths.BIDS_DERIVATIVES_WM_IEEG_METADATA) as f: JSON_iEEG_metadata = json.load(f)
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
utils.save_figure(f"{paths.FIGURES}/seizureSummaryStats/seizureCounts.pdf", save_figure = False)

#plot distribution of seizure lengths
plot_seizure_distributions.plot_distribution_seizure_length(patientsWithseizures)
utils.save_figure(f"{paths.FIGURES}/seizureSummaryStats/seizureLengthDistribution.pdf", save_figure = False)

#%% Electrode and atlas localization
atl.atlasLocalizationBIDSwrapper(iEEGpatientList,  paths.BIDS, "PIER", SESSION, IEEG_SPACE, ACQ,  paths.BIDS_DERIVATIVES_RECONALL,  paths.BIDS_DERIVATIVES_ATLAS_LOCALIZATION,
                                 paths.ATLASES, paths.ATLAS_LABELS, paths.MNI_TEMPLATE, paths.MNI_TEMPLATE_BRAIN, multiprocess=False, cores=12, rerun=False)


#%% EEG download and preprocessing of electrodes
for i in range(len(patientsWithseizures)):
    metadata_iEEG.get_precitalIctalPostictal(patientsWithseizures["subject"][i], "Ictal", patientsWithseizures["idKey"][i], USERNAME, PASSWORD,
                                        BIDS=paths.BIDS, dataset = paths.BIDS_DERIVATIVES_WM_IEEG_IEEG, session = SESSION, secondsBefore=180, secondsAfter=180, load=False)
    # get intertical
    associatedInterictal = metadata_iEEG.get_associatedInterictal(patientsWithseizures["subject"][i],  patientsWithseizures["idKey"][i])
    metadata_iEEG.get_iEEGData(patientsWithseizures["subject"][i], "Interictal", associatedInterictal, USERNAME, PASSWORD,
                          BIDS=paths.BIDS, dataset = paths.BIDS_DERIVATIVES_WM_IEEG_IEEG, session= SESSION, startKey="Start", load=False)


#%% Power analysis

###################################
#WM power as a function of distance
fname = join(paths.DATA, "white_matter_iEEG",f"power_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.pickle")
if utils.checkIfFileDoesNotExist(fname): #if power analysis already computed, then don't run
    powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = helper.power_analysis(patientsWithseizures, np.array(range(3, N)), metadata_iEEG, USERNAME, PASSWORD, SESSION, FREQUENCY_DOWN_SAMPLE, MONTAGE, paths, params.TISSUE_DEFINITION_DISTANCE[0] , params.TISSUE_DEFINITION_DISTANCE[1], params.TISSUE_DEFINITION_DISTANCE[2] )
    utils.save_pickle( [powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM], fname)
powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = utils.open_pickle(fname)
#Plots
#show figure for paper: Power vs Distance and SNR
plot_GMvsWM.plot_power_vs_distance_and_SNR(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)


utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"heatmap_pwr_vs_Distance_and_SNR_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.pdf"), save_figure= True)
#Show summary figure
plot_GMvsWM.plotUnivariate(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"summary_DEPTH_and_SNR_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.pdf"), save_figure= True)

#boxplot comparing GM vs WM for the different seizure states (interictal, preictal, ictal, postictal)
plot_GMvsWM.plot_boxplot_tissue_power_differences(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM, plot.COLORS_TISSUE_LIGHT_MED_DARK[1], plot.COLORS_TISSUE_LIGHT_MED_DARK[2])
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"boxplot_GM_vs_WM_seizure_state_{MONTAGE}_{params.TISSUE_DEFINITION_DISTANCE[0]}_GM_{params.TISSUE_DEFINITION_DISTANCE[1]}_WM_{params.TISSUE_DEFINITION_DISTANCE[2]}.pdf"), save_figure= SAVE_FIGURES)
#statistics
helper.power_analysis_stats(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)


####################################
#WM power as a function of WM percent
fname = join(paths.DATA, "white_matter_iEEG",f"power_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pickle")
if utils.checkIfFileDoesNotExist(fname): #if power analysis already computed, then don't run
    powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = helper.power_analysis(patientsWithseizures, np.array(range(3, N)), metadata_iEEG, USERNAME, PASSWORD, SESSION, FREQUENCY_DOWN_SAMPLE, MONTAGE, paths, params.TISSUE_DEFINITION_PERCENT[0] , params.TISSUE_DEFINITION_PERCENT[1],params.TISSUE_DEFINITION_PERCENT[2] )
    utils.save_pickle( [powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM], fname)
powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM = utils.open_pickle(fname)
#Plots
#Show summary figure
plot_GMvsWM.plotUnivariatePercent(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"summary_DEPTH_and_SNR_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"), save_figure= True)

#boxplot comparing GM vs WM for the different seizure states (interictal, preictal, ictal, postictal)
plot_GMvsWM.plot_boxplot_tissue_power_differences(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM, plot.COLORS_TISSUE_LIGHT_MED_DARK[1], plot.COLORS_TISSUE_LIGHT_MED_DARK[2])
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"boxplot_GM_vs_WM_seizure_state_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"), save_figure= True)
#statistics
helper.power_analysis_stats(powerGMmean, powerWMmean, powerDistAvg, SNRAll, distAll, paientList, powerGM, powerWM)


#%% Calculating functional connectivity for whole seizure segment
for fc in range(len(FC_TYPES)):
    for i in range(3, N):
        sub = patientsWithseizures["subject"][i]
        functionalConnectivityPath = join(paths.BIDS_DERIVATIVES_FUNCTIONAL_CONNECTIVITY_IEEG, f"sub-{sub}")
        utils.checkPathAndMake(functionalConnectivityPath, functionalConnectivityPath)
        metadata_iEEG.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = USERNAME, password = PASSWORD,
                                            BIDS =paths.BIDS, dataset ="derivatives/white_matter_iEEG", session = SESSION,
                                            functionalConnectivityPath = functionalConnectivityPath,
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                            fsds = FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPES[fc])


#%%
#Combine FC from the above saved calculation, and calculate differences
summaryStatsLong, FCtissueAll, seizure_number = helper.combine_functional_connectivity_from_all_patients_and_segments(patientsWithseizures, np.array(range(3, N)), metadata_iEEG, MONTAGE, FC_TYPES,
                                                                   STATE_NUMBER, FREQUENCY_NAMES, USERNAME, PASSWORD, FREQUENCY_DOWN_SAMPLE,
                                                                   paths, SESSION,  params.TISSUE_DEFINITION_PERCENT[0], params.TISSUE_DEFINITION_PERCENT[1], params.TISSUE_DEFINITION_PERCENT[2])

patient_outcomes_good = ["RID0238", "RID0267", "RID0279", "RID0294", "RID0307", "RID0309", "RID0320", "RID0365", "RID0440", "RID0424"]
patient_outcomes_poor = ["RID0274", "RID0278", "RID0371", "RID0382", "RID0405", "RID0442", "RID0322"]
patients = patient_outcomes_good + patient_outcomes_poor

summaryStatsLong = helper.add_outcomes_to_summaryStatsLong(summaryStatsLong, patient_outcomes_good, patient_outcomes_poor)

#%%
#bootstrap

medians_list = []
means_deltaT_list = []

cores = 12
iterations = 12
total = 409
func = 2
freq = 7
for i in range(total):
    simulation =  helper.deltaT_multicore_wrapper(cores, iterations, summaryStatsLong,
                                                  FCtissueAll, STATE_NUMBER,seizure_number,  FREQUENCY_NAMES,
                                                  FC_TYPES,func  ,freq  , max_connections = 50)

    medians_list.append([a_tuple[0] for a_tuple in simulation])
    medians = [item for sublist in medians_list for item in sublist]
    means_deltaT_list.append([a_tuple[1] for a_tuple in simulation])
    means_deltaT = [item for sublist in means_deltaT_list for item in sublist]
    if i+1 ==total:
        medians = np.dstack(medians)
        means_deltaT = np.vstack(means_deltaT)
    utils.printProgressBar(i +1 , total)

len(means_deltaT)
fig, axes = utils.plot_make(c = STATE_NUMBER, size_length = 12, sharey = True)


for x in range(STATE_NUMBER):

    data = pd.DataFrame(  dict( gm = medians[0, x, :], wm = medians[1, x, :]))
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG",
                  f"22boot_ECDF_all_patients_GMvsWM_ECDF2_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=False)


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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG",
                  f"22BOOTSTRAP_all_patients_GMvsWM_ECDF2_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=False)


FCtissueAll_bootstrap_flatten,_ = helper.FCtissueAll_flatten(FCtissueAll, STATE_NUMBER, func = 2 ,freq = 7, max_connections = 50)
plot_GMvsWM.plot_FC_all_patients_GMvsWM_ECDF(FCtissueAll_bootstrap_flatten, STATE_NUMBER , plot)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG",
                  f"2ECDF_all_patients_GMvsWM_ECDF2_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=False)



plot_GMvsWM.plot_boxplot_single_FC_deltaT(summaryStatsLong, FREQUENCY_NAMES, FC_TYPES, 2, 5, plot)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG",
                  f"boxplot2_single_FC_deltaT_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=False)



plot_GMvsWM.plot_boxplot_all_FC_deltaT(summaryStatsLong, FREQUENCY_NAMES, FC_TYPES, plot.COLORS_STATE4[0], plot.COLORS_STATE4[1])
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG",
                  f"boxplot2_all_FC_deltaT_Supplement_{MONTAGE}_{params.TISSUE_DEFINITION_PERCENT[0]}_GM_{params.TISSUE_DEFINITION_PERCENT[1]}_WM_{params.TISSUE_DEFINITION_PERCENT[2]}.pdf"),
                  save_figure=False)


#%%
#% Plot FC distributions for example patient
#for i in range(3,N):
i=43
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"hist_GM_vs_WM_distribution_of_FC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_example_patient_GMvsWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order,
                                           FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot,
                                           xlim = [0,0.6])

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"hist_GM_vs_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)

plot_GMvsWM.plot_FC_example_patient_GMvsWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"ECDF_GM_vs_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)

#GM-to-WM connections
plot_GMvsWM.plot_FC_example_patient_GMWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order,
                                         FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot,
                                         xlim = [0,0.6])
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"hist_GM_to_WM_distribution_of_FC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_example_patient_GMWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"ECDF_GM_to_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)


#%% Calculate FC as a function of purity

save_directory = join(paths.DATA, "GMvsWM")
func = 2; freq = 0; state = 2

## WM definition =  distance
summaryStats_Wm_FC = helper.get_FC_vs_tissue_definition(save_directory, patientsWithseizures, range(3,5), MONTAGE,  params.TISSUE_DEFINITION_DISTANCE,
                                 0, FC_TYPES, FREQUENCY_NAMES,  metadata_iEEG, SESSION, USERNAME, PASSWORD, paths, FREQUENCY_DOWN_SAMPLE, save_pickle = False , recalculate = False)

summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state, print_results = True)
plot_GMvsWM.plot_FC_vs_contact_distance(summaryStats_Wm_FC_bootstrap_func_freq_long_state)
plot_GMvsWM.plot_FC_vs_WM_cutoff(summaryStats_Wm_FC_bootstrap_func_freq_long_state)



## WM definition = percent
summaryStats_Wm_FC = helper.get_FC_vs_tissue_definition(save_directory, patientsWithseizures, range(3,5), MONTAGE,  params.TISSUE_DEFINITION_PERCENT,
                                 1, FC_TYPES, FREQUENCY_NAMES,  metadata_iEEG, SESSION, USERNAME, PASSWORD, paths, FREQUENCY_DOWN_SAMPLE, save_pickle = False , recalculate = False)
summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state, print_results = True)
plot_GMvsWM.plot_FC_vs_contact_distance(summaryStats_Wm_FC_bootstrap_func_freq_long_state,xlim = [15,80] ,ylim = [0.02,0.2])
plot_GMvsWM.plot_FC_vs_WM_cutoff(summaryStats_Wm_FC_bootstrap_func_freq_long_state)


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


means_list = []
means_delta_corr_list = []

cores = 24
iterations = 24
total = 416
func = 2
freq = 7
for i in range(total):
    utils.printProgressBar(i, total)
    simulation = helper.multicore_sfc_wrapper(cores,iterations, params.TISSUE_TYPE_NAMES2, STATE_NUMBER, patientsWithseizures, sfc_patient_list, paths,
                       FC_TYPES, STATE_NAMES, FREQUENCY_NAMES, metadata_iEEG, SESSION, USERNAME, PASSWORD,FREQUENCY_DOWN_SAMPLE, MONTAGE,
                       params.TISSUE_DEFINITION_PERCENT[0], params.TISSUE_DEFINITION_PERCENT[1], params.TISSUE_DEFINITION_PERCENT[2],
                       ratio_patients = 5, max_seizures = 1,
                       func = 2, freq = 0, print_pvalues = False )

    means_list.append([a_tuple[0] for a_tuple in simulation])
    means = [item for sublist in means_list for item in sublist]
    means_delta_corr_list.append([a_tuple[1] for a_tuple in simulation])
    means_delta_corr = [item for sublist in means_delta_corr_list for item in sublist]
    utils.printProgressBar(i+1, total)
    if i+1 == total:
        means = np.dstack(means)
        means_delta_corr =  np.vstack(means_delta_corr)

cols = pd.MultiIndex.from_product([ params.TISSUE_TYPE_NAMES2, STATE_NAMES])
means_df = pd.DataFrame(columns = ["tissue", "state", "SFC"])
for t in range(len(params.TISSUE_TYPE_NAMES)):
    df_tissue = pd.DataFrame(means[:,t,:].T, columns = STATE_NAMES)
    df_tissue = pd.melt(df_tissue, var_name = ["state"], value_name = "SFC")
    df_tissue["tissue"] = params.TISSUE_TYPE_NAMES2[t]
    means_df = pd.concat([means_df, df_tissue])

wm_preictal = means_df.query('tissue == "WM" and state == "preictal"')["SFC"]
wm_ictal = means_df.query('tissue == "WM" and state == "ictal"')["SFC"]
stats.ttest_rel(means_df.query('tissue == "WM" and state == "ictal"')["SFC"] ,means_df.query('tissue == "WM" and state == "preictal"')["SFC"])[1]
print(stats.wilcoxon(wm_preictal, wm_ictal)[1])
print(stats.mannwhitneyu(wm_preictal, wm_ictal)[1])


palette_long = ["#808080", "#808080", "#282828", "#808080"] + ["#a08269", "#a08269", "#675241", "#a08269"] + ["#8a6ca1", "#8a6ca1", "#511e79", "#8a6ca1"]+ ["#76afdf", "#76afdf", "#1f5785", "#76afdf"]
palette = ["#bbbbbb", "#bbbbbb", "#282828", "#bbbbbb"] + ["#cebeb1", "#cebeb1", "#675241", "#cebeb1"] + ["#cdc0d7", "#cdc0d7", "#511e79", "#cdc0d7"] + ["#b6d4ee", "#b6d4ee", "#1f5785", "#b6d4ee"]


fig, axes = utils.plot_make()
sns.boxplot(data = means_df , x = "tissue", y = "SFC", hue = "state", ax = axes, showfliers=False, order = ["Full Network", "GM", "GM-WM", "WM"])
plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")
#sns.stripplot(data= means_df, x = "tissue", y = "SFC", hue = "state", dodge=True, color = "#444444", ax = axes,s = 1, order = ["Full Network", "GM", "GM-WM", "WM"])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)


for a in range(len( axes.artists)):
    mybox = axes.artists[a]
    # Change the appearance of that box
    mybox.set_facecolor(palette[a])
    mybox.set_edgecolor(palette[a])
    #mybox.set_linewidth(3)
count = 0
a = 0
for line in axes.get_lines():
    line.set_color(palette[a])
    count = count + 1
    if count % 5 ==0:
        a = a +1
    if count % 5 ==0: #set mean line
        line.set_color("#222222")
        #line.set_ls("-")
        #line.set_lw(2.5)

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2SFC_bootstrap_10000.pdf"), save_figure=True)

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
#sns.kdeplot(data = df, palette =  palette3_reorder, ax = axes , lw = 5, bw_method = 0.1)
axes.set_xlim([-0.025, 0.14])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)
for l in range(len(axes.lines)):
    axes.lines[l].set_color(palette3_reorder[l])

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2SFC_bootstrap_delta_histogram_10000bootstrap.pdf"), save_figure=False)

stats.t.interval(alpha=0.95, df=len(df)-1, loc=np.mean(df), scale=stats.sem(df))


#%%#%%
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
    mybox.set_edgecolor(palette[a])
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

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"SFC_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
#%%


fig, axes = utils.plot_make()
binrange = [-0.3,0.3]
binwidth = 0.01
sns.histplot( delta_corr[:,1], ax = axes , binrange = binrange, binwidth = binwidth, kde = True, color = "red" )
sns.histplot( delta_corr[:,2], ax = axes ,  binrange = binrange, binwidth = binwidth , kde = True, color = "blue" )
sns.histplot( delta_corr[:,3], ax = axes ,  binrange = binrange, binwidth = binwidth , kde = True, color = "purple" )



func = 2
freq = 0
i=83 #3, 21, 51, 54, 63, 75, 78, 103, 104, 105, 109, 110, 111, 112, 113, 116
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_SC_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
plot_adj_heatmap(SC_order_gm, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_SC_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
plot_adj_heatmap(SC_order_wm, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_SC_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)
plot_adj_heatmap(SC_order_gmwm, cmap = cmap_structural, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_SC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = 0)

pad = 0.015
cmap_functional = sns.cubehelix_palette(start=0.7, rot=-0.1, dark=0, light=0.95, hue = 0.8, gamma = 0.8, reverse=True, as_cmap=True)
t = 4
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_0_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_1_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_2_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_3_FULL_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)

t = 5
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_0_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_1_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_2_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_3_GM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)


t = 6
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_0_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_1_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_2_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_3_WM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)



t = 7
plot_adj_heatmap(interictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_0_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(preictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_1_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(ictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_2_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)
plot_adj_heatmap(postictal[t], cmap = cmap_functional, center = 0.5)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"adj_FC_3_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.png"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)



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


utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"GLM_SFC_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES, bbox_inches = "tight", pad_inches = pad)





















#%%


#analysis of determining how good vs poor outcome have diff GM-GM activity

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



utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_FC_deltaT_PVALUES_PERMUTATION_morePatients3.pdf"), save_figure=False,
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

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_FC_deltaT_morePatient2.pdf"), save_figure=False,
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
    utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_{STATE_NAMES[s]}.pdf"), save_figure=False,
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



max_seizures=1
func = 2
freq = 7


test_statistic_array_delta_list = []
test_statistic_array_delta_permutation_list = []

ratio_patients = 5
cores = 10
iterations = 10
total = 6
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

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_PERMUTE_delta_morePatients5_19.pdf"), save_figure=False,
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_FC_ABLATION_PVALUES_PERMUTATION2.pdf"), save_figure=False,
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_ABLATION_delta_morePatients3.pdf"), save_figure=False,
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_ABLATION_difference_all_connecions_morePatients3.pdf"), save_figure=False,
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
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_ABLATION_gradient_morePatient18.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)



#%%

#REDONE
pt = 0
func = 2
freq = 7

summaryStatsLong_mean = summaryStatsLong.groupby(by=["patient", 'state', "frequency", "FC_type", "outcome", "seizure_number"]).mean().reset_index()
summaryStatsLong_outcome = summaryStatsLong_mean.query(f'outcome != "NA" and  FC_type == "{FC_TYPES[func]}"  and  frequency == "{FREQUENCY_NAMES[freq]}"    ')


summaryStatsLong_outcome_seizures_not_combined = copy.deepcopy(summaryStatsLong_outcome)
summaryStatsLong_outcome =  summaryStatsLong_outcome.groupby(by=["patient", 'state', "frequency", "FC_type", "outcome"]).mean().reset_index()


fig, axes = utils.plot_make(  size_length = 4, size_height = 6.5)
sns.pointplot(data=summaryStatsLong_outcome, x="state", y="FC_deltaT", hue = "outcome",
              ax = axes, palette = plot.COLORS_GOOD_VS_POOR4, order = STATE_NAMES,join=False, dodge=0.4,  errwidth = 7,capsize = 0.3, linestyles = ["-","--"], scale = 1.1)
plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")
sns.stripplot(data=summaryStatsLong_outcome,  x="state", y="FC_deltaT", hue = "outcome", ax = axes, palette = plot.COLORS_GOOD_VS_POOR2, dodge=True,
              size=7, order = STATE_NAMES, zorder=1, jitter = 0.25)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)

T = helper.compute_T(summaryStatsLong_outcome, group  = False, i =0)
print(T)
print(p_val := helper.compute_T(summaryStatsLong_outcome, group  = False, i =1, alternative = "two-sided"))
axes.set_title(f" {p_val}" )
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_ABLATION_delta_by_seizure.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)



T_star_list = []

cores = 16
iterations = 16
total = 1
print(iterations*total)
for i in range(total):
    simulation = helper.mutilcore_permute_deltaT_wrapper(cores, iterations, summaryStatsLong_outcome)
    T_star_list.extend(simulation)
    utils.printProgressBar(i+1, total)



T_star = np.array(T_star_list)
binrange = [-8,10]
binwidth = 0.25
fig, axes = utils.plot_make()
axes.axvline(x=T, color='black', linestyle='--', lw = 5)
sns.histplot(T_star, kde = True, ax = axes, color = "#bbbbbbbb", binwidth = binwidth, binrange = binrange, edgecolor = None, line_kws=dict(linewidth = 10))
for line in axes.get_lines():
    line.set_color("#333333")
axes.set_xlim([-3,3])
print(T_nonabs := len(np.where(T_star > T)[0]) / len(T_star))
print(T_abs := len(np.where(abs(T_star) > T)[0]) / len(T_star))
print(len(T_star))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.set_title(f" {T_nonabs}, {T_abs}" )

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2_good_vs_poor_delta_perm_by_patient.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)


##############################################


df = pd.DataFrame(columns = ["iteration", "state", "outcome", "FC_deltaT"])

cores = 17
iterations = 17
total = 1
for i in range(total):
    simulation = helper.mutilcore_deltaT_wrapper(cores, iterations, summaryStatsLong_outcome)
    df = df.append(simulation)
    utils.printProgressBar(i+1, total)


df = df.query( f"state == 'ictal'")
fig, axes = utils.plot_make()
sns.histplot(data = df, x = "FC_deltaT",  hue = "outcome", binrange = [-0.005,0.05], binwidth = 0.001, kde = True, palette = plot.COLORS_GOOD_VS_POOR, line_kws=dict(linewidth = 10), edgecolor = None)
axes.axvline(x=df.query(f"outcome == 'good' and state == 'ictal'  ")["FC_deltaT"].mean(), color='k', linestyle='--')
axes.axvline(x=df.query(f"outcome == 'poor' and state == 'ictal'  ")["FC_deltaT"].mean(), color='k', linestyle='--')
print(len(df)/2)
print(helper.compute_T(df, state = "ictal", i = 0, equal_var=True,  group  = False))
print(helper.compute_T(df, state = "ictal", i = 1, equal_var=True,  group  = False))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2_good_vs_poor_delta_boot_by_patient.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)



#%%


#ablation




delta, gm_to_wm_all , gm_to_wm_all_ablated, gm_to_wm_all_ablated_closest_wm, gm_to_wm_all_ablated_closest_wm_gradient= helper.wm_vs_gm_good_vs_poor_redone(summaryStatsLong,
                          patientsWithseizures, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                          FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, TISSUE_DEFINITION_NAME, TISSUE_DEFINITION_GM, TISSUE_DEFINITION_WM , func = 2, freq = 7,
                          closest_wm_threshold = 40)





#get the median ablated-wm connectivity for all ablated channels in a patient, then take the avg of those

delta_mean = pd.DataFrame(columns = delta.columns)
for x in range(len(delta)):
    delta_mean = delta_mean.append(copy.deepcopy(delta.iloc[x]))

    chans = delta.iloc[x]["delta"]
    chans_means = np.nanmean(np.nanmedian(chans, axis = 1)) #take median FC values, and then take means of those channels
    delta_mean.loc[ (delta_mean["patient"] ==  delta_mean.iloc[x]["patient"]) & (delta_mean["seizure_number"] ==  delta_mean.iloc[x]["seizure_number"] ) , "delta"]= chans_means
delta_mean.delta = delta_mean.delta.astype(float)
delta_means_patients =  delta_mean.groupby(by=["patient", "outcome"]).mean().reset_index()

fig, axes = plt.subplots(1, 1, figsize=(4, 4.2), dpi=300)
sns.pointplot(data = delta_mean, x= "outcome", y = "delta", ax = axes, palette = plot.COLORS_GOOD_VS_POOR4,
              join=False, dodge=0.4,  errwidth = 7,capsize = 0.3, linestyles = ["-","--"], scale = 1.3)
plt.setp(axes.lines, zorder=100); plt.setp(axes.collections, zorder=100, label="")
sns.stripplot(data = delta_mean, x= "outcome", y = "delta",ax = axes, palette = plot.COLORS_GOOD_VS_POOR2,s = 8, zorder=1, jitter = 0.3)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
pval = helper.compute_T_no_state(delta_mean, group = False, i = 1, var = "delta", alternative = "two-sided")
axes.set_title(f"{pval}" )
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2_good_vs_poor_ABLATION_delta_by_patient.pdf"), save_figure=False,
              bbox_inches = "tight", pad_inches = 0.0)



###########################3

good = []
poor = []
for x in range(len(gm_to_wm_all_ablated)):
    outcome = gm_to_wm_all_ablated["outcome"][x]
    if outcome == "good":
        good.extend(list(np.array(gm_to_wm_all_ablated["gm_to_wm_all_ablated"][x][1]) -  np.array(gm_to_wm_all_ablated["gm_to_wm_all_ablated"][x][0])))
    if outcome == "poor":
        poor.extend(list(np.array(gm_to_wm_all_ablated["gm_to_wm_all_ablated"][x][1]) -  np.array(gm_to_wm_all_ablated["gm_to_wm_all_ablated"][x][0])))
    if x+1 == len(gm_to_wm_all_ablated):
        good = np.array(good)
        poor = np.array(poor)


fig, axes = utils.plot_make()
sns.ecdfplot(good, ax = axes, color = plot.COLORS_GOOD_VS_POOR[0], lw = 5)
sns.ecdfplot(poor, ax = axes, color = plot.COLORS_GOOD_VS_POOR[1], lw = 5)
axes.set_xlim([-0.1,0.8])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2_good_vs_poor_ABLATION_difference_all_connecions_morePatients.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)




#% permute delta


delta_mean

T = helper.compute_T_no_state(delta_mean, group = True, i = 0, var = "delta")
print(T)
print(helper.compute_T_no_state(delta_mean, group = True, i = 1, var = "delta"))

T_star_list = []

cores = 16
iterations = 16
total = 1
print(iterations*total)
for i in range(total):
    simulation = helper.mutilcore_permute_deltaT_wrapper(cores, iterations, delta_mean, state_bool = False, group = True, var = "FC_deltaT")
    T_star_list.extend(simulation)
    utils.printProgressBar(i+1, total)


T_star = np.array(T_star_list)
binrange = [-8,10]
binwidth = 0.25
fig, axes = utils.plot_make()
axes.axvline(x=T, color='black', linestyle='--', lw = 5)
sns.histplot(T_star, kde = True, ax = axes, color = "#bbbbbbbb", binwidth = binwidth, binrange = binrange, edgecolor = None, line_kws=dict(linewidth = 10))
for line in axes.get_lines():
    line.set_color("#333333")
axes.set_xlim([-5,6])
print(T_nonabs := len(np.where(T_star > T)[0]) / len(T_star))
print(T_abs := len(np.where(abs(T_star) > T)[0]) / len(T_star))
print(len(T_star))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.set_title(f" {T_nonabs}, {T_abs}" )

utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2_good_vs_poor_ABLATION_perm_by_patient.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)



##############################################
#boostrap delta

df = pd.DataFrame(columns = ["iteration","outcome", "delta"])


cores = 16
iterations = 16
total = 1
for i in range(total):
    simulation = helper.mutilcore_delta_mean_wrapper(cores, iterations, delta_mean)
    df = df.append(simulation)
    utils.printProgressBar(i+1, total)


binrange = [-0.005,1]
binwidth = 0.01
fig, axes = utils.plot_make()
sns.histplot(data = df, x = "delta",  hue = "outcome", binrange =binrange, binwidth =binwidth, kde = True, palette = plot.COLORS_GOOD_VS_POOR, line_kws=dict(linewidth = 10), edgecolor = None)
axes.axvline(x=df.query(f"outcome == 'good' ")["delta"].mean(), color='k', linestyle='--')
axes.axvline(x=df.query(f"outcome == 'poor'  ")["delta"].mean(), color='k', linestyle='--')
axes.set_xlim([-0.0005, 0.25])
print(len(df)/2)
print(helper.compute_T_no_state(df, i = 0, equal_var=True,  group  = False, var = "delta"))
print(helper.compute_T_no_state(df, i = 1, equal_var=True,  group  = False, var = "delta"))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.legend([],[], frameon=False)


utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"2_good_vs_poor_ABLATION_boot_by_patient.pdf"), save_figure=False,
                  bbox_inches = "tight", pad_inches = 0)
















######################################################
######################################################
######################################################


seizure_number_good = summaryStatsLong.query(f"outcome == 'good'   ")["seizure_number"].unique()
seizure_number_poor = summaryStatsLong.query(f"outcome == 'poor'   ")["seizure_number"].unique()

FCtissueAll_ind_good = np.intersect1d(seizure_number_good, seizure_number, return_indices=True )[2]
FCtissueAll_ind_poor = np.intersect1d(seizure_number_poor, seizure_number, return_indices=True )[2]
FCtissueAll_good = [FCtissueAll[i] for i in FCtissueAll_ind_good]
FCtissueAll_poor = [FCtissueAll[i] for i in FCtissueAll_ind_poor]
FCtissueAll_outcomes = [FCtissueAll_good, FCtissueAll_poor]

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
            FCtissueAll_outcomes_single = FCtissueAll_outcomes[o]
            fc_patient = []
            for i in range(len(FCtissueAll_outcomes_single)):
                fc =  utils.getUpperTriangle(FCtissueAll_outcomes_single[i][func][freq][t][s])
                fc_patient.append(fc)
            tissue_distribution[o][t][s] = np.array([item for sublist in fc_patient for item in sublist])


for s in [1,2]:
    fig, axes = utils.plot_make(c = 2, r = 2, size_height = 5)
    axes = axes.flatten()
    for t in range(4):
        sns.ecdfplot(data = tissue_distribution[0][t][s], ax = axes[t], color = plot.COLORS_GOOD_VS_POOR[0], lw = 6)
        sns.ecdfplot(data = tissue_distribution[1][t][s], ax = axes[t], color =  plot.COLORS_GOOD_VS_POOR[1], lw = 6, ls = "-")
        axes[t].set_title(f"{TISSUE_TYPE_NAMES[t]}, {STATE_NAMES[s]}   {stats.ks_2samp( tissue_distribution[0][t][s],  tissue_distribution[1][t][s] )[1]*16  }" )
        axes[t].spines['top'].set_visible(False)
        axes[t].spines['right'].set_visible(False)
        axes[t].set_xlim([0,1])
    utils.save_figure(join(paths.FIGURES, "white_matter_iEEG", f"good_vs_poor_{STATE_NAMES[s]}.pdf"), save_figure=False,
                      bbox_inches = "tight", pad_inches = 0.0)


#good vs poor -- delta


