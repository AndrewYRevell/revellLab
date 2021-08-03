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

#plotting
from revellLab.MDPHD_THESIS.plotting import plot_GMvsWM
from revellLab.MDPHD_THESIS.plotting import plot_seizure_distributions
#%% 2/4 Paths and File names


with open(paths.METADATA_IEEG_DATA) as f: JSON_iEEG_metadata = json.load(f)
with open(paths.ATLAS_FILES_PATH) as f: JSON_atlas_files = json.load(f)
with open(paths.IEEG_USERNAME_PASSWORD) as f: IEEG_USERNAME_PASSWORD = json.load(f)

#data classes
atlases = dataclass_atlases.dataclass_atlases(JSON_atlas_files)
metadata_iEEG = dataclass_iEEG_metadata.dataclass_iEEG_metadata(JSON_iEEG_metadata)


#%% 3/4 Paramters

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




#%% 4/4 General Parameter calculation
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
                                                                   paths, SESSION,  params.TISSUE_DEFINITION_DISTANCE[0], params.TISSUE_DEFINITION_DISTANCE[1], params.TISSUE_DEFINITION_DISTANCE[2])

#bootstrap
iterations = 100
pvals = np.zeros((iterations, STATE_NUMBER))
pvals_deltaT = np.zeros((iterations, STATE_NUMBER +1))
for it in range(iterations):
    func = 1
    freq = 5
    summaryStatsLong_bootstrap, seizure_number_bootstrap = helper.summaryStatsLong_bootstrap(summaryStatsLong, ratio_patients = 0.8, max_seizures = 2)

    FCtissueAll_bootstrap = helper.FCtissueAll_bootstrap(FCtissueAll, seizure_number, seizure_number_bootstrap)
    FCtissueAll_bootstrap_flatten, pvals_iter = helper.FCtissueAll_flatten(FCtissueAll_bootstrap, STATE_NUMBER, func  ,freq  , max_connections = 20)

    pvals[it,:] = pvals_iter
    pvals_deltaT[it,:] = helper.deltaT_stats(summaryStatsLong_bootstrap, FREQUENCY_NAMES, FC_TYPES, func , freq )

    utils.printProgressBar(it +1, iterations)

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
func = 1
freq = 5
state = 2
sub = patientsWithseizures["subject"][i]
FC_type = FC_TYPES[func]

FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue = helper.get_functional_connectivity_and_tissue_subnetworks_for_single_patient(patientsWithseizures,
                                                 i, metadata_iEEG, SESSION, USERNAME, PASSWORD, paths,
                                                 FREQUENCY_DOWN_SAMPLE, MONTAGE, FC_TYPES, params.TISSUE_DEFINITION_DISTANCE[0], params.TISSUE_DEFINITION_DISTANCE[1], 3,
                                                 func, freq)

plot_GMvsWM.plot_FC_example_patient_ADJ(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
plot_GMvsWM.plot_FC_example_patient_GMWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"hist_GM_vs_WM_distribution_of_FC_GMWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_example_patient_GMvsWM(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
plt.show()
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"hist_GM_vs_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)
plot_GMvsWM.plot_FC_example_patient_GMvsWM_ECDF(sub, FC, channels, localization, localization_channels, dist, GM_index, WM_index, dist_order, FC_tissue, FC_TYPES, FREQUENCY_NAMES, state ,func, freq, TISSUE_DEFINITION_GM,  TISSUE_DEFINITION_WM, plot)
utils.save_figure(join(paths.FIGURES, "GM_vs_WM", f"ECDF_GM_vs_WM_distribution_of_FC_GMvsWM_{sub}_{FC_type}_{FREQUENCY_NAMES[freq]}.pdf"), save_figure=SAVE_FIGURES)




#%% Calculate FC as a function of purity

save_directory = join(paths.DATA, "GMvsWM")
func = 2; freq = 5; state = 2

## WM definition = percent
summaryStats_Wm_FC = helper.get_FC_vs_tissue_definition(save_directory, patientsWithseizures, range(3,5), MONTAGE,  params.TISSUE_DEFINITION_DISTANCE,
                                 0, FC_TYPES, FREQUENCY_NAMES,  metadata_iEEG, SESSION, USERNAME, PASSWORD, paths, FREQUENCY_DOWN_SAMPLE, save_pickle = False , recalculate = False)

summaryStats_Wm_FC_bootstrap_func_freq_long_state, result_lin = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC(iterations, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state, print_results = True)
plot_GMvsWM.plot_FC_vs_contact_distance(summaryStats_Wm_FC_bootstrap_func_freq_long_state)
plot_GMvsWM.plot_FC_vs_WM_cutoff(summaryStats_Wm_FC_bootstrap_func_freq_long_state)

pvalues = helper.bootstrap_FC_vs_WM_cutoff_summaryStats_Wm_FC_PVALUES(25, summaryStats_Wm_FC, FC_TYPES, FREQUENCY_NAMES, STATE_NAMES, func, freq, state)
plot_GMvsWM.plot_FC_vs_WM_cutoff_PVALUES(pvalues, 0.00005, [-0.0,0.001], plot)


## WM definition = distance
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

tractography.get_tracts(paths.BIDS, paths.DSI_STUDIO_SINGULARITY, pathDWI, pathTracts)

tractography.get_tracts_loop_through_patient_list(sfc_patient_list, paths, SESSION_RESEARCH3T = SESSION_RESEARCH3T)

patients_list = [3, 75 , 103, 104,105, 109, 110, 111, 112, 113, 115 ]
for i in sfc_patient_list:


    ses="research3T*"

    pathDWI = join(paths['qsiprep'], "qsiprep", f"sub-{sub}", f"ses-{ses}", "dwi", f"sub-{sub}_ses-{ses}_space-T1w_desc-preproc_dwi.nii.gz" )
    pathTracts = join(paths['tractography'], f"sub-{sub}", "tracts")
    utils.checkPathAndMake(pathTracts, pathTracts)
    utils.checkIfFileExistsGlob(  pathDWI)

    pathDWI = glob.glob(pathDWI)[0]

    trkName = f"{join(pathTracts, utils.baseSplitextNiiGz(pathDWI)[2])}.trk.gz"
    if False:
        tractography.get_tracts(paths['BIDS'], paths['dsiStudioSingularity'], pathDWI, pathTracts)


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
WMdef = np.arange(0.5, 1.0, 0.025 )
paientList = pd.DataFrame(columns=["patient"])
summaryStatsWmFC = pd.DataFrame( columns=["patient", "seizure_number", "FC_type", "frequency", "WM_percent", "WM_median_distance", "interictal", "preictal", "ictal", "postictal"] )
FCtissueAll = np.empty((len(FC_types), stateNum, len(FREQUENCY_NAMES), 2),dtype=object)

for func in range(len(FC_types)):
    for freq in range(len(FREQUENCY_NAMES)):
        count = 0
        for i in range(3, N):#[3, 5, 22]: #
            FC_type = FC_TYPEs[func]
            sub = patientsWithseizures["subject"][i]
            print(f"{montage}, {FC_type}, {FREQUENCY_NAMES[freq]}, {sub}")
            paientList = paientList.append(dict(patient=patientsWithseizures["subject"][i]), ignore_index=True)
            functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
            utils.checkPathAndMake(functionalConnectivityPath, functionalConnectivityPath, printBOOL=False)

            channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password,
                                                BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01",
                                                functionalConnectivityPath = functionalConnectivityPath,
                                                secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                                fsds = param.FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPE)
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
            dist_coordinates = pd.DataFrame(channels, columns=["channels"])
            dist_coordinates["x"] = np.nan
            dist_coordinates["y"] = np.nan
            dist_coordinates["z"] = np.nan
            for ch in range(len(channels)):# Getting distances of the channels
                channelName = channels[ch]
                if any(channelName == localizationChannels):
                    dist.iloc[ch, 1] = localization["percent_WM"][np.where(#percent_WM, distance_to_GM_millimeters
                        channelName == localizationChannels)[0][0]]
                    dist_coordinates.iloc[ch, 1]  = localization["x"][np.where(
                        channelName == localizationChannels)[0][0]]
                    dist_coordinates.iloc[ch, 2]  = localization["y"][np.where(
                        channelName == localizationChannels)[0][0]]
                    dist_coordinates.iloc[ch, 3]  = localization["z"][np.where(
                        channelName == localizationChannels)[0][0]]
                else:
                    # if channel has no localization, then just assume GM.
                    dist.iloc[ch, 1] = 0
            dist_coordinates_array = np.array(dist_coordinates[["x","y","z"]])
            distance_pairwise = utils.get_pariwise_distances(dist_coordinates_array)
            for wm in range(len(WMdef)):

                GMindex = np.where(dist["distance"] <= 0.5)[0]
                WMindex = np.where(dist["distance"] > WMdef[wm])[0]

                #get FC values for just the GM-GM connections and WM-WM connections
                FCtissue = [None] *2
                for t in range(len(FCtissue)):
                    FCtissue[t] = []
                for s in range(len(FC)):
                    #Reorder/get just the tissue index, and then just get the upper half of triangle (exluding diagonal)
                    FCtissue[0].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )   )
                    FCtissue[1].append(   utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )   )
                    if count == 0:
                        FCtissueAll[func][s][freq][0] = utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)    )
                        FCtissueAll[func][s][freq][1] = utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)    )
                    else:
                        FCtissueAll[func][s][freq][0] = np.concatenate([  FCtissueAll[func][s][freq][0] , utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], GMindex)       )  ]   )
                        FCtissueAll[func][s][freq][1] = np.concatenate([  FCtissueAll[func][s][freq][1] , utils.getUpperTriangle(     utils.reorderAdj(FC[s][freq], WMindex)       )  ]   )
                wm_median_distance = np.nanmedian(utils.getUpperTriangle( utils.reorderAdj(distance_pairwise, WMindex)))
                wmFC = np.array([np.nanmedian(k) for k in zip(FCtissue[1] )])
                summaryStatsWmFC = summaryStatsWmFC.append(dict(patient = sub ,seizure_number = i, FCtype = FC_TYPE, frequency = FREQUENCY_NAMES[freq], WM_percent = WMdef[wm], WM_median_distance = wm_median_distance , interictal = wmFC[0], preictal = wmFC[1], ictal = wmFC[2], postictal = wmFC[3]) , ignore_index=True   )
            count = count + 1


#%%
bootstrap_ind = []
paientList_unique = np.unique(paientList)
for i in range(len(paientList_unique)):
    ind = np.unique(summaryStatsWmFC.loc[summaryStatsWmFC["patient"] == paientList_unique[i]]["seizure_number"])
    if len(ind)> 2:
        bootstrap_ind.append(random.sample(list(ind), 2))
    if len(ind) > 1:
        bootstrap_ind.append(random.sample(list(ind), 1))
    else:
        bootstrap_ind.append(list(ind))
bootstrap_ind = [item for sublist in bootstrap_ind for item in sublist]


summaryStatsWmFC_bootstrap = summaryStatsWmFC.loc[summaryStatsWmFC["seizure_number"].isin(bootstrap_ind)]
summaryStatsWmFC_bootstrap.reset_index(inplace=True)

func = 1
freq = 5
state = 2
FCtype = FC_TYPEs[func]
tmp = summaryStatsWmFC_bootstrap.loc[summaryStatsWmFC_bootstrap["FCtype"] == FCtype].loc[summaryStatsWmFC_bootstrap["frequency"] == FREQUENCY_NAMES[freq]]

tmp = tmp.groupby(['patient', "WM_percent", "WM_median_distance"]).mean()
tmp.reset_index(inplace=True)

tmp_long = pd.melt(tmp, id_vars = ["patient", "WM_percent", "WM_median_distance"], var_name = "state", value_name = "FC")
tmp_long_tmp = tmp_long.loc[tmp_long["state"] == state_names[state]]
fig, axes = utils.plot_make()
sns.lineplot(data=tmp_long_tmp, x = "WM_percent", y = "FC")

fig, axes = utils.plot_make()
sns.regplot(data=tmp_long_tmp, x = "WM_median_distance", y = "FC")



fig, axes = utils.plot_make()
sns.lineplot(data=tmp_long_tmp, x = "WM_percent", y = "WM_median_distance")
"""
pingouin.ancova(data=tmp_long_tmp, dv = "FC", between = "WM_percent", covar = "WM_median_distance")

from pingouin import ancova, read_dataset
df_pingouin = read_dataset('ancova')
ancova(data=df_pingouin, dv='Scores', covar='Income', between='Method')
"""
model_lin = sm.OLS.from_formula("FC ~ WM_percent + WM_median_distance", data=tmp_long_tmp)
result_lin = model_lin.fit()
print(result_lin.summary())

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





#%% DWI correction and tractography
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
    func = 2
    FCtype = FC_TYPEs[func]
    freq = 7
    print(f"{FCtype}, {FREQUENCY_NAMES[freq]}, {sub}")
    functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
    channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password,
                                        BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01",
                                        functionalConnectivityPath = functionalConnectivityPath,
                                        secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                        fsds = param.FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPE)
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
    WMdef = WMdefinitionPercent3
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
        #plt.ylabel(f"FC ({FCtype} {FREQUENCY_NAMES[freq]})")
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


#%
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

"""
print(stats.ttest_1samp(all_corr[:,2] - all_corr[:,1], 0)[1] *4)
print(stats.ttest_1samp(all_corr_gm[:,2] - all_corr_gm[:,1], 0)[1] *4)
print(stats.ttest_1samp(all_corr_wm[:,2] - all_corr_wm[:,1], 0)[1] *4)
print(stats.ttest_1samp(all_corr_gmwm[:,2] - all_corr_gmwm[:,1], 0)[1] * 4)
"""

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
    FCtype = FC_TYPEs[func]
    freq = 7
    print(f"{FCtype}, {FREQUENCY_NAMES[freq]}, {sub}")
    functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
    channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password,
                                        BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01",
                                        functionalConnectivityPath = functionalConnectivityPath,
                                        secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                        fsds = param.FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPE)
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
    WMdef = WMdefinitionPercent3
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
            #plt.ylabel(f"FC ({FCtype} {FREQUENCY_NAMES[freq]})")
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

xlim = [-0.2, 0.5]
fig, axes = utils.plot_make(r =2)
sns.boxplot(x = tmp1 ,ax = axes[0]   )
sns.boxplot(x =tmp2 ,ax = axes[1]   )
sns.swarmplot(x = tmp1 ,ax = axes[0], color = "red"   )
sns.swarmplot(x =tmp2 ,ax = axes[1] , color = "red"   )
axes[0].set_xlim(xlim)
axes[1].set_xlim(xlim)
np.nanmean(tmp1)
np.nanmean(tmp2)

print(stats.ttest_rel(tmp1, tmp2)[1]*4)
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
        FCtype = FC_TYPEs[func]
        freq = 0
        #print(f"{FCtype}, {FREQUENCY_NAMES[freq]}, {sub}")
        functionalConnectivityPath = join(paths["functionalConnectivityiEEG"], f"sub-{sub}")
        channels, FC = metadata.get_FunctionalConnectivity(patientsWithseizures["subject"][i], idKey = patientsWithseizures["idKey"][i], username = username, password = password,
                                            BIDS =paths['BIDS'], dataset ="derivatives/iEEGorgDownload", session ="implant01",
                                            functionalConnectivityPath = functionalConnectivityPath,
                                            secondsBefore=180, secondsAfter=180, startKey = "EEC",
                                            fsds = param.FREQUENCY_DOWN_SAMPLE, montage = MONTAGE, FCtype = FC_TYPE)

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
        dist_coordinates = pd.DataFrame(channels, columns=["channels"])
        dist_coordinates["x"] = np.nan
        dist_coordinates["y"] = np.nan
        dist_coordinates["z"] = np.nan
        for ch in range(len(channels)):# Getting distances of the channels
            channelName = channels[ch]
            if any(channelName == localizationChannels):
                dist.iloc[ch, 1] = localization["percent_WM"][np.where(#percent_WM, distance_to_GM_millimeters
                    channelName == localizationChannels)[0][0]]
                dist_coordinates.iloc[ch, 1]  = localization["x"][np.where(
                    channelName == localizationChannels)[0][0]]
                dist_coordinates.iloc[ch, 2]  = localization["y"][np.where(
                    channelName == localizationChannels)[0][0]]
                dist_coordinates.iloc[ch, 3]  = localization["z"][np.where(
                    channelName == localizationChannels)[0][0]]
            else:
                # if channel has no localization, then just assume GM.
                dist.iloc[ch, 1] = 0
        dist_coordinates_array = np.array(dist_coordinates[["x","y","z"]])
        distance_pairwise = utils.get_pariwise_distances(dist_coordinates_array)
        closest_wm_threshold = 20 #in mm

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


        utils.get_pariwise_distances(dist_coordinates_array)


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
        for s in range(4): #getting average gm-to-wm connectivity
            state = s


            adj = FC[state][freq]

            gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex)
            gm_to_wm_median= np.nanmedian(utils.getUpperTriangle(gm_to_wm))
            #sns.histplot(utils.getUpperTriangle(gm_to_wm), bins = 20)
            gm_to_wm_per_patient.append(gm_to_wm.flatten())
        gm_to_wm_all.append(gm_to_wm_per_patient)
        for s in [1,2]:
            state = s


            adj = FC[state][freq]

            gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex)
            gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
            #sns.histplot(gm_to_wm_average, bins = 20)
            gm_to_wm_per_patient_ablated_per_channel = []
            gm_to_wm_per_patient_ablated_per_channel_closest = []
            gm_to_wm_per_patient_ablated_per_channel_closest_gradient = []
            for ch in range(len(manual_resected_electrodes)):
                #ch = 1
                #print(dist[channels == manual_resected_electrodes[ch]])
                ablated_ind = np.where(manual_resected_electrodes[ch] == channels)[0]
                ablated_wm_fc = utils.getAdjSubset(adj, ablated_ind, WMindex)[0]
                gm_to_wm_per_patient_ablated_per_channel.append(ablated_wm_fc.flatten())
                gm_to_wm = utils.getAdjSubset(adj, GMindex, WMindex  )
                gm_to_wm_average = np.nanmedian(gm_to_wm, axis = 1)
                #sns.histplot(gm_to_wm_average, bins = 20)
                #plt.show()
                value = np.nanmedian(ablated_wm_fc)
                #print(np.nanmedian(ablated_wm_fc))
                zz = stats.zscore( np.concatenate([[np.array(value)], gm_to_wm_average]) )[0]

                all_wm_connectivity = utils.getAdjSubset(adj, np.array(range(len(adj))), WMindex  )
                all_wm_connectivity_average = np.nanmedian(all_wm_connectivity, axis = 1)

                #getting FC of closest WM to ablated region
                closest_wm_index = WMindex[np.where(distance_pairwise[ablated_ind,:][0][WMindex] < closest_wm_threshold)[0]]
                ablated_closest_wm_fc = utils.getAdjSubset(adj, ablated_ind, closest_wm_index)[0]
                gm_to_wm_per_patient_ablated_per_channel_closest.append(ablated_closest_wm_fc.flatten())
                #getting FC of closest WM to ablated region GRADIENT
                closest_wm_threshold_gradient = np.arange(15, 125,5 )  #in mm
                gradient = []
                for gr in range(len(closest_wm_threshold_gradient)):
                    closest_wm_index = WMindex[np.where((distance_pairwise[ablated_ind,:][0][WMindex] < closest_wm_threshold_gradient[gr] ) & (distance_pairwise[ablated_ind,:][0][WMindex] > closest_wm_threshold_gradient[gr] - 15))[0]]
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







