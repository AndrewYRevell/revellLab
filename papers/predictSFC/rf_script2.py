"""
2020.09.30
Lena Armstrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Script for Random Forest algorithm to correlate pre-ictal functional connectivity matrices with feature matrix
    created from patient structural adjacency matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Creates file directory
    2. Creates array of sub_IDs, array of HUP_IDs, array of random atlases, array of standard atlases,
    list of permutations, arrays of starting preictal and ictal times, arrays of ending preictal ictal times
    3. Creates a feature matrix for each file
    4. Created an emtpy FC list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input: file directory, array of sub_IDs, array of HUP_IDs, array of random atlases, array of standard atlases,
list of permutations, array of starting ictal times, array of ending ictal times

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output: Random Forest predictions of FC based on SC

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import pickle
from python_files import random_forest, DataClassSfc
import pandas as pd
from python_files.create_feature_matrix import create_feature_matrix
from dataclasses import dataclass

# Each structural connectivity matrix gets a label of a functional connectivity matrix,
# which is done multiple times if there were multiple seizures

file_directory = '/Users/larmstrong2020/mount/DATA/Human_Data/BIDS_processed/'

# get data into data class
sfc_datapath = os.path.join("/media","arevell","sharedSSD","linux","papers","paper005", "data", "data_processed", "aggregated_data")
RID = "RID0278"
iEEG_filename = "HUP138_phaseII"

start_time_usec = 416023190000
stop_time_usec = 416112890000
fname = os.path.join(sfc_datapath, f"sub-{RID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle")

os.path.exists(fname)

if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = DataClassSfc.sfc(**all_data)
atlas = sfc_data.get_atlas_names()[1]

skipWindow = 0.25
secondsBeforeSpread = 60
st = int((secondsBeforeSpread-10)/skipWindow)
stp = int((secondsBeforeSpread +20)/skipWindow)

SC, SC_regions = sfc_data.get_structure(atlas)
electrodeLocalization = sfc_data.get_electrodeLocalization(atlas)

# Get electrode localization
sfc_data = sfc_data.get_electrodeLocalization()

# Creates features, which are structural connectivity matrices
features = sfc_data.create_feature_matrix()

# Accesses files for functional connectivity matrices (preictal)
FC_preictal_file_path_array = []
for x in range(len(sfc_data.get_atlas_names)):
    for s in range(len(RID)):
        file = '{0}sub-{1}/connectivity_matrices/functional/eeg/sub-{2}_{3}_{4}_{5}_functionalConnectivity.pickle' \
            .format(file_directory, RID, RID, iEEG_filename, start_time_usec, stop_time_usec)
        FC_preictal_file_path_array.append(file)

# Accesses files for functional connectivity matrices (ictal)
FC_ictal_file_path_array = []
for x in range(len(sfc_data.get_atlas_names)):
    for s in range(len(RID)):
        file = '{0}sub-{1}/connectivity_matrices/functional/eeg/sub-{2}_{3}_{4}_{5}_functionalConnectivity.pickle'\
                .format(file_directory, RID, RID, iEEG_filename, start_time_usec, stop_time_usec)
        FC_ictal_file_path_array.append(file)

# Get functional connectivity data in pickle file format (preictal)
FC_preictal_list = []
for FC_file_path in FC_preictal_file_path_array:
    with open(FC_file_path, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, \
                                    electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
    FC_preictal_list.append([broadband])

# Get functional connectivity data in pickle file format (ictal)
FC_ictal_list = []
for FC_file_path in FC_ictal_file_path_array:
    with open(FC_file_path, 'rb') as f: broadband, alphatheta, beta, lowgamma, highgamma, \
                                    electrode_row_and_column_names, order_of_matrices_in_pickle_file = pickle.load(f)
    FC_ictal_list.append([broadband])

FC_list = []

random_forest.FC_SC_random_forest(features, FC_preictal_list, FC_ictal_list, FC_list)
