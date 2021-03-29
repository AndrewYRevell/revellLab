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

# Creates features, which are structural connectivity matrices
features = []
for s in sub_ID_array:
    for ra in random_atlases:
        for p in range(len(perm_list)):
            if p < 9:
                file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected' \
                       '.nii.gz.trk.gz.{4}_Perm000{5}.count.pass.connectivity.mat'.format(file_directory, s, ra, s, ra,
                                                                                          perm_list[p])
            else:
                file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected' \
                       '.nii.gz.trk.gz.{4}_Perm00{5}.count.pass.connectivity.mat'.format(file_directory, s, ra, s, ra,
                                                                                         perm_list[p])
            features.append(create_feature_matrix(file))
    for sa in standard_atlases:
        file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.' \
               'trk.gz.{4}.count.pass.connectivity.mat'.format(file_directory, s, sa, s, sa)
        features.append(create_feature_matrix(file))

# Get electrode localization
electrode_localization_by_atlas_file_paths = []
for s in sub_ID_array:
    for ra in random_atlases:
        for p in range(len(perm_list)):
            if p < 9:
                file = '{0}sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{2}_electrode_' \
                           'coordinates_mni_{3}_Perm000{4}.csv'.format(file_directory, s, s, ra, perm_list[p])
            else:
                file = '{0}sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{2}_electrode_' \
                           'coordinates_mni_{3}_Perm00{4}.csv'.format(file_directory, s, s, ra, perm_list[p])
            electrode_localization_by_atlas_file_paths.append(file)
    for sa in standard_atlases:
        file = '{0}sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{2}_electrode_coordinates_' \
                   'mni_{3}.csv'.format(file_directory, s, s, sa)
        electrode_localization_by_atlas_file_paths.append(file)

# Get electrode localization by atlas csv file data. From get_electrode_localization.py
electrode_localization_by_atlas = []
for electrode_localization_by_atlas_file in electrode_localization_by_atlas_file_paths:
    electrode_localization_by_atlas.append(pd.read_csv(electrode_localization_by_atlas_file))



# Accesses files for functional connectivity matrices (preictal)
FC_preictal_file_path_array = []
for x in range(len(standard_atlases)):
    for s in range(len(sub_ID_array)):
        file = '{0}sub-{1}/connectivity_matrices/functional/eeg/sub-{2}_{3}_{4}_{5}_functionalConnectivity.pickle' \
            .format(file_directory, sub_ID_array[s], sub_ID_array[s], HUP_ID[s], start_preictal[s], end_preictal[s])
        FC_preictal_file_path_array.append(file)

# Accesses files for functional connectivity matrices (ictal)
FC_ictal_file_path_array = []
for x in range(len(standard_atlases)):
    for s in range(len(sub_ID_array)):
        file = '{0}sub-{1}/connectivity_matrices/functional/eeg/sub-{2}_{3}_{4}_{5}_functionalConnectivity.pickle'\
                .format(file_directory, sub_ID_array[s], sub_ID_array[s], HUP_ID[s], start_ictal[s], end_ictal[s])
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

random_forest.FC_SC_random_forest(features, FC_preictal_list, FC_ictal_list, FC_list, electrode_localization_by_atlas)
