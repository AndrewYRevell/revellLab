"""
2020.09.30
Lena Armstrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Random Forest algorithm to correlate pre-ictal functional connectivity matrices with feature matrix
    created from patient structural adjacency matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input: file directory, array of sub_IDs, array of HUP_IDs, array of random atlases, array of standard atlases,
list of permutations, array of starting preictal times, array of starting ictal times, arrays of ending preictal times,
array of ending ictal times, feature matrix, and functional connectivity matrices

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output: Random Forest predictions of FC based on SC

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np

from python_files.testing_python_toolbox import electrode_row_and_column_names


def FC_SC_random_forest(features, FC_preictal_list, FC_ictal_list, FC_list, electrode_localization_by_atlas):
    # average preictal
    FC_average_preictal = np.zeros(shape=(len(FC_preictal_list), len(FC_preictal_list[0]), len(FC_preictal_list[0][0])))
    print(FC_average_preictal.shape)

    for file in range(len(FC_average_preictal)):
        for row in range(len(FC_average_preictal[file])):
            for col in range(len(FC_average_preictal[row])):
                FC_average_preictal[row, col] = np.mean(FC_preictal_list[file], axis=2)

    # average ictal
    FC_average_ictal = np.zeros(shape=(len(FC_ictal_list), len(FC_ictal_list[0]), len(FC_ictal_list[0][0])))
    print(FC_average_ictal.shape)

    for file in range(len(FC_average_ictal)):
        for row in range(len(FC_average_ictal[file])):
            for col in range(len(FC_average_ictal[row])):
                FC_average_ictal[row, col] = np.mean(FC_ictal_list[file], axis=2)

    # subtract preictal - ictal
    FC_list = [None] * len(FC_ictal_list)

    for L in range(len(FC_ictal_list)):
        FC_list[L] = FC_ictal_list[L] - FC_preictal_list[L]

    # Remove electrodes in "electrode localization" not found in Functional Connectivity matrices
    electrode_localization_names = []
    for file in range(len(electrode_localization_by_atlas)):
        electrode_localization_names.append(np.array(electrode_localization_by_atlas[file]['electrode_name']))
        electrode_localization_by_atlas[file] = electrode_localization_by_atlas[file][
            np.in1d(electrode_localization_names[file], electrode_row_and_column_names)]

    # Remove electrodes in the Functional Connectivity matrices not found in "electrode localization"
    for file in range(len(electrode_localization_names)):
        not_in_functional_connectivity = np.in1d(electrode_row_and_column_names, electrode_localization_names)
        for i in range(len(FC_list)):
            for j in range(len(FC_list[i])):
                FC_list[file][i] = FC_list[file][i][not_in_functional_connectivity, :, :]
                FC_list[file][i] = FC_list[file][i][:, not_in_functional_connectivity, :]

    # Fisher z-transform of functional connectivity data. This is to take means of correlations and do correlations to
    # the structural connectivity
    for file in range(len(FC_list)):
        for i in range(len(FC_list[file])):
            FC_list[file][i] = np.arctanh(FC_list[file][i])

    # Remove structural ROIs not in electrode_localization ROIs
    electrode_ROIs = []
    structural_index = []
    for file in range(len(electrode_localization_by_atlas)):
        electrode_ROIs.append(np.unique(np.array(electrode_localization_by_atlas[file].iloc[:, 4])))
        electrode_ROIs[file] = electrode_ROIs[file][~(electrode_ROIs[file] == 0)]  # remove region 0
        structural_index.append(np.array(electrode_ROIs[file] - 1))  # subtract 1 because of python's zero indexing
    for row in range(len(features)):
        for col in range(len(features[row])):
            features[row][col] = features[row][col][structural_index[row], :]
            features[row][col] = features[row][col][:, structural_index[row]]

    # Taking average functional connectivity for those electrodes in same atlas regions
    ROIs = []
    for file in range(len(FC_list)):
        print('length FC_List', len(FC_list))
        for i in range(len(FC_list[file])):
            print('length i FC_List', len(FC_list[file]))
            ROIs.append(np.array(electrode_localization_by_atlas[file].iloc[:, 4]))
            for r in range(len(electrode_ROIs[file])):
                index_logical = (ROIs[file] == electrode_ROIs[file][r])
                index_first = np.where(index_logical)[0][0]
                index_second_to_end = np.where(index_logical)[0][1:]
                mean = np.mean(FC_list[file][i][index_logical, :, :], axis=0)
                # Fill in with mean.
                FC_list[file][i][index_first, :, :] = mean
                FC_list[file][i][:, index_first, :] = mean
                # delete the other rows and columns belonging to same region.
                FC_list[file][i] = np.delete(FC_list[file][i], index_second_to_end, axis=0)
                FC_list[file][i] = np.delete(FC_list[file][i], index_second_to_end, axis=1)
                # keeping track of which electrode labels correspond to which rows and columns
                ROIs[file] = np.delete(ROIs[file], index_second_to_end, axis=0)
            # remove electrodes in the ROI labels as zero
            index_logical = (ROIs[file] == 0)
            index = np.where(index_logical)[0]
            FC_list[file][i] = np.delete(FC_list[file][i], index, axis=0)
            FC_list[file][i] = np.delete(FC_list[file][i], index, axis=1)
            ROIs[file] = np.delete(ROIs, index, axis=0)

    # break down feature matrix 2D - 1D FC matrix !!!
    features_2D = []
    for row in range(len(features)):
        all_depth = []
        for col in range(len(features[row])):
            for depth in range(len(features[row])):  # Depth of 9
                for x in range(len(features[row][col])):
                    all_depth.append(features[row][col][depth])
                features_2D.append(all_depth)

    print("length features: ", len(features_2D))
    # print(features_2D)

    labels_1D = []
    for file in range(len(FC_list)):
        for row in range(len(FC_list[file])):
            for col in range(len(FC_list[file][row])):
                labels_1D.append(FC_list[file][row][col])  # only use broadband now

    print("length labels: ", len(labels_1D))
    # print(labels_1D)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features_2D, labels_1D,
                                                                                test_size=0.25, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
