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


def FC_SC_random_forest(features, FC_preictal_list, FC_ictal_list, FC_list):
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
