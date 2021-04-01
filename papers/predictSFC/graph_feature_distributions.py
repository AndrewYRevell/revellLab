"""
2020.06.17
Lena Armstrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Graph distributions of brain network features for patient structural matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. Determine patients, atlases, and permutations
    2. Creates a structural connectivity array from an array of structure file paths
    3. Graphs the permutations of each atlas
    4. Graphs total of permutations for each atlas
    5. Graphs total of permutations for each atlas for all the patients combined
    6. Saves the graphs to the appropriate output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    An array of file paths with structural matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    Graphs of edge weight distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:
    inputfile = '/USERS/arevell/mount/DATA/Human_Data/BIDS_processed/sub-RID0194_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.RA_N0100_Perm0001.count.pass.connectivity.mat'
    outputfile = '/USERS/arevell/papers/paper004/paper004/figures/supplement/edge_weight_distributions/sub-RID0194'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import bct
import numpy as np
import pandas as pd
from scipy.io import loadmat  # fromn scipy to read mat files (structural connectivity)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Patients
sub_ID_array = ['RID0139', 'RID0194', 'RID0278', 'RID0320', 'RID0309', 'RID0365', 'RID0420', 'RID0440',
                'RID0454', 'RID0490', 'RID0502', 'RID0508', 'RID0520', 'RID0522', 'RID0529', 'RID0536',
                'RID0595'] #'RID0089', 'RID0583',

# Atlases
# atlas_names = ['RA_N0075', 'RA_N0100', 'RA_N0200', 'RA_N0300', 'RA_N0400']
# 'RA_N0010', 'RA_N0030', 'RA_N0050', 'RA_N1000', 'RA_N2000''RA_N0500'

standard_atlases = ['aal_res-1x1x1', 'AAL600', 'CPAC200_res-1x1x1', 'desikan_res-1x1x1', 'DK_res-1x1x1',
                    'JHU_aal_combined_res-1x1x1', 'JHU_res-1x1x1',
                    'Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_300Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm',
                    'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm', 'Talairach_res-1x1x1']

# Permutations
perm_list = list(range(1, 31))

# Input and output file directories
file_directory = '/Users/larmstrong2020/mount/DATA/Human_Data/BIDS_processed/'
output_file_directory = '/Users/larmstrong2020/Desktop/PURM_GRAPHS/'
# output_file_directory = '/Users/larmstrong2020/mount/USERS/arevell/papers/paper004/paper004/figures/supplement/edge_weight_distributions/'
# output_file_directory = '/Users/larmstrong2020/mount/USERS/arevell/papers/paper004/paper004/figures/supplement/degree_distributions/'
# output_file_directory = '/Users/larmstrong2020/mount/USERS/arevell/papers/paper004/paper004/figures/supplement/clustering_coef_distributions/'
# output_file_directory = '/Users/larmstrong2020/mount2/USERS/arevell/papers/paper004/paper004/figures/supplement/betweenness_centrality_distributions/'
# output_file_directory = '/Users/larmstrong2020/mount/USERS/arevell/papers/paper004/paper004/figures/supplement/shortest_path_length_distributions/'


def make_structural_connectivity_array(structure_file_path_array):
    # Get Structural Connectivity data in mat file format. Output from DSI studio
    structural_connectivity_array = []
    counter = 0
    for file_path in structure_file_path_array:
        structural_connectivity_array.append(np.array(pd.DataFrame(loadmat(file_path)['connectivity'])))
        counter += 1
        print("I\'m trying...{0}".format(counter))

    # *** Conversion of connection weights to connection lengths ***
    connection_length_matrix = []
    for s in structural_connectivity_array:
        connection_length_matrix.append(bct.weight_conversion(s, 'lengths'))

    # Betweenness Centrality
    btwn_cent_arr = []
    for structural_matrix in connection_length_matrix:
        btwn_cent_arr.append(bct.betweenness_wei(structural_matrix))

    return btwn_cent_arr


'''
# DIFFERENT NETWORK FEATURES

    # Only output first half to avoid double counting, since it is an undirected
    for i in range(len(structural_connectivity_array)):
        structural_connectivity_array[i] = structural_connectivity_array[i][
            np.triu_indices(len(structural_connectivity_array[i]), k=1)]

    # Log normalizing
    for i in range(len(structural_connectivity_array)):  # do not need it by len can just be i
        structural_connectivity_array[i][structural_connectivity_array[i] == 0] = 1
        structural_connectivity_array[i] = np.log10(structural_connectivity_array[i])  # log-scaling. Converting 0s to 1 to avoid taking log of zeros
        structural_connectivity_array[i] = structural_connectivity_array[i] / np.max(structural_connectivity_array[i])  # normalization

    # Remove zeros
    for i in range(len(structural_connectivity_array)):
        structural_connectivity_array[i] = structural_connectivity_array[i][structural_connectivity_array[i] != 0]


def make_structural_connectivity_array(structure_file_path_array):
    # Get Structural Connectivity data in mat file format. Output from DSI studio
    structural_connectivity_array = []
    for file_path in structure_file_path_array:
        structural_connectivity_array.append(np.array(pd.DataFrame(loadmat(file_path)['connectivity'])))

    # Degree
    degree_arr = []
    for structural_matrix in structural_connectivity_array:
        degree_arr.append(bct.degrees_und(structural_matrix))

    return degree_arr
    
    # *** Conversion of connection weights to connection lengths ***
    connection_length_matrix = []
    for s in structural_connectivity_array:
        connection_length_matrix.append(bct.weight_conversion(s, 'lengths'))
    
    # Clustering Coefficient
    clustering_coefficients = []
    for structural_matrix in connection_length_matrix:
        clustering_coefficients.append(bct.clustering_coef_wu(structural_matrix))

    return clustering_coefficients
       
    # Betweenness Centrality
    btwn_cent_arr = []
    for structural_matrix in connection_length_matrix:
        btwn_cent_arr.append(bct.betweenness_wei(structural_matrix))

    return btwn_cent_arr
'''


# Graphs 5 permutations of each atlas
def graph_perm_level(structure_file_path_array, sub_ID, atlas, output_file, xlabel, title):
    # Get Structural Connectivity data in mat file format, log normalize, and halve data to account for undirected graph
    feature_array = make_structural_connectivity_array(structure_file_path_array)

    # Plot of totaled permutations for 3 atlases per patient
    perm_level_pdf = PdfPages(output_file)
    for perm in range(len(perm_list)):
        fig1 = plt.figure()
        plt.hist(np.ndarray.flatten(np.array(feature_array[perm])), bins=40, alpha=0.5,
                 color='#0000ff')
        if perm < 9:
            legend_patch = mpatches.Patch(color='#6666ff', label='Perm000' + str(perm + 1))
        else:
            legend_patch = mpatches.Patch(color='#6666ff', label='Perm00' + str(perm + 1))
        plt.legend(handles=[legend_patch])
        plt.ylabel('Frequency')
        plt.xlabel(xlabel)
        plt.title(title)
        fig1.text(0.025, 0.01, '{0}   {1}'.format(sub_ID, atlas))
        plt.show()
        perm_level_pdf.savefig(fig1)
    perm_level_pdf.close()


# Graphs total of permutations for each atlas
def graph_atlas_level(structure_file_path_array, sub_ID, atlas_names, perm_list, output_file, xlabel, title):
    # Get Structural Connectivity data in mat file format, log normalize, and halve data to account for undirected graph
    feature_array = make_structural_connectivity_array(structure_file_path_array)

    feature_array_atlas = []
    for a in range(len(atlas_names)):
        feature_array_total = []
        for i in range(a * len(perm_list), (a + 1) * len(perm_list)):
            for j in range(len(feature_array[i])):
                feature_array_total.append(feature_array[i][j])
        feature_array_atlas.append(feature_array_total)

    # Plot of totaled permutations for 3 atlases per patient
    atlas_level_pdf = PdfPages(output_file)
    for atlas in range(len(feature_array_atlas)):
        fig1 = plt.figure()
        plt.hist(np.ndarray.flatten(np.array(feature_array_atlas[atlas])), bins=40, alpha=0.5,
                 color='#0000ff')
        legend_patch = mpatches.Patch(color='#6666ff', label=atlas_names[atlas])
        plt.legend(handles=[legend_patch])
        plt.ylabel('Frequency')
        plt.xlabel(xlabel)
        plt.title(title)
        fig1.text(0.025, 0.01, '{0}'.format(sub_ID))
        plt.show()
        atlas_level_pdf.savefig(fig1)
    atlas_level_pdf.close()


# Graphs total of permutations for each atlas for all the patients combined
def graph_patient_level(structure_file_path_array, sub_IDs, atlas_names, perm_list, output_file, xlabel, title):
    # Get Structural Connectivity data in mat file format, log normalize, and halve data to account for undirected graph
    feature_array = make_structural_connectivity_array(structure_file_path_array)

    feature_array_all_patients = []
    for a in range(len(atlas_names)):
        structural_connectivity_array_total = []
        for i in range(a * len(perm_list) * len(sub_IDs), (a + 1) * len(perm_list) * len(sub_IDs)):
            for j in range(len(feature_array[i])):
                structural_connectivity_array_total.append(feature_array[i][j])
        feature_array_all_patients.append(structural_connectivity_array_total)

    # Plot of totaled permutations for 3 atlases per patient
    patient_level_pdf = PdfPages(output_file)
    for atlas in range(len(feature_array_all_patients)):
        fig2 = plt.figure()
        plt.hist(np.ndarray.flatten(np.array(feature_array_all_patients[atlas])), bins=40, alpha=0.5,
                     color='#0000ff')
        legend_patch = mpatches.Patch(color='#6666ff', label=atlas_names[atlas])
        plt.legend(handles=[legend_patch])
        plt.ylabel('Frequency')
        plt.xlabel(xlabel)
        plt.title(title)
        fig2.text(0.025, 0.01, 'All Patients')
        plt.show()
        patient_level_pdf.savefig(fig2)
    patient_level_pdf.close()


    '''
    # Plot of totaled permutations for 3 atlases per patient
    fig, axs = plt.subplots(5, 1)
    axs[0].title.set_text(title)
    for i in range(len(feature_array_all_patients)):
        axs[i].hist(np.ndarray.flatten(np.array(feature_array_all_patients[i])), bins=40, alpha=0.5, color='#0000ff')
        legend_patch = mpatches.Patch(color='#6666ff', label=atlas_names[i])
        axs[i].legend(handles=[legend_patch])
    axs[2].set_ylabel('Frequency')
    axs[4].set_xlabel(xlabel)
    fig.text(0.05, 0.01, 'All Patients Combined')
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.savefig(output_file)
    plt.show()
    '''


'''
# SHORTEST PATH (bar graph)

def second_largest_perm(numbers):
    max = float('-inf')
    for a in range(len(numbers)):
        for b in range(len(numbers)):
            if numbers[a][b] > max and numbers[a][b] != np.Inf:
                    max = numbers[a][b]
    return max


def second_largest_atlas(numbers):
    max = float('-inf')
    for a in range(len(numbers)):
        if numbers[a] > max and numbers[a] != np.Inf:
                max = numbers[a]
    return max


# Graphs each permutation per atlas per patient
def graph_shortest_path_perm(file_path_array, sub_ID, atlas, outputfile):
    structural_connectivity_array = []
    for file_path in file_path_array:
        structural_connectivity_array.append(np.array(pd.DataFrame(loadmat(file_path)['connectivity'])))

    shortest_path_arr = []
    for structural_matrix in structural_connectivity_array:
        shortest_path_arr.append(bct.distance_bin(structural_matrix))

    # Remove zeros - self-loops along diagonal
    # for i in range(len(shortest_path_arr)):
        # shortest_path_arr[i] = shortest_path_arr[i][shortest_path_arr[i] != 0]

    # create objects for x-axis
    perm_level_pdf = PdfPages(outputfile)
    for perm in range(len(shortest_path_arr)):
        counters = []
        for x in range(int(second_largest_perm(shortest_path_arr[perm])) + 1):
            counters.append(0)
        counters.append(0)
        objects = []
        for x in range(len(counters) - 1):
            objects.append(str(x))
        objects.append('inf')
        y_pos = np.arange(len(objects))
        for i in range(len(shortest_path_arr[perm])):
            for j in range(len(shortest_path_arr[perm])):
                for c in range(len(counters) - 1):
                    if shortest_path_arr[perm][i][j] == c:
                        counters[c] += 1
                    if shortest_path_arr[perm][i][j] == float(np.Inf):
                        counters[len(counters) - 1] += 1

        #Visualizing bar graph
        fig = plt.figure()
        plt.bar(y_pos, counters, align='center', alpha=0.5, color='#0000ff')
        if perm < 9:
            legend_patch = mpatches.Patch(color='#6666ff', label=' Perm000' + str(perm + 1))
        else:
            legend_patch = mpatches.Patch(color='#6666ff', label=' Perm00' + str(perm + 1))
        plt.legend(handles=[legend_patch])
        plt.xticks(y_pos, objects)
        plt.ylabel('Frequency')
        plt.xlabel('Shortest Path Length')
        plt.title('Shortest Path Length Distributions')
        fig.text(0.025, 0.01, '{0}   {1}'.format(sub_ID, atlas))
        plt.show()
        perm_level_pdf.savefig(fig)
    perm_level_pdf.close()


# Graphs each permutation per atlas per patient
def graph_shortest_path_atlas(file_path_array, sub_ID, atlas_names, perm_list, outputfile):
    structural_connectivity_array = []
    for file_path in file_path_array:
        structural_connectivity_array.append(np.array(pd.DataFrame(loadmat(file_path)['connectivity'])))

    shortest_path_arr = []
    for structural_matrix in structural_connectivity_array:
        shortest_path_arr.append(bct.distance_bin(structural_matrix))

    shortest_path_array_atlas = []
    for a in range(len(atlas_names)):
        shortest_path_array_total = []
        for i in range(a * len(perm_list), (a + 1) * len(perm_list)):
            for j in range(len(shortest_path_arr[i])):
                for x in range(len(shortest_path_arr[i][j])):
                    shortest_path_array_total.append(shortest_path_arr[i][j][x])
        shortest_path_array_atlas.append(shortest_path_array_total)

    # create objects for x-axis
    atlas_level_pdf = PdfPages(outputfile)
    for atlas in range(len(shortest_path_array_atlas)):
        counters = []
        for x in range(int(second_largest_atlas(shortest_path_array_atlas[atlas])) + 1):
            counters.append(0)
        counters.append(0)
        objects = []
        for x in range(len(counters) - 1):
            objects.append(str(x))
        objects.append('inf')
        y_pos = np.arange(len(objects))
        for i in range(len(shortest_path_array_atlas[atlas])):
            for c in range(len(counters) - 1):
                if shortest_path_array_atlas[atlas][i] == c:
                    counters[c] += 1
                if shortest_path_array_atlas[atlas][i] == float(np.Inf):
                    counters[len(counters) - 1] += 1

        # Visualizing bar graph
        fig = plt.figure()
        plt.bar(y_pos, counters, align='center', alpha=0.5, color='#0000ff')
        legend_patch = mpatches.Patch(color='#6666ff', label=atlas_names[atlas])
        plt.legend(handles=[legend_patch])
        plt.xticks(y_pos, objects)
        plt.ylabel('Frequency')
        plt.xlabel('Shortest Path Length')
        plt.title('Shortest Path Length Distributions')
        fig.text(0.025, 0.01, '{0}'.format(sub_ID))
        plt.show()
        atlas_level_pdf.savefig(fig)
    atlas_level_pdf.close()


# Graphs total of permutations for each atlas for all the patients combined
def graph_shortest_path_all_patients(file_path_array, sub_IDs, atlas_names, perm_list, output_file):
    structural_connectivity_array = []
    for file_path in file_path_array:
        structural_connectivity_array.append(np.array(pd.DataFrame(loadmat(file_path)['connectivity'])))

    shortest_path_arr = []
    for structural_matrix in structural_connectivity_array:
        shortest_path_arr.append(bct.distance_bin(structural_matrix))
    print("arr length: {0}".format(len(shortest_path_arr)))

    shortest_path_array_all_patients = []
    for a in range(len(atlas_names)):
        shortest_path_array_total = []
        for i in range(a * len(perm_list) * len(sub_IDs), (a + 1) * len(perm_list) * len(sub_IDs)):
            for j in range(len(shortest_path_arr[i])):
                for x in range(len(shortest_path_arr[i][j])):
                    shortest_path_array_total.append(shortest_path_arr[i][j][x])
        shortest_path_array_all_patients.append(shortest_path_array_total)

    # create objects for x-axis
    all_patients_level_pdf = PdfPages(output_file)
    for atlas in range(len(shortest_path_array_all_patients)):
        counters = []
        for x in range(int(second_largest_atlas(shortest_path_array_all_patients[atlas])) + 1):
            counters.append(0)
        counters.append(0)
        objects = []
        for x in range(len(counters) - 1):
            objects.append(str(x))
        objects.append('inf')
        y_pos = np.arange(len(objects))
        for i in range(len(shortest_path_array_all_patients[atlas])):
            for c in range(len(counters) - 1):
                if shortest_path_array_all_patients[atlas][i] == c:
                    counters[c] += 1
                if shortest_path_array_all_patients[atlas][i] == float(np.Inf):
                    counters[len(counters) - 1] += 1

        # Visualizing bar graph
        fig = plt.figure()
        plt.bar(y_pos, counters, align='center', alpha=0.5, color='#0000ff')
        legend_patch = mpatches.Patch(color='#6666ff', label=atlas_names[atlas])
        plt.legend(handles=[legend_patch])
        plt.xticks(y_pos, objects)
        plt.ylabel('Frequency')
        plt.xlabel('Shortest Path Length')
        plt.title('Shortest Path Length Distributions')
        fig.text(0.025, 0.01, 'All Patients')
        plt.show()
        all_patients_level_pdf.savefig(fig)
    all_patients_level_pdf.close()
'''

'''
# Code with appropriate input and output files
# Graphs perm-level and atlas-level plots
for s in sub_ID_array:
    output_file_atlas = ('{0}sub-{1}/{2}_random_atlases.pdf'.format(output_file_directory, s, s))
    print('Output directory: {0}'.format(output_file_atlas))
    structure_file_path_array_atlas = []
    for a in atlas_names:
        output_file_perm = ('{0}sub-{1}/{2}_{3}.pdf'.format(output_file_directory, s, s, a))
        print('Output directory: {0}'.format(output_file_perm))
        structure_file_path_array = []
        for p in range(len(perm_list)):
            if p < 9:
                file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}_Perm000{5}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a, perm_list[p])
            else:
                file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}_Perm00{5}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a, perm_list[p])
            file = '{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a)
            structure_file_path_array.append(file)
            structure_file_path_array_atlas.append(file)
        graph_perm_level(structure_file_path_array, s, a, output_file_perm, 'Betweenness Centrality', 'Betweenness Centrality Distribution')
        # graph_shortest_path_perm(structure_file_path_array, s, a, output_file_perm)
    graph_atlas_level(structure_file_path_array_atlas, s, atlas_names, perm_list, output_file_atlas, 'Betweenness Centrality', 'Betweenness Centrality Distribution')
    # graph_shortest_path_atlas(structure_file_path_array_atlas, s, atlas_names, perm_list, output_file_atlas)
'''

# Graph of all patients totaled for the network feature
output_file_patients = ('{0}All_Patients.pdf'.format(output_file_directory))
print('Output directory: {0}'.format(output_file_patients))
structure_file_path_array_all_patients = []
for a in standard_atlases:
    # for p in range(len(perm_list)):
        # for s in sub_ID_array:
            # if p < 9:
                # structure_file_path_array_all_patients.append('{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}_Perm000{5}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a, perm_list[p])) #R
                # structure_file_path_array_all_patients.append( {0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a)) # standard atlas
            # else:
                # structure_file_path_array_all_patients.append('{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}_Perm00{5}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a, perm_list[p])) #RA
    for s in sub_ID_array:
        structure_file_path_array_all_patients.append('{0}sub-{1}/connectivity_matrices/structural/{2}/sub-{3}_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.{4}.count.pass.connectivity.mat'.format(file_directory, s, a, s, a)) # standard atlas
graph_patient_level(structure_file_path_array_all_patients, sub_ID_array, standard_atlases, [1], output_file_patients, 'Betweenness Centrality', 'Betweenness Centrality Distributions')
# graph_shortest_path_all_patients(structure_file_path_array_all_patients, sub_ID_array, atlas_names, perm_list, output_file_patients)
