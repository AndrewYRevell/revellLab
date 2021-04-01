"""
2020.06.17
Lena Armstrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    Create a feature matrix to feed into the ML algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. EDGE WEIGHT (Depth 0)
    2. DEGREE (Depth 1 & 2)
    3. NODE STRENGTH (Depth 3 & 4)
    4. SHORTEST PATH LENGTH (Depth 5-9)
    5. CHARACTERISTIC PATH LENGTH (DEPTH 10-15)
    6. GLOBAL EFFICIENCY (DEPTH 16)
    7. LOCAL EFFICIENCY (DEPTH 17 & 18)
    8. BETWEENNESS CENTRALITY (Depth 19 & 20)
    9. EIGENECTOR CENTRALITY (DEPTH 21 & 22)
    10. EDGE BETWEENNESS CENTRALITY (Depth 23-25)
    11. CLUSTERING COEFFICIENTS (Depth 25 & 26)
    12. TRANSITIVITY (DEPTH 27)
    13. ASSORTATIVITY (DEPTH 28)
    14. DENSITY (Depth 29-31)
    15. PAGERANK CENTRALITY (DEPTH 32 & 33)
    16. MEAN FIRST PASSAGE OF TIME (Depth 34 & 35)
    17. SEARCH INFORMATION (Depth 36)
    18. MATCHING INDEX (37 & 38)
    19. RICH CLUB COEFFICIENT (Depth 39)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
A 2D array of a structural adjacency matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
A 3D array of brain network features based on the structural adjacency matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat  # fromn scipy to read mat files (structural connectivity)
import bct
import statistics

# Example structure file
structure_matrix_file = '/Users/larmstrong2020/Desktop/PURM_2020/Files/sub-RID0420_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.RA_N0010_Perm0001.count.pass.connectivity.mat'
structure_matrix_file2 = '/Users/larmstrong2020/Desktop/PURM_2020/Files/sub-RID0420_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.RA_N0010_Perm0001.count.pass.connectivity.mat'
structure_matrix_file3 = '/Users/larmstrong2020/mount/DATA/Human_Data/BIDS_processed/sub-RID0194/connectivity_matrices/structural/RA_N1000/sub-RID0194_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.RA_N1000_Perm0005.count.pass.connectivity.mat'
structure_matrix_file4 = '/Users/larmstrong2020/mount/DATA/Human_Data/BIDS_processed/sub-RID0420/connectivity_matrices/structural/JHU_aal_combined_res-1x1x1/sub-RID0420_ses-preop3T_dwi-eddyMotionB0Corrected.nii.gz.trk.gz.JHU_aal_combined_res-1x1x1.count.pass.connectivity.mat'
# 'aal_res-1x1x1', 'desikan_res-1x1x1', 'JHU_aal_combined_res-1x1x1'

def create_feature_matrix(structure_matrix_file):
    # Feature matrix with each element containing an NxN array
    feature_matrix = []

    # EDGE WEIGHT (Depth 0)
    # weighted & undirected network
    structural_connectivity_array = np.array(pd.DataFrame(loadmat(structure_matrix_file)['connectivity']))
    feature_matrix.append(structural_connectivity_array)

    # DEGREE (Depth 1 & 2)
    # Node degree is the number of links connected to the node.
    deg = bct.degrees_und(structural_connectivity_array)
    fill_array_2D(feature_matrix, deg)

    # *** Conversion of connection weights to connection lengths ***
    connection_length_matrix = bct.weight_conversion(structural_connectivity_array, 'lengths')
    # print(connection_length_matrix)

    # SHORTEST PATH LENGTH (Depth 3 & 4)
    '''
    The distance matrix contains lengths of shortest paths between all pairs of nodes.
    An entry (u,v) represents the length of shortest path from node u to node v.
    The average shortest path length is the characteristic path length of the network.
    '''
    shortest_path = bct.distance_wei(connection_length_matrix)
    feature_matrix.append(shortest_path[0])  # distance (shortest weighted path) matrix
    feature_matrix.append(shortest_path[1])  # matrix of number of edges in shortest weighted path

    # BETWEENNESS CENTRALITY (Depth 5 & 6)
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.
    '''
    bc = bct.betweenness_wei(connection_length_matrix)
    fill_array_2D(feature_matrix, bc)

    # CLUSTERING COEFFICIENTS (Depth 7 & 8)
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.
    '''
    cl = bct.clustering_coef_wu(connection_length_matrix)
    fill_array_2D(feature_matrix, cl)

    # Find disconnected nodes - component size set to 1
    new_array = structural_connectivity_array
    W_bin = bct.weight_conversion(structural_connectivity_array, 'binarize');
    [comps, comp_sizes] = bct.get_components(W_bin)
    print('comp: ', comps)
    print('sizes: ', comp_sizes)
    for node in range(len(comps)):
        if (comps[node] != statistics.mode(comps)):
            new_array = np.delete(new_array, new_array[node])

    return feature_matrix


# turns 2D feature into 3D
def fill_array_2D(feature_matrix, feature_array):
    feature_array_level1 = []
    for row in range(len(feature_array)):
        one_row = []
        for col in range(len(feature_array)):
            one_row.append(feature_array[row])
        feature_array_level1.append(one_row)
    feature_matrix.append(np.array(feature_array_level1))

    feature_array_level2 = []
    for row in range(len(feature_array)):
        one_row = []
        for col in range(len(feature_array)):
            one_row.append(feature_array[row])
        feature_array_level2.append(one_row)
    feature_matrix.append(np.array(feature_array_level2))


# turns 1D feature into 3D
def fill_array_1D(feature_matrix, feature_value):
    feature_array = []
    for row in range(len(feature_matrix)):
        for col in range(len(feature_matrix[row])):
            one_row = []
            for x in range(len(feature_matrix[row])):
                one_row.append(feature_value)
        feature_array.append(one_row)
    feature_matrix.append(np.array(feature_array))


feature_matrix = create_feature_matrix(structure_matrix_file4)

'''
Extra features:

    # NODE STRENGTH (Depth 3 & 4)
    # Node strength is the sum of weights of links connected to the node. The
    # instrength is the sum of inward link weights and the outstrength is the
    # sum of outward link weights.
    strength = bct.strengths_und(structure_matrix)
    fill_array_2D(feature_matrix, strength)
    
    # SHORTEST PATH LENGTH (Depth 7-9)
    # Computes the topological length of the shortest possible path connecting
    # every pair of nodes in the network (can be obtained with connection weight
    # or connection length arrays)
    shortest_path_floyd = bct.distance_wei_floyd(connection_length_matrix)
    feature_matrix.append(shortest_path_floyd[0])  # shortest path-length array
    feature_matrix.append(shortest_path_floyd[1])  # Number of edges in the shortest path array
    # [i,j]` of this array indicates the next node in the shortest path between `i` and `j`
    # This array is used as an input argument for function `retrieve_shortest_path()`
    feature_matrix.append(shortest_path_floyd[2])

    # CHARACTERISTIC PATH LENGTH (DEPTH 10-15)
    # The characteristic path length is the average shortest path length in
    # the network. 
    cpl = bct.charpath(shortest_path[0], include_diagonal=False, include_infinite=False)
    fill_array_1D(feature_matrix, cpl[0])  # characteristic path length
    fill_array_1D(feature_matrix, cpl[1])  # global efficiency
    fill_array_2D(feature_matrix, cpl[2])  # eccentricity at each vertex
    fill_array_1D(feature_matrix, cpl[3])  # radius of graph
    fill_array_1D(feature_matrix, cpl[4])  # diameter of graph

    # GLOBAL EFFICIENCY (DEPTH 16)
    # The global efficiency is the average of inverse shortest path length,
    # and is inversely related to the characteristic path length.
    ge = bct.efficiency_wei(connection_length_matrix, local=False)
    fill_array_1D(feature_matrix, ge)

    # LOCAL EFFICIENCY (DEPTH 17 & 18)
    # The local efficiency is the global efficiency computed on the
    # neighborhood of the node, and is related to the clustering coefficient.
    le = bct.efficiency_wei(connection_length_matrix, local=True)
    fill_array_2D(feature_matrix, le)
    
    # EIGENECTOR CENTRALITY (DEPTH 21 & 22)
    # Eigenector centrality is a self-referential measure of centrality:
    # nodes have high eigenvector centrality if they connect to other nodes
    # that have high eigenvector centrality. The eigenvector centrality of
    # node i is equivalent to the ith element in the eigenvector
    # corresponding to the largest eigenvalue of the adjacency matrix.
    ec = bct.eigenvector_centrality_und(structure_matrix)
    fill_array_2D(feature_matrix, ec)

    # EDGE BETWEENNESS CENTRALITY (Depth 23-25)
    # Edge betweenness centrality is the fraction of all shortest paths in
    # the network that contain a given edge. Edges with high values of
    # betweenness centrality participate in a large number of shortest paths
    ebc = bct.edge_betweenness_wei(connection_length_matrix)
    feature_matrix.append(ebc[0])  # edge betweenness centrality matrix
    fill_array_2D(feature_matrix, ebc[1])  # nodal betweenness centrality vector
    
    # TRANSITIVITY (DEPTH 27)
    # Transitivity is the ratio of 'triangles to triplets' in the network.
    # (A classical version of the clustering coefficient).
    t = bct.transitivity_wu(connection_length_matrix)
    fill_array_1D(feature_matrix, t)

    # ASSORTATIVITY (DEPTH 28)
    # The assortativity coefficient is a correlation coefficient between the
    # strengths (weighted degrees) of all nodes on two opposite ends of a link.
    # A positive assortativity coefficient indicates that nodes tend to link to
    # other nodes with the same or similar strength.
    a = bct.assortativity_wei(structure_matrix)
    fill_array_1D(feature_matrix, a)

    # DENSITY (Depth 29-31)
    # Density is the fraction of present connections to possible connections.
    density = bct.density_dir(structure_matrix)
    fill_array_1D(feature_matrix, density[0])  # density
    fill_array_1D(feature_matrix, density[1])  # vertices
    fill_array_1D(feature_matrix, density[2])  # edges

    # PAGERANK CENTRALITY (DEPTH 32 & 33)
    # The PageRank centrality is a variant of eigenvector centrality. 
    # Formally, PageRank is defined as the stationary distribution achieved
    # by instantiating a Markov chain on a graph. The PageRank centrality of
    # a given vertex, then, is proportional to the number of steps (or amount
    # of time) spent at that vertex as a result of such a process. 
    
    # The PageRank index gets modified by the addition of a damping factor,
    # d. In terms of a Markov chain, the damping factor specifies the
    # fraction of the time that a random walker will transition to one of its
    # current state's neighbors. The remaining fraction of the time the
    # walker is restarted at a random vertex. A common value for the damping
    # factor is d = 0.85.
    prc = bct.pagerank_centrality(structure_matrix, 0.85)
    fill_array_2D(feature_matrix, prc)

    # MEAN FIRST PASSAGE OF TIME (Depth 34 & 35)
    # The first passage time from i to j is the expected number of steps it takes
    # a random walker starting at node i to arrive for the first time at node j.
    mfpot = bct.mean_first_passage_time(structure_matrix)
    fill_array_2D(feature_matrix, mfpot)

    # SEARCH INFORMATION (Depth 36)
    # Computes the amount of information (measured in bits) that a random
    # walker needs to follow the shortest path between a given pair of nodes.
    si = bct.search_information(new_array)
    feature_matrix.append(si)

    # MATCHING INDEX (37 & 38)
    # Matching index is a measure of
    # similarity between two nodes' connectivity profiles (excluding their
    # mutual connection, should it exist).
    mi = bct.matching_ind_und(structure_matrix)
    fill_array_2D(feature_matrix, mi)

    # RICH CLUB COEFFICIENT (Depth 39)
    # The rich club coefficient, R, at level k is the fraction of edges that
    # connect nodes of degree k or higher out of the maximum number of edges
    # that such nodes might share
    rcc = bct.rich_club_wu(structure_matrix)
    feature_matrix.append(rcc)
'''
