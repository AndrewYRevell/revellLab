# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:31:04 2020

@author: asilv
"""

import get_network_measures as net

sub_ID='RID0278'
iEEG_filename="HUP138_phaseII"
start_times_array=[248432340000,
248525740000,
248619140000,
338848220000,
339008330000,
339168440000,
415933490000,
416023190000,
416112890000,
429398830000,
429498590000,
429598350000,
458393300000,
458504560000,
458615820000,
226925740000,
317408330000,
394423190000,
407898590000,
436904560000]
stop_times_array=[248525740000,
248619140000,
248799140000,
339008330000,
339168440000,
339348440000,
416023190000,
416112890000,
416292890000,
429498590000,
429598350000,
429778350000,
458504560000,
458615820000,
458795820000,
227019140000,
317568440000,
394512890000,
407998350000,
437015820000]

# general BIDS path
BIDS_proccessed_directory="/gdrive/public/DATA/Human_Data/BIDS_processed"
# folder to put the data in 
output_directory = "/gdrive/public/USERS/arevell/papers/paper005/data_processed/network_measures"
# path to the mni tissue segmentation assigment 
electrode_localization_by_classification_atlas_file_path = '{0}/sub-{1}/electrode_localization/electrode_localization_by_atlas/sub-{1}_electrode_coordinates_mni_tissue_segmentation_dist_from_grey.csv'.format(BIDS_proccessed_directory,sub_ID)
# loop over the times 
for i in range(len(start_times_array)):
    print("Starting time",i)
    #Making Correct file paths and names based on above input
    start_time_usec=start_times_array[i]
    stop_time_usec=stop_times_array[i]
    function_file_path = "{0}/sub-{1}/connectivity_matrices/functional/eeg/sub-{1}_{2}_{3}_{4}_functionalConnectivity.pickle".format(BIDS_proccessed_directory,sub_ID,iEEG_filename,start_time_usec,stop_time_usec)
    outputfile_name = "sub-{0}_{1}_{2}_{3}_true_and_resampled_network.pickle".format(sub_ID,iEEG_filename,start_time_usec,stop_time_usec)
    outputfile = '{0}/sub-{1}/{2}'.format(output_directory, sub_ID,outputfile_name)
    net.get_true_and_resampled_network(function_file_path,outputfile,electrode_localization_by_classification_atlas_file_path)




