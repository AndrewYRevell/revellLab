"""
07/05/2020
Dhanya Mahesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: This code will take a processed patient file along with the MNI coordinates Excel file
and output a 3D matrix: (frequency, PSD, distance from grey matter)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:
    1. def processEEG(eegpath, coorpath):
        PARAMS: eegpath: path of processed EEG pickle file
                coorpath: path of MNI coordinates for above pickle file
        RETURN:
        This method carries out the PSD processing and plotting for one EEG file.
    2. toComparePSDvdist(dat, amt):
        PARAMS: dat = EEG data
                amt = amount of channels/electrodes
        RETURN: freqs = frequency information for all electrodes
                PSD = power spectral density info for all electrodes
                dist = distance from grey matter structure (0 = grey matter electrode, >0 = WM electrode)
        This method calculates PSD and distance measures.
    3. plotPSDvsdist(tmpf, tmpp, tmpd, type_time)
        PARAMS: tmpf = a subset of frequencies to plot
                tmpd = a subset of distances to plot
                tmpp = a subset of PSD to plot
                type_time = title of plot
        This method creates a 3D plot for PSD, frequency, distance
    4. Code for user to run/change

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    eegpath = "data/sub-RID0278_HUP138_phaseII_415933490000_416023190000_EEG_filtered.pickle"
    coorpath = "greywhite/sub-RID0278_electrode_coordinates_mni_tissue_segmentation_dist_from_grey.csv"
    ref_elec = 'RF04' REFERENCE ELECTRODE
    patient_name = 'RID0278'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    patient_name + '_PSD_array.csv' = csv with PSD, freq, distance info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    #preictal
    inputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_processed/eeg_filtered/sub-RID0440/sub-RID0440_HUP172_phaseII_402651841658_402704260829_EEG_filtered.pickle'
    outputfile ='/Users/andyrevell/mount/USERS/arevell/papers/paper005/analysis/eeg_spectral_properties/sub-RID0440_PSD_array.csv'



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import matplotlib.pyplot as plt
from matplotlib import *
import pickle
from scipy import signal
from scipy import stats
from mpl_toolkits import mplot3d
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
# broadband_conn is slightly changed to only do pre-processing
from scipy.stats import zscore
from skimage.transform import rescale
from echobase import broadband_conn, multiband_conn
from spectrum import *
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
""" Start of Code: Preprocessing """
def processEEG(eegpath, coorpath):
    with open(eegpath, 'rb') as f: time_series, fs = pickle.load(f)
    electrode_info = pd.read_csv(coorpath)

    freqs_full, psd_full, dist_full = toComparePSDvdist(time_series, fs, electrode_info)
    export_arr = np.vstack([freqs_full, psd_full, dist_full])
    print("freqs, psd, dist = " + str(export_arr.shape[0]))
    #psd_full = stats.zscore(psd_full, axis=None)

    plotPSDvsdist(freqs_full[0:44, 0:20], psd_full[0:44, 0:20], dist_full[0:44, 0:20], type_time=0)
    numpy.savetxt('/Users/andyrevell/mount/USERS/arevell/papers/paper005/analysis/eeg_spectral_properties/' + patient_name + '_PSD_array.csv', export_arr, delimiter=',', comments="freqs, psd, dist = " + str(export_arr.shape[0]/3))

""" Power Spectral Density Plots (3D Plot) """
def toComparePSDvdist(time_series, fs, electrode_info):
    electrode_names = list(time_series) # get list of electrodes from the data
    dist = electrode_info['distance from grey matter']
    dat = time_series.to_numpy() # preictal time series
    freqs, psd = signal.welch(np.reshape(dat[:, [1]], dat[:, [1]].shape[0]), fs, nperseg=int(fs) * 1,
                              window=signal.hamming(int(fs) * 1))
    freqs_arr = np.zeros(freqs.shape)
    psd_arr = np.zeros(psd.shape)
    dist_arr = np.zeros(freqs.shape)
    for x in range(time_series.shape[1]):
        if (electrode_names[x] != ref_elec):
            x_ray = dat[:, [x]]  # for every channel
            x_ray = np.reshape(x_ray, dat.shape[0])
            ind = np.where(electrode_info['electrode_name'] == electrode_names[x])  # find the index of the electrode for the white matter distance
            dist_arr = np.vstack([dist_arr, np.full((513), dist[ind[0][0]])])  # get distance
            freqs, psd = signal.welch(x_ray, fs, nperseg=int(fs) * 1, window=signal.hamming(int(fs) * 1),)  # collect psd data
            freqs_arr = np.vstack([freqs_arr, freqs])
            psd_arr = np.vstack([psd_arr, psd])
    freqs_arr = numpy.delete(freqs_arr, (0), axis=0)
    psd_arr = np.delete(psd_arr, (0), axis=0)
    dist_arr = np.delete(dist_arr, (0), axis=0)
    return freqs_arr, psd_arr, dist_arr

def plotPSDvsdist(tmpf, tmpp, tmpd, type_time):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(tmpd, tmpf, tmpp, rstride=1, cstride=1, cmap='viridis')
    ax.set_xlabel('distance from grey matter electrode')
    ax.set_ylabel('frequency (Hz)')
    ax.set_zlabel('PSD (z scored)')
    ax.set_ylim([0, 20])
    if(type_time == 0):
        ax.set_title('Preictal Electrodes for RID0278')
    elif(type_time == 1):
        ax.set_title('Ictal Electrodes for RID0278')
    elif(type_time == 2):
        ax.set_title('Interictal Electrodes for RID0278')
    else:
        ax.set_title('Postictal Electrodes for RID0278')
    plt.show()

"""Run the following code"""
eegpath = "data/sub-RID0278_HUP138_phaseII_415933490000_416023190000_EEG_filtered.pickle"
coorpath = "greywhite/sub-RID0278_electrode_coordinates_mni_tissue_segmentation_dist_from_grey.csv"
ref_elec = 'RF04'
patient_name = 'sub-RID0278'
processEEG(eegpath, coorpath)