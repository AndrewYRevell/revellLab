"""
Dhanya Mahesh
06/15/2020 - 06/23/2020

1. Open all the filtered pickle files and make lists with distance, electrode labels, etc.
2. plotPSDvsdist(dat, amt, type_time) method: This method plots a 3D PSD vs frequency vs distance from grey matter electrode
    plot
    PARAMS:
    dat = dat_preictal, dat_ictal, dat_postictal, dat_interictal
    amt = number of electrodes
    type_time = label with either "ictal", "postictal", etc for title purposes
3. specplot(dat,chann,lab) method: This method plots a spectrogram for a certain channel in a dataset
    PARAMS:
    dat = dat_preictal, dat_ictal, dat_postictal, dat_interictal
    chann = channel number (use electrode_names_'time' to find out index)
    lab = label for title purposes
4. calc_MI(x, y, bins) method: This method is a calculation for mutual information from x and y time series
5. MIplot(dat,lab) method: This method plots the mutual information measures for all a time series
    PARAMS:
    dat = dat_preictal, dat_ictal, dat_postictal, dat_interictal
    lab = label for legend purposes
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
# preictal
with open("data/sub-RID0278_HUP138_phaseII_415933490000_416023190000_EEG_filtered.pickle", 'rb') as f: time_series_preictal, fs = pickle.load(f)
# ictal
with open("data/sub-RID0278_HUP138_phaseII_416023190000_416112890000_EEG_filtered.pickle", 'rb') as f: time_series_ictal, fs = pickle.load(f)
# interictal
with open("data/sub-RID0278_HUP138_phaseII_394423190000_394512890000_EEG_filtered.pickle", 'rb') as f: time_series_interictal, fs = pickle.load(f)
# postictal
with open("data/sub-RID0278_HUP138_phaseII_416112890000_416292890000_EEG_filtered.pickle", 'rb') as f: time_series_postictal, fs = pickle.load(f)
electrode_names_preictal = list(time_series_preictal) # get list of electrodes from the data
electrode_names_ictal = list(time_series_ictal) # get list of electrodes from the data
electrode_names_interictal = list(time_series_interictal) # get list of electrodes from the data
electrode_names_postictal = list(time_series_postictal) # get list of electrodes from the data

electrode_info = pd.read_csv("greywhite/sub-RID0278_electrode_coordinates_mni_tissue_segmentation_dist_from_grey.csv")
dist = electrode_info['distance from grey matter']
dat_preictal = time_series_preictal.to_numpy() # preictal time series
dat_ictal = time_series_ictal.to_numpy() # ictal time series
dat_interictal = time_series_interictal.to_numpy()
dat_postictal = time_series_postictal.to_numpy()



""" Power Spectral Density Plots (3D Plot) """
def toComparePSDvdist(dat, amt):
    # initializing arrays for x, y, z plotting
    # need to figure out how long each segment is
    freqs, psd = signal.welch(np.reshape(dat[:, [1]], dat[:, [1]].shape[0]), fs, nperseg=512 * 1,
                              window=signal.hamming(512 * 1))
    # initializing white matter arrays
    freqs_white = np.zeros(freqs.shape)
    psd_white = np.zeros(psd.shape)
    dist_white = np.zeros(freqs.shape)
    for x in range(amt):
        if (electrode_names_preictal[x] != 'RF04'):
            x_ray = dat[:, [x]]  # for every channel
            x_ray = np.reshape(x_ray, dat.shape[0])
            ind = np.where(electrode_info['electrode_name'] == electrode_names_preictal[
                x])  # find the index of the electrode for the white matter distance
            # if dist[ind[0][0]] > 0:
            dist_white = np.vstack([dist_white, np.full((257), dist[ind[0][0]])])  # get distance
            freqs, psd = signal.welch(x_ray, fs, nperseg=512 * 1, window=signal.hamming(512 * 1),
                                      scaling='spectrum')  # collect psd data
            # print(psd.max())
            # print(electrode_names_preictal[x])
            # psd = stats.zscore(psd) # z score psd
            freqs_white = np.vstack([freqs_white, freqs])
            psd_white = np.vstack([psd_white, psd])
    # delete the first row of the array because it is all zeros
    freqs_white = numpy.delete(freqs_white, (0), axis=0)
    psd_white = np.delete(psd_white, (0), axis=0)
    dist_white = np.delete(dist_white, (0), axis=0)
    # focus on certain frequencies of interest
    tmpf = freqs_white[0:44, 0:20]
    tmpp = stats.zscore(psd_white[0:44, 0:20], axis=None)
    tmpd = dist_white[0:44, 0:20]
    return tmpf, tmpp, tmpd

def plotPSDvsdist(tmpf, tmpp, tmpd, type_time):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(tmpd, tmpf, tmpp, rstride=1, cstride=1, cmap='viridis')
    ax.set_xlabel('distance from grey matter electrode')
    ax.set_ylabel('frequency (Hz)')
    ax.set_zlabel('PSD (z scored)')
    if(type_time == 0):
        ax.set_title('Pre-ictal Electrodes for RID0278')
    elif(type_time == 1):
        ax.set_title('Ictal Electrodes for RID0278')
    elif(type_time == 2):
        ax.set_title('Interictal Electrodes for RID0278')
    else:
        ax.set_title('Postictal Electrodes for RID0278')
    plt.show()

tmpf, tmpp, tmpd = toComparePSDvdist(dat_preictal,time_series_preictal.shape[1])
plotPSDvsdist(tmpf,tmpp,tmpd,type_time = 0)
tmpf, tmpp, tmpd = toComparePSDvdist(dat_ictal,time_series_ictal.shape[1])
plotPSDvsdist(tmpf,tmpp,tmpd,type_time = 1)
tmpf, tmpp, tmpd = toComparePSDvdist(dat_interictal,time_series_interictal.shape[1])
plotPSDvsdist(tmpf,tmpp,tmpd,type_time = 2)
tmpf, tmpp, tmpd = toComparePSDvdist(dat_postictal,time_series_postictal.shape[1])
plotPSDvsdist(tmpf,tmpp,tmpd,type_time = 3)



"""Spectrogram  """
def specplot(dat,chann,lab):
    x_ray = dat[:,[chann]]
    x_ray = np.reshape(x_ray, dat.shape[0])
    fig = plt.figure()
    Pxx, freqs, bins, im = plt.specgram(x_ray, NFFT=512, Fs=fs, scale_by_freq = True,cmap = 'RdBu')
    ax = fig.add_subplot(111)
    ax.set_title('Spectrogram: '+lab)
    ax.set_xlabel('Time (in seconds)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im).set_label('Intensity [dB]')
    fig.show()

specplot(dat_preictal,5,lab = 'Preictal '+electrode_names_preictal[5])
specplot(dat_ictal,5,lab = 'Ictal '+electrode_names_ictal[5])
specplot(dat_interictal,5,lab = 'Interictal '+electrode_names_interictal[5])
specplot(dat_postictal,5,lab = 'Postictal '+electrode_names_postictal[5])


""" Mutual Information (Basic Time Series) """
def calc_MI(x, y, bins):
#def calc_MI(x,y):
    c_xy = np.histogram2d(x, y, bins)[0]
    #mi = normalized_mutual_info_score(x,y)
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# Note: need to find other ways of calculating MI
elec_info = pd.read_csv("greywhite/sub-RID0278_electrode_coordinates_mni_tissue_segmentation_dist_from_grey.csv")
coor = elec_info[['mni_x_coordinate','mni_y_coordinate','mni_z_coordinate']] # coordinate list
grey_elec_info = np.where(elec_info['distance from grey matter'] == 0) # find all the grey electrodes
dist = elec_info['distance from grey matter'] # array with all the distances
# need some way to get rid of extra electrodes in this excel sheet that are not in pickle file
def MIplot(dat,dat_names, lab):
    dist_act = []
    mi_act = []
    elec_names = elec_info['electrode_name']
    grey_elec_names = [elec_names[i] for i in grey_elec_info[0]]
    coh_coor = cdist(coor,coor) # Euclidiean distances between each electrode
    for x in range(coor.shape[0]): # for each electrode
        if dist[x] > 0 and x < dat.shape[1] and x != 43 and x != 75: # if the electrode is WM, and within the pickle file dataset
            #print(x)
            tmp_coor = coh_coor[x] # temporary distance matrix
            tmp_coor = tmp_coor[grey_elec_info] # only take distances from WM to GM
            result = list(np.where(tmp_coor == min(tmp_coor)))[0][0]
            dist_act.append(min(tmp_coor))
            #result = min(enumerate(tmp_coor), key=lambda x: x[1] if x[1] > 0 else float('inf'))
            ind_GM = dat_names.index(grey_elec_names[result])
            ind_WM = dat_names.index(elec_names[x])
            y_ray = dat[:, ind_GM] #GM electrode time series array
            y_ray = np.reshape(y_ray, dat.shape[0])
            x_ray = dat[:, ind_WM]
            x_ray = np.reshape(x_ray, dat.shape[0])
            #if(np.where(coh_coor[x] == result[1])[0][0] < dat.shape[1]):
                #y_ray = dat[:, [np.where(coh_coor[x] == result[1])[0][0]]] # ictal grey
                #y_ray = np.reshape(y_ray, dat.shape[0])
                #print(result)
                #dist_act.append(result[1])
                #mi_act.append(calc_MI(x_ray,y_ray))
            mi_act.append(calc_MI(x_ray,y_ray,int(sqrt(x_ray.shape[0]/5))))
            #dist_act.append(dist[x])
    plt.scatter(dist_act,mi_act,label = lab)
    plt.title('Mutual Info vs Distance from GM electrode')
    plt.xlabel('Distance from GM electrode')
    plt.ylabel('Mutual Information (Time Series) ')
    plt.show()

MIplot(dat_preictal, electrode_names_preictal, lab = "Preictal")
MIplot(dat_ictal, lab = "Ictal")
MIplot(dat_interictal, lab = "Interictal")
MIplot(dat_postictal, lab = "Postictal")
plt.title('Mutual Info vs Distance from GM electrode')
plt.xlabel('Distance from GM electrode')
plt.ylabel('Mutual Information (Time Series) ')
plt.legend(loc="upper right")
plt.show()


