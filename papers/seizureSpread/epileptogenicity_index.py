# Imports
import numpy as np
from scipy.signal import convolve2d
from scipy import signal

# EI functions
def compute_hfer(target_data, base_data, fs):
    '''
    This function computes the high frequency energy ratio for the target and baseline data.
    It normalizes the energy in the target and baseline data to the average energy in each the target
    and baseline respectively.
    The input data, both target and baseline, must be bandpassed into the gamma range of either

    :param target_data: (Time x Channel) data with pre-ictal to ictal transition
    :param base_data: (Time x Channel) pre-ictal baseline data
    :param fs: sampling frequency
    :return: normalized high frequency energy for baseline and target data
    '''
    target_data = target_data.T
    base_data = base_data.T
    target_sq = target_data ** 2
    base_sq = base_data ** 2
    window = int(fs / 2.0)
    target_energy=convolve2d(target_sq,np.ones((1,window)),'same')
    base_energy=convolve2d(base_sq,np.ones((1,window)),'same')
    base_energy_ref = np.sum(base_energy, axis=1) / base_energy.shape[1]
    target_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, target_energy.shape[1]))
    base_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, base_energy.shape[1]))
    norm_target_energy = target_energy / target_de_matrix.astype(np.float32)
    norm_base_energy = base_energy / base_de_matrix.astype(np.float32)
    return norm_target_energy, norm_base_energy

def determine_threshold_onset(target, base):
    '''
    Computes the threshold value that must be surpassed in order to detect the transition from interictal to ictal
    in the target signal (10 sd above mean baseline signal)

    :param target: (channels x time) normalized baseline energy data
    :param base: (channels x time) normalized baseline energy data
    :return: onset_location: time index for each channel at which transition interictal-ictal takes place
    '''
    base_data = base.copy()
    target_data = target.copy()
    sigma = np.std(base_data, axis=1, ddof=1)
    channel_max_base = np.max(base_data, axis=1)
    thresh_value = channel_max_base + 10 * sigma
    onset_location = np.zeros(shape=(target_data.shape[0],))
    for channel_idx in range(target_data.shape[0]):
        logic_vec = target_data[channel_idx, :] > thresh_value[channel_idx]
        if np.sum(logic_vec) == 0:
            onset_location[channel_idx] = len(logic_vec)
        else:
            onset_location[channel_idx] = np.where(logic_vec != 0)[0][0]
    return onset_location


def compute_ei_index(target, base, fs):
    '''
    Function to actually compute the EI. The input is in Time x Channels, but it gets transposed
    inside compute_hfer, therefore careful when debugging.

    :param target: (Time x Channel) data with pre-ictal to ictal transition
    :param base: (Time x Channel) pre-ictal baseline data
    :param fs: sampling frequency
    :return: ei: List of EI values for each of the input channels
    '''
    target, base = compute_hfer(target, base, fs)
    ei = np.zeros([1, target.shape[0]])
    hfer = np.zeros([1, target.shape[0]])
    onset_rank = np.zeros([1, target.shape[0]])
    channel_onset = determine_threshold_onset(target, base)
    seizure_location = np.min(channel_onset)
    onset_channel = np.argmin(channel_onset)
    hfer = np.sum(target[:, int(seizure_location):int(seizure_location + 0.25 * fs)], axis=1) / (fs * 0.25)
    print(hfer)
    onset_asend = np.sort(channel_onset)
    time_rank_tmp = np.argsort(channel_onset)
    onset_rank = np.argsort(time_rank_tmp) + 1
    onset_rank = np.ones((onset_rank.shape[0],)) / np.float32(onset_rank)
    ei = np.sqrt(hfer * onset_rank)
    for i in range(len(ei)):
        if np.isnan(ei[i]) or np.isinf(ei[i]):
            ei[i] = 0
    if np.max(ei) > 0:
        ei = ei / np.max(ei)
    return ei


def get_ei_from_data(target, base, fs):
    '''
    :param target: (Time x Channel) data with pre-ictal to ictal transition
    :param base: (Time x Channel) pre-ictal baseline data
    :param fs: sampling frequency
    :return: ei: List of EI values for each of the input channels
    '''
    # Define bandpass filter between 70Hz and either fs/2 or 140Hz (whichever is smaller)
    if int(fs/2)<140:
        b, a = signal.butter(4, [70,int(fs/2)-1], 'bandpass', fs=fs)
    else:
        b, a = signal.butter(4, [70, 140], 'bandpass', fs=fs)

    # Get the Gamma components of the target and baseline signals
    target_he = signal.filtfilt(b, a, target, axis=0)
    base_he = signal.filtfilt(b, a, base, axis=0)

    # Compute the EI in the high frequency signal
    ei = compute_ei_index(target_he, base_he, fs=fs)
    return ei