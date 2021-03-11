"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    #preictal
    inputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_raw/eeg/sub-RID0440/sub-RID0440_HUP172_phaseII_402651841658_402704260829_EEG.pickle'
    outputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_processed/eeg_filtered/sub-RID0440/sub-RID0440_HUP172_phaseII_402651841658_402704260829_EEG_filtered.pickle'

    #ictal
    inputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_raw/eeg/sub-RID0440/sub-RID0440_HUP172_phaseII_402704260829_402756680000_EEG.pickle'
    outputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_processed/eeg_filtered/sub-RID0440/sub-RID0440_HUP172_phaseII_402704260829_402756680000_EEG_filtered.pickle'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pickle


#def removeElectrodes():
    


def filter_eeg_data(inputfile, outputfile):
    print("opening {0} ".format(inputfile))

    with open(inputfile, 'rb') as f:
        data, fs = pickle.load(f)
    data_array = np.array(data)
    low = 0.16
    high = 127
    print("Filtering Data between {0} and {1} Hz".format(low, high))
    fc = np.array([low, high])  # Cut-off frequency of the filter
    w = fc / np.array([(fs / 2), (fs / 2)])  # Normalize the frequency
    b, a = signal.butter(4, w, 'bandpass')
    filtered = np.zeros(data_array.shape)
    for i in range(data_array.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data_array[:, i])
    filtered = filtered + (data_array[0] - filtered[0])  # correcting offset created by filtfilt
    # output2 = output + (signala.mean() - output.mean()   )
    print("Filtering Data done")
    print("Notch Filtering Between at 60 Hz ")
    f0 = 60  # Cut-off notch filter
    Q = 30
    b, a = signal.iirnotch(f0, Q, fs)
    notched = np.zeros(data_array.shape)
    for i in range(data_array.shape[1]):
        notched[:, i] = signal.filtfilt(b, a, filtered[:, i])
    print("Notch Filtering Done")
    notched_df = pd.DataFrame(notched, columns=data.columns)
    # save file
    print("Saving file to {0}\n\n".format(outputfile))
    with open(outputfile, 'wb') as f:
        pickle.dump([notched_df, fs], f)


"""
#plotting
elec = 2
t2=np.arange(0,512,1)
plt.plot(t2, data_array[t2,elec], label=data.columns[elec])
plt.plot(t2, filtered[t2,elec], label='filtered')
plt.plot(t2, notched[t2,elec], label='notched')
plt.legend()
plt.show()
plt.close()


"""




