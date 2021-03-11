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
inputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_processed/eeg_filtered/sub-RID0440/sub-RID0440_HUP172_phaseII_402651841658_402704260829_EEG_filtered.pickle'

#ictal
inputfile = '/Users/andyrevell/mount/USERS/arevell/papers/paper005/data_processed/eeg_filtered/sub-RID0440/sub-RID0440_HUP172_phaseII_402704260829_402756680000_EEG_filtered.pickle'

~~~~~~~
"""
from scipy import signal
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open(inputfile, 'rb') as f: data, fs = pickle.load(f)
data_array = np.array(data)

i = 122
data_tmp = data_array[:,i]
# Define window length (4 seconds)
win = 4 * fs
freqs, psd = signal.welch(data_tmp, fs, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, 2000 * 1.1])
plt.title("Welch's periodogram {0}".format(data.columns[i]))
plt.xlim([5, freqs.max()/10 ])
sns.despine()
plt.show()
plt.close()