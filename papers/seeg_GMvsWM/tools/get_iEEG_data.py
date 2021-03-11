""""
2020.04.06. Python 3.7
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
To get iEEG data from iEEG.org. Note, you must download iEEG python package from GitHub - instructions are below
1. Gets time series data and sampling frequency information. Specified electrodes are removed.
2. Saves as a pickle format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input
    username: your iEEG.org username
    password: your iEEG.org password
    iEEG_filename: The file name on iEEG.org you want to download from
    start_time_usec: the start time in the iEEG_filename. In microseconds
    stop_time_usec: the stop time in the iEEG_filename. In microseconds.
        iEEG.org needs a duration input: this is calculated by stop_time_usec - start_time_usec
    ignore_electrodes: the electrode/channel names you want to exclude. EXACT MATCH on iEEG.org. Caution: some may be LA08 or LA8
    outputfile: the path and filename you want to save.
        PLEASE INCLUDE EXTENSION .pickle.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    Saves file outputfile as a pickel. For more info on pickeling, see https://docs.python.org/3/library/pickle.html
    Briefly: it is a way to save + compress data. it is useful for saving lists, as in a list of time series data and sampling frequency together along with channel names

    List index 0: Pandas dataframe. T x C (rows x columns). T is time. C is channels.
    List index 1: float. Sampling frequency. Single number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example usage:

username = 'arevell'
password = 'password'
iEEG_filename='HUP138_phaseII'
start_time_usec = 248432340000
stop_time_usec = 248525740000
removed_channels = ['EKG1', 'EKG2', 'CZ', 'C3', 'C4', 'F3', 'F7', 'FZ', 'F4', 'F8', 'LF04', 'RC03', 'RE07', 'RC05', 'RF01', 'RF03', 'RB07', 'RG03', 'RF11', 'RF12']
outputfile = '/Users/andyrevell/mount/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_248432340000_248525740000_EEG.pickle'
get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, outputfile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run from command line:
python3.6 -c 'import get_iEEG_data; get_iEEG_data.get_iEEG_data("arevell", "password", "HUP138_phaseII", 248432340000, 248525740000, ["EKG1", "EKG2", "CZ", "C3", "C4", "F3", "F7", "FZ", "F4", "F8", "LF04", "RC03", "RE07", "RC05", "RF01", "RF03", "RB07", "RG03", "RF11", "RF12"], "/gdrive/public/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_D01_248432340000_248525740000_EEG.pickle")'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#How to get back pickled files
with open(outputfile, 'rb') as f: data, fs = pickle.load(f)
"""
from ieeg.auth import Session
import pandas as pd
import pickle

def get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, ignore_electrodes, outputfile):
    print("\n\nGetting data from iEEG.org:")
    print("iEEG_filename: {0}".format(iEEG_filename))
    print("start_time_usec: {0}".format(start_time_usec))
    print("stop_time_usec: {0}".format(stop_time_usec))
    print("ignore_electrodes: {0}".format(ignore_electrodes))
    print("Saving to: {0}".format(outputfile))
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    s = Session(username, password)
    ds = s.open_dataset(iEEG_filename)
    channels = list(range(len(ds.ch_labels)))
    data = ds.get_data(start_time_usec, duration, channels)
    df = pd.DataFrame(data, columns=ds.ch_labels)
    df = pd.DataFrame.drop(df, ignore_electrodes, axis=1)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate #get sample rate
    with open(outputfile, 'wb') as f: pickle.dump([df, fs], f)
    print("...done\n")


""""
Download and install iEEG python package - ieegpy
GitHub repository: https://github.com/ieeg-portal/ieegpy

If you downloaded this code from https://github.com/andyrevell/paper001.git then skip to step 2
1. Download/clone ieepy. 
    git clone https://github.com/ieeg-portal/ieegpy.git
2. Change directory to the GitHub repo
3. Install libraries to your python Path. If you are using a virtual environment (ex. conda), make sure you are in it
    a. Run:
        python setup.py build
    b. Run: 
        python setup.py install
              
"""