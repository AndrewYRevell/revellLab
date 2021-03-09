""""
2020.09.17 Python 3.8.5
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:
    To get iEEG data from iEEG.org. Note, you must download iEEG python package from GitHub - instructions are below
    1. Gets time series data and sampling frequency information. Specified electrodes are removed.
    2. Saves as a pickle format\
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input
    username: your iEEG.org username
    password: your iEEG.org password
    iEEG_filename: The file name on iEEG.org you want to download from
    startUsec: the start time in the iEEG_filename. In microseconds
    stopUsec: the stop time in the iEEG_filename. In microseconds.
        iEEG.org needs a duration input: this is calculated by stopUsec - startUsec
    ignoreElectrodes: the electrode/channel names you want to exclude. EXACT MATCH on iEEG.org. Caution: some may be LA08 or LA8
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

username = 'username'
password = 'password'
iEEG_filename='HUP138_phaseII'
startUsec = 248432340000
stopUsec = 248525740000
ignoreElectrodes = ['EKG1', 'EKG2', 'CZ', 'C3', 'C4', 'F3', 'F7', 'FZ', 'F4', 'F8', 'LF04', 'RC03', 'RE07', 'RC05', 'RF01', 'RF03', 'RB07', 'RG03', 'RF11', 'RF12']
outputfile = '/home/arevell/papers/paper002/data/raw/eeg/sub-RID0278/sub-RID0278_HUP138_phaseII_248432340000_248525740000_EEG.pickle'
get_iEEG_data(username, password, iEEG_filename, startUsec, stopUsec, removed_channels, outputfile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run from command line:
python3.6 -c 'import get_iEEG_data; get_iEEG_data.get_iEEG_data("arevell", "password", "HUP138_phaseII", 248432340000, 248525740000, ["EKG1", "EKG2", "CZ", "C3", "C4", "F3", "F7", "FZ", "F4", "F8", "LF04", "RC03", "RE07", "RC05", "RF01", "RF03", "RB07", "RG03", "RF11", "RF12"], "/gdrive/public/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_D01_248432340000_248525740000_EEG.pickle")'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#How to extract pickled files
with open(outputfile, 'rb') as f: data, fs = pickle.load(f)
"""

from ieeg.auth import Session
import pandas as pd
import numpy as np

def get_iEEG_data(username, password, fname_iEEG, startUsec, stopUsec, ignoreElectrodes, channels = "all"):
    print("\nGetting data from iEEG.org:")
    print(f"fname_iEEG: {fname_iEEG}")
    print(f"startUsec: {startUsec}")
    print(f"stopUsec: {stopUsec}")
    print(f"channels: {channels}")
    print(f"ignoreElectrodes: {ignoreElectrodes}")
    startUsec = int(startUsec)
    stopUsec = int(stopUsec)
    duration = stopUsec - startUsec
    s = Session(username, password)
    ds = s.open_dataset(fname_iEEG)
    if channels == "all":
        channels_ind = ds.get_channel_indices(ds.ch_labels)
        channel_labels = ds.ch_labels
    else: #get channel number from channel label
        channels_ind = ds.get_channel_indices(channels)
        channel_labels = channels
        
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate #get sample rate
    
    #if duration is greater than ~10 minutes, then break up the API request to iEEG.org. 
    #The server blocks large requests, so the below code breaks up the request and 
    #concatenates the data
    server_limit_minutes = 10
    if duration < server_limit_minutes*60*1e6:
        for c in range(len(channels_ind)):
            if c == 0: #initialize
                data = ds.get_data(startUsec, duration,[ channels_ind[c]])
            else:
                data = np.concatenate([data, ds.get_data(startUsec, duration, [channels_ind[c]])], axis=1)
        #data = ds.get_data(startUsec, duration, channels_ind)
    if duration >= server_limit_minutes*60*1e6:
        break_times = np.ceil(np.linspace(startUsec, stopUsec, num=int(np.ceil(duration/(server_limit_minutes*60*1e6))+1), endpoint=True))
        break_data = np.zeros(shape = (int(np.ceil(duration/1e6*fs)), len(channels_ind)))#initialize
        print("breaking up data request from server because length is too long")
        for i in range(len(break_times)-1):
            print("{0}/{1}".format(i+1, len(break_times)-1))
            break_data[range(int( np.ceil((break_times[i]-break_times[0])/1e6*fs) ), int(  np.ceil((break_times[i+1]- break_times[0])/1e6*fs) )  ),:] = ds.get_data(break_times[i], break_times[i+1]-break_times[i], channels_ind)
        data = break_data
        
        
    index = np.round(np.linspace(startUsec, stopUsec, num = len(data))).astype(int)
    df = pd.DataFrame(data, columns=channel_labels, index = index)
    df = pd.DataFrame.drop(df, ignoreElectrodes, axis=1, errors='ignore') #errors = ignore means only drop electrodes if actually contained within
    return df, fs
    #print("Saving to: {0}".format(outputfile))
    #with open(outputfile, 'wb') as f: pickle.dump([df, fs], f)
    print("...done\n")


def get_iEEG_annotations(username, password, fname_iEEG, annotationLayerName):
    print("\nGetting data from iEEG.org:")
    print(f"fname_iEEG: {fname_iEEG}")
    print(f"Annotation Layer: {annotationLayerName}")

    s = Session(username, password)
    ds = s.open_dataset(fname_iEEG)

    annotations = pd.DataFrame(columns=(["file", "electrode", "start", "stop"]))
    
    if annotationLayerName in ds.get_annotation_layers(): #if annotations exists, get them
        annotationsLayer = ds.get_annotations(annotationLayerName)
    

        for j in range(len(annotationsLayer)):
            start = annotationsLayer[j].start_time_offset_usec
            stop = annotationsLayer[j].end_time_offset_usec
            for k in range(len(annotationsLayer[j].annotated)):
                channel_label = annotationsLayer[j].annotated[k].channel_label
                annotations = annotations.append({'file': fname_iEEG, 'electrode':channel_label, 'start':start, 'stop':stop}, ignore_index=True)
        return annotations
    else:
        return print(f"Annotation layer does not exist: {annotationLayerName}")
    print("...done\n")






def get_fs(username, password, fname_iEEG):
    print("\nGetting sampling frequency from iEEG.org:")
    print(f"fname_iEEG: {fname_iEEG}")
    s = Session(username, password)
    ds = s.open_dataset(fname_iEEG)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate #get sample rate
    return fs



def get_natus(username, password, fname_iEEG = "HUP172_phaseII", annotationLayerName = "Imported Natus ENT annotations"):
    print("\nFinding seizures")
    print(f"fname_iEEG: {fname_iEEG}")
    
    s = Session(username, password)
    ds = s.open_dataset(fname_iEEG)
   
    annotation_layers = np.array(list(ds.get_annotation_layers()))
    if not any(annotationLayerName == annotation_layers): #if there are no annotation layer names matchingt the user input, then list the layer names
        raise Exception(f"\n{annotationLayerName} does not match any layer names.\n\nThe existing annotation layer names are:\n\n{annotation_layers}")
    annotationsLayer = ds.get_annotations(annotationLayerName)
        
    annotations = pd.DataFrame(columns=(["file", "annotationLayer", "description", "start", "stop"]))
    annotationsSeizure = pd.DataFrame(columns=(["file", "annotationLayer", "description", "start", "stop"]))
    annotationsUEOEEC = pd.DataFrame(columns=(["file", "annotationLayer", "description", "start", "stop"]))
    
    for j in range(len(annotationsLayer)):
            start = annotationsLayer[j].start_time_offset_usec
            stop = annotationsLayer[j].end_time_offset_usec
            description = annotationsLayer[j].description
            
            annotations = annotations.append({'file': fname_iEEG, "annotationLayer":annotationLayerName,  'description':description, 'start':start, 'stop':stop}, ignore_index=True)
            
            if any(["seizure" in description.lower(),  "sz" in description.lower() ]) :
                annotationsSeizure = annotationsSeizure.append({'file': fname_iEEG, "annotationLayer":annotationLayerName,  'description':description, 'start':start, 'stop':stop}, ignore_index=True)
                
            if any(["ueo" in description.lower(),  "eec" in description.lower() ]) :
                annotationsUEOEEC = annotationsUEOEEC.append({'file': fname_iEEG, "annotationLayer":annotationLayerName,  'description':description, 'start':start, 'stop':stop}, ignore_index=True)
                
    return annotations, annotationsSeizure, annotationsUEOEEC

    
    
    









