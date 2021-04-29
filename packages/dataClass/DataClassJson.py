#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:50:54 2021

@author: arevell
"""

import sys
import copy
import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from os.path import join
from sklearn.preprocessing import OneHotEncoder

#import custom
path = join("/media","arevell","sharedSSD","linux","papers","paper002") #Parent directory of project
from revellLab.packages.eeg.echobase import echobase
from revellLab.packages.seizureSpread import echomodel
from revellLab.packages.eeg.ieegOrg import downloadiEEGorg


#%%


@dataclass
class DataClassJson:
    jsonFile: dict = "unknown"
    
    def get_iEEGData(self, sub, eventsKey, idKey, username, password,  BIDS = None, dataset= None, session = None, startUsec = None, stopUsec= None, startKey = "EEC", IGNORE_ELECTRODES = True, channels = "all", load = True):
        fname_iEEG = self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey]["FILE"]
        if IGNORE_ELECTRODES == True: #if you want to ignore electrodes, then set to True
            ignoreElectrodes =  self.jsonFile["SUBJECTS"][sub]["IGNORE_ELECTRODES"]
        else: #else if you want to get all electrodes, set to False
            ignoreElectrodes = []
        if startUsec is None:
            startUsec = int(float(self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey][startKey])*1e6)
        if stopUsec is None:
            stopUsec = int(float(self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey]["Stop"])*1e6)
        if not BIDS == None:
            fpath = self.getBIDSdirectoryToSaveiEEG(BIDS, dataset, sub, session )
        else: fpath = None
        
        if not fpath == None:
            echobase.check_path(fpath)
            fname = os.path.join(fpath, f"sub-{sub}_ses-{session}_task-{eventsKey}_acq-{startUsec}to{stopUsec}_ieeg.eeg")
            fnameMetadata = os.path.join(fpath, f"sub-{sub}_ses-{session}_task-{eventsKey}_acq-{startUsec}to{stopUsec}_ieeg.json")
            fnameEvents = os.path.join(fpath, f"sub-{sub}_ses-{session}_task-{eventsKey}_acq-{startUsec}to{stopUsec}_events.tsv")
            
            if os.path.exists(fname):
                print(f"\nFile exist {fname}")
                if load:
                    print("Loading")
                    df = pd.read_csv(fname, index_col=0)
                    with open(fnameMetadata) as f: metadata = json.load(f)
                    fs = metadata["SamplingFrequency"]
            else:
                print(f"\nFile does not exist. Saving to\n{fname}")
                df, fs = downloadiEEGorg.get_iEEG_data(username, password, fname_iEEG, startUsec, stopUsec, channels, ignoreElectrodes = [])
                self.saveEEG(fname, fnameMetadata, fnameEvents, df, fs, eventsKey, fname_iEEG, startUsec, stopUsec, eventsSave = False )
        else: 
            print("\nNo file path given. Not saving data. Downloading...")
            df, fs = downloadiEEGorg.get_iEEG_data(username, password, fname_iEEG, startUsec, stopUsec, channels, ignoreElectrodes = [])
        if load:
            print(f"ignored electrodes: {ignoreElectrodes}")
            df = pd.DataFrame.drop(df, ignoreElectrodes, axis=1, errors='ignore') 
            return df, fs
        
    
    def get_precitalIctalPostictal(self, sub, eventsKey, idKey, username, password, BIDS = None, dataset= None, session = None, secondsBefore = 30, secondsAfter = 30, startKey = "EEC", IGNORE_ELECTRODES = True, channels = "all", load = True):
        fname_iEEG = self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey]["FILE"]
        if IGNORE_ELECTRODES: 
            ignoreElectrodes =  self.jsonFile["SUBJECTS"][sub]["IGNORE_ELECTRODES"]
        else:
            ignoreElectrodes = []
        #calculate exactly the start/stop times to pull
        ictalStartUsec = int(float(self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey][startKey])*1e6)
        precitalStartUsec = int(ictalStartUsec - secondsBefore*1e6)
        ictalStopUsec = int(float(self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey]["Stop"])*1e6)
        postictalStopUsec = int(ictalStopUsec + secondsAfter*1e6)
        startUsec = precitalStartUsec
        stopUsec = postictalStopUsec
        duration = (ictalStopUsec - ictalStartUsec)/1e6
        
        if not BIDS == None:
            fpath = self.getBIDSdirectoryToSaveiEEG(BIDS, dataset, sub, session )
        else: fpath = None
        
        if not fpath == None:
            echobase.check_path(fpath)
            fname = os.path.join(fpath, f"sub-{sub}_ses-{session}_task-{eventsKey}_acq-{startUsec}to{stopUsec}_ieeg.eeg")
            fnameMetadata = os.path.join(fpath, f"sub-{sub}_ses-{session}_task-{eventsKey}_acq-{startUsec}to{stopUsec}_ieeg.json")
            fnameEvents = os.path.join(fpath, f"sub-{sub}_ses-{session}_task-{eventsKey}_acq-{startUsec}to{stopUsec}_events.tsv")
            
            if os.path.exists(fname):
                print(f"\nFile exist {fname}")
                if load:
                    print("Loading")
                    df = pd.read_csv(fname, index_col=0)
                    with open(fnameMetadata) as f: metadata = json.load(f)
                    fs = metadata["SamplingFrequency"]
            else:
                print(f"\nFile does not exist. Saving to\n{fname}")
                df, fs = downloadiEEGorg.get_iEEG_data(username, password, fname_iEEG, startUsec, stopUsec , channels, ignoreElectrodes = [])
                self.saveEEG(fname, fnameMetadata, fnameEvents, df, fs, eventsKey, fname_iEEG, startUsec, stopUsec, secondsBefore, duration, ictalStartUsec, ictalStopUsec , eventsSave = True)
        else: 
            print("\nNo file path given. Downloading data")
            df, fs = downloadiEEGorg.get_iEEG_data(username, password, fname_iEEG, startUsec, stopUsec, channels, ignoreElectrodes = [])
        if load:
            print(f"ignored electrodes: {ignoreElectrodes}")
            ictalStartIndex  = int(secondsBefore*fs)
            ictalStopIndex = int(secondsBefore*fs + (ictalStopUsec - ictalStartUsec)/1e6*fs)
            df = pd.DataFrame.drop(df, ignoreElectrodes, axis=1, errors='ignore') 
            return df, fs, ictalStartIndex, ictalStopIndex
        

    def getBIDSdirectoryToSaveiEEG(self, BIDS, dataset, sub, session ):
        echobase.check_path(join(BIDS, dataset))
        directory = join(BIDS, dataset,  f"sub-{sub}", f"ses-{session}", "ieeg" )
        echobase.makePath(directory)    
        return directory

        
    def saveEEG(self, fname, fnameMetadata, fnameEvents, df, fs, eventsKey, fname_iEEG, startUsec, stopUsec, secondsBefore = None, duration = None, ictalStartUsec = None, ictalStopUsec  = None , eventsSave = False):
        df.to_csv(fname, index=True, header=True, sep=',')
        #saving metadata
        metadata ={"TaskName": eventsKey, "TaskDescription": fname_iEEG,
                   "iEEGReference": "unknown",
                   "RecordingDuration": (stopUsec-startUsec)/1e6,
                   "SamplingFrequency": fs,
                   "PowerLineFrequency": 60, "SoftwareFilters": "n/a"}   
        with open(fnameMetadata, 'w', encoding='utf-8') as f: json.dump(metadata, f, ensure_ascii=False, indent=4)
        if eventsSave:
            ictalStartIndex  = int(secondsBefore*fs)
            ictalStopIndex = int(secondsBefore*fs + (ictalStopUsec - ictalStartUsec)/1e6*fs)
            events = pd.DataFrame([dict(onset = secondsBefore, duration = duration, ictalStartUsecEEC = ictalStartUsec, ictalStopUsec = ictalStopUsec, ictalStartIndexEEC = ictalStartIndex,  ictalStopIndex = ictalStopIndex )])
            fnameEvents = events.to_csv(fnameEvents, index=False, header=True, sep='\t')

    def downsample(self, data, fs, fsds):
        #fsds = fs_downsample: the frequency to downsample to
        downsampleFactor = int(fs/fsds) #downsample to specified frequency
        data_downsampled = signal.decimate(data, downsampleFactor, axis=0)#downsample data
        return data_downsampled
    
    def get_annotations(self, sub, eventsKey, idKey, annotationLayerName, username, password):
        annotations = downloadiEEGorg.get_iEEG_annotations(username, password, self.jsonFile["SUBJECTS"][sub]["Events"][eventsKey][idKey]["FILE"], annotationLayerName)    
        return annotations
    
    #extract the annotated segments in data 
    def get_annotations_iEEG(self, annotations, data, channels, index):
        dataAnnotation = []
        dataAnnotationChannels = []
        for c in range(len(annotations)):
            annElec = annotations["electrode"][c]
            #convert name to standard channel name:
            annElec = echobase.channel2std(    np.array([annElec]  ).astype("object")   ) 
            annStr = annotations["start"][c]
            annStp = annotations["stop"][c]
            
            annIndex = np.where((index >=annStr) & (index <=annStp))
            if len(annIndex[0]) > 0:
                if any(channels == annElec): #if data contains electrode in annotations
                    col = np.where(channels == annElec)[0][0]
                    dataAnnotation.append(  data[annIndex, col ].T   )
                    dataAnnotationChannels.append( annElec  )
        return dataAnnotation, dataAnnotationChannels

    def get_associatedInterictal(self, sub, idKey):
        AssociatedInterictal = self.jsonFile["SUBJECTS"][sub]["Events"]["Ictal"][idKey]["AssociatedInterictal"]
        return AssociatedInterictal
    #%Extracting which patients actually have annotations
    def get_patientsWithSeizureChannelAnnotations(self):
        patientsWithAnnotations = pd.DataFrame(columns = ["subject", "idKey", "AssociatedInterictal"])
        subjects = list(self.jsonFile["SUBJECTS"].keys())
        for s in range(len(subjects)):
            idKeys = list(self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"].keys())
            for i in  range(len(idKeys)):
                if "SeizureChannelAnnotations" in self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]:
                    if self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]["SeizureChannelAnnotations"] == "yes":
                        AssociatedInterictal = self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]["AssociatedInterictal"]
                        patientsWithAnnotations = patientsWithAnnotations.append(dict(subject =  subjects[s], idKey = idKeys[i], AssociatedInterictal = AssociatedInterictal),ignore_index=True)
        return patientsWithAnnotations
    
    def get_patientsWithSeizuresAndInterictal(self):
        patientsWithseizures = pd.DataFrame(columns = ["subject", "idKey", "AssociatedInterictal", "EEC", "UEO", "stop"])
        subjects = list(self.jsonFile["SUBJECTS"].keys())
        for s in range(len(subjects)):
            idKeys = list(self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"].keys())
            for i in range(len(idKeys)):
                if "AssociatedInterictal" in self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]:
                    AssociatedInterictal = self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]["AssociatedInterictal"]
                    if not AssociatedInterictal == "missing":
                        EEC = float(self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]["EEC"])
                        UEO = float(self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]["UEO"])
                        stop = float(self.jsonFile["SUBJECTS"][subjects[s]]["Events"]["Ictal"][idKeys[i]]["Stop"])
                        patientsWithseizures = patientsWithseizures.append(dict(subject =  subjects[s], idKey = idKeys[i], AssociatedInterictal = AssociatedInterictal, EEC = EEC, UEO = UEO, stop = stop),  ignore_index=True)
        return patientsWithseizures
    
    def preprocessNormalizeDownsample(self, df, df_interictal, fs, fsds, montage = "bipolar", prewhiten = True):
        #% Preprocessing
        data, data_ref, _, data_filt, channels = echobase.preprocess(df, fs, fsds, montage=montage, prewhiten = prewhiten)
        dataII, _, _, dataII_filt, channels = echobase.preprocess(df_interictal, fs, fsds, montage=montage, prewhiten = prewhiten)
        #normalize
        dataII_scaler = echomodel.scaleData(dataII_filt, dataII_filt)
        data_scaler = echomodel.scaleData(data_filt, dataII_filt)
        #get annotated segments of preprocessed data
        #alter annotations to include time_step seconds beforehand
        #downsample
        dataII_scalerDS = self.downsample(dataII_scaler, fs, fsds)
        data_scalerDS = self.downsample(data_scaler, fs, fsds)
        return dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels
       

        
    def get_dataXY(self, sub, idKey, AssociatedInterictal, username, password, annotationLayerName, BIDS = None, dataset= None, session = None, fsds = 128, window = 10 , skipWindow = 0.25, secondsBefore = 60, secondsAfter = 60, montage = "bipolar", prewhiten = True):
        print("\nGetting ictal data")
        df, fs, _, _ = self.get_precitalIctalPostictal(sub, "Ictal", idKey, username, password, BIDS = BIDS, dataset= dataset, session = session, secondsBefore = secondsBefore, secondsAfter = secondsAfter)
        print("\nGetting interictal data")
        df_interictal, _ = self.get_iEEGData(sub, "Interictal", AssociatedInterictal, username, password, BIDS = BIDS, dataset= dataset, session = session, startKey = "Start")
        print(f"\nPreprocessing data\nMontage: {montage}\nPrewhiten: {prewhiten}")
        dataII_scaler, data_scaler, dataII_scalerDS, data_scalerDS, channels = self.preprocessNormalizeDownsample(df, df_interictal, fs, fsds, montage = montage, prewhiten = prewhiten)
        time_step, skip = int(window*fsds), int(skipWindow*fsds)
        annotations = self.get_annotations(sub, "Ictal", idKey, annotationLayerName, username, password) 
        print("\nAnnotations")
        #get annotated segments of preprocessed data
        #alter annotations to include time_step seconds beforehand
        annotations_altered = copy.deepcopy(annotations)
        annotations_altered["start"] = annotations_altered["start"] - int(window*1e6)
        index = np.array(df.index)
        dataAnnotation, dataAnnotationChannels = self.get_annotations_iEEG(annotations_altered, data_scaler, channels, index[range(len(data_scaler))])
        #downsample
        dataAnnotationDS = copy.deepcopy(dataAnnotation)
        for i in range(len(dataAnnotationDS)):
            dataAnnotationDS[i] = self.downsample(dataAnnotationDS[i], fs, fsds)
        #d.plot_eeg(dataAnnotationDS[10], fsds, nchan = 1, dpi = 300)    
        #d.plot_eeg(data_scalerDS, fsds, nchan = 1, dpi = 300)
        print("Constructing classes")
        #% Classes
        #generate classes. class 0 = not seizing (interictal). class 1 = seizing annotations
        CLASS0all = echomodel.overlapping_windows(dataII_scalerDS, time_step, skip)
        CLASS0all = CLASS0all.reshape( CLASS0all.shape[0] * CLASS0all.shape[2], CLASS0all.shape[1] , 1  )
        for i in range(len(dataAnnotationDS)):
            if i == 0:
                CLASS1 = echomodel.overlapping_windows(dataAnnotationDS[i], time_step, skip)
            else:
                CLASS1 = np.concatenate(  [CLASS1, echomodel.overlapping_windows(dataAnnotationDS[i], time_step, skip) ]   )
        #pick same number of class0 as class1
        CLASS0 = CLASS0all[np.random.choice(len(CLASS0all), size=len(CLASS1), replace=False),:]
        print("Preprocessing classes")
        #% Preproccess classes
        #generate X vectors
        X = np.concatenate( [ CLASS0, CLASS1 ])
        #generate class Y values
        Y = np.concatenate( [ np.repeat(0, len(CLASS1)), np.repeat(1, len(CLASS1)) ])
        Y = Y.reshape(Y.shape[0], 1)
        #One hot encoding
        ohe = OneHotEncoder(sparse=False)
        Y = ohe.fit_transform(Y)
        #Suffle
        shuffle = np.random.permutation(X.shape[0])
        X = X[shuffle,:,:]
        Y = Y[shuffle,:]
        return X, Y, data_scalerDS, dataII_scalerDS, dataAnnotationDS
    
    ##Plotting functions
    def plot_eeg(self, data, fs, startSec = None, stopSec = None, nchan = None, markers = [], aspect = 20, height = 0.3, hspace = -0.3, dpi = 300, lw=1, fill = False, savefig = False, pathFig = None):
        if stopSec == None:
            stopSec = len(data)/fs
        if startSec == None:
            startSec = 0
        if nchan == None:
            nchan = data.shape[1]
        df_wide = pd.DataFrame(data[   int(fs * startSec): int(fs * stopSec),  range(nchan)]    )
        df_long = pd.melt(df_wide, var_name = "channel", ignore_index = False)
        df_long["index"] = df_long.index
        if fill == True:
            pal = sns.cubehelix_palette(nchan, rot=-.25, light=.7)
        else:
            pal = sns.cubehelix_palette(nchan, rot=-.25, light=0)
        sns.set(rc={"figure.dpi":dpi})
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(df_long, row="channel", hue="channel", aspect=aspect, height=height, palette=pal)
        if fill == True:
            g.map(sns.lineplot,"index", "value", clip_on=False, color="w", lw=0.8)
            g.map(plt.fill_between,  "index", "value")
                
        else:
            g.map(sns.lineplot, "index", "value", clip_on=False, alpha=1, linewidth=lw)
        if len(markers) > 0:    
            axes = g.axes  
            if len(markers) > 1:
                for c in range(len(axes)):
                    axes[c][0].axvline(x=markers[c])
            else:
                for c in range(len(axes)):
                    axes[c][0].axvline(x=markers[0])
        g.fig.subplots_adjust(hspace=hspace)
        g.set_titles("")
        g.set(yticks=[])
        g.set(xticks=[])
        g.set_axis_labels("", "")
        g.despine(bottom=True, left=True)        
        
        if savefig:
            if pathFig == None:
                print("Must provide figure path and filename to save")
            else: plt.savefig(pathFig, transparent=True)

    
    
    

#%%
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
