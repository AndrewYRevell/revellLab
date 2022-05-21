

import json
import copy
import pandas as pd
import numpy as np
from os.path import join, splitext, basename

from revellLab.paths import constants_paths as paths

#%%
#% 02 Paths and files

metadataDir =  paths.METADATA
fnameJSON = join(metadataDir, "iEEGdataRevell.json")
fnameJSON_seizure_severity = join(metadataDir, "iEEGdataRevell_seizure_severity.json")


#%%

metadata_from_akash = join(metadataDir, "metadata_for_seizure_spread.csv")
seizure_akash = pd.read_csv(metadata_from_akash)

RID_HUP = join(metadataDir, "RID_HUP.csv")
RID_HUP = pd.read_csv(RID_HUP)

with open(fnameJSON) as f: jsonFile = json.load(f)



#%% loop thru patients to replace these times


template_ictal = {'FILE': 'NA',
   'SeizureType': 'missing',
   'EMU_Report_Event_Number': 'missing',
   'EEC': np.nan,
   'UEO': np.nan,
   'Stop': np.nan,
   'OnsetElectrodes': ['missing'],
   'SeizurePhenotype': 'missing',
   'AssociatedInterictal': '1',
   'SeizureChannelAnnotations': 'no'}

template_interictal =  {'FILE': 'NA',
  'Start': np.nan,
  'Stop': np.nan,
  'AssociatedIctal': '1'}



template_id = {'RID': np.nan,
 'HUP': np.nan,
 '3T_ID': 'NA',
 '7T_ID': 'NA',
 'Outcome 6 month': 'missing',
 'Outcome 12 month': 'missing',
 'Outcome 24 month': 'missing',
 'Sex': 'missing',
 'AgeOnset': 'missing',
 'AgeSurgery': 'missing',
 'Location': 'missing',
 'Lesion Status': 'missing',
 'Previous Surgery': 'missing',
 'Pathology': 'missing',
 'Resection Type': 'missing',
 'Events': {'Ictal': {},
            'Interictal': {}
            },
 'ELECTRODE_COORDINATES': 'missing',
 'ELECTRODE_LABELS': 'missing',
 'MANUAL_RESECTED_ELECTRODES': ['missing'],
 'IGNORE_ELECTRODES': ["C3",
                "C4",
                "CZ",
                "EKG",
                "EKG1",
                "EKG2",
                "ECG",
                "ECG1",
                "ECG2",
                "LOC",
                "ROC",
                "FZ",
                "FP1",
                "FZ",
                "O1",
                "P3",
                "F3",
                "F7",
                "T3",
                "T5",
                "EEG EKG 01-Ref",
                "EEG EKG 02-Ref",
                "EEG EKG-Ref",
                "EEG EKG1-Ref",
                "EEG EKG2-Ref",
                "EEG Rate-Ref",
                "EEG RR-Ref",
                "LDA"
                ]
}

#%%

seizure_severity_dict = dict(SUBJECTS = {})
previous_hup = 0
event_num = 1

#%%

for i in range(len(seizure_akash)):
    print(f"\r{i}:   {np.round((i+1)/(len(seizure_akash))*100,1 )}%     ", end = "\r")


    hup_num = int( seizure_akash.Patient.iloc[i][3:])
    pd.to_numeric(RID_HUP.hupsubjno)
    loc = list(RID_HUP.hupsubjno).index(hup_num)
    rid_num = int(RID_HUP.record_id[loc])
    RID = f"RID{rid_num:04d}"
    
    
    if previous_hup != hup_num:
        event_num = 1
        event_num_str = str(event_num)
        #get ignore electrode is present
        RID_keys =  list(jsonFile["SUBJECTS"].keys() )
        hup_num_all = [jsonFile["SUBJECTS"][x]["HUP"]  for  x   in  RID_keys]
            
        
        new_id = copy.deepcopy(template_id)
        
        new_id["RID"] = rid_num
        new_id["HUP"] = hup_num
        
        if hup_num in hup_num_all:
            loc = hup_num_all.index(hup_num)
            RID_key = RID_keys[loc]
            if jsonFile["SUBJECTS"][RID_key]["IGNORE_ELECTRODES"] != ["missing"]:
                IGNORE_ELECTRODES = jsonFile["SUBJECTS"][RID_key]["IGNORE_ELECTRODES"]
                new_id["IGNORE_ELECTRODES"] = copy.deepcopy(IGNORE_ELECTRODES)

        
        
        new_id_event = copy.deepcopy(template_ictal)
        new_id_interictal  = copy.deepcopy(template_interictal)
        
        new_id_event["FILE"] = seizure_akash.iloc[i]["iEEG Filename"]
        new_id_event["EEC"] = seizure_akash.iloc[i]["Seizure EEC"]
        new_id_event["UEO"] = seizure_akash.iloc[i]["Seizure UEO"]
        new_id_event["Stop"] = seizure_akash.iloc[i]["Seizure end"]
        new_id_event["AssociatedInterictal"] = '1'
        
        new_id_interictal["FILE"] = seizure_akash.iloc[i]["iEEG Filename"]
        new_id_interictal["Start"] = float(seizure_akash.iloc[i]["Interictal clip start"])
        new_id_interictal["Stop"] = float(seizure_akash.iloc[i]["Interictal clip start"] + 180)
        new_id_interictal["AssociatedIctal"] ='1'
        
        new_id["Events"]["Ictal"][event_num_str] = new_id_event
        new_id["Events"]["Interictal"][event_num_str] = new_id_interictal
        
        seizure_severity_dict["SUBJECTS"][RID] = new_id
        previous_hup = hup_num
    else:   
        event_num = event_num + 1
        event_num_str = str(event_num)
        new_id_event = copy.deepcopy(template_ictal)
        new_id_interictal  = copy.deepcopy(template_interictal)
        
        new_id_event["FILE"] = seizure_akash.iloc[i]["iEEG Filename"]
        new_id_event["EEC"] = seizure_akash.iloc[i]["Seizure EEC"]
        new_id_event["UEO"] = seizure_akash.iloc[i]["Seizure UEO"]
        new_id_event["Stop"] = seizure_akash.iloc[i]["Seizure end"]
        new_id_event["AssociatedInterictal"] = event_num_str
        
        new_id_interictal["FILE"] = seizure_akash.iloc[i]["iEEG Filename"]
        new_id_interictal["Start"] = float(seizure_akash.iloc[i]["Interictal clip start"])
        new_id_interictal["Stop"] = float(seizure_akash.iloc[i]["Interictal clip start"] + 180)
        new_id_interictal["AssociatedIctal"] = event_num_str
        
        new_id["Events"]["Ictal"][event_num_str] = new_id_event
        new_id["Events"]["Interictal"][event_num_str] = new_id_interictal

with open(fnameJSON_seizure_severity, 'w') as f: json.dump(seizure_severity_dict, f,  sort_keys=False, indent=4)

#%%