import numpy as np




#montaging
MONTAGE_BIPOLAR = "bipolar"
MONTAGE_CAR = "car"

#Frequency
FREQUENCY_NAMES = ["Broadband", "delta", "theta", "alpha", "beta", "gammaLow", "gammaMid", "gammaHigh"]
FREQUENCY_DOWN_SAMPLE = 256 #sampleing frequency for down-sampled data

FC_TYPES = ["pearson", "coherence", "crossCorrelation"]

#states
STATE_NAMES = ["interictal", "preictal", "ictal", "postictal"]
STATE_NUMBER_TOTAL = len(STATE_NAMES)


#WM vs GM parameters
TISSUE_DEFINITION_DISTANCE = ["distance_to_GM_millimeters", 0, 3, np.arange(0,8 + 0.5, 0.5), 6] #(Name, GM definition < or =, WM definition > or =, sequence, prefference ind of sequence)
TISSUE_DEFINITION_PERCENT = ["percent_WM", 0.5, 0.9,  np.arange(0, 1 + 0.025, 0.025), 36]  #(Name, GM definition < or =, WM definition > or =, sequence, prefference ind of sequence)



#Imaging parameters
SESSION_IMPLANT = "implant01"
SESSION_RESEARCH3T = "research3Tv[0-9][0-9]"
ACQUISITION_RESEARCH3T_T1_MPRAGE = "3D"
IEEG_SPACE = ["T00"]



OUTCOME_NAMES = ["good", "poor"]
TISSUE_TYPE_NAMES = ["Full Network", "GM-only", "WM-only", "GM-WM"]
TISSUE_TYPE_NAMES2 = ["Full Network", "GM", "WM", "GM-WM"]
