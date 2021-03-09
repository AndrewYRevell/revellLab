"""

"""
import os
import inspect
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder

# %%
""""
Note 2020.05.06
To install mtspec:
See https://krischer.github.io/mtspec/ for more documentation
1. Need to have gfortran installed on computer
2. It is different for Linux and Mac
Linux:
#apt-get install gfortran
#pip install mtspec
#or
# conda config --add channels conda-forge
# conda install mtspec
Mac OS:
Need homebrew, then do:
#brew install gcc
#brew cask install gfortran
#pip install mtspec
"""

"""
A. Main
"""
def scaleData(data, dataBASE, scaleType = "RobustScaler"):
    if scaleType == "RobustScaler":
        scaler = RobustScaler()
    if scaleType == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scaleType == "StandardScaler":
        scaler = StandardScaler()
    if scaleType == "MaxAbsScaler":
        scaler = MaxAbsScaler()
    scaler.fit(dataBASE)
    data_scaled = scaler.transform(data)
    return data_scaled


def overlapping_windows(arr, time_step, skip):
	# flatten data
	X = list()
	start = 0
	# step over the entire history one time step at a time
	for ii in range(len(arr)):
		# define the end of the input sequence
		end = start + time_step
		# ensure we have enough data for this instance
		if end <= len(arr):
			X.append(arr[start:end,:])
		# move along one time step
		start = start + int(skip)
	return np.array(X)

def splitDataframeTrainTest(dataframe, colname, trainSize = 0.66):
        uniqueIDs = np.unique(dataframe[colname])
        trainNum= int(np.round(trainSize * len(uniqueIDs)))
        subIndex = np.random.permutation(len(uniqueIDs))
        trainSubjects = uniqueIDs[subIndex[0:trainNum]]
        testSubjects = uniqueIDs[subIndex[trainNum:len(subIndex)]]
        train = dataframe[dataframe[colname].isin(trainSubjects)]
        test = dataframe[dataframe[colname].isin(testSubjects)]
        return train, test

def build_model_wavenet(learn_rate, beta_1, beta_2, input_shape, dropout):
	#CNN model
    rate = 2
    #rate_exp_add = 2
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=beta_2)
    model = Sequential()
    
    model.add(Conv1D(filters=128, kernel_size=128,data_format="channels_last", activation='relu', dilation_rate = 2**rate, input_shape=input_shape,  padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    rate = rate #+ rate_exp_add * 2
    model.add(Conv1D(filters=64, kernel_size=128, activation='relu', dilation_rate = 2**rate,padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    rate = rate #+ rate_exp_add * 2
    model.add(Conv1D(filters=8, kernel_size=128, activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    
    model.add(Conv1D(filters=8, kernel_size=128, activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    #model.add(GlobalAveragePooling1D())
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())
    return model

def build_model_1dCNN(learn_rate, beta_1, beta_2, input_shape, dropout):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=beta_2)
    model = Sequential()
    
    model.add(Conv1D(filters=8, kernel_size=256, strides = 2, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=6, kernel_size=128, strides = 2, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(filters=3, kernel_size=64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters=4, kernel_size=8, strides = 2, activation='relu', ))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())
    return model

def build_model_LSTM(learn_rate, beta_1, beta_2, input_shape, dropout):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=beta_2)
    model = Sequential()
    model.add(LSTM(4, activation='relu', input_shape=input_shape))

    model.add(Dropout(dropout))

    model.add(Dense(128, activation='relu'))

    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())
    return model
        


def modelTrain(X_train, y_train, X_test, y_test, callbacks_list, modelName = "wavenet", learn_rate=0.01, beta_1=0.9, beta_2=0.999, input_shape=None, dropout=0.3, training_epochs=3, batch_size=2**11, validation_split = 0.2, verbose = 1):
    if input_shape == None:
        input_shape = (X_train.shape[1], X_train.shape[2])
    if modelName == "wavenet":
        model = build_model_wavenet(learn_rate, beta_1, beta_2, input_shape, dropout)
    if modelName == "1dCNN":
        model = build_model_1dCNN(learn_rate, beta_1, beta_2, input_shape, dropout)
    if modelName == "lstm":
        model = build_model_LSTM(learn_rate, beta_1, beta_2, input_shape, dropout)
    history = model.fit(X_train, y_train, epochs=training_epochs, verbose=verbose, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks_list)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    return accuracy, history.history['accuracy'][-1]


def modelPredict(fpath_model, X_test):
    model = load_model(fpath_model)
    print(model.summary())
    
    yPredictProbability =  model.predict(X_test, verbose=1)
    return yPredictProbability

def modelEvaluate(yPredictProbability, X_test, y_test, title = "Model Performance"):
    Y_predict = np.argmax(yPredictProbability, axis=-1)
    Y = np.argmax(y_test, axis=-1).reshape(y_test.shape[0], 1)
    
    #Sensitivity, specificit, PPV, NPC, and accuracy
    positives = Y[np.where(Y_predict == 1)] 
    positives_true = np.where(positives.flatten() == 1)[0]
    positives_false = np.where(positives.flatten()  == 0)[0]
    negatives = Y[np.where(Y_predict == 0)] 
    negatives_true = np.where(negatives.flatten() == 0)[0]
    negatives_false = np.where(negatives.flatten() == 1)[0]
    TP = len(positives_true)
    FP = len(positives_false)
    TN = len(negatives_true)
    FN = len(negatives_false)
    acc = (TP+TN )/(TP + FP +  TN + FN)#accuracy
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    PPV = TP/(TP + FP)
    if (TN+FN) > 0:
        NPV = TN/(TN + FN)
    else:
        NPV = 0
    
    #AUC
    fpr, tpr, threshold = metrics.roc_curve(Y, yPredictProbability[:,1] )
    roc_auc = metrics.auc(fpr, tpr)
    
    #precision-recall, and PR-AUC
    precision, recall, thresholds = metrics.precision_recall_curve(Y, yPredictProbability[:,1])
    f1 = metrics.f1_score(Y, Y_predict)
    
    pr_auc = metrics.auc(recall, precision)
    
    print(f"\nTP: {TP} \nFP: {FP} \nTN: {TN} \nFN: {FN}")
    print(f"\naccuracy: {acc} \nSensitivity: {sensitivity} \nSpecificity: {specificity} \nPPV: {PPV} \nNPV: {NPV} ")
    
    #plotting 
    fig, ax = plt.subplots(2,2, figsize = (6.75,6.75), dpi = 300)
    ax[0,0].text(x= 0.05, y = 0.97, ha='left', va = "top" ,
                 s = f"{title}\n\nTP: {TP} \nFP: {FP} \nTN: {TN} \nFN: {FN}\
                 \n\nAccuracy: {acc:0.3f} \nSensitivity: {sensitivity:0.3f} \
                 \nSpecificity: {specificity:0.3f} \nPPV: {PPV:0.3f} \nNPV: {NPV:0.3f} ")
    ax[0,0].set_xlim([0, 1])
    ax[0,0].set_ylim([0, 1])
    ax[0,0].set_xlabel('')
    ax[0,0].set_ylabel('')
    ax[0,0].get_xaxis().set_ticks([])
    ax[0,0].get_yaxis().set_ticks([])
    sns.despine(bottom=True, left=True, ax=ax[0,0])
    
    
    sns.histplot(data = yPredictProbability[:,1], ax = ax[0,1])
    ax[0,1].axes.axvline(x = 0.5, linestyle='--')
    ax[0,1].set_xlabel('Seizure Probability')
    ax[0,1].set_ylabel('Count')
    sns.despine(bottom=False, left=False, ax=ax[0,1])
    ax[0,1].text(x= 0.5, y = 0.00, s = "Negatives  \n\n", ha='right', va = "bottom") 
    ax[0,1].text(x= 0.5, y = 0.00, s = "  Positives\n\n", ha='left', va = "bottom") 
    
    sns.lineplot(fpr, tpr, ci = None, ax = ax[1,0], color = "darkorange" )
    sns.lineplot([0, 1], [0, 1], ci = None, ax = ax[1,0], color = "navy", linestyle='--' )
    ax[1,0].set_xlim([-0.05, 1.05])
    ax[1,0].set_ylim([-0.05, 1.05])
    ax[1,0].set_xlabel('FPR')
    ax[1,0].set_ylabel('TPR (Recall, Sensitivity)')    
    ax[1,0].text(x= 1, y = 0.05, s = f'ROC Curve\nAUC = {roc_auc:0.3f}', ha='right', va = "bottom")
    sns.despine(bottom=False, left=False, ax=ax[1,0])
    
    
    sns.lineplot(x = recall, y = precision, ax = ax[1,1], linewidth=1, ci=None,  color = "darkorange")
    sns.lineplot([0, 1], [0, 0], ci = None, ax = ax[1,1], color = "navy", linestyle='--' )
    ax[1,1].set_xlim([-0.05, 1.05])
    ax[1,1].set_ylim([-0.05, 1.05])
    ax[1,1].set_xlabel('Recall (TPR, Sensitivity)')
    ax[1,1].set_ylabel('Precision (PPV)')
    ax[1,1].text(x= 1, y = 0.05, s = f'PR Curve\nF1 score = {f1:0.3f} \nAUC = {pr_auc:0.3f}', ha='right', va = "bottom")
    sns.despine(bottom=False, left=False, ax=ax[1,1])


# %%
"""
C. Utilities:
"""

def check_path(path):
    '''
    Check if path exists
    Parameters
    ----------
        path: str
            Check if valid path
    '''
    if not os.path.exists(path):
        raise IOError('%s does not exists' % path)


def make_path(path):
    '''
    Make new path if path does not exist
    Parameters
    ----------
        path: str
            Make the specified path
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise IOError('Path: %s, already exists' % path)


def check_path_overwrite(path):
    '''
    Prevent overwriting existing path
    Parameters
    ----------
        path: str
            Check if path exists
    '''
    if os.path.exists(path):
        raise IOError('%s cannot be overwritten' % path)


def check_has_key(dictionary, key_ref):
    '''
    Check whether the dictionary has the specified key
    Parameters
    ----------
        dictionary: dict
            The dictionary to look through
        key_ref: str
            The key to look for
    '''
    if key_ref not in dictionary.keys():
        raise KeyError('%r should contain the %r key' % (dictionary, key_ref))


def check_dims(arr, nd):
    '''
    Check if numpy array has specific number of dimensions
    Parameters
    ----------
        arr: numpy.ndarray
            Input array for dimension checking
        nd: int
            Number of dimensions to check against
    '''
    if not arr.ndim == nd:
        raise Exception('%r has %r dimensions. Must have %r' % (arr, arr.ndim, nd))


def check_type(obj, typ):
    '''
    Check if obj is of correct type
    Parameters
    ----------
        obj: any
            Input object for type checking
        typ: type
            Reference object type (e.g. str, int)
    '''
    if not isinstance(obj, typ):
        raise TypeError('%r is %r. Must be %r' % (obj, type(obj), typ))


def check_function(obj):
    '''
    Check if obj is a function
    Parameters
    ----------
        obj: any
            Input object for type checking
    '''
    if not inspect.isfunction(obj):
        raise TypeError('%r must be a function.' % (obj))


# Progress bar function
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill="X", printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()





# %%
# visulaize
"""
vmin = -0.5; vmax = 0.9; title_size = 8
fig,axes = plt.subplots(4,2,figsize=(8,16), dpi = 300)
sns.heatmap(adj_xcorr, square=True, ax = axes[0][0], vmin = vmin, vmax = vmax); axes[0][0].set_title("X corr; tau: 0.25 ; elliptic", size=title_size)
sns.heatmap(np.abs(adj_xcorr), square=True, ax = axes[0][1], vmin = 0, vmax = vmax); axes[0][1].set_title("X corr Abs; tau: 0.25; elliptic", size=title_size)

sns.heatmap(adj_pear, square=True, ax = axes[1][0], vmin = vmin, vmax = vmax); axes[1][0].set_title("Pearson; elliptic", size=title_size)
sns.heatmap(adj_spear, square=True, ax = axes[1][1], vmin = vmin, vmax = vmax); axes[1][1].set_title("Spearman; elliptic", size=title_size)
sns.heatmap(adj_cohe_bb_m, square=True, ax = axes[2][0]); axes[2][0].set_title("Coherence: mt_spec; elliptic", size=title_size)
sns.heatmap(adj_cohe_bb, square=True, ax = axes[2][1]); axes[2][1].set_title("Coherence: Scipy; elliptic", size=title_size)

sns.heatmap(adj_MI, square=True, ax = axes[3][0]); axes[3][0].set_title("Mutual Information; elliptic", size=title_size)
fig,axes = plt.subplots(2,2,figsize=(8,8), dpi = 300)
sns.heatmap(adj_butter_xcorr, square=True, ax = axes[0][0], vmin = vmin, vmax = vmax)
sns.heatmap(np.abs(adj_butter_xcorr), square=True, ax = axes[0][1], vmin = vmin, vmax = vmax)
sns.heatmap(adj_butter_pear, square=True, ax = axes[1][0], vmin = vmin, vmax = vmax)
sns.heatmap(adj_butter_spear, square=True, ax = axes[1][1], vmin = vmin, vmax = vmax)
###########
###########
###########
###########
ch = 1
data_ch = data[:,ch]
data_ch_hat = data_hat[:,ch]

fig,axes = plt.subplots(1,2,figsize=(8,4), dpi = 300)
st = 0; sp = 15
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch[range(fs*st,fs*sp)], ax = axes[0] , linewidth=0.5 )
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch_hat[range(fs*st,fs*sp)] , ax = axes[1], linewidth=0.5 )
data_ch = data[:,ch]
data_ch_hat = data_butter[:,ch]

fig,axes = plt.subplots(1,2,figsize=(8,4), dpi = 300)
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch[range(fs*st,fs*sp)], ax = axes[0] , linewidth=0.5 )
sns.lineplot(x =  np.array(range(fs*st,fs*sp))/1e6*fs, y = data_ch_hat[range(fs*st,fs*sp)] , ax = axes[1], linewidth=0.5 )
###########
###########
###########
###########
fig,axes = plt.subplots(2,2,figsize=(8,8), dpi = 300)
sns.histplot(adj_xcorr_025[np.triu_indices( len(adj_xcorr_025), k = 1)], ax = axes[0][0])
sns.histplot(adj_pear[np.triu_indices( len(adj_pear), k = 1)], ax = axes[0][1])
sns.histplot(adj_spear[np.triu_indices( len(adj_spear), k = 1)], ax = axes[1][0])
fig,axes = plt.subplots(2,2,figsize=(8,8), dpi = 300)
sns.histplot(adj_butter_xcorr[np.triu_indices( len(adj_butter_xcorr), k = 1)], ax = axes[0][0])
sns.histplot(adj_butter_pear[np.triu_indices( len(adj_butter_pear), k = 1)], ax = axes[0][1])
sns.histplot(adj_butter_spear[np.triu_indices( len(adj_butter_spear), k = 1)], ax = axes[1][0])
###########
###########
###########
###########
n1=18; n2 = 37
d1= data_hat[:,n1]
d2= data_hat[:,n2]
print(f"\nx_xorr:   {np.round( adj_xcorr_025[n1,n2],2 )}"  ) 
print(f"Pearson:  {np.round( pearsonr(d1, d2)[0],2 )}; p-value: {np.round( pearsonr(d1, d2)[1],2 )}"  ) 
print(f"Spearman: {np.round( spearmanr(d1, d2)[0],2 )}; p-value: {np.round( spearmanr(d1, d2)[1],2 )}"  ) 
adj_xcorr_025[n1,n2]; adj_pear[n1,n2]; adj_spear[n1,n2]
fig,axes = plt.subplots(1,1,figsize=(8,4), dpi = 300)
sns.regplot(  x = data_hat[range(fs*st, fs*sp),  n1], y= data_hat[range(fs*st, fs*sp),n2], ax = axes , scatter_kws={"s":0.05})
d1= data_hat[:,n1]
d2= data_hat[:,n2]
print(f"\nx_xorr:   {np.round( adj_butter_xcorr[n1,n2],2 )}"  ) 
print(f"\nPearson:  {np.round( pearsonr(d1, d2)[0],2 )}; p-value: {np.round( pearsonr(d1, d2)[1],2 )}"  ) 
print(f"Spearman: {np.round( spearmanr(d1, d2)[0],2 )}; p-value: {np.round( spearmanr(d1, d2)[1],2 )}"  ) 
fig,axes = plt.subplots(1,1,figsize=(8,4), dpi = 300)
sns.regplot(  x = data_butter[range(fs*st, fs*sp),  n1], y= data_butter[range(fs*st, fs*sp),n2], ax = axes , scatter_kws={"s":0.1})
elecLoc["Tissue_segmentation_distance_from_label_2"]   
elecLoc["electrode_name"]   


eeg.columns    


tmp = np.intersect1d(elecLoc["electrode_name"]   , eeg.columns  , return_indices = True )    


elecLoc["Tissue_segmentation_distance_from_label_2"]   
tmp2 = np.array(elecLoc.iloc[tmp[1],:]["Tissue_segmentation_distance_from_label_2"]    )

adjjj = adj_xcorr
adjjj = adj_pear
adjjj = adj_spear
adjjj = adj_MI
ind_wm = np.where(tmp2 > 0)[0]
ind_gm = np.where(tmp2 <= 0)[0]
tmp_wm = adjjj[ind_wm[:,None], ind_wm[None,:]]
tmp_gm = adjjj[ind_gm[:,None], ind_gm[None,:]]
np.mean(tmp_gm)
np.mean(tmp_wm)
order = np.argsort(tmp2)
tmp2[order][63]
adj_xcorr_ord = adj_xcorr[order[:,None], order[None,:]]
adj_pear_ord = adj_pear[order[:,None], order[None,:]]
adj_spear_ord = adj_spear[order[:,None], order[None,:]]
adj_MI_ord = adj_MI[order[:,None], order[None,:]]
"""