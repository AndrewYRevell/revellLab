"""
Created on Sat Aug 21 10:16:44 2021

@author: arevell


Purpose of code: simulate permutation hypothesis testing in tandem with bootstrapping.


Defining the problem:
Categorical variables: X is "good"; Y is "poor"

X and Y both have four levels of data: Patients, seizures, contacts, and contact localization

levels:
- Both X and Y have different N patients.
- Each patient has different M seizures.
- Each patient/seizure has Q contacts
- Each contact is either in GM or WM.

- The measurements of GM and WM both follow gamma distributions

Question: Is the differentce between GM and WM larger for Y than X (poor vs good)?

Definitions:
    DT (delta T = median measurement of WM - median measurement of GM)
    X_DT = the delta T of good outcome patients
    Y_DT = the delta T of poor outcome patients


In other words, is Y_DT > X_DT

Null hypothesis, H0: Y_DT - X_DT = 0
Alternative Hypothesis: H1: Y_DT - X_DT != 0
(but actually, I think Y_DT - X_DT > 0 based on prior knowledge of epilepsy pathophysiology in poor outcome patients)


"""


import numpy as np #for working with array data
import pandas as pd #for working with data tables
import scipy #for sceintifc computing like stats
import random #for generating random numbers
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for plotting
import copy #for copying data

#defining helper functions
def plot_make(r = 1, c = 1, size_length = None, size_height = None, dpi = 300, sharex = False, sharey = False , squeeze = True):
    """
    helper function to make subplots in python quickly
    """
    if size_length == None:
       size_length = 4* c
    if size_height == None:
        size_height = 4 * r
    fig, axes = plt.subplots(r, c, figsize=(size_length, size_height), dpi=dpi, sharex =sharex, sharey = sharey, squeeze = squeeze)
    return fig, axes


#%%
#defining the groud truth parameters

mean_gm_X = 1 #this is the mean of the GM contact measurement, GOOD outcome patients
mean_wm_X = 2 #this is the mean of the WM contact measurement, GOOD outcome patients

mean_gm_Y = 1 #this is the mean of the GM contact measurement, POOR outcome patients
mean_wm_Y = 5 #this is the mean of the WM contact measurement, POOR outcome patients

#gamma distribution parameters. Sclae is the shape of the gamma distribution. We assume to be the same
scale_gm_X = 1
scale_wm_X = 1
scale_gm_Y = 1
scale_wm_Y = 1
scale = 1

#defining the number of contacts for both GM and WM. Assume equal contacts in GM and WM (approximately true). Generally, there are at least a few and max 100 contacts in each
Q_gm = 40
Q_wm = 40
Q = 40 #because they are the same
Q_sigma = 10 #the st deviation of contacts
Q_lower = 5 #the minimum number of contacts existing
Q_upper = 80 #the maximum number of contacts existing
#these numbers are based on prior knowledge of a "typical" implant in a patient
#https://stackoverflow.com/questions/27831923/python-random-number-generator-with-mean-and-standard-deviation

#%%
# the number of measurements of GM and WM (based on Q contacts) based on functional conenctivity is given by formula:
#measurements = (Q * Q - Q )/ 2
def calculate_number_of_measurements(Q, Q_sigma, Q_lower, Q_upper):
    """
    simulate n contacts with upper and lower bounds. And then output number of measurements
    :param Q: mean of contact N
    :param Q_sigma: sigma dev of contact N
    :param Q_lower: lower bound of contact N
    :param Q_upper: Upper bound of contact N
    :return: return number of measurements

    """
    distribution = scipy.stats.truncnorm((Q_lower - Q) / Q_sigma, (Q_upper - Q) / Q_sigma, loc=Q, scale=Q_sigma) #simulate n contacts with upper and lower bound
    n = distribution.rvs(1).round(0).astype(int)[0]
    measurements = int((n * n - n )/ 2)
    return measurements

#simulate and plot the number of measurements
measurements = np.zeros(1000)
for N in range(1000):
    measurements[N] = calculate_number_of_measurements(Q_gm, Q_sigma, Q_lower, Q_upper)
sns.histplot(measurements)

#%%
#simulate the gamma distributions of GM and WM
def simulate_single_seizure(mean_gm, mean_wm, scale_gm, scale_wm, Q_gm, Q_wm, Q_sigma, Q_lower, Q_upper ):
    data_gm = np.random.gamma(mean_gm, scale_gm, calculate_number_of_measurements(Q_gm, Q_sigma, Q_lower, Q_upper))
    data_wm = np.random.gamma(mean_wm, scale_wm, calculate_number_of_measurements(Q_wm, Q_sigma, Q_lower, Q_upper))
    df_gm = pd.DataFrame(data_gm, columns = ["measurement"])
    df_gm["tissue"] = "gm"
    df_wm = pd.DataFrame(data_wm, columns = ["measurement"])
    df_wm["tissue"] = "wm"
    df = pd.concat([df_gm, df_wm])
    return df

def add_categorical_variables(df, outcome):
    df["outcome"] = "good"
    df["outcome-tissue"] = df["outcome"] + "_" +  df["tissue"]
    return df
df_x = add_categorical_variables(simulate_single_seizure(mean_gm_X, mean_wm_X, scale, scale, Q, Q, Q_sigma, Q_lower, Q_upper ), "good")
df_y = add_categorical_variables(simulate_single_seizure(mean_gm_Y, mean_wm_Y, scale, scale, Q, Q, Q_sigma, Q_lower, Q_upper ), "poor")

df = pd.concat([df_x, df_y])


#% plotting
fig, axes = plot_make(c = 2)
sns.histplot(df, x = "measurement", hue = "outcome-tissue", ax = axes[0], binrange = binrange, binwidth = binwidth, kde = True, edgecolor = None)
sns.ecdfplot(df, x = "measurement", hue = "outcome-tissue", ax = axes[1])




#%% Now simulate patients will have multiple seizures, each with their own variation in GM and WM measurements

mean_gm_sigma =0.5 #the mean itself for a single patient will have variation between seizures
mean_wm_sigma =0.5 #the mean itself for a single patient will have variation between seizures

mean_gm= 1
mean_wm = 5
M_seizures = 5

def simulate_M_seizures(mean_gm, mean_wm, scale_gm, scale_wm, Q_gm, Q_wm, Q_sigma, Q_lower, Q_upper,
                        mean_gm_sigma, mean_wm_sigma, M_seizures):

    mean_gm_patient = np.random.gamma(mean_gm, mean_gm_sigma, M_seizures)
    mean_wm_patient = np.random.gamma(mean_wm, mean_wm_sigma, M_seizures)
    for M in range(M_seizures):

        df_seizure =simulate_single_seizure(mean_gm_patient[M], mean_wm_patient[M], scale_gm, scale_wm, Q_gm, Q_wm, Q_sigma, Q_lower, Q_upper )
        df_seizure["seizure"] = M
        if M == 0: #initialize
            df = df_seizure
        else:
            df = pd.concat([df, df_seizure])
    return df

#%%

N_X = 10 #number of good
N_Y = 7 #number of poor
mean_seizures = 2.5 #mean number of seizures per patient
#zero-truncated poisson distribution of seizures. Never zero.
def zpt_cdf(lam, size=None):
    #http://blog.richardweiss.org/2017/10/18/pymc3-truncated-poisson.html
    # get the lam value and produce the scipy frozen
    # poisson distribution using it
    lam = np.asarray(lam)
    dist = scipy.stats.distributions.poisson(lam)
    # find the lower CDF value, the probability of a
    # zero (or lower) occuring.
    lower_cdf = dist.cdf(0)
    # The upper CDF is 1, as we are not placing an upper
    # bound on this distribution
    upper_cdf = 1
    # Move our random sample into the truncated area of
    # the distribution
    shrink_factor = upper_cdf - lower_cdf
    sample = np.random.rand(size) * shrink_factor + lower_cdf
    # and find the value of the poisson distribution
    # at those sampled points. In this case, this will
    # mean no zeros.
    return dist.ppf(sample).astype(int)
sns.histplot( zpt_cdf(2.5, size=10000 ), binwidth = 1, binrange = [0.5,10])


def simulate_N_patients(N, mean_seizures, outcome, mean_gm, mean_wm, scale_gm, scale_wm, Q_gm, Q_wm, Q_sigma, Q_lower, Q_upper, mean_gm_sigma, mean_wm_sigma, count = 0):
    for n in range(N):
        M_seizures = zpt_cdf(mean_seizures, size=1 )[0]
        df_patient = simulate_M_seizures(mean_gm, mean_wm, scale_gm, scale_wm, Q_gm, Q_wm, Q_sigma, Q_lower, Q_upper,
                                mean_gm_sigma, mean_wm_sigma, M_seizures)
        df_patient["patient"] = n + count
        if n == 0: #initialize
            df= df_patient
        else:
            df = pd.concat([df, df_patient])

        df["outcome"] = outcome
    return df


#%%
X = simulate_N_patients(N_X, mean_seizures, "good", mean_gm_X, mean_wm_X, scale, scale, Q, Q, Q_sigma, Q_lower, Q_upper, mean_gm_sigma, mean_wm_sigma)
Y = simulate_N_patients(N_Y, mean_seizures, "poor", mean_gm_Y, mean_wm_Y, scale, scale, Q, Q, Q_sigma, Q_lower, Q_upper, mean_gm_sigma, mean_wm_sigma, count = N_X )
df = pd.concat([X,Y])
df["outcome-tissue"] = df["outcome"] + "_" +  df["tissue"]


fig, axes = plot_make(c = 2)
sns.histplot(df, x = "measurement", hue = "outcome-tissue", ax = axes[0], binrange = [0,10], binwidth = 1, kde = True, edgecolor = None)
sns.ecdfplot(df, x = "measurement", hue = "outcome-tissue", ax = axes[1])


#%%
#calculate DT (delta T)


COLORS_GOOD_VS_POOR = ["#7400b3", "#da8d00"]


df_medians = df.groupby(by=["patient", "seizure", "tissue", "outcome"]).median().reset_index()
df_medians = df_medians.pivot(index=["patient", "seizure", "outcome"], columns='tissue', values='measurement').reset_index()
df_medians["DT"] =  df_medians["wm"]  - df_medians["gm"]

df_DT = copy.deepcopy(df_medians)

def compute_T(df_DT, group  =True):
    if group:
        df_DT_patient= df_DT.groupby(by=["patient", "outcome"]).mean().reset_index()
    else:
        df_DT_patient = copy.deepcopy(df_DT)
    v1 = df_DT_patient.query('outcome == "good"')["DT"]
    v2 = df_DT_patient.query('outcome == "poor"')["DT"]
    T = scipy.stats.ttest_ind(v2, v1)[0]
    return T

df_DT_patient= df_DT.groupby(by=["patient", "outcome"]).mean().reset_index()
T = compute_T(df_DT)
fig, axes = plot_make()
sns.boxplot(data=df_DT_patient, x="outcome", y="DT", ax = axes, showfliers = False, palette =COLORS_GOOD_VS_POOR )
sns.swarmplot(data=df_DT_patient, x="outcome", y="DT", ax = axes, color ="black")
print(T)
axes.set_title(f"{T}" )

#%%% Bootstrapping Original data to calculate T_R

#bootstrap df_DT
ratio_patients = 1

def bootstrap_df_DT(df_DT, ratio_patients):
    df_DT_bootstrap = pd.DataFrame(columns = df_DT.columns)

    for o in ["good", "poor"]:
        df_DT_outcome = df_DT.query(f'outcome == "{o}"')
        patients = np.unique(df_DT_outcome["patient"])
        boot_patients = np.random.choice(list(patients.astype(int)), len(patients) * ratio_patients)
        for p in range(len(boot_patients)):
            pt = boot_patients[p]
            seizures = np.unique(df_DT_outcome.query(f'patient == {pt}')["seizure"])
            boot_seizures = np.random.choice(list(seizures.astype(int)), len(seizures))
            for n in range(len(boot_seizures)):
                df_DT_bootstrap = df_DT_bootstrap.append(df_DT_outcome.query(f'patient == {pt} and seizure == {boot_seizures[n]}'))
    return df_DT_bootstrap



df_DT_bootstrap = bootstrap_df_DT(df_DT, ratio_patients)
T_R = compute_T(df_DT_bootstrap, group  = False) #don't need to groupby because bootstrapping by patients

fig, axes = plot_make()
sns.boxplot(data=df_DT_bootstrap, x="outcome", y="DT", ax = axes, showfliers = False, palette =COLORS_GOOD_VS_POOR )
sns.swarmplot(data=df_DT_bootstrap, x="outcome", y="DT", ax = axes, color ="black")
axes.set_title(f"{T_R}" )



#Simulate N times
iterations = 100
T_R = np.zeros(iterations)
for it in range(iterations):
    df_DT_bootstrap = bootstrap_df_DT(df_DT, ratio_patients)
    T_value = compute_T(df_DT_bootstrap, group = False)
    T_R[it] = T_value
    print(f"\r{it}: {T_value}", end = "\r")


binrange = [0,12]
binwidth = 0.5
fig, axes = plot_make()
sns.histplot(T_R, kde = True, ax = axes, color = "#222222", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_R.mean(), color='k', linestyle='--')
axes.axvline(x=T, color='b', linestyle='--')
axes.set_xlim(binrange)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
#pvalue = len( np.where(permute >= Tstat.mean()) [0]) / len(Tstat)
axes.set_title(f"{abs(T_R).mean()}" )



#%% permutation hypothesis testing, to simulate T*


#permute
def permute_df_DT(df_DT):
    df_DT_permute = copy.deepcopy(df_DT)
    patients = np.unique(df_DT_permute["patient"])
    for pt in range(len(patients)):
        outcome = np.array(df_DT_permute.query(f'patient == {patients[pt]}')["outcome"])[0]
        switch = scipy.stats.binom.rvs(1, 0.5, size=1)[0]#simulating probability of switching outcomes
        if switch == 1:
            if outcome == "good": outcome = "poor"
            else: outcome = "good"
        df_DT_permute.loc[ df_DT_permute["patient"] == patients[pt]     , "outcome"] = outcome
    return df_DT_permute

def permute_df_DT(df_DT):
    permute = copy.deepcopy(df_DT)
    patients = np.unique(permute["patient"])
    l_good = len(np.unique(permute.query(f"outcome == 'good'")["patient"]))
    l_poor = len(np.unique(permute.query(f"outcome == 'poor'")["patient"]))
    p_good = np.random.choice(patients, int(l_good))
    p_poor = np.random.choice(patients, int(l_poor))
    permute.loc[permute.patient.isin(p_good),"outcome"] = "good"
    permute.loc[~permute.patient.isin(p_good),"outcome"] = "poor"
    return permute



df_DT_permute = permute_df_DT(df_DT)
T_B = compute_T(df_DT_permute)
print(T_B)

df_DT_patient_permute = df_DT_permute.groupby(by=["patient", "outcome"]).mean().reset_index()
fig, axes = plot_make()
sns.boxplot(data=df_DT_patient_permute, x="outcome", y="DT", ax = axes, showfliers = False, palette =COLORS_GOOD_VS_POOR )
sns.swarmplot(data=df_DT_patient_permute, x="outcome", y="DT", ax = axes, color ="black")
axes.set_title(f"{T_B}" )

#Run many simulationg
iterations = 10000
T_permute = np.zeros(iterations)
for it in range(iterations):
    df_DT_permute = permute_df_DT(df_DT)
    T_B = compute_T(df_DT_permute)
    T_permute[it] = T_B
    print(f"\r{it}: {T_B}", end = "\r")

binrange = [-12,12]
binwidth = 0.5
fig, axes = plot_make()
sns.histplot(T_permute, kde = True, ax = axes, color = "#bbbbbb", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_permute.mean(), color='k', linestyle='--')
axes.axvline(x=T, color='b', linestyle='--')
axes.set_xlim(binrange)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
pvalue = len( np.where(T_permute >= T )[0]) / len(T_permute)
axes.set_title(f"{T_permute.mean()}: {pvalue}" )



#%% permutation hypothesis testing + bootstrapping, to simulate T*


#Run many simulationg
iterations = 100
T_permute_boot = np.zeros(iterations)
for it in range(iterations):
    df_DT_permute = permute_df_DT(df_DT)
    df_DT_permute_bootstrap = bootstrap_df_DT(df_DT_permute, ratio_patients)
    T_B_permute = compute_T(df_DT_permute, group  = False)
    T_permute_boot[it] = T_B_permute
    print(f"\r{it}: {T_B_permute}", end = "\r")

binrange = [-12,12]
binwidth = 0.5
fig, axes = plot_make()
axes.axvline(x=T, color='b', linestyle='--')

sns.histplot(T_R, kde = True, ax = axes, color = "#222222", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_R.mean(), color='k', linestyle='--')

sns.histplot(T_permute, kde = True, ax = axes, color = "blue", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_permute.mean(), color='blue', linestyle='--')

sns.histplot(T_permute_boot, kde = True, ax = axes, color = "#aaaaaa", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_permute_boot.mean(), color='white', linestyle='--')

axes.set_xlim(binrange)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)



#%%

#Run simulation start to finish


X = simulate_N_patients(N_X, mean_seizures, "good", mean_gm_X, mean_wm_X, scale, scale, Q, Q, Q_sigma, Q_lower, Q_upper, mean_gm_sigma, mean_wm_sigma)
Y = simulate_N_patients(N_Y, mean_seizures, "poor", mean_gm_Y, mean_wm_Y, scale, scale, Q, Q, Q_sigma, Q_lower, Q_upper, mean_gm_sigma, mean_wm_sigma, count = N_X )
df = pd.concat([X,Y])
df["outcome-tissue"] = df["outcome"] + "_" +  df["tissue"]


df_medians = df.groupby(by=["patient", "seizure", "tissue", "outcome"]).median().reset_index()
df_medians = df_medians.pivot(index=["patient", "seizure", "outcome"], columns='tissue', values='measurement').reset_index()
df_medians["DT"] =  df_medians["wm"]  - df_medians["gm"]

df_DT = copy.deepcopy(df_medians)
df_DT_patient= df_DT.groupby(by=["patient", "outcome"]).mean().reset_index()
T = compute_T(df_DT)
print(T)

#Simulate Bootstrapping original data  N times and calculate distribution of T_R
iterations = 10000
T_R = np.zeros(iterations)
for it in range(iterations):
    df_DT_bootstrap = bootstrap_df_DT(df_DT, ratio_patients)
    T_value = compute_T(df_DT_bootstrap, group = False)
    T_R[it] = T_value
    print(f"\r{it}: {T_value}", end = "\r")

print("")
#Run many simulationg
T_permute = np.zeros(iterations)
for it in range(iterations):
    df_DT_permute = permute_df_DT(df_DT)
    T_B = compute_T(df_DT_permute)
    T_permute[it] = T_B
    print(f"\r{it}: {T_B}", end = "\r")

print("")
#Run many simulationg
T_permute_boot = np.zeros(iterations)
for it in range(iterations):
    df_DT_permute = permute_df_DT(df_DT)
    df_DT_permute_bootstrap = bootstrap_df_DT(df_DT_permute, ratio_patients)
    T_B_permute = compute_T(df_DT_permute, group  = False)
    T_permute_boot[it] = T_B_permute
    print(f"\r{it}: {T_B_permute}", end = "\r")



binrange = [-12,12]
binwidth = 0.5
fig, axes = plot_make()
#true T
axes.axvline(x=T, color='b', linestyle='--')

#distribution of Ts
sns.histplot(T_R, kde = True, ax = axes, color = "#222222", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_R.mean(), color='k', linestyle='--')

#distribution of permuting original data
sns.histplot(T_permute, kde = True, ax = axes, color = "blue", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_permute.mean(), color='blue', linestyle='--')

#distribution of perumting original data and then boostrap
sns.histplot(T_permute_boot, kde = True, ax = axes, color = "#aaaaaa", binwidth = binwidth, binrange = binrange, edgecolor = None)
axes.axvline(x=T_permute_boot.mean(), color='white', linestyle='--')

axes.set_xlim(binrange)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

T_permute
T_permute_boot

scipy.stats.ttest_ind(T_permute, T_permute_boot)

len(T_permute)
scipy.stats.ttest_ind(T_permute, T_permute_boot)[0]



