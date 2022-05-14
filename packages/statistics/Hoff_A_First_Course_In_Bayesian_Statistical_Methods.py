"""
Created on Sat Apr 30 18:11:41 2022

@author: arevell
"""

import numpy as np
import seaborn as sns


custom_params = {"axes.spines.right": False, "axes.spines.top": False, 'figure.dpi': 300,
                 "legend.frameon": False, "savefig.transparent": True}
sns.set_theme(style="ticks", rc=custom_params,  palette="pastel")
sns.set_context("talk")

kde_kws = {"bw_adjust": 2}

#%% Doing some binomial distribution
binwidth= 0.5
trials = 1000

n=20

data1 = np.random.binomial(n, 0.2, trials)
data2 = np.random.binomial(n, 0.1, trials)
data3 = np.random.binomial(n, 0.05, trials)

sns.histplot(data1, kde= True, binrange=[0,n] , binwidth=binwidth , kde_kws=kde_kws , stat = "probability" , color = "#772222")
sns.histplot(data2, kde= True, binrange=[0,n] , binwidth=binwidth , kde_kws=kde_kws , stat = "probability", color = "#227722" )
sns.histplot(data3, kde= True, binrange=[0,n] , binwidth=binwidth , kde_kws=kde_kws , stat = "probability", color = "#772277")


t = 0.25
e = (0.25) * (0.11+0.19+0.31+0.53)


y=0.11
b = y*t/e
print(np.round(b,2))

y=0.19
b = y*t/e
print(np.round(b,2))

y=0.31
b = y*t/e
print(np.round(b,2))

y=0.53
b = y*t/e
print(np.round(b,2))
