# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:10:12 2020

@author: walke
"""

import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import math
style.use('seaborn')
sns.set_style(style='darkgrid')

import statsmodels.api as sm
from glm.glm import GLM
from glm.families import Gaussian
from sklearn.linear_model import LinearRegression

#%%
data = pd.read_csv("C:/Users/walke/Documents/galvanize/case_studies/regression-case-study/data/train.csv")
data['SalePrice'] = np.log(data['SalePrice'])
data = data[data['YearMade']>1940]
data = data[['SalePrice','MachineHoursCurrentMeter']][data['MachineHoursCurrentMeter']!=0]
data = data.dropna()
#data['MachineHoursCurrentMeter'] = np.log(data['MachineHoursCurrentMeter'])
hours_threshold = np.quantile(data['MachineHoursCurrentMeter'],0.99)
data = data[data['MachineHoursCurrentMeter']<np.quantile(data['MachineHoursCurrentMeter'],0.99)]
hours_mean = np.mean(data['MachineHoursCurrentMeter'])
y = data['SalePrice']
X = data[['MachineHoursCurrentMeter']]
model = sm.OLS(y, X)
X = sm.add_constant(X)
results = model.fit()
plt.scatter(X['MachineHoursCurrentMeter'],y,alpha=0.1)
plt.plot(X['MachineHoursCurrentMeter'],X['MachineHoursCurrentMeter'].map(lambda x: const + x*hours),color='r')
plt.title('Usage Price Model')
plt.xlabel('Machine Hours')
plt.ylabel('Log(Sale Price)')
#%%
data = pd.read_csv("C:/Users/walke/Documents/galvanize/case_studies/regression-case-study/data/train.csv")
data['SalePrice'] = np.log(data['SalePrice'])
#data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter']!=0 & ~ data['MachineHoursCurrentMeter'].isnull()] = np.log(data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter']!=0 & ~ data['MachineHoursCurrentMeter'].isnull()])
data = data[data['YearMade']>1940]
data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter']==0] = hours_mean
data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter'].isnull()] = hours_mean
data = data[data['MachineHoursCurrentMeter']<hours_threshold]
y = data['SalePrice']
X = data[['MachineHoursCurrentMeter']]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
plt.scatter(X['MachineHoursCurrentMeter'],y,alpha=0.1)
plt.plot(X['MachineHoursCurrentMeter'],X['MachineHoursCurrentMeter'].map(lambda x: const + x*hours),color='r')
plt.title('Usage Price Model')
plt.xlabel('Machine Hours')
plt.ylabel('Log(Sale Price)')
#%%
data = pd.read_csv("C:/Users/walke/Documents/galvanize/case_studies/regression-case-study/data/train.csv")
data['SalePrice'] = np.log(data['SalePrice'])
