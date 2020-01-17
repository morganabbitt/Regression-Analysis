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
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
const, hours = results.params
plt.scatter(X['MachineHoursCurrentMeter'],y,alpha=0.1)
plt.plot(X['MachineHoursCurrentMeter'],X['MachineHoursCurrentMeter'].map(lambda x: const + x*hours),color='r')
plt.title('Usage Price Model')
plt.xlabel('Machine Hours')
plt.ylabel('Log(Sale Price)')
#%%
data = pd.read_csv("C:/Users/walke/Documents/galvanize/case_studies/regression-case-study/data/train.csv")
data['SalePrice'] = np.log(data['SalePrice'])
data = data[data['YearMade']!=1000]
#data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter']!=0 & ~ data['MachineHoursCurrentMeter'].isnull()] = np.log(data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter']!=0 & ~ data['MachineHoursCurrentMeter'].isnull()])
data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter']==0] = hours_mean
data['MachineHoursCurrentMeter'][data['MachineHoursCurrentMeter'].isnull()] = hours_mean
data = data[data['MachineHoursCurrentMeter']<hours_threshold]
y = data['SalePrice']
X = data[['MachineHoursCurrentMeter']]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
const, hours = results.params
plt.scatter(X['MachineHoursCurrentMeter'],y,alpha=0.1)
plt.plot(X['MachineHoursCurrentMeter'],X['MachineHoursCurrentMeter'].map(lambda x: const + x*hours),color='r')
plt.title('Usage Price Model')
plt.xlabel('Machine Hours')
plt.ylabel('Log(Sale Price)')
#%%
data['ProductSize'] = data['ProductSize'].fillna('Compact').replace({'Mini':0, 'Compact':1, 'Small': 2, 'Medium':3, 'Large':5, 'Large / Medium':4})

data["Age"] = (data["saledate"].map(lambda x: int(x.split(" ")[0][-4:])))-data["YearMade"]

final_train_1940_below = data[data['YearMade']<1940]
final_train_1940_above = data[data['YearMade']>1940]
final_train = final_train_1940_below[['SalePrice','Age','MachineHoursCurrentMeter','ProductSize']]
y = final_train['SalePrice']
X = final_train[['Age','MachineHoursCurrentMeter','ProductSize']]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
coeffs = results.params

final_train = final_train_1940_above[['SalePrice','Age','MachineHoursCurrentMeter','ProductSize']]
y = final_train['SalePrice']
X = final_train[['Age','MachineHoursCurrentMeter','ProductSize']]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
coeffs = results.params

"""X_new = sm.add_constant(data['ProductSize'])
y_new = data['SalePrice']
model_new = sm.OLS(y_new, X_new)
results_new = model_new.fit()
results_new.summary()
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(y_new, results_new.resid, alpha=0.01)
plt.axhline(0, linestyle='--',color='r')
plt.title('Residuals of Numeric Rank of Product Size', size=25)
plt.xlabel('Log Price', size=20)
plt.ylabel('Residuals')

const, size = results_new.params
fig, ax = plt.subplots(figsize=(15,10))
plt.scatter(data['ProductSize'], data['SalePrice'], alpha=0.002, s=120)
plt.plot(data['ProductSize'], data['ProductSize'].map(lambda x: const + size * x), color='r', linestyle='--', linewidth=5)"""