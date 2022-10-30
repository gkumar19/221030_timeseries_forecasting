# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:01:03 2022

@author: gkum7098
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4,1,sharex=True)

df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)

df['1-diff'] = df['Passengers'].diff()
df['log'] = np.log(df['Passengers'])
df['log-1-diff'] = np.log(df['Passengers']).diff()

df[['Passengers', '1-diff']].plot(ax=axs[0])
df[['log']].plot(ax=axs[1])
df[['log-1-diff']].plot(ax=axs[2]) #stationary

from statsmodels.tsa.arima.model import ARIMA

df.index.freq = 'MS'
N_test = 24 # number of data points to be tested

model = ARIMA(df['Passengers'].iloc[:-N_test], order=(1,0,0))
result = model.fit()
df['arima_AR1'] = result.predict(start=df.index[0], end=df.index[-1])
df[['arima_AR1']].plot(ax=axs[0])

#two ways of finding orders:
    
#1. gridsearch

from itertools import product
from sklearn.metrics import r2_score

p = range(13)
d = range(3)
q = range(3)
pdq = product(p, d, q)
max_score = float('-inf')
for p, d, q in pdq:
    try:
        model = ARIMA(df['Passengers'].iloc[:-N_test], order=(p,d,q))
        result = model.fit()
        test_score = r2_score(df['Passengers'].iloc[-N_test:], result.forecast(N_test))
        if test_score > max_score:
            max_score = test_score
            best_params = p, d, q #10, 2, 1
            best_model = model
    except:
        pass
model = ARIMA(df['Passengers'].iloc[:-N_test], order=(10,2,1))
result = model.fit()
prediction = result.get_forecast(N_test)
df['arima_best'] = prediction.predicted_mean
df['arima_best'].plot(ax=axs[0])
axs[0].fill_between(df.index[-N_test:] ,
                    prediction.conf_int()['lower Passengers'], 
                    prediction.conf_int()['upper Passengers'],color='red', alpha=0.3)

#2. statistical techniques

#q ma component: acf plot
#p ar component: pacf plot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(df['Passengers'])
plot_acf(df['Passengers'])

#d. stationarity
from statsmodels.tsa.stattools import adfuller
test_statistics, p_value, _,_,_,_ = adfuller(df['Passengers'])
test_statistics, p_value, _,_,_,_ = adfuller(df['log-1-diff'].dropna()) #p_value less than 0.05 --> stationary

#3. auto arima

import pmdarima as pm

model = pm.auto_arima(df['Passengers'].iloc[:-N_test],
                      trace=True,
                      suppress_warnings=True,
                      stepwise=False, #False means full grid search
                      max_p=12,
                      max_q=2,
                      max_order=14, #sum of all orders
                      seasonal=True, m=12)

model.summary() #sarima model (p, d, q) * (P, D, Q) --> seasonal part

test_pred, confint = model.predict(n_periods=N_test, return_conf_int=True)
train_pred = model.predict_in_sample(start=0, end=-1)




