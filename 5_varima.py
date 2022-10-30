# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:56:36 2022

@author: gkum7098
"""

import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('temperature.csv')

def create_date(x):
    month = x['month']
    day = x['day']
    year = x['year']
    date = datetime.date(year, month, day)
    return date
df['date'] = pd.to_datetime(df.apply(create_date, axis=1))
df = df[['date', 'City', 'AverageTemperatureFahr']]
df = pd.pivot_table(df, 
                    values='AverageTemperatureFahr',
                    index='date',
                    columns='City', aggfunc=np.mean)
df = df.interpolate()
df = df[['Auckland', 'Stockholm']].dropna()
df.plot()


N_test = 100

scale = StandardScaler() #necessary for VARMA / VAR
df.loc[df.index[:-N_test], ['Auckland_scaled', 'Stockholm_scaled']] = scale.fit_transform(df[['Auckland', 'Stockholm']].iloc[:-N_test,:])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(df['Auckland_scaled'].iloc[:-N_test])
plot_acf(df['Auckland_scaled'].iloc[:-N_test])

plot_pacf(df['Stockholm_scaled'].iloc[:-N_test])
plot_acf(df['Stockholm_scaled'].iloc[:-N_test])

df.index.freq = 'MS'

#granger casuality test for predictability between time series, test is for if second column effects the first column or not, effects if p value < 0.05
from statsmodels.tsa.stattools import grangercausalitytests
granger_result = grangercausalitytests(df[['Auckland_scaled', 'Stockholm_scaled']].dropna(),
                      maxlag=15)


from statsmodels.tsa.statespace.varmax import VARMAX #VARMA model are no unique, and we cannot dervice p and q uniquely
model = VARMAX(df[['Auckland_scaled', 'Stockholm_scaled']].iloc[:-N_test,:],
               order=(10,10)) #time consuming
result = model.fit(maxiter=100)

forecast = result.get_forecast(N_test)
result.fittedvalues #train set
forecast.predicted_mean #test set

df.loc[df.index[-N_test:] , ['Auckland_predicted', 'Stockholm_predicted']] = scale.inverse_transform(forecast.predicted_mean)

fig, axs = plt.subplots(2,1, sharex=True)
df[['Auckland', 'Auckland_predicted']].plot(ax=axs[0])
df[['Stockholm', 'Stockholm_predicted']].plot(ax=axs[1])

from statsmodels.tsa.api import VAR #most common model, becuase of non-unique issue with VARMA

model = VAR(df[['Auckland_scaled', 'Stockholm_scaled']].iloc[:-N_test,:])

lag_order_results = model.select_order(maxlags=30)
print(lag_order_results.selected_orders)

results = model.fit(maxlags=15, ic='aic')
lag_order = results.k_ar
prior = df[['Auckland_scaled', 'Stockholm_scaled']].iloc[:-lag_order,:].to_numpy()
pridiction = results.forecast(prior, N_test)
train_fitted_values = results.fittedvalues