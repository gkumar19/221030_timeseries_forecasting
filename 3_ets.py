# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:33:14 2022

@author: gkum7098
"""

import pandas as pd
from sklearn.metrics import r2_score
df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
df['simple_moving_average'] = df['Passengers'].rolling(5).mean().shift(1)

alpha = 0.5
df['exponential_moving_average'] = df['Passengers'].ewm(alpha=alpha, adjust=False).mean().shift(1) #same as SimpleExpSmoothing

N_test = 12 # number of data points to be tested

#ETS model : Exponential Smoothing Methods : Error Trend Season 

from statsmodels.tsa.holtwinters import SimpleExpSmoothing #non-tresnding and non-seasonal
df.index.freq = 'MS' #timeseries alias: https://pandas.pydata.org/docs/user_guide/timeseries.html
model = SimpleExpSmoothing(df['Passengers'].iloc[:-N_test], initialization_method='legacy-heuristic')
result = model.fit(smoothing_level=alpha, optimized=False)
df['simple_exponential_smoothing'] = result.predict(start=df.index[0], end=df.index[-1])
df.loc[df.index[:-N_test],'simple_exponential_smoothing2'] = result.fittedvalues #alternate of above line
df.loc[df.index[-N_test:], 'simple_exponential_smoothing2'] = result.forecast(N_test) #alternate of second above line

from statsmodels.tsa.holtwinters import Holt #trending but non-seasonal
model = Holt(df['Passengers'].iloc[:-N_test], initialization_method='legacy-heuristic')
result = model.fit()
df['holt_linear_trend'] = result.predict(start=df.index[0], end=df.index[-1])

from statsmodels.tsa.holtwinters import ExponentialSmoothing #trending and seasonal
model = ExponentialSmoothing(df['Passengers'].iloc[:-N_test],
                             initialization_method='legacy-heuristic',
                             trend='add', seasonal='add', seasonal_periods=12)
result = model.fit()
df['holt_winters_add'] = result.predict(start=df.index[0], end=df.index[-1])
print('r2_square holt_winters_add:', r2_score(df['Passengers'].iloc[-N_test:] ,df['holt_winters_add'].iloc[-N_test:]))
model = ExponentialSmoothing(df['Passengers'].iloc[:-N_test],
                             initialization_method='legacy-heuristic',
                             trend='add', seasonal='mul', seasonal_periods=12)
result = model.fit()
df['holt_winters_multiply'] = result.predict(start=df.index[0], end=df.index[-1])
print('r2_square holt_winters_multiply:', r2_score(df['Passengers'].iloc[-N_test:] ,df['holt_winters_multiply'].iloc[-N_test:]))

#Arima Models

df.plot()