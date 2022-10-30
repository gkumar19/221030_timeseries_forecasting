# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:44:38 2022

@author: gkum7098
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
df_close = pd.read_csv('sp500_close.csv', index_col=0, parse_dates=True)

fig, axs = plt.subplots(4, 1, sharex=True)
df_goog = df_close[['GOOG']].dropna()
df_goog.plot(ax=axs[0], title='goog_price')
np.log(df_goog).plot(ax=axs[1], title='goog_log_price')
np.log(df_goog.pct_change(1) + 1).plot(ax=axs[2], title='goog_log_return')

#normalcy test to check random walk hypthesis, if p > 0.05 probabiliy gaussian
from scipy.stats import shapiro
stat, p = shapiro(np.log(df_goog.pct_change(1) + 1))

df_goog['simple_moving_average'] = df_goog['GOOG'].rolling(window=10).mean().shift(1)
df_goog.plot(ax=axs[3], title='simple_moving_average')

print('r2_score', r2_score(df_goog.dropna()['GOOG'], df_goog.dropna()['simple_moving_average']))
print('mean_absolute_percentage_error', mean_absolute_percentage_error(df_goog.dropna()['GOOG'], df_goog.dropna()['simple_moving_average']))
print('mean_squared_error', mean_squared_error(df_goog.dropna()['GOOG'], df_goog.dropna()['simple_moving_average']))
print('mean_absolute_error', mean_absolute_error(df_goog.dropna()['GOOG'], df_goog.dropna()['simple_moving_average']))