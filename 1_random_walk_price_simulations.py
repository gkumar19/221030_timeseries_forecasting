# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:32:36 2022

@author: gkum7098
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_prices = [] #Log price chart has same return % shown with same length vertical lines in the chart
mu, sigma = 0, 0.1

log_price0 = 0
log_prices.append(log_price0)
for _ in range(10000):
    log_return = np.random.normal(mu, sigma)
    log_price = log_prices[-1] + log_return
    log_prices.append(log_price)
    
prices = [np.exp(log_price) for log_price in log_prices]

fig, axs = plt.subplots(2, 1, figsize=(10,5))
pd.Series(prices).plot(title='prices', ax= axs[0])
pd.Series(log_prices).plot(title='log_prices', ax=axs[1])
