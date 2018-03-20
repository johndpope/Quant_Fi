import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import pandas as pd
import numpy as np
import seaborn as sns
plt.style.use('ggplot')


def check_for_stationarity(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print ('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.')
        return False

df = pd.read_csv('stock_dfs/MSFT.csv', parse_dates=True, index_col='Date')

X = df.loc['2014-01-01':'2015-01-01']['Adj Close']
X.name = 'MSFT'
check_for_stationarity(X)

X1 = X.diff()[1:]
X1.name = X.name + ' Additive Returns'
check_for_stationarity(X1)

X1 = X.pct_change()[1:]
X1.name = X.name + ' Multiplicative Returns'
check_for_stationarity(X1)