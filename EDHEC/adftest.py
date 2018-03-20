#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:49:47 2018

@author: nicob
"""
import matplotlib.pyplot as plt
import time
plt.style.use('seaborn')
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter


main_data = pd.read_csv('sp500_joined_closes.csv', index_col='Date'
                        , parse_dates=True)


data = main_data.iloc[-3000:].dropna(axis=1, how='any')

df = data['AAPL']

df2 = data['MSFT']


X = df.values


Y = df2.values


plt.plot(X)
plt.plot(Y)
plt.show()

result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
 

    
def get_hurst(X):
    lag1 = 2
    lags = range(lag1, 20)
    tau = [np.sqrt(np.std(np.subtract(X[lag:], X[:-lag]))) for lag in lags]
    
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2
    return hurst

hurst = get_hurst(X)
print('\n')
print('Hurst Adj Close= {:.4f}'.format(hurst))

def get_halflife(X):
    X_lag = np.roll(X,1)
    X_lag[0] = 0
    X_ret = X - X_lag
    X_ret[0] = 0
    
    #adds intercept terms to X variable for regression
    X_lag2 = sm.add_constant(X_lag)
    
    model = sm.OLS(X_ret,X_lag2)
    res = model.fit()
    
    halflife = round(-np.log(2) / res.params[1],1)
    return halflife

def get_hedge_ratio(X, Y):
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    return results.params

print('\nHalf life= {} days'.format(get_halflife(X)))


alpha, beta = get_hedge_ratio(X,Y)

Z = Y - (alpha + beta * X)


def draw_scatterplot(X, Y):

    plen = len(X)
    colour_map = plt.cm.get_cmap('GnBu')
    colours = np.linspace(0.1, 1, plen)

    # Create the scatterplot object
    scatterplot = plt.scatter(X, Y, 
                              s=30, c=colours, cmap=colour_map, 
                              edgecolor='k', alpha=0.8)

    # Add a colour bar for the date colouring and set the
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in X[::plen//9].index]
    )
    plt.show()




plt.plot(Z, linewidth=1)
plt.title('Residuals (EWC - HedgeRatio * EWA) Analysis',
          fontname="Times New Roman Bold", fontsize=16)
plt.xlabel('Apr 2006 - Apr 2012')
plt.ylabel('EWC - hedgeRatioEWA')
plt.show()




result = adfuller(Z)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

def calc_slope_intercept_kalman(X, Y):
    """
    Utilise the Kalman Filter from the pyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [X, np.ones(X.shape)]
    ).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2.0,
        transition_covariance=trans_cov
    )

    state_means, state_covs = kf.filter(Y)
    return state_means, state_covs

tic = time.time()
state_means, state_covs = calc_slope_intercept_kalman(X, Y)
tac = time.time()

print('{:.3f}ms'.format((tac-tic)*100))

beta_kf = pd.DataFrame({'Slope': state_means[:, 0], 'Intercept': state_means[:, 1]},
                   index=df.index)


beta_kf.plot(subplots=True,linestyle='--')
plt.suptitle('Alpha & Beta over time from Kalman Filter',
          fontname="Times New Roman Bold", fontsize=16)
plt.show()


Z2 = Y - (state_means[:, 1] + state_means[:, 0] * X)
zscore = (Z2 - Z2.mean()) / Z2.std()


plt.plot(Z2, linewidth=1)

plt.title('ERatio w Kalman Filter',
          fontname="Times New Roman Bold", fontsize=16)
plt.xlabel('Apr 2006 - Apr 2012')
plt.ylabel('e(t) = EWC - (alpha + beta *EWA)')
plt.show()

result = adfuller(Z2)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

print('\nDynamic Ratio std: {:.3f}'.format(np.std(Z2)))
print('\nDynamic Ratio mean: {:.3f}'.format(np.mean(Z2)))




