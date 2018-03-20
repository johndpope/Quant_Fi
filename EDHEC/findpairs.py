#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:15:48 2018

@author: nicob
"""
import matplotlib.pyplot as plt
import time
plt.style.use('seaborn')
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

def pca_clustering(data):
    
    returns = data.pct_change()
    returns = returns.dropna(axis=0, how='any')
    
    N_PRIN_COMPONENTS = 50
    pca = PCA(n_components=N_PRIN_COMPONENTS)
    pca.fit(returns)
    
    X = pca.components_.T
    X = preprocessing.StandardScaler().fit_transform(X)
    
    clf = DBSCAN(eps=2.9, min_samples=3)
    clf.fit(X)
    labels = clf.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("\nClusters discovered: %d" % n_clusters_)
    
    clustered = clf.labels_
    
    ticker_count = len(returns.columns)
    print("Total pairs possible in universe: %d " % (ticker_count*(ticker_count-1)/2))
    
    clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    
    CLUSTER_SIZE_LIMIT = 9999
    counts = clustered_series.value_counts()
    
    ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
    
    print("Clusters formed: %d" % len(ticker_count_reduced))
    print("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())
    return ticker_count_reduced, clustered_series

def find_cointegrated_pairs(data, significance=0.05):
    # This function is from https://www.quantopian.com/lectures/introduction-to-pairs-trading
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j], score))
                
    return score_matrix, pvalue_matrix, pairs

def get_pairs(ticker_count_reduced, clustered_series, data):
    cluster_dict = {}
    for i, which_clust in enumerate(ticker_count_reduced.index):
        tickers = clustered_series[clustered_series == which_clust].index
        
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(np.log(data[tickers]))
            
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs
        
    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])
    print("We found %d pairs." % len(pairs))
    print("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))
    return pairs

def calc_slope_intercept_kalman(X, Y):
    """
    Use the Kalman Filter from the pyKalman package
    to calculate the slope and intercept of the regressed
    prices
    """
    delta = 1e-5 # To Optimize
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
    return state_means[:, 1]

def calc_zscore(X, Y):
    
    state_means, state_covs = calc_slope_intercept_kalman(X, Y)
    kalman_spread = Y - (state_means[:, 1] + state_means[:, 0] * X)
    z_score = (kalman_spread[-1] - np.mean(kalman_spread)) / np.std(kalman_spread)
    return z_score, kalman_spread

def get_halflife(spread):
    #Run OLS regression on spread series and lagged version of itself
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    
    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()
  
    halflife = (-np.log(2) / res.params[1],0)
    return halflife

def main():
    main_data = pd.read_csv('sp500_joined_closes.csv', index_col='Date'
                            , parse_dates=True)
    
    data = main_data.iloc[-504:].dropna(axis=1, how='any')
    
    tic = time.time()
    
    ticker_count_reduced, clustered_series = pca_clustering(data)
    print('\n')
    pairs = get_pairs(ticker_count_reduced, clustered_series, data)

    from operator import itemgetter
    
    top_pairs=[]
    tickers = []
    for pair in sorted(pairs, key=itemgetter(2))[:]:
        tickers.append([pair[i] for i in range(2)])
        top_pairs.append("KalmanPairTrade(symbol('"+str(pair[0])+"'), symbol('"+str(pair[1])+"'), initial_bars=300, freq='1m', delta=1e-3, maxlen=300)")
    
    top_pairs = (",".join(map(str,top_pairs)))
    
    tickers = np.unique(tickers)
    list_ticker = []
    
    for ticker in tickers:
        list_ticker.append("symbol('"+str(ticker)+"')")
    
    list_ticker = (",".join(map(str,list_ticker)))
    print(top_pairs)
    print(list_ticker)
    
if __name__ == "__main__":
    main()

