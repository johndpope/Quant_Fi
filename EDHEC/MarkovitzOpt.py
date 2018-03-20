import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

class Stock:
    pass


def quotes(Tickerlist):
    dfs = []
    for ticker in Tickerlist:
        df = pd.read_csv(ticker + str('.csv'), parse_dates=True, index_col='Date')
        dfs.append(df['Close'])

    df = pd.concat(dfs, axis=1)
    df.columns = Tickerlist
    df = df.resample('BMS').first()
    df = df / df.shift(1) - 1
    return df


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''

    k = np.random.rand(n)
    return k / sum(k)


a = ['AIR.PA', 'SAF.PA', 'MC.PA']


def random_portfolio(returns):

    p = np.asmatrix(returns.mean() * 12)
    w = np.asmatrix(rand_weights(returns.shape[1]))
    C = returns.cov().values

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma


returns = quotes(a)


