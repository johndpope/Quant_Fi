import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import datetime
import time




def quotes(Tickerlist, startdate):
    yf.pdr_override()
    dfs = []
    for ticker in Tickerlist:
        #df = pd.read_csv(ticker + str('.csv'), parse_dates=True, index_col='Date', usecols=['Date', 'Adj Close'])
        df = pdr.get_data_yahoo(ticker, start=startdate, end=datetime.date.today(), usecols=['Date', 'Adj Close'])
        dfs.append(df['Adj Close'])

    df = pd.concat(dfs, axis=1)
    df.columns = Tickerlist
    df = df.resample('BMS').last()
    df.to_csv('ASS1.csv')
    return df


def minvar_portfolio(num_portfolios):
    returns = quotes(stocks)
    results = np.zeros((4 + len(stocks) - 1, num_portfolios))
    cov_matrix = returns.cov()
    mean_monthly_returns = returns.mean()

    for i in range(num_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(len(stocks)))

        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_monthly_returns * weights) * 12
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)

        # store results in results array
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, i] = results[0, i] / results[1, i]
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j + 3, i] = weights[j]

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T, columns=['ret','stdev','sharpe'] + [stock for stock in stocks])

    # locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

    return min_vol_port


def calc_min_variance_portfolio(return_vector, covariance_matrix, target_return):
    """
    Given return, variance, and correlation data on multiple assets and a target portfolio
    return, calculate the minimum variance portfolio that achieves the target_return, if possible.
    """
    MU = np.asmatrix(return_vector).T
    m = target_return
    COV = np.asmatrix(covariance_matrix)
    ONE = np.matrix((1,) * COV.shape[0]).T

    A = ONE.T * COV.I * ONE
    a3 = float(A)
    B = MU.T * COV.I * ONE
    a2 = float(B)
    C = MU.T * COV.I * MU
    a1 = float(C)

    LAMBDA = (a3 * m - a2) / (a3 * a1 - (a2 * a2))
    GAMMA = ((a1 - a2 * m) / ((a3 * a1) - (a2 * a2)))

    WSTAR = COV.I * ((LAMBDA * MU) + (GAMMA * ONE))
    STDDEV = np.sqrt(WSTAR.T * COV * WSTAR)

    WMIN = (COV.I * ONE) / (ONE.T * COV.I * ONE)
    STDDEVMIN = np.sqrt(1 / (ONE.T * COV.I * ONE))

    return WSTAR, STDDEV, WMIN, STDDEVMIN

def calc_outsample_portfolio(returns, weight, date2):
        return np.matrix(returns.loc[date2:].mean() * 12) * weight

#['AIR.PA', 'MC.PA', 'SAF.PA']

stocks = [x.upper() for x in input('Ticker --> ').split()]
#stocks = ['GS', 'JPM', 'AAPL']


target_return = float(input('Target Return %:  ')) / 100
date_entry = input('Enter an in-sample start date in YYYY-MM-DD format: ')
year, month, day = map(int, date_entry.split('-'))
date1 = datetime.date(year, month, day)


date_entry = input('Enter an in-sample end date in YYYY-MM-DD format: ')
year, month, day = map(int, date_entry.split('-'))
date2 = '{}-{:02}'.format(year, month)
date3 = datetime.date.today()



returns = quotes(stocks, date1)
cov_matrix = np.matrix(returns.cov() * 12)
return_vect = np.matrix(returns.loc[:date2].mean() * 12)



allocations, stddev, allocationsmin, stddevmin = calc_min_variance_portfolio(return_vect, cov_matrix, target_return)
df = pd.DataFrame(allocations, index=stocks, columns=['Weight'])
dfmin = pd.DataFrame(allocationsmin, index=stocks, columns=['Weight'])


print("-" * 40)
print("Scenario 1 - Optimized Portfolio for target return\n")
print("Min variance portfolio:")
print(df)
print("\nTarget Return: {:.2f}%".format(target_return * 100.0))
print("Portfolio std deviation: %.2f%%" % (stddev * 100.0))
print("-" * 40)
print("Scenario 2 - GMV Portfolio\n")
print("Global Minimum Variance:")
print(dfmin)
print("\nGMV Portfolio Return: %.2f%%" % (return_vect * allocationsmin * 100))
print("GMV Portfolio std deviation: %.2f%%" % (stddevmin * 100.0))
print("-" * 40)
print("Scenario 3 - Out-of-sample results")
print("\nPortfolio Return: %.2f%%\n" % (calc_outsample_portfolio(returns, allocations, date2) * 100))
print("-" * 40)

