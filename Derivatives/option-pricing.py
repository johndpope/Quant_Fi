#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:39:00 2018

@author: nicob
"""
import time
import math
import numpy as np
from scipy import stats

class EuropeanOption(object):
    def __init__(self, S0, K, r, T, sigma, is_call):
        self.S0 = S0 # Stock index at time 0
        self.K = K # Strike price
        self.r = r # Riskfree rate
        self.T = T # Time maturity, in years
        self.sigma = sigma # Volatility
        self.is_call = is_call  # Call or put

class BlackScholes(EuropeanOption):
    def __init__(self, S0, K, r, T, sigma, div, is_call):
        super().__init__(S0, K, r, T, sigma, is_call)
        self.div = div

    def __setup_parameters__(self):        
        d1 = ( math.log(self.S0 / self.K) + (self.r - self.div +0.5*math.pow(self.sigma,2)) *self.T) \
        / (self.sigma*math.sqrt(self.T))
        d2 = d1 - self.sigma*math.sqrt(self.T)
        self.Nd1 = stats.norm.cdf(self.is_call * d1)
        self.Nd2 = stats.norm.cdf(self.is_call * d2)
    
    def value(self):
        self.__setup_parameters__()
        value = self.is_call * self.S0 * math.exp(-self.div * self.T)*self.Nd1 \
                - self.is_call*self.K*math.exp(-self.r*self.T)*self.Nd2 
        return value

class MonteCarloVanilla(EuropeanOption):
    def __init__(self, S0, K, r, T, sigma, div, is_call, niter):
        super().__init__(S0, K, r, T, sigma, is_call)
        self.niter = niter
        
    def _termvalues_(self):
        """Vectorized Weiner Process"""
        X = X = np.random.randn(self.niter)
        return self.S0 * np.exp( (self.r- (self.sigma**2) / 2)*self.T \
                                + self.sigma *math.sqrt(self.T) * X)

    def _payoffs_(self):
        terminal_prices = self._termvalues_()
        return np.maximum(self.is_call*(terminal_prices-self.K), 0)
    
    def value(self):
        return np.mean(self._payoffs_())*math.exp(-self.r*self.T)
    
    
    
class MonteCarloDigitale(MonteCarloVanilla):
    def __init__(self, S0, K, r, T, sigma, div, is_call, niter):
        super().__init__(S0, K, r, T, sigma, div, is_call, niter)
        
    def _payoffs_(self):
        """Payoff of 1 if terminal price above K"""
        terminal_prices = self._termvalues_()
        return np.where(self.is_call*(terminal_prices-self.K) > 0, 1, 0) 
        
class MonteCarloAsian(MonteCarloVanilla):
    def __init__(self, S0, K, r, T, sigma, div, is_call, niter, nsteps):
        super().__init__(S0, K, r, T, sigma, div, is_call, niter)
        self.nsteps = nsteps
        self.STs = None
    
    def _termvalues_(self):
        """Calculate the arithmetic average price of the paths"""
        self.dt = self.T / self.nsteps
        self.STs = np.full((self.niter, self.nsteps), self.S0, dtype=np.float32)
        for i in range(self.nsteps):
            X = np.random.randn(self.niter)
            self.STs[:,i] = self.STs[:,i-1] * np.exp( (self.r- (self.sigma**2) / 2)*self.dt \
                    + self.sigma*np.sqrt(self.dt) * X)
        self.STs = self.STs.mean(axis=1)
        return self.STs
    
class BinomCRR(EuropeanOption):
    """Cox-Ross-Rubinstein European option valuation."""
    def __init__(self, S0, K, r, T, sigma, is_call, N):
        super().__init__(S0, K, r, T, sigma, is_call)
        self.N = max(1, N) # At least 1 period
        self.STs = None #Tree stock prices
        
    def __setup_parameters__(self):
        """ Required calculations for the model """
        self.dt = self.T/float(self.N)  # Single time step, in years
        self.df = math.exp(-(self.r) * self.dt)  # Discount factor
        self.M = self.N + 1  # Number of terminal nodes of tree
        self.u = math.exp(self.sigma * math.sqrt(self.dt)) # Up movement
        self.d = math.exp(-(self.sigma * math.sqrt(self.dt))) # Down movement
        self.pu = (math.exp((self.r)*self.dt) # Risk Neutral Probability of up-move
        - self.d) / (self.u-self.d) 
        self.pd = 1-self.pu

    def _initialize_stock_price_tree_(self):
        # Initialize terminal price nodes to zeros
        self.STs = np.zeros(self.M)
        # Calculate expected stock prices for each final node
        for i in range(self.M):
            self.STs[i] = self.S0*(self.u**(self.N-i))*(self.d**i)
 
    def _initialize_payoffs_tree_(self):
    # Get payoffs when the option expires at terminal nodes
        payoffs = np.maximum( 0, (self.STs-self.K) if self.is_call
        else(self.K-self.STs))
        return payoffs
    
    def _traverse_tree_(self, payoffs):
    # Starting from the time the option expires, traverse
    # backwards and calculate discounted payoffs at each node
        for i in range(self.N):
            payoffs = (payoffs[:-1] * self.pu + payoffs[1:] * self.pd) * self.df  
        return payoffs
    
    def __begin_tree_traversal__(self):
        payoffs = self._initialize_payoffs_tree_()
        return self._traverse_tree_(payoffs)

    def value(self):
        """ The pricing implementation """
        self.__setup_parameters__()
        self._initialize_stock_price_tree_()
        payoffs = self.__begin_tree_traversal__()
        return payoffs[0]  # Option value converges to first node
    
    def __str__(self):
        """Give description of the option when class is called"""
        return "Option: {}\nPrice: {}\nStrike: {}\nSigma: {:.2f}%".format(self.is_call, self.S0, self.K, self.sigma*100)



#Option = BinomCRR(50, 50, 0.1, 0.4167, 0.4, 1, 10000)
#Option = BlackScholes(50, 50, 0.1, 0.4167, 0.4, 0, 1)
#Option = MonteCarloVanilla(50, 50, 0.1, 0.4167, 0.4, 0, 1, 100000)
#Option = MonteCarloDigitale(50, 50, 0.1, 0.4167, 0.4, 0, 1, 100000)
#Option = MonteCarloAsian(40, 40, 0.05, 0.5, 0.4, 0, 1, 10000, 1000)

tic = time.time()
#print(Option.value())
tac = time.time()
print("Exec time: {:.10f}s".format((tac-tic)))