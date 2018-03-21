#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:39:00 2018

@author: nicob
"""
import time
import math
import numpy as np

class Option(object):
    def __init__(self, S0, K, r, T, N, params):
        self.S0 = S0 # Stock index at time 0
        self.K = K # Strike price
        self.r = r # Riskfree rate
        self.T = T # Time maturity, in years
        self.N = max(1, N) # At least 1 period
        self.STs = None #Tree stock prices
        
        """ Optional parameters used by derived classes """
        self.sigma = params.get("sigma", 0)  # Volatility
        self.is_call = params.get("is_call", True)  # Call or put
        
        """ Computed values """
        self.dt = T/float(N)  # Single time step, in years
        self.df = math.exp(-(r) * self.dt)  # Discount factor

        
class EuropeanCRR(Option):
    """Cox-Ross-Rubinstein European option valuation."""
    
    def __setup_parameters__(self):
        """ Required calculations for the model """
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

    def price(self):
        """ The pricing implementation """
        self.__setup_parameters__()
        self._initialize_stock_price_tree_()
        payoffs = self.__begin_tree_traversal__()
        return payoffs[0]  # Option value converges to first node
    
    def __str__(self):
        """Give description of the option when class is called"""
        return "Call Option: {}\nPrice: {}\nStrike: {}\nSigma: {:.2f}%".format(self.is_call, self.S0, self.K, self.sigma*100)



eu_opt = EuropeanCRR(50, 50, 0.1, 0.4167, 10000, {"sigma":0.4, "is_call": False})
tic = time.time()
print(eu_opt.price())
tac = time.time()
print("Exec time: {:.10f}s".format((tac-tic)))