from scipy import stats
import math
import time

def BSM(S, K, T, V, r, div, cp):
    """
    S : S0 initial stock price
    K : Strike
    T : Exp time in years
    V : Volatility (sigma)
    r : Risk free rate
    div : Dividend or not
    cp : +1 / -1 for call / put
    """
    d1 = (math.log(S/K)+(r-div+0.5*math.pow(V,2))*T) / (V*math.sqrt(T))
    d2 = d1 - V*math.sqrt(T)
    
    optprice = cp * S * math.exp(-div*T)*stats.norm.cdf(cp*d1) - cp*K*math.exp(-r*T)*stats.norm.cdf(cp*d2)
    return optprice




tic = time.time()
Opt = BSM(50,50,0.4167,0.4,0.1,0,-1)
tac = time.time()
print("The Option Price is: {0:.4f}".format(Opt))
print("{:.8f}ms".format((tac-tic)*1000))