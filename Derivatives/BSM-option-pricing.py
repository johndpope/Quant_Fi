from scipy import stats
import math
import time
import numpy as np

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
    print(d1)
    print(d2)
    
    optprice = cp * S * math.exp(-div*T)*stats.norm.cdf(cp*d1) - cp*K*math.exp(-r*T)*stats.norm.cdf(cp*d2)
    print("BS- The Option Price is: {0:.4f}".format(optprice))


def Weiner(S,R,Vol,T,X):
    return S * np.exp( (R- (Vol**2) / 2)*T + Vol*math.sqrt(T) * X)

    
def Monte_Carlo_Vanilla_Vectorization(S_0,K,R,Vol,T,cp, N_simu):
    X = np.random.randn(N_simu)
    assets_final_value = Weiner(S_0,R,Vol,T,X)
    payoffs = np.maximum(cp*(assets_final_value-K), 0)
 
    print("MC- Option Price is: {0:.4f}".format(np.mean(payoffs)*math.exp(-R*T)))

def Monte_Carlo_Digital_Vectorization(S_0,K,R,Vol,T, cp, N_simu):
    X = np.random.randn(N_simu)
    assets_final_value = Weiner(S_0,R,Vol,T,X)
    payoffs = np.where(cp*(assets_final_value-K) > 0, 1, 0)
 
    print("MC- Digital Option Price is: {0:.4f}".format(np.mean(payoffs)*math.exp(-R*T)))

S_0 = 50
K = 50
R = 0.1
T = 0.4167
Vol = 0.4
div = 0
cp = 1
N_simu = 10000000

BSM(S_0, K, T, Vol, R, div, cp)
Monte_Carlo_Vanilla_Vectorization(S_0,K,R,Vol,T,cp, N_simu)
Monte_Carlo_Digital_Vectorization(S_0,K,R,Vol,T,cp, N_simu)

