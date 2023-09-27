import numpy as np
import numba as nb

from GEModelTools import lag, lead

# lags and leads of unknowns and shocks
# K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
# K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]


@nb.njit
def production_firm(par,ini,ss,Gamma,K,L1,L2,rK,w1,w2,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*Gamma*K_lag**(par.alpha-1.0)*L1**((1.0-par.alpha)/2.0)*L2**((1.0-par.alpha)/2.0)
    w1[:] = Gamma*(K_lag)**par.alpha*((1.0-par.alpha)/2.0)*L1**(-(1.0+par.alpha)/2.0)*L2**((1.0-par.alpha)/2.0)
    w2[:] = Gamma*(K_lag)**par.alpha*L1**((1.0-par.alpha)/2.0)*((1.0-par.alpha)/2.0)*L2**(-(1.0+par.alpha)/2.0)
    
    # b. production and investment
    Y[:] = Gamma*K_lag**(par.alpha)*L1**((1.0-par.alpha)/2.0)*L2**((1.0-par.alpha)/2.0)


@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L1,L1_hh,L2,L2_hh,Y,C_hh,K,I,clearing_A,clearing_L1,clearing_L2,clearing_Y):

    clearing_A[:] = K-A_hh #Asset market clearing
    clearing_L1[:] = L1-L1_hh #Labor market clearing for labor type 1
    clearing_L2[:] = L2-L2_hh #Labor market clearing for labor type 2
    I = K-(1-par.delta)*lag(ini.K,K) #Law of motion for capital
    clearing_Y[:] = Y-C_hh-I #Goods market clearing
