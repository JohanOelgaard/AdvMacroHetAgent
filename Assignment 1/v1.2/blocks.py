import numpy as np
import numba as nb

from GEModelTools import lag, lead

# lags and leads of unknowns and shocks
# K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
# K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]


@nb.njit
def production_firm(par,ini,ss,Gamma,K,phi1,L0,L1,rK,w0,w1,Y):

    K_lag = lag(ini.K,K)
    L1 = par.chi1*phi1
    Gamma = par.Gamma_ss

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha * Gamma * K_lag**(par.alpha-1.0) *L0**((1.0-par.alpha)/2.0) * L1**((1.0-par.alpha)/2.0)
    w0[:] = Gamma * (K_lag)**par.alpha * ((1.0-par.alpha)/2.0) * L0**(((1.0-par.alpha)/2.0)-1) * L1**((1.0-par.alpha)/2.0)
    w1[:] = Gamma * (K_lag)**par.alpha * ((1.0-par.alpha)/2.0) * L1**(((1.0-par.alpha)/2.0)-1) * L0**((1.0-par.alpha)/2.0)
    
    # b. production and investment
    Y[:] = Gamma * K_lag**(par.alpha) * L0**((1.0-par.alpha)/2.0) * L1**((1.0-par.alpha)/2.0)

# @nb.njit
# def mutual_fund(par,ini,ss,K,rK,A,r):
#     # a. total assets
#     A[:] = K
#     # b. return
#     r[:] = rK - par.delta

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L0,L0_hh,L1,L1_hh,Y,C_hh,K,I,clearing_A,clearing_L0,clearing_L1,clearing_Y):
    #total assets
    A[:] = K

    # clearing
    clearing_A[:] = K-A_hh #Asset market clearing
    clearing_L0[:] = L0-L0_hh #Labor market clearing for labor type 1
    clearing_L1[:] = L1-L1_hh #Labor market clearing for labor type 2
    I = K-(1-par.delta)*lag(ini.K,K) #Law of motion for capital
    clearing_Y[:] = Y-C_hh-I #Goods market clearing
