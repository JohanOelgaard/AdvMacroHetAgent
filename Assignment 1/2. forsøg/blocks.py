import numpy as np
import numba as nb

from GEModelTools import lag, lead

# lags and leads of unknowns and shocks
# K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
# K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]


@nb.njit
def production_firm(par,ini,ss,K, phi_0, phi_1 ,rK,w0, w1,Y,L):
    K_lag = lag(ini.K,K)
    Gamma = par.Gamma_ss
    L0 = 2/3 * phi_0
    L1 = 1/3 * phi_1

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha* Gamma *K_lag**(par.alpha-1)*L0**((1-par.alpha)/2)*L1**((1-par.alpha)/2)
    w0[:] = (1.0-par.alpha)/2* Gamma *K_lag**par.alpha*L1**((1-par.alpha)/2)*L0**((1-par.alpha)/2-1)
    w1[:] = (1.0-par.alpha)/2* Gamma *K_lag**par.alpha*L0**((1-par.alpha)/2)*L1**((1-par.alpha)/2-1)
 
    # b. production and investment
    Y[:] = Gamma *K_lag**(par.alpha)*L0**((1-par.alpha)/2)*L1**((1-par.alpha)/2)
    L[:] = L0+L1
    
# @nb.njit
# def mutual_fund(par,ini,ss,K,rK,A,r):

#     # a. total assets
#     A[:] = K

#     # b. return
#     r[:] = rK-par.delta

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L,L_hh,Y,C_hh,K,I,clearing_A,clearing_L,clearing_Y):
    # a. total assets
    A[:] = K

    # b. clearing
    clearing_A[:] = A-A_hh
    clearing_L[:] = L-L_hh
    I = K-(1.0-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I
