import numpy as np
import numba as nb

from GEModelTools import prev,next

import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,K,L,rK,w,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*par.Gamma_Y*(K_lag/L)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*par.Gamma_Y*(K_lag/L)**par.alpha
    
    # b. production and investment
    Y[:] = par.Gamma_Y*K_lag**(par.alpha)*L**(1-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K + ss.B

    # b. return
    r[:] = rK-par.delta

@nb.njit
def government(par,ini,ss,B,G,tau,Lg,L,w,wt):
 
    tau[:] = ss.tau
    B[:] = ss.B
    Lg[:] = (tau*w*L-par.chi)/(par.Gamma_G+w-tau*w)
    G[:] = par.Gamma_G*Lg
    wt[:] = (1-tau)*w
    

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L,Lg,L_hh,Y,C_hh,K,I,clearing_A,clearing_L,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L[:] = L+Lg-L_hh
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I