import time
import numpy as np
import numba as nb
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):

        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    for i_fix in range(par.Nfix):
        
        # a. raw value
        ell = 1.0
        y = ss.wt*ell*par.z_grid+ss.chi
        c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
        v_a = (1+ss.r)*c**(-par.sigma)

        # b. expectation
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a

def obj_ss(x,model,do_print=False):

    KL = x[0]

    par = model.par
    ss = model.ss

    # a. firms
    par.Gamma_Y
    ss.rK = par.alpha*par.Gamma_Y*(KL)**(par.alpha-1)
    ss.w = (1.0-par.alpha)*par.Gamma_Y*(KL)**par.alpha

    # b. arbitrage
    ss.r = ss.rK - par.delta

    # c. government
    ss.tau = par.tau_ss
    ss.chi = par.chi_ss

    # d. households
    ss.wt = (1-ss.tau)*ss.w
    
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # e. market clearing
    ss.Lg = (ss.L_hh * ss.w*ss.tau-ss.chi) / (ss.w+par.Gamma_G)
    ss.G = par.Gamma_G*ss.Lg  
    ss.B = 0.0
    ss.L = ss.L_hh - ss.Lg
    ss.K = KL*ss.L
    ss.Y = par.Gamma_Y*ss.K**(par.alpha)*ss.L**(1-par.alpha)
    ss.I = par.delta*ss.K
    ss.A = ss.K + ss.B
    ss.clearing_A = ss.A - ss.A_hh
    ss.clearing_L = ss.L + ss.Lg - ss.L_hh
    ss.clearing_Y = ss.Y - (ss.C_hh+ss.I+ss.G)
    ss.clearing_G = ss.G + ss.w*ss.Lg + ss.chi - ss.tau*ss.w*ss.L_hh

    return ss.clearing_A

def find_ss(model,do_print=False):
    """ find the steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss
    
    KL_min = ((1/par.beta+par.delta-1)/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1)) + 1e-2
    KL_max = (par.delta/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1))-1e-2
    KL_mid = (KL_min+KL_max)/2 # middle point between max values as initial capital labor ratio

    # a. solve for K and L
    initial_guess =  np.array([KL_mid])
    if do_print: print(f'starting at [{initial_guess[0]:.4f}]')

    res = optimize.root(obj_ss, initial_guess, args=(model,))
    if do_print: 
        print('')
        print(res)
        print('')
    
    # b. final evaluations
    obj_ss(res.x,model)

    # c. show
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.K = :6.3f}')
        #print(f'{ss.B = :6.3f}')
        print(f'{ss.A_hh = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.C_hh = :6.3f}')
        print(f'{ss.G = :6.3f}')
        print(f'{ss.I = :6.3f}')
        print(f'{ss.L = :6.3f}')
        print(f'{ss.Lg = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.wt = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_L = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')
        print(f'{ss.clearing_G = :.2e}')


def optimize_social_welfare(model,tau_guess,chi_guess=np.NaN,do_print=False):
    """ optimizer for social welfare based on taxes and chi"""
    par = model.par
    ss = model.ss
    # a. guess
    tau = tau_guess
    if np.isnan(chi_guess): #setting chi to 0 if no guess is given
        chi = 0.0
    else:
        chi = chi_guess
    
    # b. setup objective function
    def obj(tau,chi,model):
        """ objective function for social welfare maximization """

        par = model.par
        ss = model.ss

        par.tau_ss = tau
        par.chi_ss = chi
    
        model.find_ss()
        return -np.sum([par.beta**t * ((np.sum((ss.u+(ss.G+par.S)**(1-par.omega)/(1-par.omega)) * ss.D / (np.sum(ss.D))))) for t in range(par.T)])
    
    # c. solve
    t0 = time.time()
    if np.isnan(chi_guess): #implicitly it assumes that when no guess is given for chi, we are only optimizing for tau
        res = optimize.minimize_scalar(obj,args=(chi,model),bounds=[0.0,0.99],method='bounded')
        par.tau_ss = res.x
    else:
        res = optimize.minimize(lambda x: obj(x[0],x[1],model),x0=[tau,chi],method='Nelder-Mead')
        par.tau_ss = res.x[0]
        par.chi_ss = res.x[1]
    
    # d. final evaluation
    model.find_ss()

    # e. print
    print(f'Optimal taxes found in {elapsed(t0)}')
    if do_print:
        #print tau and chi values
        print(f'Optimal wage tax: {ss.tau = :6.4f}')
        print(f'Optimal lump sum transfer: {ss.chi = :6.4f}')
        #print ss values
        print('')
        print('Steady state values with optimized tax levels')
        print(f'{ss.K = :6.3f}')
        print(f'{ss.A_hh = :6.3f}')
        #print(f'{ss.B = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.C_hh = :6.3f}')
        print(f'{ss.G = :6.3f}')
        print(f'{ss.I = :6.3f}')
        print(f'{ss.L = :6.3f}')
        print(f'{ss.Lg = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.wt = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_L = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')
        print(f'{ss.clearing_G = :.2e}')

    return par.tau_ss, par.chi_ss
        


