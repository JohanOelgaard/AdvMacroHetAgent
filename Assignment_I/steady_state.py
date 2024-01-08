import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,np.max([ss.w0,ss.w1])*par.a_max,par.Na)
    # par.a_grid[:] = equilogspace(0.0,np.max([ss.w0,ss.w1])*par.a_max,par.Na)

    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    # c. beta
    par.beta_grid[:] = np.tile(np.array([par.beta_mean-par.sigma_beta,par.beta_mean,par.beta_mean+par.sigma_beta]),2)

    # e. eta
    par.eta0_grid[:] = np.repeat(np.array([1.0,0.0]),3) # grid for labor type 0
    par.eta1_grid[:] = np.repeat(np.array([0.0,1.0]),3) # grid for labor type 1

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    for i_fix in range(par.Nfix):
        if i_fix < 3:
            ss.z_trans[i_fix,:,:] = z_trans
            ss.Dbeg[i_fix,:,0] = (2*z_ergodic/9) #par.Nfix # ergodic at a_lag = 0.0
            ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0
        else:
            ss.z_trans[i_fix,:,:] = z_trans
            ss.Dbeg[i_fix,:,0] = (z_ergodic/9) #par.Nfix # ergodic at a_lag = 0.0
            ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0
            
    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = par.z_grid
    c = m = (1+ss.rK-par.delta) * par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    u = c**(1-par.sigma)/(1-par.sigma) - par.nu
    v_a = (1+ss.rK-par.delta) * c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans @ v_a
    
def obj_ss(K_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. production
    ss.A = ss.K = K_ss
    ss.L0 = par.chi0*ss.phi0 # by distribution
    ss.L1 = par.chi1*ss.phi1 # by distribution
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L0**((1-par.alpha)/2)*ss.L1**((1-par.alpha)/2)    

    # b. implied prices
    ss.rK = par.alpha*ss.Gamma*ss.K**(par.alpha-1.0)*ss.L0**((1-par.alpha)/2)*ss.L1**((1-par.alpha)/2)    
    ss.w0 = ss.Gamma*(ss.K)**par.alpha*((1.0-par.alpha)/2.0)*ss.L0**(((1.0-par.alpha)/2.0)-1)*ss.L1**((1.0-par.alpha)/2.0)
    ss.w1 = ss.Gamma*(ss.K)**par.alpha*((1.0-par.alpha)/2.0)*ss.L1**(((1.0-par.alpha)/2.0)-1)*ss.L0**((1.0-par.alpha)/2.0)

    # c. household behavior
    if do_print:

        print(f'guess {ss.K = :.4f}')    
        print(f'implied {ss.rK = :.4f}')
        print(f'implied {ss.w0 = :.4f}')
        print(f'implied {ss.w1 = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. market clearing
    ss.clearing_A = ss.A - ss.A_hh
    ss.clearing_L0 = ss.L0-ss.L0_hh
    ss.clearing_L1 = ss.L1-ss.L1_hh
    ss.I = ss.K - (1-par.delta)*ss.K
    ss.clearing_Y = ss.Y - ss.C_hh - ss.I

    return ss.clearing_A # target to hit

def find_ss(model,method='direct',do_print=False,K_min=1.0,K_max=10.0,NK=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,K_min=K_min,K_max=K_max,NK=NK)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')
def find_ss_direct(model,do_print=False,K_min=1.0,K_max=10.0,NK=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    K_ss_vec = np.linspace(K_min,K_max,NK) # trial values
    clearing_A = np.zeros(K_ss_vec.size) # asset market errors

    for i,K_ss in enumerate(K_ss_vec):
        
        try:
            clearing_A[i] = obj_ss(K_ss,model,do_print=do_print)
        except Exception as e:
            clearing_A[i] = np.nan
            if do_print: print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    K_min = np.max(K_ss_vec[clearing_A < 0])
    K_max = np.min(K_ss_vec[clearing_A > 0])

    if do_print: print(f'K in [{K_min:12.8f},{K_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,K_min,K_max,args=(model,),do_print=do_print,
        varname='K_ss',funcname='A-A_hh'
    )
