import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w0','w1','phi0','phi1'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c','l0','l1'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['phi1'] # exogenous shocks
        self.unknowns = ['K','L0','L1'] # endogenous unknowns
        # self.unknowns = ['L0', 'L1'] # endogenous unknowns (not used today)
        self.targets = ['clearing_A'] # targets = 0 (not used today)
        # self.targets = ['clearing_L1'] # targets = 0 (not used today)

        # d. all variables
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'hh', # household block
            'blocks.market_clearing']

        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 6 # number of fixed discrete states (none here)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta_mean = 0.975 # discount factor
        par.sigma_beta = 0.010 # half width of beta grid
        par.sigma = 2.0 # CRRA coefficient
        par.nu = 0.5 # disutility of labor factor NB not relevant
        par.epsilon = 1.0 # disutility of labor factor NB not relevant

        # b. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # c. labour productivity
        par.phi0 = 1.0 # productivity of type 0
        par.phi1 = 2.0 # productivity of type 1

        # d. production
        par.Gamma_ss = 1.0 # production function scale parameter
        par.alpha = 0.36 # capital share
        par.delta = 0.10 # depreciation rate

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem

        # h. for transition path
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

        # i. initial distribution
        par.chi0 = 2/3
        par.chi1 = 1/3
          
    def allocate(self):
        """ allocate model """

        par = self.par

        self.allocate_GE() # should always be called here

        # a. grids
        par.Nbeta = par.Nfix
        par.beta_grid = np.zeros(par.Nbeta)
        par.eta0_grid = np.zeros(par.Nbeta)
        par.eta1_grid = np.zeros(par.Nbeta)

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss