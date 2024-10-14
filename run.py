# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:29:48 2024

@author: Yehonathan Barda
"""

#imports 
import numpy as np
import pandas as pd
from sim import Simulation
from scipy.constants import hbar



'''
PART A: 
'''


def run_part_A(trap_omega, Nbids):
    beta_list = np.array([6]) / (hbar * trap_omega)
    params = {'omega': trap_omega}
    dt = 0.1E-15
    
    Initial_pos_A = np.random.uniform(-1, 1, size=(Nbids, 1)) * 1E-10
    Initial_mom_A = np.zeros((Nbids, 1))


    for beta in beta_list:
        xyz_file = "results\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.xyz'
        energy_file = "results\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, \
                           outname= energy_file ,R = Initial_pos_A, p = Initial_mom_A,\
                              Nsteps=10000,printfreq=1, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids, beta=beta)
        mysim.run(**params)


'''
PART B - Beads
'''

def run_part_Beads(trap_omega, beta): # Beads
    Nbids_list = np.arange(2, 101, 2)
    params = {'omega': trap_omega}
    dt = 0.1E-15
    
    for Nbids in Nbids_list:
        Initial_pos_A = np.random.uniform(-1, 1, size=(Nbids, 1)) * 1E-10
        Initial_mom_A = np.zeros((Nbids, 1))

        xyz_file = "results\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.xyz'
        energy_file = "results\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, seed = 134892987, \
                           outname= energy_file ,R = Initial_pos_A, p = Initial_mom_A,\
                              Nsteps=10000,printfreq=10, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids, beta=beta)
        mysim.run(**params)

'''
Part C - Temperature
'''
def run_part_A(n):
    beta = n / (hbar * trap_omega)
    Nbids = 10 * n
    params = {'omega': trap_omega}
    dt = 0.1E-15
    
    Initial_pos_A = np.random.uniform(-1, 1, size=(Nbids, 1)) * 1E-10
    Initial_mom_A = np.zeros((Nbids, 1))

    seed_list = [134892987, 134892988, 134892989]

    for j, seed in enumerate(seed_list):
        xyz_file = "results\A beta = {:.1e} bids = {:} run = {j}".format(beta, Nbids) + '.xyz'
        energy_file = "results\A beta = {:.1e} bids = {:} run = {j}".format(beta, Nbids) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, seed= seed, \
                           outname= energy_file ,R = Initial_pos_A, p = Initial_mom_A,\
                              Nsteps=10000,printfreq=1, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids, beta=beta)
        mysim.run(**params)

## change

if __name__ == "__main__":
    # Constants
    trap_omega = 50 * 1.602176634E-22 / hbar # from meV to J

    run_part_A(trap_omega, Nbids = 10)

    # run_part_Beads(trap_omega, beta = 6 / (hbar * trap_omega))
    print('Done')