# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:29:48 2024

@author: Yehonathan Barda
"""

#imports 
import numpy as np
from sim import Simulation
from scipy.constants import hbar



'''
PART A: 
'''


def run_part_A(trap_omega, Nbids):
    beta_list = np.array([6]) / (hbar * trap_omega)
    params = {'omega': trap_omega}
    dt = 0.1E-15
    
    Initial_pos =  np.zeros((Nbids, 1))


    for beta in beta_list:
        xyz_file = "results2\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.xyz'
        energy_file = "results2\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, \
                           outname= energy_file ,R = Initial_pos,\
                              Nsteps=10000,printfreq=1, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids, beta=beta)
        mysim.sampleMB(beta, removeCM = True)
        mysim.run(**params)


'''
PART B - Beads
'''

def run_part_Beads(trap_omega, beta): # Beads
    Nbids_list = np.arange(2, 101, 4)
    params = {'omega': trap_omega}
    dt = 0.1E-15
    
    for Nbids in Nbids_list:
        Initial_pos = np.zeros((Nbids, 1))

        xyz_file = "results2\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.xyz'
        energy_file = "results2\A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, seed = 134892987, \
                           outname= energy_file ,R = Initial_pos, \
                              Nsteps=10000,printfreq=10, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids, beta=beta)
        mysim.sampleMB(beta, removeCM = False)
        mysim.run(**params)
        print('Done: number of beads=' + str(Nbids))

'''
Part C - Temperature
'''
def run_part_C(n):
    N6 = 70 # Number of beads for n = 6
    beta = n / (hbar * trap_omega)
    Nbids = int(np.round(N6 * 6 / n / 2) * 2) # Number of beads for n, rounded to the nearest even number
    params = {'omega': trap_omega}
    dt = 0.1E-15
    
    Initial_pos = np.zeros((Nbids, 1))

    seed_list = [134892987, 134892988, 134892989]

    for j, seed in enumerate(seed_list):
        xyz_file = "results2\A beta = {:.1e} bids = {:} run = {:}".format(beta, Nbids, j) + '.xyz'
        energy_file = "results2\A beta = {:.1e} bids = {:} run = {:}".format(beta, Nbids,j) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, seed= seed, \
                           outname= energy_file ,R = Initial_pos, \
                              Nsteps=10000,printfreq=10, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids, beta=beta)
        mysim.sampleMB(beta, removeCM = True)
        mysim.run(**params)


if __name__ == "__main__":
    # Constants
    trap_omega = 50 * 1.602176634E-22 / hbar # from meV to J

    # for n in range(1,4):
    #     run_part_C(n)
    #     print('Done: ' + str(n))

    run_part_Beads(trap_omega, beta = 6 / (hbar * trap_omega))
    print('Done')