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

def run_part_A(trap_omega, Nbids = 3):
    beta_list = np.array([1]) / (hbar * trap_omega)
    Initial_pos_A = np.array([[i] for i in range(Nbids)]) * 1E-10
    Initial_mom_A = np.zeros((Nbids, 3))
    params = {'omega': trap_omega}
    dt = 0.1E-15
    for beta in beta_list:
        xyz_file = "A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.xyz'
        energy_file = "A beta = {:.1e} bids = {:}".format(beta, Nbids) + '.erg'
        mysim = Simulation( dt=dt, ftype="Harm",  xyzname = xyz_file, \
                           outname= energy_file ,R = Initial_pos_A, p = Initial_mom_A,\
                              Nsteps=10000,printfreq=10, K = 0, mass = 6.6335209E-26, kind = ['Ar'] * Nbids)
        mysim.run(**params)


'''
PART B
'''
def run_part_B(trap_omega):
    dt = 10E-15
    mysim = Simulation( dt = dt, L = 22.6E-10, ftype="LJ", xyzname = 'PartB T=40.pd' , \
                       outname='PartB T=40.erg', Nsteps=10000, printfreq=100, K=0, kind=['Ar'] * 256,\
                       R = np.zeros((256,3)), mass =6.6335209E-26,   PBC=True)
    mysim.readXYZ('Ar_init_super.xyz')
    mysim.R *= 1E-10
    mysim.mass = np.ones((256,1)) * 6.6335209E-26
    mysim.sampleMB(40)
    params = {'eps': 1.656778224E-21, 'sig': 3.4E-10}
    mysim.run(**params)
    print('Done')

## change

if __name__ == "__main__":
    # Constants
    trap_omega = 50 * 1.602176634E-22 / hbar # from meV to J

    run_part_A(trap_omega, Nbids=3)
