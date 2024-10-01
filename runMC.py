# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bardalas
"""

#imports
import numpy as np
import pandas as pd
from sim import Simulation
from scipy.constants import hbar, Boltzmann
import os
import time
t0 = time.time()

def sec_to_hms(sec):
    hours, remainder = divmod(sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

'''
PART A:
'''
if False:
    dr_max_list = np.array([10,6,4,2,1.5,1,0.7,0.5,0.1,0.001,0.0001]) * 1E-11 #np.linspace(0.001,10,10) * 1E-11 #np.array([0.2,0.15, 0.1,0.08,0.05,0.01, 0.005]) * 1E-10
    acc_file = open('AII_acc_ratio.log', 'w')
    acc_file.write("Delta ratio\n")
    trap_omega = 50 * 1.602176634E-22 / hbar # from meV to J
    params = {'omega': trap_omega}
    for delta in dr_max_list:
        Initial_pos_A = np.array([[1,0,0]]) * 5E-10  # 5 Angstrom
        xyz_file = 'AII Δ=' + "{:.1e}".format(delta) + '.xyz'
        energy_file = 'AII Δ=' + "{:.1e}".format(delta) + '.log'
        mysim =  Simulation(dt = 1,L=11.3E-10, ftype="Harm",  xyzname = xyz_file, \
                           outname= energy_file ,R = Initial_pos_A, drmax = delta, Temp = 298,PBC = False, \
                              Nsteps=10000,printfreq=10, K = 0, mass = np.array([[6.6335209E-26]]), kind = ['Ar'])
        mysim.runMC(**params)
        acc_file.write('{} {}\n'.format(delta,mysim.accept))
        print('for Δ={:.1e}    |   accepted ratio={:.0f}%'.format(delta,mysim.accept * 100))
    acc_file.close()


'''
PART B:
'''

if False:
    dr_max_list = np.array([100,20,10,6,4,2,1.77,1.3,1,0.7,0.5,0.0001]) * 1E-11 
    acc_file = open('BII_acc_ratio.log', 'w')
    acc_file.write("Delta ratio\n")
    Lambda = 7.757E23
    params = {'Lambda': Lambda}
    for delta in dr_max_list:
        Initial_pos_B = np.array([[1,0,0]]) * 0.5E-10  # 5 Angstrom
        xyz_file = 'BII Δ=' + "{:.1e}".format(delta) + '.xyz'
        energy_file = 'BII Δ=' + "{:.1e}".format(delta) + '.log'
        mysim =  Simulation(dt = 1,L=11.3E-10, ftype="Anharm",  xyzname = xyz_file, \
                           outname= energy_file ,R = Initial_pos_B, drmax = delta, Temp = 298,PBC = False, \
                              Nsteps=10000,printfreq=10, K = 0, mass = np.array([[6.6335209E-26]]), kind = ['Ar'])
        mysim.runMC(**params)
        acc_file.write('{} {}\n'.format(delta,mysim.accept))
        print('for Δ={:.1e}    |   accepted ratio={:.0f}%'.format(delta,mysim.accept * 100))
    acc_file.close()


'''
PART C
'''
if True:    
    dr_max_list = np.array([2.8 *5]) *1E-11 # np.array([20,15,10,8,5,4,3.1,2.9,2.8,2,1.2,0.5]) * 1E-11
    sim_type = 113 # 0 for 298K 22.6A every atom, 2 for 298K 22.6A random, 40 for 40K and 22.6A evry atom, 42 for 40K and 22.6A random, 113 for 113A and 298K
    acc_file = open('CII_acc_ratio.log', 'a') 
    if os.path.getsize('CII_acc_ratio.log') == 0:
        acc_file.write("Delta ratio type time Nsteps Temp L\n")
    params = {'eps': 1.656778224E-21, 'sig': 3.4E-10}

    for delta in dr_max_list:
        # xyz_file = 'C2II Δ=' + "{:.1e}".format(delta) + '.xyz'
        # energy_file = 'C2II Δ=' + "{:.1e}".format(delta) + '.log' 
        xyz_file = 'CII L=113.xyz'
        energy_file = 'CII L=113.log'

        mysim =  Simulation(dt = 1,L=113E-10, ftype="LJ",  xyzname = xyz_file,  Nsteps=10000, \
                           outname= energy_file ,R = np.zeros(( 256,3)), drmax = delta, Temp = 298,PBC = True, \
                            printfreq=100, K = 0, mass = np.array([[6.6335209E-26]]), kind = ['Ar'])
        mysim.readXYZ('Ar_init_super.xyz')
        mysim.R *= 1E-10 * 5# *5 for 113
        mysim.mass = np.ones((256,1)) * 6.6335209E-26
        
        start_time = time.time()

        mysim.runMC(**params)

        end_time = time.time()
        elapsed_time = sec_to_hms(end_time - start_time)
        acc_file.write('{} {} {} {}:{}:{} {} {} {}\n'.format(delta,mysim.accept, sim_type, *elapsed_time, mysim.Nsteps,298, mysim.L))

        print('for Δ={:.1e}    |   accepted ratio={:.0f}%   |   Time taken: {:02}:{:02}:{:02}'.format(delta, mysim.accept * 100, *elapsed_time))
    acc_file.close()

t_end = time.time()
elapsed_time = t_end - t0
print('Done. time: {:02}:{:02}:{:02}'.format(*sec_to_hms(elapsed_time))) 