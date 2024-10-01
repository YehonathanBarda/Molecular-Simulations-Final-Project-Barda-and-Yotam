# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:24:36 2022

@author: hirshb
"""

#imports
import numpy as np
import pandas as pd
from sim import Simulation
from scipy.constants import hbar

"""
HERE, TO RUN THE SIMULATION, YOU WILL NEED TO DO THE FOLLOWING THINGS:
    
    1. CREATE AN OBJECT OF THE SIMULATION CLASS. A MINIMAL EXAMPLE IS
        >>> mysim = Simulation( dt=0.1E-15, L=11.3E-10, ftype="LJ" )
    
    2. DEFINE THE PARAMETERS FOR THE POTENTIAL. USE A DICTIONARY, FOR EXAMPLE
    FOR THE LJ MODEL OF ARGON, IN SI UNITS:
        >>> params = { "eps":  1.656778224E-21, "sig": 3.4E-10 }
    
THEN, CALLING THE METHODS YOU IMPLEMENTED IN sim.py, YOU NEED TO
    3. READ THE INITIAL XYZ FILE PROVIDED OR SAMPLE INITIAL COORDINATES.
    4. SAMPLE INITIAL MOMENTA FROM MB DISTRIBUTION.
    5. REMOVE COM MOTION.
    6. RUN THE SIMULATION, INCLUDING PRINTING XYZ AND ENERGIES TO FILES.

THE SPECIFIC SIMULATIONS YOU NEED TO RUN, AND THE QUESTIONS YOU NEED TO ANSWER,
ARE DEFINED IN THE COURSE NOTES.
    
NOTE THAT TO CALL A METHOD OF A CLASS FOR THE OBJECT mysim, THE SYNTAX IS
    >>> mysim.funcName( args )
    
FINALLY, YOU SHOULD
    7. ANALYZE YOUR RESULTS. THE SPECIFIC GRAPHS TO PLOT ARE EXPLAINED IN THE
    COURSE NOTES.

NOTE: THE INPUT FILE GIVEN HAS THE COORDINATES IN ANGSTROM.
*YOUR OUTPUT XYZ FILE SHOULD BE PRINTED IN ANGSTROM TOO*, BUT YOU CAN USE
ANY UNITS YOU WANT IN BETWEEN, I SUGGEST USING SI UNITS.

NOTE: YOU WILL BE ASKED TO IMPLEMENT DIFFERENT METHODS TO EVALUATE THE FORCES,
e.g., evalLJ, evalHarm etc. HOWEVER, WHEN YOU WISH YOU CALL THEM, CALL mysim.evalForce().

IT WILL TAKE CARE AUTOMATICALLY, TO CALL THE WRITE METHOD BASED ON mysim.ftype.

"""

'''
PART A:
'''
dt_list = np.arange(1,2) * 1E-16
if False:
    Initial_pos_A = np.array([[1,0,0]]) * 5E-10  # 5 Angstrom
    Initial_mom_A = np.zeros((1,3))
    trap_omega = 50 * 1.602176634E-22 / hbar # from meV to J
    params = {'omega': trap_omega}
    for dt in dt_list:
        xyz_file = 'A dt = ' + "{:.1e}".format(dt) + '.xyz'
        energy_file = 'A dt = ' + "{:.1e}".format(dt) + '.erg'
        mysim = Simulation( dt=dt, L=11.3E-10, ftype="Harm",  xyzname = xyz_file, \
                           outname= energy_file ,R = Initial_pos_A, p = Initial_mom_A,\
                              Nsteps=10000,printfreq=10, K = 0, mass = np.array([[6.6335209E-26]]), kind = ['Ar'])
        mysim.run(**params)


'''
PART B
'''
if True:
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

## errrwerer