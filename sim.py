#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  6 11:55:59 2022

@author: hirshb


WELCOME TO YOUR FIRST PROJECT! THIS BIT OF TEXT IS CALLED A DOCSTRING.
BELOW, I HAVE CREATED A CLASS CALLED "SIMULATION" FOR YOUR CONVENIENCE.
I HAVE ALSO IMPLEMENTED A CONSTRUCTOR, WHICH IS A METHOD THAT IS CALLED 
EVERY TIME YOU CREATE AN OBJECT OF THE CLASS USING, FOR EXAMPLE, 
    
    >>> mysim = Simulation( dt=0.1E-15, L=11.3E-10, ftype="LJ" )

I HAVE ALSO IMPLEMENTED SEVERAL USEFUL METHODS THAT YOU CAN CALL AND USE, 
BUT DO NOT EDIT THEM. THEY ARE: evalForce, dumpXYZ, dumpThermo and readXYZ.

YOU DO NOT NEED TO EDIT THE CLASS ITSELF. 

YOUR JOB IS TO IMPLEMENT THE LIST OF CLASS METHODS DEFINED BELOW WHERE YOU 
WILL SEE THE FOLLOWING TEXT: 

        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################

YOU ARE, HOWEVER, EXPECTED TO UNDERSTAND WHAT ARE THE MEMBERS OF THE CLASS
AND USE THEM IN YOUR IMPLEMENTATION. THEY ARE ALL EXPLAINED IN THE 
DOCSTRING OF THE CONSTRUCTOR BELOW. FOR EXAMPLE, WHENEVER YOU WISH 
TO USE/UPDATE THE MOMENTA OF THE PARTICLES IN ONE OF YOUR METHODS, YOU CAN
ACCESS IT BY USING self.p. 

#    >>> self.p = np.zeros( (self.Natoms,3) )
        
FINALLY, YOU WILL NEED TO EDIT THE run.py FILE WHICH RUNS THE SIMULATION.
SEE MORE INSTRUCTIONS THERE.

"""
################################################################
################## NO EDITING BELOW THIS LINE ##################
################################################################

#imports
import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as BOLTZMANN
import matplotlib.pyplot as plt
from scipy.constants import hbar

class Simulation:
    
    def __init__( self, dt, L, Nsteps=0, R=None, mass=None, kind=None, \
                 p=None, F=None, U=None, K=None, seed=937142, ftype=None, \
                 step=0, printfreq=1000, xyzname="sim.xyz", fac=1.0, \
                 outname="sim.log", debug=False, PBC=False, drmax = None, Temp = 273.15 ):
        """
        THIS IS THE CONSTRUCTOR. SEE DETAILED DESCRIPTION OF DATA MEMBERS
        BELOW. THE DESCRIPTION OF EACH METHOD IS GIVEN IN ITS DOCSTRING.

        Parameters
        ----------
        dt : float
            Simulation time step.
            
        L : float
            Simulation box side length.
            
        Nsteps : int, optional
            Number of steps to take. The default is 0.
            
        R : numpy.ndarray, optional
            Particles' positions, Natoms x 3 array. The default is None.
            
        mass : numpy.ndarray, optional
            Particles' masses, Natoms x 1 array. The default is None.
            
        kind : list of str, optional
            Natoms x 1 list with atom type for printing. The default is None.
            
        p : numpy.ndarray, optional
            Particles' momenta, Natoms x 3 array. The default is None.
            
        F : numpy.ndarray, optional
            Particles' forces, Natoms x 3 array. The default is None.
            
        U : float, optional
            Potential energy . The default is None.
            
        K : float, optional
            Kinetic energy. The default is None.
        
        E : float, optional
            Total energy. The default is None.
                    
        seed : int, optional
            Big number for reproducible random numbers. The default is 937142.
            
        ftype : str, optional
            String to call the force evaluation method. The default is None.
            
        step : INT, optional
            Current simulation step. The default is 0.
            
        printfreq : int, optional
            PRINT EVERY printfreq TIME STEPS. The default is 1000.
            
        xyzname : TYPE, optional
            DESCRIPTION. The default is "sim.xyz".
            
        fac : float, optional
            Factor to multiply the positions for printing. The default is 1.0.
            
        outname : TYPE, optional
            DESCRIPTION. The default is "sim.log".
            
        debug : bool, optional
            Controls printing for debugging. The default is False.

        PBC : bool, optional #Mine
            whether  PBC are aplleid. The default is False.

        drmax : float, optional #Mine
            Half the length of the box in which MC steps are proposed. default is None.

        Temp : float, optional #Mine
            The temperature for the Canonical ensemble, set the value of beta. default is 273.15K.
    

        Returns
        -------
        None.

        """
        
        #general        
        self.debug=debug 
        self.printfreq = printfreq 
        self.xyzfile = open( xyzname, 'w' ) 
        self.outfile = open( outname, 'w' ) 
        
        #simulation
        self.Nsteps = Nsteps 
        self.dt = dt 
        self.L = L 
        self.seed = seed 
        self.step = step         
        self.fac = fac
        self.PBC = PBC  ## Mine
        self.drmax = drmax
        self.beta = 1 / (Temp * BOLTZMANN)
        self.accept = 0

        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
        else:
            self.R = np.zeros( (1,3) )
            self.mass=np.array([1.6735575E-27]) #H mass in kg as default
            self.kind = ["H"]
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
            
        if p is not None:
            self.p = p
            self.K = K
        else:
            self.p = np.zeros( (self.Natoms,3) )
            self.K = 0.0
        
        if F is not None:
            self.F = F
            self.U = U
        else:
            self.F = np.zeros( (self.Natoms,3) )
            self.U = 0.0
            
        self.E = self.K + self.U
               
        #set RNG seed
        np.random.seed( self.seed )
        
        #check force type
        if ( ftype == "LJ" or ftype == "Harm" or ftype == "Anharm"):
            self.ftype = "eval" + ftype
        else:
            raise ValueError("Wrong ftype value - use LJ or Harm or Anharm.")
        
        #Constant for 1-D ring polymer Path integral simulation:
        self.Cjk = None
        self.Ckj = None
        self.omega_p = self.Natoms / (self.beta * hbar)
        self.omega_k = 2 * self.omega_p * np.sin(np.pi * np.arange(0, self.Natoms) / self.Natoms)

        self.dict = {'a':np.cos(self.omega_k * self.dt)[:, np.newaxis],\
                'b':- (self.mass * self.omega_k * np.sin(self.omega_k * self.dt))[:, np.newaxis],\
                    'c':((1 / self.mass * self.omega_k) * np.sin(self.omega_k * self.dt))[:, np.newaxis]}
    
    
    def __del__( self ):
        """
        THIS IS THE DESCTRUCTOR. NOT USUALLY NEEDED IN PYTHON. 
        JUST HERE TO CLOSE THE FILES.

        Returns
        -------
        None.

        """
        self.xyzfile.close()
        self.outfile.close()
    
    def evalForce( self, **kwargs ):
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).

        Returns
        -------
        None. Calls the correct method based on self.ftype.

        """
        
        getattr(self, self.ftype)(**kwargs)
            
    def dumpThermo( self ):
        """
        THIS FUNCTION DUMPS THE ENERGY OF THE SYSTEM TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
        if( self.step == 0 ):
            self.outfile.write( "step K U E\n" )
        
        self.outfile.write( str(self.step) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + "\n" )
                
    def dumpXYZ( self ):
        """
        THIS FUNCTION DUMP THE COORDINATES OF THE SYSTEM IN XYZ FORMAT TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
        self.xyzfile.write( str( self.Natoms ) + "\n")
        self.xyzfile.write( "Step " + str( self.step ) + "\n" )
        
        for i in range( self.Natoms ):
            self.xyzfile.write( self.kind[i] + " " + \
                              "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,2]*self.fac ) + "\n" )
    

    def dumpXYZ_pandas( self ):
        """
        THIS FUNCTION DUMP THE COORDINATES OF THE SYSTEM IN CSV (sep = " ") FORMAT TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
        if( self.step == 0 ):
            self.xyzfile.write( "step kind atom_num x y z px py pz\n")
        
        for i in range( self.Natoms ):
            self.xyzfile.write( str(self.step) + " " + \
                              self.kind[i] + " " + \
                              str(i) + " " + \
                              "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,2]*self.fac ) + " " + \
                              "{:.6e}".format( self.p[i,0]*self.fac ) + " " + \
                              "{:.6e}".format( self.p[i,1]*self.fac ) + " " + \
                              "{:.6e}".format( self.p[i,2]*self.fac ) + "\n" )
    

    
    
    def readXYZ( self, inpname ):
        """
        THIS FUNCTION READS THE INITIAL COORDINATES IN XYZ FORMAT.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
           
        df = pd.read_csv( inpname, sep="\s+", skiprows=2, header=None )
        
        self.kind = df[ 0 ]
        self.R = df[ [1,2,3] ].to_numpy()
        self.Natoms = self.R.shape[0]
        
################################################################
################## NO EDITING ABOVE THIS LINE ##################
################################################################
    
    
    def sampleMB( self, temp, removeCM=True ):
        """
        THIS FUNCTIONS SAMPLES INITIAL MOMENTA FROM THE MB DISTRIBUTION.
        IT ALSO REMOVES THE COM MOMENTA, IF REQUESTED.

        Parameters
        ----------
        temp : float
            The temperature to sample from.
        removeCM : bool, optional
            Remove COM velocity or not. The default is True.

        Returns
        -------
        None. Sets the value of self.p.
        
        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, LJ POTENTIAL,
        MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. THE MEAN MOMENTUM OF EACH COORDINATE SHOULD BE VERY CLOSE TO ZERO
        2. THE STD SHOULD BE CLOSE TO SQRT(m*Kb*T)
        NOTE: IN THIS CASE, WE SHOW A TEST FOR THE MEAN, WRITE THE OTHERS YOURSELF.
        
        Example: 
        >>> temp = 300  # temperature in K, using default values for mass and Natoms. 
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="LJ")
        >>> mysim.Natoms = 10000
        >>> mysim.mass = np.array([[1.6735575E-27]] * mysim.Natoms)
        >>> mysim.sampleMB(temp, removeCM=False)
        >>> mean_momentum = np.mean(mysim.p, axis=0)
        >>> np.linalg.norm(mean_momentum[0]) < 1e-20  
        True
        >>> np.linalg.norm(mean_momentum[1]) < 1e-20
        True
        >>> np.linalg.norm(mean_momentum[2]) < 1e-20
        True
        >>> std_momentum = np.std(mysim.p, axis=0)
        >>> theo_std = np.sqrt(mysim.mass[0,0] * BOLTZMANN * temp)
        >>> abs(std_momentum[0] - theo_std) < 5e-26
        True
        >>> abs(std_momentum[1] - theo_std) < 5e-26
        True
        >>> abs(std_momentum[2] - theo_std) < 5e-26
        True

        """
        std = np.sqrt(self.mass * BOLTZMANN * temp)
        self.p = np.random.normal(0,std, (self.Natoms, 3))
        x = np.random.normal()
        if removeCM:
            self.p = self.p - np.mean(self.p, axis=0)

    
    def applyPBC( self ):
        """
        THIS FUNCTION APPLIES PERIODIC BOUNDARY CONDITIONS.

        Returns
        -------
        None. Sets the value of self.R.

        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, LJ POTENTIAL,
        MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. A PARTICLE WITH X > L/2
        2. A PARTICLE WITH X < -L/2
        3. REPEAT 1 AND 2 FOR Y AND Z.

        Example:
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="LJ", mass = np.array([1.6735575E-27]*6))
        >>> mysim.R = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]) * mysim.L
        >>> mysim.applyPBC()
        >>> np.all(np.abs(mysim.R) < 1E-20)
        True
        """
        self.R[self.R > self.L / 2] -= self.L
        self.R[self.R < - self.L / 2] += self.L
        # is_over = self.R > (self.L / 2)
        # is_under = self.R < -(self.L / 2)
        # self.R += (is_under.astype(int) - is_over.astype(int)) * self.L

                    
    def removeRCM( self ):
        """
        THIS FUNCTION ZEROES THE CENTERS OF MASS POSITION VECTOR.

        Returns
        -------
        None. Sets the value of self.R.


        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, LJ POTENTIAL
        MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. CREATE THREE PARTICLES WITH THE SAME MASS AND A NON-ZERO RCM, REMOVE IT.
        2. REPEAT FOR THREE PARTICLES WITH DIFFERENT MASSES: 1M, 2M, 3M.

        Example:
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="LJ")
        >>> mysim.R = np.array([[0.3,-0.2,0.4],[0.7,-0.2,-0.1],[0.4,0.2,0]]) * mysim.L
        >>> mysim.mass =  np.array([[1],[1],[1]]) * 1.6735575E-27
        >>> mysim.removeRCM()
        >>> np.all(np.abs(np.sum(mysim.R * mysim.mass, axis = 0) / np.sum(mysim.mass)) < 1E-22)
        True
        >>> mysim.R = 1.6735575E-27
        >>> mysim.mass = np.array([[0.4],[3],[1]]) * 1.6735575E-27
        >>> mysim.removeRCM()
        >>> np.all(np.abs(np.sum(mysim.R * mysim.mass, axis = 0) / np.sum(mysim.mass)) < 1E-22)
        True

        """   

        self.R = self.R - np.sum(self.R * self.mass, axis = 0) / np.sum(self.mass)


                
             
    def evalLJ( self, eps, sig ):
        """
        THIS FUNCTION EVALUTES THE LENNARD-JONES POTENTIAL AND FORCE.

        Parameters
        ----------
        eps : float
            epsilon LJ parameter.
        sig : float
            sigma LJ parameter.

        Returns
        -------
        None. Sets the value of self.F and self.U.

        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, LJ POTENTIAL
        EPSILON 1, SIMGA 1, MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. CHECK THAT THE VALUE OF THE POTENTIAL AT THE MINIMUM IS -EPSILON.
        2. CHECK THAT THE VALUE OF THE POTENTIAL AT SIGMA IS ZERO

        Example:
        >>> eps = 1
        >>> sig = 1
        >>> pos_min = np.array([[0,0,0], [2 ** (1/6) * sig,0,0]])
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="LJ", R = pos_min, PBC = False)
        >>> mysim.mass = np.ones((mysim.Natoms, 1)) * 1.6735575E-27
        >>> mysim.evalLJ(eps,sig)
        >>> abs(mysim.U + eps) < 1E-20
        True
        >>> mysim.Natoms = 3 # testing the function with 3 atoms, the third atom is far so it doesn't really effect F and U 
        >>> mysim.mass = np.ones((mysim.Natoms, 1)) * 1.6735575E-27
        >>> mysim.R = np.array([[0,0,0], [sig,0,0],[0,0,30000.0]])  
        >>> mysim.evalLJ(eps,sig)
        >>> abs(mysim.U) < 1E-20
        True
        """
        
        self.U = 0
        r_ij2 = np.zeros((self.Natoms, self.Natoms - 1, 3)) # array for force evaluation 

        for i in range(self.Natoms):
            r_ij2[i] = self.R[i] - np.delete(self.R, i, axis=0)

            if i == self.Natoms - 1:
                break

            r_ij1 = ( self.R[i] - self.R[i+1:]) # array for energy evaluation 
            if self.PBC:
                r_ij1[r_ij1 > self.L / 2] -= self.L # energy PBC
                r_ij1[r_ij1 < - self.L / 2] += self.L # energy PBC

            norm_ij1 = np.linalg.norm(r_ij1, axis = 1) # enrgy norm
            self.U += 4 * eps * np.sum((sig / norm_ij1) ** 12 - (sig / norm_ij1) ** 6)
        if self.PBC:
            r_ij2[(r_ij2 > self.L / 2)] -= self.L  # force PBC
            r_ij2[(r_ij2 < - self.L / 2)] += self.L # force PBC
        norm_ij2 = np.repeat(np.linalg.norm(r_ij2, axis = 2)[:,:,np.newaxis] , 3 , axis=2) # force norm
        self.F = 4 * eps * np.sum((12 * sig ** 12 / norm_ij2 ** 14 - 6 * sig ** 6 / norm_ij2 ** 8) * r_ij2, axis = 1)

            
    def evalHarm( self, omega ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.

        Parameters
        ----------
        omega : float
            The frequency of the trap.

        Returns
        -------
        None. Sets the value of self.F and self.U.

        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, HARMONIC POTENTIAL, 
        OMEGA 1, MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example: 
        1. CHECK THAT THE VALUE OF THE POTENTIAL AT THE MINIMUM IS ZERO.
        2. CHECK THAT THE VALUE OF THE POTENTIAL AT 5A IS 2.09194687e-46. 

        Example:
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="Harm", mass = 1.6735575E-27)
        >>> mysim.mass = np.array([[1.6735575E-27]])
        >>> mysim.R = np.array([[0,0,0]])
        >>> mysim.evalHarm(omega = 1)
        >>> abs(mysim.U) < 1E-20
        True
        >>> mysim.R = np.array([[5E-10,0,0]])
        >>> mysim.evalHarm(omega = 1)
        >>> abs(mysim.U-2.09194687e-46) < 1E-50
        True
        
        """
        self.F = - self.mass * omega ** 2 * self.R * np.array([1,0,0])
        self.U = 1 / 2 * omega ** 2 * self.mass * np.sum(self.R[:,0] ** 2)


    def evalAnharm( self, Lambda ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR AN ANHARMONIC TRAP.

        Parameters
        ----------
        Lambda : float
            The parameter of the trap U = 0.25 * Lambda * x**4

        Returns
        -------
        None. Sets the value of self.F and self.U.

        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, ANHARMONIC POTENTIAL, 
        LAMBDA 1, MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. CHECK THAT THE VALUE OF THE POTENTIAL AT THE MINIMUM IS ZERO.
        2. CHECK THAT THE VALUE OF THE POTENTIAL AT 5A IS 1.5625E-38

        Example:
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="Anharm", mass = 1.6735575E-27)
        >>> mysim.mass = np.array([[1.6735575E-27]])
        >>> mysim.R = np.array([[0,0,0]])
        >>> mysim.evalAnharm(Lambda = 1)
        >>> abs(mysim.U) < 1E-20
        True
        >>> mysim.R = np.array([[5E-10,0,0]])
        >>> mysim.evalAnharm(Lambda = 1)
        >>> abs(mysim.U-1.5625E-38) < 1E-50
        True
        """
        self.F = - Lambda * self.R ** 3 * np.array([1,0,0])
        self.U = 0.25 * Lambda * self.R[0,0] ** 4

    def evalring( self, omega2 ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR FREE RING POLYMER WITH HARMONIC INTERACTIONS BETWEEN NEIGHBORS (one dimension).

        Returns
        -------
        None. Adds to the value of self.F and self.U.

        Tests
        -----
        for a ring of 3 atoms with omega2 = 1 and mass = 1:
        1. CHECK THAT THE VALUE OF THE POTENTIAL AT THE MINIMUM IS 0.
        2. for the possitions: [[0,0,0],[1,0,0],[2,0,0]] the potential energy should be 0.5
        Example:

        """
        
    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.

        Returns
        -------
        None. Set s the value of self.K.

        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15,BOX SIDE LENGTH 11.3E-10, LJ POTENTIAL, 
        MOMENTUM 1 (ALL DIRECTIONS), KINETIC ENERGY 1,
        MASS 1.6735575E-27
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. CHECK THAT THE VALUE OF THE KINETIC ENERGY IS 8.96294271e+26.
        
        Example:
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="LJ", mass = np.ones((1,1)) * 1.6735575E-27)
        >>> mysim.mass = np.ones((1,1)) * 1.6735575E-27
        >>> mysim.K = 1
        >>> mysim.p = np.array([[1,1,1]])
        >>> mysim.CalcKinE()
        >>> abs(mysim.K - 8.96294271e+26) < 4E17
        True
        """
        self.K = np.sum(np.sum( self.p**2, axis = 1) / (2 * self.mass[0,0])  )
        # self.K = np.sum(np.linalg.norm(self.p, axis = 1) ** 2 / (2 * self.mass))


    def VVstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.

        Returns
        -------
        None. Sets self.R, self.p.

        Tests
        -----
        CREATE AN OBJECT OF THE SIMULATION CLASS WITH FOLLOWING PARAMS:
        TIMESTEP 0.1E-15, BOX SIDE LENGTH 11.3E-10, LJ POTENTIAL, 
        MOMENTUM 1 (ALL DIRECTIONS), KINETIC ENERGY 0.5, 
        ATOM 1 POSITION (0,0,0), ATOM 2 POSITION (1,1,1)
        FORCE 2E15, POTENTIAL 0.5, MASS 1E-15 (BOTH ATOMS),
        EPSILON 0.5, SIMGA 0.5
        
        IMPLEMENT THE FOLLOWING TESTS BELOW THE WORD Example:
        1. CHECK THAT AFTER ONE VVSTEP THE POSITIONS OF THE ATOMS ARE UPDATED BY 0.11.
        2. CHECK THAT AFTER ONE VVSTEP THE MOMENTUMS OF THE ATOMS ARE 1.1 (ALL DIRECTIONS).

        Example:
        >>> mysim = Simulation(dt=0.1E-15, L=11.3E-10, ftype="LJ", R = np.array([[0,0,0],[1,1,1]]).astype(float))
        >>> mysim.mass = np.ones((mysim.Natoms,1))*1E-15
        >>> mysim.p = np.ones((mysim.Natoms, 3)).astype(float)
        >>> mysim.K = 0.5
        >>> mysim.F = np.ones((mysim.Natoms, 3)) * 2E15
        >>> mysim.U = 0.5
        >>> params = {'eps': 0.5, 'sig': 0.5}
        >>> mysim.VVstep(**params)
        >>> np.all(np.abs(mysim.R - np.array([[0,0,0],[1,1,1]]).astype(float) - 0.11) < 1E-15)
        True
        >>> np.all(np.abs(mysim.p - 1.1) < 1E-20)
        True
        """

        self.p += self.F * self.dt / 2
        self.R += (self.p / self.mass) * self.dt
        self.evalForce(**kwargs)
        self.p += self.F * self.dt / 2


    def CulcCjk(self):
        """
        This function calculates the Cjk matrix for the ring polymer.

        Returns
        -------
        None. Sets self.Cjk, self.Ckj.
        """
        C = np.zeros((self.Natoms, self.Natoms))
        for j in range(self.Natoms):
            for k in range(self.Natoms):
                if k == 0:
                    C[j, k] = np.sqrt(1 / np.float(self.Natoms))
                elif 1 <= k <= self.Natoms / 2 - 1:
                    C[j, k] = np.sqrt(2 / np.float(self.Natoms)) * np.cos(2 * np.pi * (j+1) * k / np.float(self.Natoms))
                elif k == self.Natoms / 2:
                    C[j, k] = np.sqrt(1 / np.float(self.Natoms)) * (-1) ** (j+1)
                elif self.Natoms / 2 + 1 <= k <= self.Natoms - 1:
                    C[j, k] = np.sqrt(2 / np.float(self.Natoms)) * np.sin(2 * np.pi * (j+1) * k / np.float(self.Natoms))
        self.Cjk = C
        self.Ckj = np.transpose(C)
   

    def PolyRingStep(self, **kwargs):
        """
        THIS FUNCTION PERFORMS ONE STEP FOR A RING POLYMER.

        Returns
        -------
        None. Sets self.R, self.p.
        """
        
        self.p += self.F * self.dt / 2

        Rtild = np.dot(self.Cjk, self.R)
        ptild = np.dot(self.Cjk, self.p)

        # cos_omegak_dt = np.cos(self.omega_k * self.dt)
        # sin_omegak_dt = np.sin(self.omega_k * self.dt)
        # mass_omegak = self.mass * self.omega_k

        # self.Rtild = cos_omegak_dt[:, np.newaxis] * self.Rtild - mass_omegak[:, np.newaxis] * sin_omegak_dt[:, np.newaxis] * self.ptild
        # self.ptild = (1 / mass_omegak)[:, np.newaxis] * sin_omegak_dt[:, np.newaxis] * self.Rtild + cos_omegak_dt[:, np.newaxis] * self.ptild

        Rtild = self.dict['a'] * Rtild + self.dict['b'] * ptild
        ptild = self.dict['c'] * Rtild + self.dict['a'] * ptild

        self.R = np.dot(self.Ckj, Rtild)
        self.p = np.dot(self.Ckj, ptild)

        self.evalForce(**kwargs)

        self.p += self.F * self.dt / 2


    
    def evalLJ_atom(self, R, atom, eps, sig):
        """
        THIS FUNCTION EVALUTES THE LENNARD-JONES POTENTIAL FOR A SINGLE ATOM.
        
        Returns
        -------
        float. The value of the potential energy for a single atom.
        """

        r_ij = R[atom] - np.delete(R,atom, axis=0) # array for energy evaluation 
        if self.PBC:
            r_ij[r_ij > self.L / 2] -= self.L # energy PBC
            r_ij[r_ij < - self.L / 2] += self.L # energy PBC
        norm_ij = np.linalg.norm(r_ij, axis = 1) # enrgy norm
        return 4 * eps * np.sum((sig / norm_ij) ** 12 - (sig / norm_ij) ** 6)


  
    def MCstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE METROPOLIS MC STEP IN THE NVT ENSEMBLE.
        YOU WILL NEED TO PROPOSE TRANSLATION MOVES, APPLY  
        PBC, CALCULATE THE CHANGE IN POTENTIAL ENERGY, ACCEPT OR REJECT, 
        AND CALCULATE THE ACCEPTANCE PROBABILITY. 

        Returns
        -------
        None. Sets self.R.
        """
        
        # def evalLJ_atom(R,atom,eps,sig):
        #     r_ij = R[atom] - np.delete(R,atom, axis=0) # array for energy evaluation 
        #     if self.PBC:
        #         r_ij[r_ij > self.L / 2] -= self.L # energy PBC
        #         r_ij[r_ij < - self.L / 2] += self.L # energy PBC
        #     norm_ij = np.linalg.norm(r_ij, axis = 1) # enrgy norm
        #     return 4 * eps * np.sum((sig / norm_ij) ** 12 - (sig / norm_ij) ** 6)
        

        # previous energy and positions arrays
        U0 = np.copy(self.U) 

        for atom in range(self.Natoms): # np.random.randint(0,self.Natoms , self.Natoms): # preform MC pass.
            R0 = np.copy(self.R)
            # PROPOSE TRANSLATION MOVE
            R1 = R0[atom] + np.random.uniform(-1,1, (1,3)) * self.drmax

            # apply PBC
            if self.PBC:
                R1[R1 > self.L / 2] -= self.L
                R1[R1 < - self.L / 2] += self.L
            self.R[atom] = R1

            # eval energy
            if self.ftype == 'evalLJ':
                U0 = self.evalLJ_atom(R0, atom, **kwargs)
                U1 = self.evalLJ_atom(self.R, atom, **kwargs)
            else: 
                self.evalForce(**kwargs)
                U1 = self.U
            
            # accept or reject
            ratio = np.exp(-self.beta * (U1 - U0))
            xsi = np.random.uniform(0,1)

            if ratio >= xsi: # accept
                # self.R[atom] = R1
                self.accept += 1
            else:  # reject
                self.R[atom] = R0[atom]




        
    def runMC(self, **kwargs):
        """ 
        THIS FUNCTION DEFINES AN MC SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO LOOP OVER MC STEPS, 
        PRINT THE COORDINATES AND ENERGIES EVERY PRINTFREQ TIME STEPS 
        TO THEIR RESPECTIVE FILES, SIMILARLY TO YOUR MD CODE.

        Returns
        -------
        None.

        """ 
        for step in range(self.Nsteps):
            self.step = step
            self.evalForce(**kwargs)
            if self.step % self.printfreq == 0:
                self.dumpThermo()
                self.dumpXYZ_pandas()
             
            self.MCstep(**kwargs)
        self.accept /=  self.Nsteps * self.Natoms
        self.evalForce(**kwargs)
        self.dumpThermo()
        self.dumpXYZ_pandas()


        
    def run( self, **kwargs ):
        """
        THIS FUNCTION DEFINES A SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO:
            1. EVALUATE THE FORCES (USE evalForce() AND PASS A DICTIONARY
                                    WITH ALL THE PARAMETERS).
            2. PROPAGATE FOR NS TIME STEPS USING THE VELOCITY VERLET ALGORITHM.
            3. APPLY PBC.
            4. CALCULATE THE KINETIC, POTENTIAL AND TOTAL ENERGY AT EACH TIME
            STEP. 
            5. YOU WILL ALSO NEED TO PRINT THE COORDINATES AND ENERGIES EVERY 
        PRINTFREQ TIME STEPS TO THEIR RESPECTIVE FILES, xyzfile AND outfile.

        Returns
        -------
        None.

        """      
        
        self.evalForce(**kwargs)
        for step in range(self.Nsteps):
            self.step = step

            # self.VVstep(**kwargs)

            self.CalcKinE()
            self.E = self.K + self.U

            if self.step % self.printfreq == 0:
                self.dumpThermo()
                self.dumpXYZ_pandas() 

            self.VVstep(**kwargs)

            if self.PBC:
                self.applyPBC()
             

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)


