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
from scipy.constants import hbar

class Simulation:
    
    def __init__( self, dt, L = 11.3E-10, Nsteps=0, R=None, mass=None, kind=None, \
                 p=None, F=None, U=None, K=None, seed=937142, ftype=None, \
                 step=0, printfreq=1000, xyzname="sim.xyz", fac=1.0, \
                 outname="sim.log", debug=False, PBC=False, drmax = None, beta = None):
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
        self.beta = beta
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

        if self.Natoms%2 != 0:
            raise ValueError("Number of atoms must be even.")
        
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
        
        self.omega_p = self.Natoms / (self.beta * hbar) # eq 6
        self.omega_k = np.reshape(2 * self.omega_p * np.sin(np.pi * np.arange(0, self.Natoms) / self.Natoms),(self.Natoms,1)) # eq 20
        c = np.zeros((self.Natoms, 1))
        c[0] = self.dt / self.mass
        c[1:] = ((1 / (self.mass * self.omega_k[1:])) * np.sin(self.omega_k[1:]* self.dt))

        # Dictionary for for the matrix in eq 2.36 in notes
        self.dict = {'cos':np.cos(self.omega_k * self.dt),\
            '-sin':-self.mass * self.omega_k * np.sin(self.omega_k * self.dt),\
                '1/sin': c}
    
    
    def __del__( self ):
        """
        THIS IS THE DESTRUCTOR. NOT USUALLY NEEDED IN PYTHON. 
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
            self.xyzfile.write( "step kind bead_num x p\n")
        
        for i in range( self.Natoms ):
            self.xyzfile.write( str(self.step) + " " + \
                              self.kind[i] + " " + \
                              str(i) + " " + \
                              "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \

                              "{:.6e}".format( self.p[i,0]*self.fac ) + "\n" )
    

    
    
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
    
    
    def sampleMB( self, removeCM=True ):
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
        std = np.sqrt(self.mass / self.beta)
        self.p = np.random.normal(0,std, (self.Natoms, 1))
        if removeCM:
            self.p = self.p - np.mean(self.p, axis=0)


                                
    def evalHarm( self, omega):
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
        self.F = - self.mass * omega ** 2 * self.R
        self.U = 1 / 2 * omega ** 2 * self.mass * np.mean(self.R ** 2)
        # print(f'step: {self.step}\nforce:\n{self.F}')


    def CalcKinE_PI( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE RING POLYMER.

        Returns
        -------
        None. Sets the value of self.K.

        """
        self.K = 1/(2 * self.beta) + 0.5 * np.mean(-self.F * (self.R - np.mean(self.R, axis = 0))) # from notes
        
        # self.K = self.Natoms / (2*self.beta) - (self.mass * self.Natoms) / (2 * self.beta ** 2 * hbar ** 2) * np.sum((np.roll(self.R,-1) - self.R)**2) # from takerman 1
        # self.K = -1 / 2 * np.mean(self.R * self.F) # from takerman 2 (viral estimator)


    def CalcCjk(self):
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
                    C[j, k] = np.sqrt(1 / np.float64(self.Natoms))
                elif 1 <= k and k <= self.Natoms / 2 - 1:
                    C[j, k] = np.sqrt(2 / np.float64(self.Natoms)) * np.cos(2 * np.pi * (j+1) * k / np.float64(self.Natoms))
                elif k == self.Natoms / 2:
                    C[j, k] = np.sqrt(1 / np.float64(self.Natoms)) * (-1) ** (j+1)
                elif self.Natoms / 2 + 1 <= k and k <= self.Natoms - 1:
                    C[j, k] = np.sqrt(2 / np.float64(self.Natoms)) * np.sin(2 * np.pi * (j+1) * k / np.float64(self.Natoms))
                else:
                    raise ValueError('k out of bounds')
        self.Cjk = C
        self.Ckj = np.transpose(C)
   

    def PolyRingStep(self, **kwargs):
        """
        THIS FUNCTION PERFORMS ONE STEP FOR A RING POLYMER.

        Returns
        -------
        None. Sets self.R, self.p.
        """ 
        gamma = 2 * self.omega_k # eq 36
        gamma[0] = 1/(100 * self.dt) # Baraks mail
 
        ptild = np.dot(self.Cjk, self.p) # 27

        ptild = np.exp(- gamma * self.dt / 2) * ptild + np.sqrt( (self.mass * self.Natoms / self.beta) * (1 - np.exp(- gamma * self.dt))) * np.random.normal(0, 1, size = (self.Natoms, 1)) # 28 Langevin part

        self.p = np.dot(self.Ckj, ptild) # 29

        self.evalForce(**kwargs)

        self.p += self.F * self.dt / 2 # 21

        Rtild = np.dot(self.Cjk, self.R) # 22
        ptild = np.dot(self.Cjk, self.p) # 22

        ptild_New = self.dict['cos'] * ptild + self.dict['-sin'] * Rtild # 23
        Rtild = self.dict['1/sin'] * ptild + self.dict['cos'] * Rtild # 23
        ptild = np.copy(ptild_New) # 23
        
        self.R = np.dot(self.Ckj, Rtild) # 24
        self.p = np.dot(self.Ckj, ptild) # 24

        self.evalForce(**kwargs)

        self.p += self.F * self.dt / 2 # 25

        ptild = np.dot(self.Cjk, self.p) # 27

        ptild = np.exp(- gamma * self.dt / 2) * ptild + np.sqrt( (self.mass * self.Natoms / self.beta) * (1 - np.exp(- gamma * self.dt))) * np.random.normal(0, 1, size = (self.Natoms, 1)) # 28 Langevin part

        self.p = np.dot(self.Ckj, ptild) # 29


        
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
        
        self.CalcCjk()
        for step in range(self.Nsteps):
            self.step = step        
            self.evalForce(**kwargs)
            self.CalcKinE_PI()
            self.E = 2 * self.U # virial estimator for the total energy (Takerman), detiled in the rapport

            if self.step % self.printfreq == 0:
                self.dumpThermo()
                self.dumpXYZ_pandas() 

            self.PolyRingStep(**kwargs)
             

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)


