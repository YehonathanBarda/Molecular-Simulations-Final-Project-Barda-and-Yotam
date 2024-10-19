import numpy as np
from scipy.constants import hbar
trap_omega = 50 * 1.602176634E-22 / hbar # from meV to J

# def CalcCjk():
#         """
#         This function calculates the Cjk matrix for the ring polymer.

#         Returns
#         -------
#         None. Sets self.Cjk, self.Ckj.
#         """
#         Natoms = 4

#         C = np.zeros((Natoms, Natoms))
#         for j in range(Natoms):
#             for k in range(Natoms):
#                 if k == 0:
#                     C[j, k] = np.sqrt(1 / np.float64(Natoms))
#                 elif 1 <= k and k <= Natoms / 2 - 1:
#                     C[j, k] = np.sqrt(2 / np.float64(Natoms)) * np.cos(2 * np.pi * (j+1) * k / np.float64(Natoms))
#                 elif k == Natoms / 2:
#                     C[j, k] = np.sqrt(1 / np.float64(Natoms)) * (-1) ** (j+1)
#                 elif Natoms / 2 + 1 <= k and k <= Natoms - 1:
#                     C[j, k] = np.sqrt(2 / np.float64(Natoms)) * np.sin(2 * np.pi * (j+1) * k / np.float64(Natoms))
#                 else:
#                     raise ValueError('k out of bounds')
#         Cjk = C
#         Ckj = np.transpose(C)
#         print(Cjk)
#         # print(Ckj)
# CalcCjk()

# v = np.array([[1], [2], [3]])
# # v = np.array([1, 2, 3]) 
# M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(M, '\n\n', v, '\n\n', M @ v)
# print(np.dot(M, v))
Natoms = 10
beta =  6 / (hbar * trap_omega)
mass = 6.6335209E-26
dt = 0.1E-15

omega_p = Natoms / (beta * hbar) # eq 6
omega_k = np.reshape(2 * omega_p * np.sin(np.pi * np.arange(0, Natoms) / Natoms),(Natoms,1)) # eq 20
c = np.zeros((Natoms, 1))
c[0] = dt / mass
c[1:] = ((1 / (mass * omega_k[1:])) * np.sin(omega_k[1:]* dt))

dict = {'cos':np.cos(omega_k * dt),\
        '-sin':- (mass * omega_k * np.sin(omega_k * dt)),\
            '1/sin': c}

print(dict['cos'])
# print(omega_k)
# print(np.pi * np.arange(0, Natoms) / Natoms)