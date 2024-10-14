import numpy as np
from scipy.constants import hbar
Natoms = 4

# R = np.array([[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0]])
# print(R)
# print('\n_____________________________________\n')
# avg = np.mean(R, axis=0)
# print(avg)


# a = np.arange(5)
# print(a)
# print('\n_____________________________________\n')
# a2 = a[:, np.newaxis]
# print(a2)
# print('\n_____________________________________\n')
# print(a2 * R)
# C = np.zeros((Natoms, Natoms))
# for j in range(Natoms):
#     for k in range(Natoms):
#         if k == 0:
#             C[j, k] = np.sqrt(1 / np.float(Natoms))
#         if 1 <= k and k <= Natoms / 2 - 1:
#             C[j, k] = np.sqrt(2 / np.float(Natoms)) * np.cos(2 * np.pi * (j+1) * k / np.float(Natoms))
#         if k == Natoms / 2:
#             C[j, k] = np.sqrt(1 / np.float(Natoms)) * (-1) ** (j+1)
#         if Natoms / 2 + 1 <= k and k <= Natoms - 1:
#             C[j, k] = np.sqrt(2 / np.float(Natoms)) * np.sin(2 * np.pi * (j+1) * k / np.float(Natoms))

# print(C)
# print('\n_____________________________________\n')

# p = np.array([[1],[2],[3],[4]])
# print(p)
# print('\n_____________________________________\n')

# print(np.dot(C, p))

# x = np.array([1,2,3,4])
# print(x)
# print(np.reshape(x, (4,1)))
# print(x * p)
dt = 0.1E-15
trap_omega = 50 * 1.602176634E-22 / hbar
beta = 1 / (hbar * trap_omega)
mass = 6.6335209E-26

omega_p = Natoms / (beta * hbar)
omega_k = 2 * omega_p * np.sin(np.pi * np.arange(0, Natoms) / Natoms)

c = ((1 / (mass * omega_k)) * np.sin(omega_k * dt))[:, np.newaxis]
c[0] = dt / mass
        
dict = {'cos':np.cos(omega_k * dt)[:, np.newaxis],\
    '-sin':- (mass * omega_k * np.sin(omega_k * dt))[:, np.newaxis],\
         '1/sin': c}

print(dict['1/sin'])