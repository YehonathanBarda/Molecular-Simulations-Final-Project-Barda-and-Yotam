import numpy as np
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
Natoms = 3
C = np.zeros((Natoms, Natoms))
for j in range(Natoms):
    for k in range(Natoms):
        if k == 0:
            C[j, k] = np.sqrt(1 / np.float(Natoms))
        if 1 <= k and k <= Natoms / 2 - 1:
            C[j, k] = np.sqrt(2 / np.float(Natoms)) * np.cos(2 * np.pi * (j+1) * k / np.float(Natoms))
        if k == Natoms / 2:
            C[j, k] = np.sqrt(1 / np.float(Natoms)) * (-1) ** (j+1)
        if Natoms / 2 + 1 <= k and k <= Natoms - 1:
            C[j, k] = np.sqrt(2 / np.float(Natoms)) * np.sin(2 * np.pi * (j+1) * k / np.float(Natoms))

print(C)
print('\n_____________________________________\n')
Nbids = 4
x = np.random.uniform(-5, 5, size=(Nbids, 1)) * 1E-10
print(x)