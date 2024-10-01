import numpy as np
R = np.array([[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0]])
print(R)
print('\n_____________________________________\n')
a = np.arange(5)
print(a)
print('\n_____________________________________\n')
a2 = a[:, np.newaxis]
print(a2)
print('\n_____________________________________\n')
print(a2 * R)