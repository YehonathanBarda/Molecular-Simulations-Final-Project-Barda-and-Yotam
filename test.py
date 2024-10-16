import numpy as np

def CalcCjk():
        """
        This function calculates the Cjk matrix for the ring polymer.

        Returns
        -------
        None. Sets self.Cjk, self.Ckj.
        """
        Natoms = 4

        C = np.zeros((Natoms, Natoms))
        for j in range(Natoms):
            for k in range(Natoms):
                if k == 0:
                    C[j, k] = np.sqrt(1 / np.float64(Natoms))
                elif 1 <= k and k <= Natoms / 2 - 1:
                    C[j, k] = np.sqrt(2 / np.float64(Natoms)) * np.cos(2 * np.pi * (j+1) * k / np.float64(Natoms))
                elif k == Natoms / 2:
                    C[j, k] = np.sqrt(1 / np.float64(Natoms)) * (-1) ** (j+1)
                elif Natoms / 2 + 1 <= k and k <= Natoms - 1:
                    C[j, k] = np.sqrt(2 / np.float64(Natoms)) * np.sin(2 * np.pi * (j+1) * k / np.float64(Natoms))
                else:
                    raise ValueError('k out of bounds')
        Cjk = C
        Ckj = np.transpose(C)
        print(Cjk)
        # print(Ckj)
CalcCjk()