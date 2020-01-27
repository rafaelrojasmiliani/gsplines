"""
    Test functions generation of piecewise functions.
    This test is done just by plooting a piecewise functions
    randomly made.
"""
import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
import unittest
from gsplines.piecewisefunction import cPiecewiseFunction
from gsplines.basis1010 import cBasis1010




class cMyTest(unittest.TestCase):
    def test(self):
        '''
        '''
        np.set_printoptions(linewidth=500, precision=4)

        for i in range(10):
            dim = np.random.randint(1, 20)
            N = np.random.randint(1, 20)
            y = (np.random.rand(6*dim*N)*2-1)*np.pi 
            tauv = np.random.rand(N)
            T = np.sum(tauv)
            alpha = np.random.rand()
            basis = cBasis1010(alpha)
            pw = cPiecewiseFunction(tauv, y, dim, basis)

            t = np.arange(0, T, 0.001)

            res = pw(t)

            assert res.shape[1] == dim and res.shape[0] == t.shape[0]





def main():
    unittest.main()


if __name__ == '__main__':
    main()

