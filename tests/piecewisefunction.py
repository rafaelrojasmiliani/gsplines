"""
    Test functions generation of piecewise functions.
    This test is done just by plooting a piecewise functions
    randomly made.
"""
import numpy as np
import sympy as sp
import unittest
from gsplines.piecewisefunction import cPiecewiseFunction
from gsplines.basis1010 import cBasis1010
from gsplines.basis0010 import cBasis0010
from gsplines.basis1000 import cBasis1000




class cMyTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)
        self.N_ = np.random.randint(3, 10)
        self.dim_ = np.random.randint(3, 10)

    def test(self):
        '''
        '''

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

    def test_l2_norms(self):
        ''' Test L2 norm'''
        B = cBasis1000()
        bdim = B.dim_
        N = self.N_
        dim = self.dim_
        y = np.random.rand(bdim*dim*N)
        tauv = np.random.rand(N)
        T = np.sum(tauv)
        
        pw = cPiecewiseFunction(tauv, y, dim, B)

        f = pw
        dt = 0.001
        fv = np.array([np.linalg.norm(f(t))**2 for t in np.arange(0, T, dt)])

        Itest = np.sum(fv[1:]+fv[:-1])*dt/2.0
        Inom = pw.l2_norm()

        err = np.abs(Itest - Inom)
        assert err < 5.0e-2, '''
        Error in integral = {:.4f}
        Nominal integral = {:.4f}
        Test integral = {:.4f}
        '''.format(err, Inom, Itest)

def main():
    unittest.main()


if __name__ == '__main__':
    main()

