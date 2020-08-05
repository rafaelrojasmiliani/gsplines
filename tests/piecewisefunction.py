"""
    Test functions generation of piecewise functions.
    This test is done just by plooting a piecewise functions
    randomly made.
"""
import numpy as np
import unittest
from gsplines.piecewisefunction import cPiecewiseFunction
from gsplines.basis1010 import cBasis1010
from gsplines.basis0010 import cBasis0010
from gsplines.basis1000 import cBasis1000


import os
import unittest
import functools
import traceback
import sys
import pdb

def debug_on(*exceptions):
    ''' Decorator for entering in debug mode after exceptions '''
    if not exceptions:
        exceptions = (Exception, )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])
                sys.exit(1)

        return wrapper

    return decorator


class cMyTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)
        self.N_ = np.random.randint(3, 10)
        self.dim_ = np.random.randint(3, 10)

    @debug_on()
    def test(self):
        ''' Test that there are not exceptions
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

    @debug_on()
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

    @debug_on()
    def testplot(self):
        from gsplines.gspline import cSplineCalc
        dim = np.random.randint(6, 8)
        N = np.random.randint(2, 20)
        tauv = 0.5 + np.random.rand(N) * 3.0
        T = np.sum(tauv)
        wp = (np.random.rand(N + 1, dim) - 0.5) * 2 * np.pi
        splcalc = cSplineCalc(dim, N, cBasis0010())
        q = splcalc.getSpline(tauv, wp)
        try:
            import matplotlib.pyplot as plt
            t = np.arange(0, T, 0.01)

            plt.plot(t, q(t)[:, 0])
            plt.show()
        except:
            pass

    @debug_on()
    def test_time(self):
        ''' Test the evaluation time
        '''

        dim = np.random.randint(6, 8)
        N = np.random.randint(2, 20)
        tauv = 0.5 + np.random.rand(N) * 3.0
        wp = (np.random.rand(N + 1, dim) - 0.5) * 2 * np.pi
        splcalc = cSplineCalc(dim, N, cBasis0010())
        q = splcalc.getSpline(tauv, wp)
        dt = 0
        Ntest = 1000
        for _ in range(Ntest):
            t = tauv[0]
            t0 = time.time()
            q0 = q(t)[0]
            t1 = time.time()
            dt += t1 - t0

        print('mean evaluation time ', dt/Ntest)



def main():
    unittest.main()


if __name__ == '__main__':
    main()

