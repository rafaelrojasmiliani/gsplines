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
from gsplines.gspline import cSplineCalc
import time

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
        '''
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

