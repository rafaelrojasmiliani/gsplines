"""
"""
import numpy as np
import sympy as sp
import unittest
from gsplines.gspline import cSplineCalc
from gsplines.distfunction import cDistFunction, cDistSpeed
from gsplines.basis1010 import cBasis1010

import functools
import traceback
import sys
import pdb


def debug_on(*exceptions):
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

        return wrapper

    return decorator


class scalarmap(object):
    def __init__(self):
        self.dim_ = 0
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)
        pass

    def __call__(self, q, qd):
        return 0.0

    def deriv_wrt_q(q, qd):
        return 0.0

    def deriv_wrt_qd(q, qd):
        return 0.0


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)
        self.N_ = np.random.randint(3, 10)
        self.dim_ = np.random.randint(3, 10)

    @debug_on()
    def testConstructorAndTimes(self):
        ''' Test that the constructor works and that getTimes give the vector
        with the correct time instants'''
        g = scalarmap()
        df = cDistFunction(
            g,
            self.N_,
        )
        tauv = np.random.rand(self.N_)
        ti, si, iintr, _ = df.getTimes(tauv)

    @debug_on()
    def testCall(self):
        ''' Test that the caller of the funciton works, given tauv and
        waypoints
        '''
        dim = self.dim_

        class myScalarmap(object):
            def __init__(self):
                self.dim_ = dim
                pass

            def __call__(self, q, qd):
                return q.dot(q)

            def deriv_wrt_q(self, q, qd):
                return 0.0

            def deriv_wrt_qd(self, q, qd):
                return 0.0

        g = myScalarmap()
        df = cDistFunction(g, self.N_)
        tauv = 0.5 + np.random.rand(self.N_)
        wp = np.random.rand(self.N_ + 1, self.dim_)
        v = df(tauv, wp)
        tis, sis, iints, _ = df.getTimes(tauv)
        q = df.splcalc_.getSpline(tauv, wp)

        qt = q(tis)
        res = np.array([qi.dot(qi) for qi in qt])
        assert np.max(np.abs(res - v)) < 1.0e-10

    @debug_on()
    def testDerivative(self):
        ''' Test the derivtibve of the funciton w.r.t. tauv and u'''
        dim = self.dim_

        class myScalarmap(object):
            def __init__(self):
                self.dim_ = dim
                pass

            def __call__(self, q, qd):
                return q.dot(q) + qd.dot(qd)

            def deriv_wrt_q(self, q, qd):
                return 2 * q

            def deriv_wrt_qd(self, q, qd):
                return 2 * qd

        g = myScalarmap()
        df = cDistFunction(g, self.N_)
        tauv = 0.5 + np.random.rand(self.N_)/2.0
        wp = np.random.rand(self.N_ + 1, self.dim_)

        wpidx = [(i, j) for i in range(self.N_ + 1) for j in range(dim)]

        jac = df.jacobian(tauv, wp, wpidx)
        dtau = 1.0e-6
        for i, taui in enumerate(tauv):
            tauv_aux = tauv.copy()
            tauv_aux[i] += -dtau
            v0 = df(tauv_aux, wp).copy()
            tauv_aux[i] += 2 * dtau
            v1 = df(tauv_aux, wp)

            dvdtaui = 0.5 * (v1 - v0) / dtau

            err = np.abs(dvdtaui - jac[:, i])

            e = np.max(err)

            assert e < 1.0e-6

        dwp = 1.0e-6
        for i, (j, k) in enumerate(wpidx, start=self.N_):
            wp_aux = wp.copy()
            wp_aux[j, k] += -3 * dwp
            v0 = df(tauv, wp_aux).copy() * (-1.0 / 60.0)
            wp_aux[j, k] += dwp
            v1 = df(tauv, wp_aux).copy() * (3.0 / 20.0)
            wp_aux[j, k] += dwp
            v2 = df(tauv, wp_aux).copy() * (-3.0 / 4.0)
            wp_aux[j, k] += 2 * dwp
            v3 = df(tauv, wp_aux).copy() * (3.0 / 4.0)
            wp_aux[j, k] += dwp
            v4 = df(tauv, wp_aux).copy() * (-3.0 / 20.0)
            wp_aux[j, k] += dwp
            v5 = df(tauv, wp_aux).copy() * (1.0 / 60.0)

            dvdwp_jk = (v0 + v1 + v2 + v3 + v4 + v5) / dwp

            err = np.abs(dvdwp_jk - jac[:, i])

            e = np.max(err)

            assert e < 1.0e-6

    @debug_on()
    def testSpeed(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
