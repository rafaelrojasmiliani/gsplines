"""
    Test the cost function from the problem 1010
"""
import numpy as np
import sympy as sp
import quadpy
import unittest
from opttrj.cost1000 import cCost1000

from itertools import tee


def pairwise(iterable):
    '''s -> (s0,s1), (s1,s2), (s2, s3), ...'''
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)

    def testRun(self):
        N = 20  # np.random.randint(2, 10)
        dim = 8  # np.random.randint(1, 8)
        wp = np.random.rand(N + 1, dim)
        tauv = np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost1000(wp, T)

        res = np.zeros((N, ))
        for i in range(5):
            a = cost(tauv)
            g = cost.gradient(tauv, res)

        cost.printPerformanceIndicators()

    def testValue(self):
        N = np.random.randint(2, 60)
        dim = np.random.randint(1, 8)
        wp = np.random.rand(N + 1, dim)
        tauv = np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost1000(wp, T)
        q = cost.splcalc_.getSpline(tauv, wp)

        qd = q.deriv(1)

        def qd1norm(t):
            return np.einsum('ij,ij->i', qd(t), qd(t))

        def runningcost(t):
            return qd1norm(t)

        Inom = cost(tauv)
        err = 1.e100
        badtrentCounter = 0
        print('')
        for Ngl in range(100, 500, 30):
            scheme = quadpy.line_segment.gauss_legendre(Ngl)
            time_partition = np.linspace(0, T, 50)
            Itest = 0.0
            for t0, tf in pairwise(time_partition):
                Itest += scheme.integrate(runningcost, [t0, tf])
            if abs(Itest - Inom) > err:
                badtrentCounter += 1
            else:
                badtrentCounter = 0
            assert badtrentCounter < 3
            err = abs(Itest - Inom)
            print('Test value    = {:.3f}, Error = {:.3f}'.format(Itest, err))
            print('Nominal value = {:.3f}'.format(Inom))
            if err < 1.0e-4:
                break

    def testGradient(self):
        ''' Compare the value of the integral with the finite-difference
        version'''
        print('')

        N = np.random.randint(5, 10)
        dim = np.random.randint(1, 5)
        wp = np.random.rand(N + 1, dim)
        tauv = 0.5+2.0*np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost1000(wp, T)

        dtau = 1.0e-7
        gradTest = np.zeros(tauv.shape)
        for j_tau in range(0, N):
            print('computing partial derivative w.r.t tau_{:d}'.format(j_tau))
            tauv_aux = tauv.copy()
            tauv_aux[j_tau] += -2.0 * dtau
            I0 = cost(tauv_aux) * (1.0 / 12.0)
            tauv_aux[j_tau] += dtau
            I1 = cost(tauv_aux) * (-2.0 / 3.0)
            tauv_aux[j_tau] += 2.0 * dtau
            I2 = cost(tauv_aux) * (2.0 / 3.0)
            tauv_aux[j_tau] += dtau
            I3 = cost(tauv_aux) * (-1.0 / 12.0)

            gradTest[j_tau] = (I0 + I1 + I2 + I3) / dtau

        res = np.zeros(gradTest.shape)

        gradNom = cost.gradient(tauv, res)

        ev = np.abs(gradNom - gradTest)

        e = np.max(ev)

        assert e < 1.0e-6, '''
            Error in gradient:
            Test Gradient    = {}
            Nominal Gradient = {}
        '''.format(*[np.array2string(vi) for vi in [gradTest, gradNom]])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
