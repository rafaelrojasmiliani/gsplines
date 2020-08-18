"""
    Test the cost function from the problem 1010 canonic
"""
import numpy as np
import sympy as sp
import quadpy
import unittest
from opttrj.cost0010canonic import cCost0010Canonic

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
        N = 10  # np.random.randint(2, 10)
        dim = 6  # np.random.randint(1, 8)
        wp = np.random.rand(N + 1, dim)
        tauv = np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost0010Canonic(wp, T)

        res = np.zeros((N, ))
        for i in range(5):
            a = cost(tauv)
            g = cost.gradient(tauv, res)

        cost.printPerformanceIndicators()

    def testValue(self):
        ''' Compare the integral with the value of the class'''
        N = np.random.randint(2, 60)
        dim = np.random.randint(1, 8)
        wp = np.random.rand(N + 1, dim)
        tauv = np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost0010Canonic(wp, T)
        q = cost.splcalc_.getSpline(tauv, wp)

        qddd = q.deriv(3)

        def qd3norm(t):
            return np.einsum('ij,ij->i', qddd(t), qddd(t))

        def runningcost(t):
            return qd3norm(t)

        INominal = cost(tauv)
        err = 1.e100
        badtrentCounter = 0
        for Ngl in range(500, 1000, 50):
            scheme = quadpy.line_segment.gauss_legendre(Ngl)
            time_partition = np.linspace(0, T, 50)
            ITest = 0.0
            for t0, tf in pairwise(time_partition):
                ITest += scheme.integrate(runningcost, [t0, tf])
            if abs(ITest - INominal) > err:
                badtrentCounter += 1
            else:
                badtrentCounter = 0
            assert badtrentCounter < 3
            err = abs(ITest - INominal)
            print('Test Cost function    = {:.3f}'.format(ITest))
            print('Nominal Cost function = {:.3f}'.format(INominal))
            print('Error                 = {:.3f}'.format(err))
#            if err < 1.0e-6:
#                break

    def testGradient(self):
        ''' Compare the value of the integral with the finite-difference
        version'''
        print('')

        N = np.random.randint(2, 6)
        dim = np.random.randint(1, 5)
        wp = np.random.rand(N + 1, dim)
        tauv = 0.5+2.0*np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost0010Canonic(wp, T)

        dtau = 1.0e-7
        gradTest = np.zeros(tauv.shape)
        for j_tau in range(0, N):
            print('computinf gradietn w.r.t tau_{:d}'.format(j_tau))
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
