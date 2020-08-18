"""
    Test the cost function from the problem 1010
"""
import numpy as np
import sympy as sp
import unittest
from scipy.special import roots_legendre
from opttrj.cost1010 import cCost1010

from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
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
        cost = cCost1010(_wp=wp, _alpha=0.5, _T=T)

        res = np.zeros((N, ))
        for i in range(5):
            a = cost(tauv)
            g = cost.gradient(tauv, res)

        cost.printPerformanceIndicators()

    def testValue(self):
        N = np.random.randint(2, 60)
        dim = np.random.randint(1, 8)
        wp = np.random.rand(N + 1, dim)
        alpha = np.random.rand()
        tauv = np.random.rand(N)
        T = np.sum(tauv)
        cost = cCost1010(_wp=wp, _alpha=alpha, _T=T)
        q = cost.splcalc_.getSpline(tauv, wp)

        qddd = q.deriv(3)
        qd = q.deriv(1)

        def qd3norm(t):
            return np.einsum('ij,ij->i', qddd(t), qddd(t))

        def qdnorm(t):
            return np.einsum('ij,ij->i', qd(t), qd(t))

        def runningcost(t):
            return alpha * qdnorm(t) + (1-alpha)*qd3norm(t)

        err = 1.e100
        Inom = cost(tauv)
        badtrentCounter = 0
        for Ngl in range(100, 500, 30):
            lr, lw = roots_legendre(Ngl)
            time_partition = np.linspace(0, T, 50)
            Itest_1 = 0.0
            for t0, tf in pairwise(time_partition):
                Itest_1 += sum([w*runningcost(s)[0]*(tf-t0)/2.0 for w, s in zip(lw, (lr+1.0)/2.0*(tf-t0)+t0)])
            Itest_2 = alpha*q.l2_norm(1) + (1.0 - alpha) * q.l2_norm(3)
            if abs(Itest_1 - Inom) > err:
                badtrentCounter += 1
            else:
                badtrentCounter = 0
            assert badtrentCounter < 3
            err = abs(Itest_1 - Inom)
            err2 = abs(Itest_2 - Inom)
            print('Error w.r.t. quadrature = {:.3f}'.format(err))
            print('Error w.r.t. gspl. impl = {:.3f}'.format(err2))
            if err < 1.0e-4:
                break



def main():
    unittest.main()


if __name__ == '__main__':
    main()
