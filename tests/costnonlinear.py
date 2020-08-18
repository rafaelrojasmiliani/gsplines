"""
    Test the cost function from the problem 1010
"""
import numpy as np
import sympy as sp
import quadpy
import unittest
from opttrj.costnonlinear import cCostNonLinear

from itertools import tee


class cMyCost(cCostNonLinear):
    def runningCost(self, _t, _tauv, _u):
        pass

    def runningCostGradient(self, _t, _tauv, _u):
        pass


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.random.seed()
        self.N_ = np.random.randint(2, 6)
        self.dim_ = np.random.randint(2, 8)
        self.wp_ = np.random.rand(self.N_ + 1, self.dim_)
        self.T_ = 100.0

    def testWaypoints(self):
        wp = np.random.rand(self.N_ + 1, 2)
        Ni = 10
        Ngl = 10
        cost = cMyCost(wp, self.T_, Ni, Ngl)
        from matplotlib import pyplot as plt
        plt.plot(wp[:, 0], wp[:, 1], 'b-')
        plt.plot(cost.wp_[:, 0], cost.wp_[:, 1], 'ro')
        plt.plot(wp[:, 0], wp[:, 1], 'b*')

        plt.show()

    def test_u2wp(self):
        dim = 2
        wp = np.random.rand(self.N_ + 1, dim)
        Ni = np.random.randint(1, 10)
        Ngl = 10
        cost = cMyCost(wp, self.T_, Ni, Ngl)
        u = np.zeros((cost.ushape_, ))
        u = cost.wp2u(u)
        U = u.reshape(-1, dim)
        from matplotlib import pyplot as plt
        plt.plot(wp[:, 0], wp[:, 1], 'b-')
        plt.plot(U[:, 0], U[:, 1], 'ro')
        plt.plot(wp[:, 0], wp[:, 1], 'b*')
        plt.plot(cost.wp_[:, 0], cost.wp_[:, 1], 'g+')

        plt.title('Ni = {:d}, N = {:d}'.format(Ni, self.N_))
        plt.show()
        plt.clf()
        u += 0.01 * (np.random.rand(u.shape[0]) - 0.5)
        u = cost.u2wp(u)
        plt.plot(wp[:, 0], wp[:, 1], 'b-')
        plt.plot(wp[:, 0], wp[:, 1], 'b*')
        plt.plot(cost.wp_[:, 0], cost.wp_[:, 1], 'g+')

        plt.title('Ni = {:d}, N = {:d}'.format(Ni, self.N_))
        plt.show()

    def test_uwpindexing(self):
        Ni = 10
        Ngl = 10
        cost = cMyCost(self.wp_, self.T_, Ni, Ngl)
        u = np.zeros((cost.ushape_, ))
        u = cost.wp2u(u)

        for ui, wipx_i in enumerate(cost.uToWp_):
            e = abs(cost.wp_[wipx_i[0], wipx_i[1]] - u[ui])
            assert e < 1.0e-10

        u = np.random.rand(cost.ushape_)

        cost.u2wp(u)

        for ui, wipx_i in enumerate(cost.uToWp_):
            e = abs(cost.wp_[wipx_i[0], wipx_i[1]] - u[ui])
            assert e < 1.0e-10

    def test_run_eval_grad(self):

        Ni = 10
        Ngl = 10

        cost = myCost(self.wp_, self.T_, Ni, Ngl)

        u = np.random.rand(cost.ushape_)
        tauv = 0.5 + np.random.rand(cost.N_)

        mygradient = np.vectorize(
            lambda t, inter: cost.runningCostGradient(t, tauv, u),
            signature='(),()->(n)')

        grad = mygradient(0.0, 0.0)
        assert grad.ndim == 1 and grad.shape[0] == cost.ushape_ + cost.N_
        grad = mygradient([0, 1, 2], 0.0)
        assert grad.ndim == 2 and grad.shape[1] == cost.ushape_ + \
            cost.N_ and grad.shape[0] == 3

        x = np.hstack([tauv, u])
        res = cost(x)
        assert np.isscalar(res)
        res = cost.gradient(x)
        assert res.ndim == 1 and res.shape[0] == cost.ushape_ + cost.N_

    def testdomain2window(self):
        Ni = 10
        Ngl = 10

        cost = myCost(self.wp_, self.T_, Ni, Ngl)
        tauv = 0.5 + np.random.rand(cost.N_)

        t0 = 0.0
        for iinter, taui in enumerate(tauv):
            tf = t0 + taui
            tarray = np.arange(t0, tf, 0.05)[1:]
            for t in tarray:
                s, taui2, iinter2 = cost.domain2window(t, tauv)

                assert iinter2 == iinter and taui2 - taui < 1.0e-9, '''
                Interval fro domain2window (Nominal) = {:d}
                Interval fro iteration     (Testing) = {:d}
                size of tauv                         = {:d}
                taui Nominal                         = {:14.7e}
                taui Test                            = {:14.7e}
                sNom                                 = {:14.7e}
                t                                    = {:14.7e}
                t0                                   = {:14.7e}
                tf                                   = {:14.7e}
                '''.format(iinter2, iinter, tauv.shape[0], taui2, taui, s, t,
                           t0, tf)
            t0 = tf


class myCost(cCostNonLinear):
    def __init__(self, _wp, _T, _Ni, _Ngl):
        super().__init__(_wp, _T, _Ni, _Ngl)
        self.runninf_cost_gradient_buff = np.zeros((self.ushape_ + self.N_, ))

    def runningCost(self, _t, _tauv, _u, _y=None, _inter=None):
        return 0.0

    def runningCostGradient(self,
                            _t,
                            _tauv,
                            _u,
                            _y=None,
                            _inter=None,
                            _dydtau=None,
                            _dydu=None):
        return self.runninf_cost_gradient_buff


def main():
    unittest.main()


if __name__ == '__main__':
    main()
