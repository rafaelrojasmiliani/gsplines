"""
"""
import numpy as np
import sympy as sp
import quadpy
import unittest
from opttrj.costarclenjerk import cCostArcLenJerk
from opttrj.opttrj0010 import opttrj0010
from itertools import tee
import sys


def pairwise(iterable):
    '''s -> (s0,s1), (s1,s2), (s2, s3), ...'''
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+14.7e}'.format},
            threshold=sys.maxsize)
        np.random.seed()
        self.N_ = np.random.randint(2, 4)
        self.dim_ = np.random.randint(2, 3)
        self.wp_ = (np.random.rand(self.N_ + 1, self.dim_) - 0.5) * 2.0
        self.T_ = 10.0
        self.Ni_ = 3
        self.Ngl_ = 30

    def __testValue(self):

        cost = cCostArcLenJerk(self.wp_, self.T_, self.Ni_, self.Ngl_)
        wp = cost.wp_.copy()
        print('''\n---- Computing min jerk for cost value testing ----
                   ---- N= {:d}, dim = {:d}----'''.format(
            wp.shape[0] - 1, wp.shape[1]))
        q = opttrj0010(wp, self.T_, _printPerformace=True)
        qd = q.deriv()
        qddd = q.deriv(3)

        def runningCost(_t):
            if np.isscalar(_t):
                qd_ = np.linalg.norm(qd(_t)[0])
                qddd_ = np.linalg.norm(qddd(_t)[0])
                res = np.power(qd_ * qddd_, 2)
            else:
                qd_ = qd(_t)
                qddd_ = qddd(_t)
                res0 = np.einsum('ij,ij->i', qd_, qd_)
                res1 = np.einsum('ij,ij->i', qddd_, qddd_)
                res = np.multiply(res0, res1)
            return res

        ytest = q.y_
        tauv = q.tau_
        ts = np.arange(0, self.T_, 0.5)
        u = np.zeros((self.Ni_ * self.N_ * self.dim_, ))
        u = cost.wp2u(u)

        ynom = cost.waypointConstraints(tauv, u)
        assert np.linalg.norm(ytest - ynom) < 1.0e-8
        # Test value of the runnign cost
        print('---- Testing value of the running cost function ----')
        for ti in ts:
            rctest = runningCost(ti)
            rcnom = cost.runningCost(ti, tauv, u)
            e = abs(rctest - rcnom)
            assert e < 1.0e-8, '''
            error   = {:14.7e}
            test    = {:14.7e}
            nominal = {:14.7e}'''.format(e, rctest, rcnom)

        print('    Value of the running cost Ok')

        x = np.hstack([tauv, u])
        Inom = cost(x)
        err = 1.e100
        badtrentCounter = 0
        print('---- Testing value of the cost function ----')
        for Ngl in range(100, 500, 30):
            scheme = quadpy.line_segment.gauss_legendre(Ngl)
            time_partition = np.linspace(0, self.T_, cost.N_)
            Itest = 0.0
            for t0, tf in pairwise(time_partition):
                Itest += scheme.integrate(runningCost, [t0, tf])
            if abs(Itest - Inom) > err:
                badtrentCounter += 1
            else:
                badtrentCounter = 0
            assert badtrentCounter < 3
            e = abs(Itest - Inom)
            ep = e / Itest
            assert ep < 1.0e-6, '''
            error   = {:14.7e}
            test    = {:14.7e}
            nominal = {:14.7e}'''.format(e, Itest, Inom)
        print('    Value of the cost function Ok')

    def testGradient(self):

        cost = cCostArcLenJerk(self.wp_, self.T_, self.Ni_, self.Ngl_)
        wp = cost.wp_.copy()
        print('''\n---- Computing min jerk for gradient testing ----
                   ---- N= {:d}, dim = {:d}----'''.format(
            wp.shape[0] - 1, wp.shape[1]))
        q = opttrj0010(wp, self.T_, _printPerformace=True)
        qd = q.deriv()
        qddd = q.deriv(3)

        def runningCost(_t):
            if np.isscalar(_t):
                qd_ = np.linalg.norm(qd(_t)[0])
                qddd_ = np.linalg.norm(qddd(_t)[0])
                res = np.power(qd_ * qddd_, 2)
            else:
                qd_ = qd(_t)
                qddd_ = qddd(_t)
                res0 = np.einsum('ij,ij->i', qd_, qd_)
                res1 = np.einsum('ij,ij->i', qddd_, qddd_)
                res = np.multiply(res0, res1)
            return res

        tauv = q.tau_
        gradTest = np.zeros((cost.N_ + cost.ushape_, ))
        u = np.zeros((cost.ushape_, ))
        cost.wp2u(u)
        du = 1.0e-8
        for j_u in range(cost.ushape_):
            print('computinf gradietn w.r.t u_{:d}'.format(j_u))
            u_aux = u.copy()
            u_aux[j_u] += -du
            x = np.hstack([tauv, u_aux])
            I0 = cost(x)
            u_aux[j_u] += 2.0 * du
            x = np.hstack([tauv, u_aux])
            I1 = cost(x)
            gradTest[j_u + cost.N_] = 0.5 * (I1 - I0) / du

        dtau = 1.0e-8
        for j_tau in range(0, cost.N_):
            print('computinf gradietn w.r.t tau_{:d}'.format(j_tau))
            tauv_aux = tauv.copy()
            tauv_aux[j_tau] += -2.0 * dtau
            x = np.hstack([tauv_aux, u])
            I0 = cost(x) * (1.0 / 12.0)
            tauv_aux[j_tau] += dtau
            x = np.hstack([tauv_aux, u])
            I1 = cost(x) * (-2.0 / 3.0)
            tauv_aux[j_tau] += 2.0 * dtau
            x = np.hstack([tauv_aux, u])
            I2 = cost(x) * (2.0 / 3.0)
            tauv_aux[j_tau] += dtau
            x = np.hstack([tauv_aux, u])
            I3 = cost(x) * (-1.0 / 12.0)
            gradTest[j_tau] = (I0 + I1 + I2 + I3) / dtau

        x = np.hstack([tauv, u])
        gradNom = cost.gradient(x)

        ev = np.abs(gradNom - gradTest)
        e = np.max(ev)
        epNom = e / np.max(gradNom)
        epTest = e / np.max(gradTest)
        print('Error')
        print(ev)
        print('Nominal Value')
        print(gradNom)
        print('Test Value')
        print(gradTest)

        assert e < 1.0e-5, '''
        Maximum error                = {:14.7e}
        Error relative to nomial val = {:14.7e}
        Error relative to test val   = {:14.7e}
        '''.format(e, epNom, epTest)

    def testaQderivatives(self):

        cost = cCostArcLenJerk(self.wp_, self.T_, self.Ni_, self.Ngl_)

        dtaui = 0.0001

        for i in range(0, 100):

            taui = 0.1 + np.random.rand() * 2
            s = np.random.rand() * 2.0 - 1.0

            Q10 = cost.buildQ1(s, taui - 2 * dtaui) * (1.0 / 12.0)
            Q11 = cost.buildQ1(s, taui - dtaui) * (-2.0 / 3.0)
            Q12 = cost.buildQ1(s, taui + dtaui) * (2.0 / 3.0)
            Q13 = cost.buildQ1(s, taui + 2 * dtaui) * (-1.0 / 12.0)

            dQ1dtauTest = (Q10 + Q11 + Q12 + Q13) / dtaui

            dQ1dtauNom = cost.buildQ1(s, taui, 'derivative_tau')

            ev = np.abs(dQ1dtauTest - dQ1dtauNom)
            epTest = np.max(np.divide(ev, np.max(np.abs(dQ1dtauTest))))
            epNom = np.max(np.divide(ev,  np.max(np.abs(dQ1dtauNom))))
            e = np.max(ev)

            from textwrap import dedent
            assert epTest < 1.0e-8 and epNom < 1.0e-8, dedent('''
                        Error
                        {}
                        Nominal Value
                        {}
                        Test Value
                        {}
                        {}
                      ''').format(
                *[np.array2string(v) for v in [ev, dQ1dtauNom, dQ1dtauTest, epTest]])

            Q30 = cost.buildQ3(s, taui - 2 * dtaui) * (1.0 / 12.0)
            Q31 = cost.buildQ3(s, taui - dtaui) * (-2.0 / 3.0)
            Q32 = cost.buildQ3(s, taui + dtaui) * (2.0 / 3.0)
            Q33 = cost.buildQ3(s, taui + 2 * dtaui) * (-1.0 / 12.0)

            dQ3dtauTest = (Q30 + Q31 + Q32 + Q33) / dtaui

            dQ3dtauNom = cost.buildQ3(s, taui, 'derivative_tau')

            ev = np.abs(dQ3dtauTest - dQ3dtauNom)
            e = np.max(ev)
            epTest = np.max(np.divide(ev, np.max(np.abs(dQ3dtauTest))))
            epNom = np.max(np.divide(ev,  np.max(np.abs(dQ3dtauNom))))

            from textwrap import dedent
            assert epTest < 1.0e-8 and epNom < 1.0e-8, dedent('''
                        Error
                        {}
                        Nominal Value
                        {}
                        Test Value
                        {}
                        {}
                      ''').format(
                *[np.array2string(v) for v in [ev, dQ3dtauNom, dQ3dtauTest, epTest]])


#    def testFirstGuess(self):
#
#        wp = np.random.rand(self.N_+1, 2)
#        cost = cCostArcLenJerk(wp, self.T_, self.Ni_, self.Ngl_)
#
#        x0 = cost.getFirstGuess()
#
#        tauv0 = x0[:cost.N_]
#        u0 = x0[cost.N_:]
#
#        qminjerk = cost.qminjerk_
#
#        wp2 = cost.wp_
#
#        t = np.arange(0, self.T_, 0.001)
#        q_ = qminjerk(t)
#
#        from matplotlib import pyplot as plt
#
#        plt.plot(wp2[:, 0], wp2[:, 1], 'ro')
#
#        plt.plot(q_[:, 0], q_[:, 1], 'b')
#
#
#        plt.show()
#


def main():
    unittest.main()


if __name__ == '__main__':
    main()
