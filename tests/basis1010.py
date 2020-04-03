"""
    Test functions in the space of solutions of the
    Euler Lagrange equations of

    \int_{-1}^{1} (2/tau) \alpha dq/ds + (2/tau)^5 (1-\alpha) d^3 q / ds^3 dt
"""
import unittest
import numpy as np
import sympy as sp
from gsplines.basis1010 import cBasis1010


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)
        s, a, tau = sp.symbols('s a tau', positive=True)

        k = sp.sqrt(2) / 4.0 * sp.Pow((a / (1.0 - a)), 0.25)
        p = tau * k * s
        B0 = sp.sympify(1)
        B1 = p
        B2 = sp.sin(p) / sp.exp(p)
        B3 = sp.cos(p) / sp.exp(p)
        B4 = sp.sin(p) * sp.exp(p)
        B5 = sp.cos(p) * sp.exp(p)
        self.Bsym = [B5, B4, B3, B2, B1, B0]
        self.ssym_ = s
        self.alphasym_ = a
        self.tausym_ = tau


    def test(self):
        self.test_basis()

    def test_basis(self):
        '''
            Here we rest the correctness of the numerical output of the basis
            class comparing it with its analitical form optained using sympy
        '''

        for item in range(0, 100):
            a = np.random.rand()
            Bimpl = cBasis1010(a)
            tau_ = 10.0 * np.random.rand()
            s_ = np.random.rand() * 2.0 - 1.0
            B = [
                Bi.subs({
                    self.alphasym_: a,
                    self.tausym_: tau_
                }) for Bi in self.Bsym
            ]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            e = np.max(B - Bimpl.evalOnWindow(s_, tau_))
            # print('error = {:.3e}\r'.format(e), end='')

            assert (e < 1.0e-10)


    def test_l2_norms(self):
        ''' Test L2 norms '''
        tau = np.random.rand()*5.0
        a = 0.9
        Bimpl = cBasis1010(a)
        Qd3 = np.zeros((6, 6))
        Qd1 = np.zeros((6, 6))
        Bimpl.l2_norm(tau, Qd1, 1)
        Bimpl.l2_norm(tau, Qd3, 3)
        y = np.random.rand(Bimpl.dim_)

        def qd3norm2(s):
            res = Bimpl.evalDerivOnWindow(s, tau, 3).dot(y)
            return np.power(res, 2.0)*tau/2.0
        def qd1norm2(s):
            res = Bimpl.evalDerivOnWindow(s, tau, 1).dot(y)
            return np.power(res, 2.0)*tau/2.0

        dt = 5.0e-5
        i = 0
        for f, Q in [(qd1norm2, Qd1), (qd3norm2, Qd3)]:
            i += 1
            fv = np.array([f(t) for t in np.arange(-1, 1, dt)])
            Itest = np.sum(fv[1:]+fv[:-1])*dt/2.0

            Inom = Q.dot(y).dot(y)

            err = np.abs(Itest - Inom)
            assert err < 5.0e-2, '''
            Error in integral = {:.4f}
            Nominal integral = {:.4f}
            Test integral = {:.4f}
            Error in {:d}
            '''.format(err, Inom, Itest, i)
def main():
    unittest.main()


if __name__ == '__main__':
    main()
