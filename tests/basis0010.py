'''
Test basis, which represents the basis of the Legendre Polynomials
'''
import numpy as np
import sympy as sp
import unittest
from gsplines.basis.basis0010 import cBasis0010
    
class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        ''' Initialize the symbolic expression of the Legendre Polynomials
        '''
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)
        tau = sp.symbols('tau', positive=True)
        s = sp.symbols('s', real=True)
        basis = [0, 0, 0, 0, 0, 0]
        basis[0] = sp.sympify(1.0)
        basis[1] = s
        for i in range(1, 5):
            basis[i + 1] = 1.0 / (
                i + 1.0) * ((2.0 * i + 1.0) * s * basis[i] - i * basis[i - 1])

        self.Bsym_ = basis
        self.tausym_ = tau
        self.ssym_ = s

    def test_value(self):
        ''' Compare the value of the symbolic and the implemented Legendre
        Polynomials'''
        for item in range(0, 100):
            Bimpl = cBasis0010()
            tau_ = 10.0 * np.random.rand()
            s_ = np.random.rand() * 2.0 - 1.0
            B = [Bi.subs({self.tausym_: tau_}) for Bi in self.Bsym_]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            e = np.max(np.abs(B - Bimpl.evalOnWindow(s_, tau_)))
            # print('error = {:.3e}\r'.format(e), end='')

            assert (e < 1.0e-10)


    def test_derivatives_wrt_t(self):
        ''' Compare the derivative w.r.t. t of the symbolic and the implemented
        Legendre Polynomials'''
        for item in range(0, 100):
            Bimpl = cBasis0010()
            tau_ = 10.0 * np.random.rand()
            s_ = np.random.rand() * 2.0 - 1.0
            ddeg = np.random.randint(1, 6)
            B = [
                Bi.diff(self.ssym_, ddeg) * sp.Pow(2 / self.tausym_, ddeg)
                for Bi in self.Bsym_
            ]
            B = [Bi.subs({self.tausym_: tau_}) for Bi in B]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            e = np.max(np.abs(B - Bimpl.evalDerivOnWindow(s_, tau_, ddeg)))

            assert (
                e < 5.0e-3
            ), 'Large error on derivatives wrt t,    error = {:+.3e}'.format(e)


    def test_derivatives_wrt_tau(self):
        ''' Compare the derivative w.r.t. tau of the symbolic and the implemented
        Legendre Polynomials'''
        for item in range(0, 100):
            Bimpl = cBasis0010()
            tau_ = 10.0 * np.random.rand()
            s_ = np.random.rand() * 2.0 - 1.0
            ddeg = np.random.randint(1, 6)
            B = [
                Bi.diff(self.ssym_, ddeg) * sp.Pow(2 / self.tausym_, ddeg)
                for Bi in self.Bsym_
            ]
            B = [Bi.diff(self.tausym_) for Bi in B]
            B = [Bi.subs({self.tausym_: tau_}) for Bi in B]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            e = np.max(
                np.abs(B - Bimpl.evalDerivWrtTauOnWindow(s_, tau_, ddeg)))

            assert (
                e < 5.0e-2
            ), 'Large error on derivatives wrt tau error = {:+.3e}'.format(e)


    def test_l2_norms(self):
        ''' Test L2 norms '''
        tau = np.random.rand()*5.0
        Bimpl = cBasis0010()
        Qd3 = np.zeros((6, 6))
        Qd1 = np.zeros((6, 6))
        Bimpl.l2_norm(tau, Qd3, 3)
        Bimpl.l2_norm(tau, Qd1, 1)
        y = np.random.rand(Bimpl.dim_)

        def qd3norm2(s):
            res = Bimpl.evalDerivOnWindow(s, tau, 3).dot(y)
            return np.power(res, 2.0)*tau/2.0
        def qd1norm2(s):
            res = Bimpl.evalDerivOnWindow(s, tau, 1).dot(y)
            return np.power(res, 2.0)*tau/2.0

        dt = 5.0e-6
        for f, Q in [(qd1norm2, Qd1), (qd3norm2, Qd3)]:
            fv = np.array([f(t) for t in np.arange(-1, 1, dt)])
            Itest = np.sum(fv[1:]+fv[:-1])*dt/2.0



            Inom = Q.dot(y).dot(y)

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
