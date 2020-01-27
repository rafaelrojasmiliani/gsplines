"""
"""
import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
import unittest
from gsplines.basis0010 import cBasis0010
from gsplines.basis0010 import cBasis0010canonic
from gsplines.piece import cPiece


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
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

        tau52 = sp.Pow(tau, 5 / 2)
        from sympy import sqrt
        basis = [
            tau52 / 8,
            sqrt(3) * s * tau52 / 8,
            3 * sqrt(5) * tau52 * (sp.Pow(s, 2) - 1 / 3) / 16,
            sp.Pow(s, 3) * tau52 / 48,
            sqrt(3) * sp.Pow(s, 4) * tau52 / 192,
            sqrt(5) * tau52 * (sp.Pow(s, 5) - 10 * sp.Pow(s, 3) / 3) / 320
        ]

        self.BsymCan_ = basis

    def test_basis(self):
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

            Bimpl = cBasis0010canonic()

            B = [Bi.subs({self.tausym_: tau_}) for Bi in self.BsymCan_]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            e = np.max(np.abs(B - Bimpl.evalOnWindow(s_, tau_)))
            # print('error = {:.3e}\r'.format(e), end='')

            assert (e < 1.0e-10)
        pass

    def test_derivatives_wrt_t(self):
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

            Bimpl = cBasis0010canonic()
            B = [
                Bi.diff(self.ssym_, ddeg) * sp.Pow(2 / self.tausym_, ddeg)
                for Bi in self.BsymCan_
            ]
            B = [Bi.subs({self.tausym_: tau_}) for Bi in B]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            Bnom = Bimpl.evalDerivOnWindow(s_, tau_, ddeg)
            e = np.max(np.abs(B - Bnom))

            assert (e < 5.0e-3), '''Large error on derivatives wrt t,
                    error = {:+.3e}
                    deg   = {:d}
                    Bnom  = {}
                    Btest = {}
                '''.format(e, ddeg, np.array2string(Bnom), np.array2string(B))

    def test_derivatives_wrt_tau(self):
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

            Bimpl = cBasis0010canonic()
            B = [
                Bi.diff(self.ssym_, ddeg) * sp.Pow(2 / self.tausym_, ddeg)
                for Bi in self.BsymCan_
            ]
            B = [Bi.diff(self.tausym_) for Bi in B]
            B = [Bi.subs({self.tausym_: tau_}) for Bi in B]
            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
            B = np.array([Bi(s_) for Bi in B])

            Bnom = Bimpl.evalDerivWrtTauOnWindow(s_, tau_, ddeg)
            e = np.max(np.abs(B - Bnom))

            assert (e < 5.0e-2), '''
            Large error on derivatives wrt tau
            error = {:+.3e}
            BTest = {}
            Bnom  = {}
            deg   = {:d}
            '''.format(e, np.array2string(B), np.array2string(Bnom), ddeg)

    def testintegration(self):
        tau_ = 10.0 * np.random.rand()
        ddeg = 3
        B = [
            Bi.diff(self.ssym_, ddeg) * sp.Pow(2 / self.tausym_, ddeg)
            for Bi in self.BsymCan_
        ]
        B = [Bi.subs({self.tausym_: tau_}) for Bi in B]
        B = [sp.lambdify(self.ssym_, Bi) for Bi in B]

        BBNom = np.eye(6)
        for i in range(0, 3):
            BBNom[i, i] = 0.0
        BBTest = np.zeros(2 * (6, ))
        import quadpy
        scheme = quadpy.line_segment.gauss_legendre(1000)

        for i in range(6):
            for j in range(6):

                def BBfun(s):
                    return B[i](s) * B[j](s)

                f = np.vectorize(BBfun, signature='()->()')
                BBTest[i, j] = scheme.integrate(f, [-1, 1])*tau_/2

        ev = np.abs(BBTest - BBNom)

        print('')
        print(ev)
        e = np.max(ev)
        print(e)

        assert e < 1.0e-9

    def testspline(self):
        from gsplines.gspline import cSplineCalc
        print('Test continuity constraints with plot')
        for i in range(3):
            dim = np.random.randint(2, 3)
            N = np.random.randint(3, 10)
            wp = (np.random.rand(N + 1, dim) - 0.5) * 2 * np.pi
            tauv = 0.5 + np.random.rand(N) * 3.0
            tis = [np.sum(tauv[0:i]) for i in range(0, N + 1)]
            T = np.sum(tauv)
            splcalc = cSplineCalc(dim, N, cBasis0010())
            spln = splcalc.getSpline(tauv, wp)
            from matplotlib import pyplot as plt

            t = np.arange(0, T, 0.005)
            q_list = [spln.deriv(i)(t) for i in range(0, 6)]
            q_num = [spln(t)]
            for i in range(0, 6):
                q_num.append((q_num[i][1:, :] - q_num[i][:-1, :]) / 0.005)

            fig, axs = plt.subplots(6, dim)

            for i in range(0, 6):
                for j in range(0, dim):
                    axs[i, j].plot(t, q_list[i][:, j])
                    if i == 0:
                        tt = t
                    else:
                        tt = t[:-i]
                    axs[i, j].plot(tt, q_num[i][:, j], 'm+')
                    axs[i, j].grid()
                    for ti in tis:
                        axs[i, j].axvline(x=ti, color='b', linestyle='--')

            plt.show()

        for i in range(3):
            dim = np.random.randint(2, 3)
            N = np.random.randint(3, 10)
            wp = (np.random.rand(N + 1, dim) - 0.5) * 2 * np.pi
            tauv = 0.5 + np.random.rand(N) * 3.0
            tis = [np.sum(tauv[0:i]) for i in range(0, N + 1)]
            T = np.sum(tauv)
            splcalc = cSplineCalc(dim, N, cBasis0010canonic())
            spln = splcalc.getSpline(tauv, wp)

            t = np.arange(0, T, 0.005)
            q_list = [spln.deriv(i)(t) for i in range(0, 6)]
            q_num = [spln(t)]
            for i in range(0, 6):
                q_num.append((q_num[i][1:, :] - q_num[i][:-1, :]) / 0.005)

            fig, axs = plt.subplots(6, dim)

            for i in range(0, 6):
                for j in range(0, dim):
                    axs[i, j].plot(t, q_list[i][:, j])
                    if i == 0:
                        tt = t
                    else:
                        tt = t[:-i]
                    axs[i, j].plot(tt, q_num[i][:, j], 'm+')
                    axs[i, j].grid()
                    for ti in tis:
                        axs[i, j].axvline(x=ti, color='b', linestyle='--')

            plt.show()


def main():
    unittest.main()


if __name__ == '__main__':
    main()
