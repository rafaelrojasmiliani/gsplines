"""
    Test functions in the space of solutions of the
    Euler Lagrange equations of

    \int_{-1}^{1} (2/tau) \alpha dq/ds + (2/tau)^5 (1-\alpha) d^3 q / ds^3 dt
"""
import numpy as np
import sympy as sp
import unittest
from gsplines.basis1010 import cBasis1010
from gsplines.piece import cPiece


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)

    def test(self):
        basis = cBasis1010
        coeff = np.random.rand(basis.dim_)
        domain = (np.random.rand(2) + np.array([-2, 1])) * 1
        tau = domain[1] - domain[0]
        alpha = np.random.rand()
        basis = cBasis1010(alpha)
        piece = cPiece(coeff, domain, basis)

        B = cBasis1010(alpha)

        t = np.arange(domain[0], domain[1], 0.2)
        res0 = piece(t)

        s = 2.0 * (t - domain[0]) / tau - 1.0
        res1 = np.array([coeff.dot(B.evalOnWindow(si, tau)) for si in s])

        e = np.max(np.abs(res0 - res1))

        assert e < 1.0e-5

    def test_basis(self):
        pass


#        '''
#            Here we rest the correctness of the numerical output of the basis
#            class comparing it with its analitical form optained using sympy
#        '''
#
#        for item in range(0, 100):
#            a = np.random.rand()
#            Bimpl = cBasis1010(a)
#            tau_ = 10.0 * np.random.rand()
#            s_ = np.random.rand() * 2.0 - 1.0
#            B = [
#                Bi.subs({
#                    self.alphasym_: a,
#                    self.tausym_: tau_
#                }) for Bi in self.Bsym
#            ]
#            B = [sp.lambdify(self.ssym_, Bi) for Bi in B]
#            B = np.array([Bi(s_) for Bi in B])
#
#            e = np.max(B - Bimpl.evalOnWindow(s_, tau_))
#            # print('error = {:.3e}\r'.format(e), end='')
#
#            assert (e < 1.0e-10)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
