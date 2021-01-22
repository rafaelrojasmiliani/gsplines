"""
    Test functions in the space of solutions of the
    Euler Lagrange equations of

    \int_{-1}^{1} (2/tau) \alpha dq/ds + (2/tau)^5 (1-\alpha) d^3 q / ds^3 dt
"""
import unittest
from gsplines.basis import cBasis1010
from gsplines.basis import cBasis0010
from gsplines.basis import cBasisLagrange
from gsplines.operations import FunctionSum
import numpy as np
from numpy.random import randint
from random import choice

from gsplines.interpolator import rand_interpolate
from .tools import debug_on


class cMyTest(unittest.TestCase):
    @debug_on()
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        import sys
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)
        self.dim_ = 4
        basis = [cBasis1010(0.9), cBasis0010()]

        self.functions_ = [rand_interpolate(
            randint(2, 10), 3, choice(basis)) for _ in range(30)]
        self.scalfunctions_ = [rand_interpolate(
            randint(2, 8), 1, choice(basis)) for _ in range(30)]
        self.exec_time_ = self.functions_[0].T_

    @debug_on()
    def test_sum(self):
        res = choice(self.functions_)

        time_span = np.arange(0, self.exec_time_, 0.05)

        for _ in range(1):
            f2 = choice(self.functions_)

            f3 = FunctionSum(res, f2)

            for deg in range(5):

                fun = f3.deriv(deg)

                nom_val = res.deriv(deg)(time_span) + f2.deriv(deg)(time_span)

                test_val = f3.deriv(deg)(time_span)

                err = nom_val - test_val
                assert np.linalg.norm(err, ord=np.inf) < 1.0e-10


if __name__ == '__main__':
    unittest.main()
