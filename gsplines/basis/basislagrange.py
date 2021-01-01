"""
    Definition of the fundamental class of functions.
"""
import copy as cp
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial


class cBasisLagrange(object):
    def __init__(self, _dom_points):
        self.dom_points_ = np.atleast_1d(_dom_points)
        self.dim_ = len(self.dom_points_)
        lag_pols = [
            lagrange(self.dom_points_, [
                     1.0 if i == k else 0.0 for i in range(self.dim_)])
            for k in range(self.dim_)
        ]
        self.pols_ = [
            Polynomial(lag.coef[::-1], domain=[-1, 1])
            for lag in lag_pols
        ]
        deriv_matrix = np.zeros(2*(self.dim_,))

        for i in range(self.dim_):
            for j in range(self.dim_):
                pol_deriv = self.pols_[j].deriv()
                ri = self.dom_points_[i]
                deriv_matrix[j, i] = pol_deriv(ri)

        self.Dmat_ = deriv_matrix
        self.buff_ = np.zeros((self.dim_,))

    def derivMatrixOnWindow(self, _tau, _deg):
        res = np.linalg.matrix_power(2.0/_tau * self.Dmat_, _deg)
        return np.ravel(res)

    def evalDerivOnWindow(self, _s, _tau, _deg):
        assert np.isscalar(_s)
        v = self.evalOnWindow(_s, _tau)
        dmat = self.derivMatrixOnWindow(_tau, _deg)
        return np.ravel(dmat.dot(v))

    def evalOnWindow(self, _s, _tau):
        """Eval on window evaluate in [-1, 1]
        returns the cFundFuncBasis instance which contains
        the time derivate of the current instance."""
        assert np.isscalar(_s)
        assert -1.001 <= _s <= 1.001
        result = np.array([pol(_s) for pol in self.pols_])
        return result

    def evalDerivWrtTauOnWindow(self, _s, _tau, _deg=1):
        """Eval on window evaluate in [-1, 1] returns the
        derivates wrt tau of the _deg derivate of each basis
        wrt t"""
        assert np.isscalar(_s)
        # Compute the derivative wet t
        v0 = self.evalDerivOnWindow(_s, _tau, _deg)
        v0 *= -0.5 * _deg * (2.0 / _tau)
        return np.ravel(v0)
