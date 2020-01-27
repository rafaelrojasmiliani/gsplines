"""
    Definition of the fundamental class of functions.
"""
import copy as cp
import numpy as np


class cBasis1010(object):
    dim_ = 6

    def __init__(self, _params=None):

        assert _params is not None
        self.Dmat_ = np.array(
            [[1, -1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, -1, -1, 0, 0],
             [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
            dtype=np.float)
        self.buff_ = np.ones((6, ))
        self.buff_[5] = 1.0
        self.dim_ = 6
        self.params_ = cp.deepcopy(_params)

    def derivMatrixOnWindow(self, _tau, _deg):
        alpha = self.params_
        k = np.sqrt(2) / 4.0 * np.power(alpha, 0.25) / \
            np.power((1.0 - alpha), 0.25)
        tauk = k * 2.0
        res = np.linalg.matrix_power(tauk * self.Dmat_, _deg)
        return res
        


    def evalDerivOnWindow(self, _s, _tau, _deg):
        assert np.isscalar(_s)
        alpha = self.params_
        k = np.sqrt(2) / 4.0 * np.power(alpha, 0.25) / \
            np.power((1.0 - alpha), 0.25)
        # rememver ds/dt = 2.0/tau
        aux = np.power(2.0 * k, _deg)
        v = self.evalOnWindow(_s, _tau)
        return np.ravel(aux * np.linalg.matrix_power(self.Dmat_, _deg).dot(v))

    def evalOnWindow(self, _s, _tau):
        """
        Eval on window evaluate in [-1, 1]
        returns the cFundFuncBasis  instance which contains the time
        derivate of the current instance.
        """
        assert np.isscalar(_s)
        alpha = self.params_
        k = np.sqrt(2) / 4.0 * np.power(alpha, 0.25) / \
            np.power((1.0 - alpha), 0.25)
        p = _tau * k * _s
        expp = np.exp(p)
        cosp = np.cos(p)
        sinp = np.sin(p)
        self.buff_[0] = expp * cosp
        self.buff_[1] = expp * sinp
        self.buff_[2] = cosp / expp
        self.buff_[3] = sinp / expp
        self.buff_[4] = p

        return self.buff_

    def evalDerivWrtTauOnWindow(self, _s, _tau, _deg=1):
        """Eval on window evaluate in [-1, 1] returns the derivates wrt tau
        of the _deg derivate of each basis wrt t"""
        assert np.isscalar(_s)
        # Compute the derivative wet t
        alpha = self.params_
        k = np.sqrt(2) / 4.0 * np.power(alpha, 0.25) / \
            np.power((1.0 - alpha), 0.25)
        aux = np.power(2.0 * k, _deg)
        v0 = self.evalOnWindow(_s, _tau)
        v1 = np.ravel(aux * np.linalg.matrix_power(self.Dmat_, _deg).dot(v0))
        return np.ravel(k * _s * self.Dmat_.dot(v1))


#class cPiece1010(object):
#    '''
#        This class is intended to handle a curve defined by a linear
#        combinarion of basis.
#    '''
#    def __init__(self, _coeff=None, _alpha=None, _domain=None, _params=None):
#        self.coeff_ = cp.copy(_coeff)
#        self.alpha_ = cp.copy(_params)
#        self.domain_ = [_domain[0], _domain[1]]
#        self.tau_ = self.domain_[1] - self.domain_[0]
#
#        self.Dmat = np.array(
#            [[1, 1, 0, 0, 0, 0], [-1, 1, 0, 0, 0, 0], [0, 0, -1, 1, 0, 0],
#             [0, 0, -1, -1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]],
#            dtype=np.float)
#
#    def __call__(self, _t):
#
#        _t = 2.0 * (_t - self.domain_[0]) / self.tau_ - 1.0
#        alpha = self.alpha_
#        k = np.sqrt(2) / 4.0 * np.power(alpha, 0.25) / \
#            np.power((1.0 - alpha), 0.25)
#        p = self.tau_ * k * _t
#        expp = np.exp(p)
#        cosp = np.cos(p)
#        sinp = np.sin(p)
#
#        result = self.coeff_[0] * np.multiply(expp, cosp)
#        result += self.coeff_[1] * np.multiply(expp, sinp)
#        result += self.coeff_[2] * np.divide(cosp, expp)
#        result += self.coeff_[3] * np.divide(sinp, expp)
#        result += self.coeff_[4] * p
#        result += self.coeff_[5]
#
#        return result
#
#    def deriv(self, _deg=1):
#        alpha = self.alpha_
#        k = np.sqrt(2) / 4.0 * np.power(alpha, 0.25) / \
#            np.power((1.0 - alpha), 0.25)
#        tauk = k * 2.0
#        y = np.linalg.matrix_power(tauk * self.Dmat, _deg).dot(self.coeff_)
#        return cPiece1010(y, self.alpha_, self.domain_)
