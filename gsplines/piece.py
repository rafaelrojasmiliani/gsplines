'''
    Generic function that is a linear combination of basis.
'''
import copy as cp
import numpy as np


class cPiece(object):
    '''
        This class is intended to handle a curve defined by a linear
        combinarion of basis.
    '''

    def __init__(self, _coeff, _domain, _basis):
        assert _domain[1] > _domain[0]
        self.coeff_ = cp.copy(_coeff)
        self.domain_ = [_domain[0], _domain[1]]
        self.tau_ = self.domain_[1] - self.domain_[0]

        self.basis_ = cp.deepcopy(_basis)

    def __call__(self, _t):
        ''' Compute the value of the function with a linear combination of
            the basis
            :_t: float, time'''

        _t = np.atleast_1d(_t)

        _t[np.where(_t < self.domain_[0])] = self.domain_[0]
        _t[np.where(_t > self.domain_[1])] = self.domain_[1]
        _s = 2.0 * (_t - self.domain_[0]) / self.tau_ - 1.0
        result = np.array([
            self.basis_.evalOnWindow(si, self.tau_).dot(self.coeff_)
            for si in _s
        ])
        return result

    def deriv(self, _deg=1):
        ''' Returns the derivative of the current function'''
        D = self.basis_.derivMatrixOnWindow(self.tau_, _deg)
        D = D.T
        res = cp.deepcopy(self)
        res.coeff_[:] = D.dot(self.coeff_)
        return res
