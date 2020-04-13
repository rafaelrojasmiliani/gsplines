import numpy as np


class cBasis1000(object):
    ''' Basis of first order polynomials'''
    dim_ = 2

    def __init__(self):

        self.buff_ = np.ones((2, ))

        self.Dmat_ = np.array([[0.0, 0.0], [1.0, 0.0]])

    def evalDerivOnWindow(self, _s, _tau, _deg):
        assert np.isscalar(_s)
        # rememver ds/dt = 2.0/tau
        v = self.evalOnWindow(_s, _tau)
        aux = 2.0 / _tau
        return np.ravel(np.linalg.matrix_power(aux * self.Dmat_, _deg).dot(v))

    def evalOnWindow(self, _s, _tau):
        """Eval on window evaluate in [-1, 1] returns the cFundFuncBasis
        instance which contains the time derivate of the current instance.
        """
        assert np.isscalar(_s)
        self.buff_[0] = 1.0
        self.buff_[1] = _s
        return self.buff_

    def evalDerivWrtTauOnWindow(self, _s, _tau, _deg=1):
        """Eval on window evaluate in [-1, 1] returns the first derivate wrt
        tau of the _deg derivate of each basis wrt t"""
        assert np.isscalar(_s)
        # Compute the derivative wet t

        if _deg == 0:
            self.buff_.fill(0.0)
            return self.buff_
        v0 = self.evalDerivOnWindow(_s, _tau, _deg)
        v0 *= -0.5 * _deg * (2.0 / _tau)
        return np.ravel(v0)

    def derivMatrixOnWindow(self, _tau, _deg):
        aux = 2.0 / _tau
        return np.linalg.matrix_power(aux * self.Dmat_, _deg)

    def l2_norm(self, _tau, _Q, _deg=0):
        _Q.fill(0.0)
        if _deg == 0:
            _Q[:, :] = np.array([[2.0, 0.0], 
                                 [0.0, 2.0/3.0]], dtype=np.float)
            _Q *= _tau/2.0
        elif _deg == 1:
            _Q[:, :] = np.array([[0.0, 0.0], 
                                 [0.0, 2.0]], dtype=np.float)

            _Q *= 2.0/_tau
