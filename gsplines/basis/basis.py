''' Base clase to define a basis '''


class cBasis(object):
    ''' class define a basis'''

    def __init__(self):
        pass

    def evalOnWindow(self, _s, _tau):
        pass

    def evalDerivOnWindow(self, _s, _tau, _deg):
        assert np.isscalar(_s)
        # rememver ds/dt = 2.0/tau
        v = self.evalOnWindow(_s, _tau)
        aux = 2.0 / _tau
        return np.ravel(np.linalg.matrix_power(aux * self.Dmat_, _deg).dot(v)).copy()

    def evalDerivWrtTauOnWindow(self, _s, _tau, _deg=1):
        """Eval on window evaluate in [-1, 1] returns the first derivate wrt
        tau of the _deg derivate of each basis wrt t"""
        assert np.isscalar(_s)
        # Compute the derivative wet t

        if _deg == 0:
            self.buff_.fill(0.0)
            return self.buff_.copy()
        v0 = self.evalDerivOnWindow(_s, _tau, _deg)
        v0 *= -0.5 * _deg * (2.0 / _tau)
        return np.ravel(v0)

    def derivMatrixOnWindow(self, _tau, _deg):
        aux = 2.0 / _tau
        return np.linalg.matrix_power(aux * self.Dmat_, _deg)

    def l2_norm(self, _tau, _Q, _deg=0):
        _Q.fill(0.0)
        if _deg == 0:
            raise ValueError()
        elif _deg == 1:
            _Q[:, :] = np.array([[0, 0, 0, 0, 0, 0],
                                 [0, 2, 0, 2, 0, 2],
                                 [0, 0, 6, 0, 6, 0],
                                 [0, 2, 0, 12, 0, 12],
                                 [0, 0, 6, 0, 20, 0],
                                 [0, 2, 0, 12, 0, 30]], dtype=np.float)

            _Q *= 2.0/_tau

        elif _deg == 2:
            raise ValueError()
        elif _deg == 3:
            _Q[3, 3] = 450.0
            _Q[3, 5] = 3150.0
            _Q[5, 3] = 3150.0
            _Q[4, 4] = 7350.0
            _Q[5, 5] = 61740.0
            _Q *= np.power(2.0 / _tau, 5.0)
        else:
            raise ValueError()


def getDmatLeg(n):
    ''' Compute the derivative matrix for
    the legendre polynomials
    '''

    def alpha(i):
        return float(i + 1.0) / float(2.0 * i + 1.0)

    def gamma(i):
        return float(i) / float(2.0 * i + 1.0)

    D = np.zeros((n, n))

    for i in range(1, n):
        for j in range(0, n - 1):
            if i == j + 1:
                D[i, j] = (i) / alpha(i - 1)
            elif i > j + 1:
                firstTerm = 0.0
                secondTerm = alpha(j - 1) * D[i - 1, j - 1] if j >= 1 else 0.0
                thirdTerm = gamma(j + 1) * D[i - 1, j + 1] if i >= 1 else 0.0
                fourthTerm = gamma(i - 1) * D[i - 2, j] if i >= 2 else 0.0

                D[i, j] = 1.0/alpha(i-1) *\
                    (firstTerm+secondTerm+thirdTerm-fourthTerm)

    return D


class cBasis0010canonic(object):
    ''' Legendre basis of fifth order polynomials'''
    dim_ = 6

    def __init__(self):

        from numpy import sqrt
        Dm = np.array([[0.0,           0.0,      0.0,         0.0,  0.0, 0.0],
                       [sqrt(3.0),     0.0,      0.0,         0.0,  0.0, 0.0],
                       [0.0,           sqrt(15.0), 0.0,
                        0.0,  0.0, 0.0],
                       [1.0/6.0,       0.0,
                           sqrt(5.0)/15.0, 0.0, 0.0, 0.0],
                       [0.0,           0.0,      0.0,
                           sqrt(3.0), 0.0, 0.0],
                       [-sqrt(5.0)/12.0, 0.0,      -1.0/6.0,    0.0, sqrt(15.0), 0.0]])

        self.Dmat_ = Dm
        self.buff_ = np.ones((6, ))
        self.dim_ = 6

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
        s = _s
        taui52 = np.power(_tau, 5.0 / 2.0)
        self.buff_[0] = (1.0 / 8.0) * taui52
        self.buff_[1] = np.sqrt(3) / 8.0 * s * taui52
        s2 = s * s
        self.buff_[2] = (3.0 * np.sqrt(5) / 16.0) * taui52 * (s2 - 1.0 / 3.0)
        s3 = s2 * s
        self.buff_[3] = (1.0 / 48.0) * s3 * taui52
        s4 = s3 * s
        self.buff_[4] = (np.sqrt(3) / 192.0) * s4 * taui52
        s5 = s4 * s
        self.buff_[5] = (np.sqrt(5) / 320.0) * taui52 * (s5 - 10.0 / 3.0 * s3)
        return self.buff_.copy()

    def evalDerivWrtTauOnWindow(self, _s, _tau, _deg=1):
        """Eval on window evaluate in [-1, 1] returns the first derivate wrt
        tau of the _deg derivate of each basis wrt t"""
        assert np.isscalar(_s)
        # Compute the derivative wet t

        if _deg == 0:
            self.buff_.fill(0.0)
            return self.buff_.copy()
        v0 = self.evalDerivOnWindow(_s, _tau, _deg)
        v0 *= (5.0/2.0 - _deg) / _tau
        return np.ravel(v0)

    def derivMatrixOnWindow(self, _tau, _deg):
        aux = 2.0 / _tau
        return np.linalg.matrix_power(aux * self.Dmat_, _deg)
