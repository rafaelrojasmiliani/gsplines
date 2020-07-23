from .piece import cPiece

import itertools
import numpy as np

import copy as cp


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class cPiecewiseFunction(object):
    """
    This Piecewise polynomial has N points and N-1 functions.
    This class is an abstraction of a table of lambda
    functions which evaluates each cell to get the component
    of a piecewise function at the correct interval.  However
    this class is suited to represent functions which returns
    a linear combination of other functions.

            Table inside this classe
                 t0<=t<t1    t0<=t<t1  t0<=t<t1
                +----------+---------+--------+
component 1     |  BF_11    |  BF_12   |   BF_13 |
                +-----------------------------+
component 2     |  BF 21    |  BF22    |   BF23  |
                +-----------------------------+
                |          |         |        |
                +-----------------------------+
component n     |  BF n1    |   BFn2   |  BFn3   |
                +----------+---------+--------+
    """

    def __init__(self, _tauv, _y, _dim, _basis):
        """
        Constructor. Creates a piecewise map given the
        canonical representation given in _x and the
        funcition to evaluate

        Parameters:
        ----------
         _dim: int
               dimension of the ambient space. How many
               components does the polynomial functions have.
         _x: array, double
             array with the coefficients of the polynomials
             of each components and the intervals IN
             "CANONICAL" FORM i.e. THE FORM MORE CONVENIENT
             FOR THE OPTIMIZER.

                 comp. 1 int 1 comp. 2 int 1           comp 1 inter 2
                +-------------+-----------+--------+
                |  BF_11       |           | ...
                +-------------+-----------+--------+

                _N: Number of intervals of the pw functions."""

        N = _tauv.shape[0]
        self.basis_ = cp.deepcopy(_basis)
        self.bdim_ = self.basis_.dim_
        self.y_ = _y.copy()  # pointer to the coefficient array
        self.N_ = N  # number of intervals
        self.dim_ = _dim  # dimension of the ambient space
        self.tau_ = _tauv.copy()  # pointer to the vector of intervals

        # self.tis_ is a list with the boundaries of the intervals
        self.tis_ = [np.sum(self.tau_[0:i]) for i in range(0, N + 1)]

        self.T_ = self.tis_[-1]
        self.domain_ = [0.0, self.T_]

        self.functions_table_ = []  # components of the curve
        self.Qbuff = np.zeros((self.bdim_, self.bdim_))

        bdim = self.bdim_
        for idim in range(0, self.dim_):
            func_row = [
                cPiece(
                    self.y_[idim * bdim + i * bdim * self.dim_:
                            idim * bdim + i * bdim * self.dim_ + bdim],
                    _domain=[self.tis_[i], self.tis_[i + 1]],
                    _basis=self.basis_) for i in range(0, self.N_)
            ]
            self.functions_table_.append(func_row)

        self.wp_ = self(self.tis_)

    def __call__(self, _t):

        #        if hasattr(_t, '__len__'):
        #            pass
        #        else:
        #            _t = float(_t)
        #
        _t = np.atleast_1d(_t)
        assert _t.ndim == 1
        _t[_t < self.domain_[0]] = self.domain_[0]
        _t[_t > self.domain_[1]] = self.domain_[1]

        intervals = np.zeros(_t.shape, dtype=np.int)
        result = np.zeros((_t.shape[0], self.dim_))
        for i, ti in enumerate(_t):
            for Ni, (tl, tr) in enumerate(pairwise(self.tis_)):
                if tl <= ti and ti <= tr:
                    intervals[i] = Ni
                    break



#        cond_list = [
#            np.logical_and(tl <= _t, _t <= tr)
#            for tl, tr in pairwise(self.tis_)
#        ]
#
#        result2 = np.vstack([
#            np.piecewise(_t, cond_list, self.functions_table_[i])
#            for i in range(0, self.dim_)
#        ]).transpose()

        bdim = self.basis_.dim_
        dim = self.dim_
        for i, Ni in enumerate(intervals):
            s = 2.0 * (_t[i] - self.tis_[Ni]) / self.tau_[Ni] - 1.0
            Bi = self.basis_.evalOnWindow(s, self.tau_[Ni])
            for j in range(dim):
                y = self.y_[j * bdim + Ni * bdim * dim:
                            j * bdim + Ni * bdim * dim + bdim]
                qij = Bi.dot(y)
                result[i, j] = qij
#                assert np.linalg.norm(result[i, j] - result2[i, j] ) < 1.0e-10
        return result
    def deriv(self, _deg=1):
        ''' Get the derivative of the curve.
        This funciton returns another pice-wise function which is the _m-th
        derivative of the current instance (self).
        Paramenters:
        ---------
            _deg: int
                Degree f the derivative
        '''
        result = cp.deepcopy(self)
        bdim = result.basis_.dim_
        dim = result.dim_
        for Ni, tau in enumerate(result.tau_):
            D = result.basis_.derivMatrixOnWindow(tau, _deg)
            DT = D.T
            for j in range(dim):
                y = result.y_[j * bdim + Ni * bdim * dim:
                            j * bdim + Ni * bdim * dim + bdim]
                y[:] = DT.dot(y)

        return result

    def __intervals_union(self, other):
        ti1 = [np.sum(self.tau_[0:i]) for i in range(0, self.N_ + 1)]
        ti2 = [np.sum(other.tau_[0:i]) for i in range(0, other.N_ + 1)]
        tis = np.sort(np.concatenate((ti1, ti2)))
        indexes_to_remove = []
        i = 0
        for ti, tj in pairwise(tis):
            if abs(ti - tj) < 1.0e-5:
                indexes_to_remove.append(i)
            i += 1
        tis = np.delete(tis, indexes_to_remove)
        return tis

    def __add__(self, other):
        print(self.tis_)
        print(other.tis_)
        assert self.alpha_ - other.alpha_ < 1.0e-5
        tis = self.__intervals_union(other)
        print(tis)

        wp = self(tis) + other(tis)
        tauv = np.array([tr - tl for tl, tr in pairwise(tis)])
        from .c4b6constraint import cC4B6Constraint
        constraints = cC4B6Constraint(_q=wp)
        return constraints.solve(_tauv=tauv, _alpha=self.alpha_)

    def __sub__(self, other):
        assert self.alpha_ - other.alpha_ < 1.0e-5
        tis = self.__intervals_union(other)

        wp = self(tis) - other(tis)
        tauv = np.array([tr - tl for tl, tr in pairwise(tis)])
        from .c4b6constraint import cC4B6Constraint
        constraints = cC4B6Constraint(_q=wp)
        return constraints.solve(_tauv=tauv, _alpha=self.alpha_)

    def l2_norm(self, _deg=0):
        '''Compute the L2 noorm of the _deg-th derivative of the curve
        Paramenters:
        ------------
            _deg: int
                Degree of the derivative that we desire the L2 norm
        '''
        bdim = self.bdim_
        basis = self.basis_
        tauv = self.tau_
        y = self.y_
        Q = np.zeros((bdim, bdim))
        res = 0.0
        for iinter in range(0, self.N_):
            i0 = iinter * bdim * self.dim_
            basis.l2_norm(tauv[iinter], Q, _deg)
            for idim in range(0, self.dim_):
                j0 = i0 + idim * bdim
                yi = y[j0:j0 + bdim]
                res += Q.dot(yi).dot(yi)
        return res


class cPiecewiseFunction_2(object):
    """
        This Piecewise polynomial has N points and N-1 functions.

        This class is an abstraction of a table of lambda functions
        which evaluates each cell to get the component of a piecewise
        function at the correct interval.

        However this class is suited to represent functions which
        returns a linear combination of other functions.

            Table inside this classe
                 t0<=t<t1    t0<=t<t1  t0<=t<t1
                +----------+---------+--------+
component 1     |  BF_11    |  BF_12   |   BF_13 |
                +-----------------------------+
component 2     |  BF 21    |  BF22    |   BF23  |
                +-----------------------------+
                |          |         |        |
                +-----------------------------+
component n     |  BF n1    |   BFn2   |  BFn3   |
                +----------+---------+--------+
    """

    def __init__(self, _tauv, _y, _dim, _basis):
        """ Constructor. Creates a piecewise map given the canonical
            representation given in _x and the funcition to evaluate

            Parameters:
            ----------
                _dim: int
                    dimension of the ambient space. How many components
                    does the polynomial functions have.
                _x: array, double
                    array with the coefficients of the polynomials of
                    each components and the intervals IN "CANONICAL" FORM
                    i.e. THE FORM MORE CONVENIENT FOR THE OPTIMIZER.

                 comp. 1 int 1 comp. 2 int 1           comp 1 inter 2
                +-------------+-----------+--------+
                |  BF_11       |           | ...
                +-------------+-----------+--------+

                _N: Number of intervals of the pw functions."""

        N = _tauv.shape[0]
        self.basis_ = cp.deepcopy(_basis)
        self.y_ = _y.copy()  # pointer to the coefficient array
        self.N_ = N  # number of intervals
        self.dim_ = _dim  # dimension of the ambient space
        self.tau_ = _tauv.copy()  # pointer to the vector of intervals

        # self.tis_ is a list with the boundaries of the intervals
        self.tis_ = [np.sum(self.tau_[0:i]) for i in range(0, N + 1)]

        self.T_ = self.tis_[-1]

        self.functions_table_ = []  # components of the curve
        self.Qbuff = np.zeros((6, 6))

        self.wp_ = self(self.tis_)

    def __domain2window(self, _t):
        '''domain2window

        :param _t:
        :param _tauv:
            Returns a triple
            (s, taui, i):
            s, real,
                paramenter i [-1, 1]
            taui, real,
                lengh of the interval
            i, int
                index of the interval

        '''
        _tauv = self.tau_
        assert np.isscalar(_t)
        if _t <= 0.0:
            return -1.0, _tauv[0], 0

        tis = [np.sum(_tauv[0:i]) for i in range(0, self.N_ + 1)]

        for iint, (t0, tf) in enumerate(zip(tis[:-1], tis[1:])):
            if t0 <= _t and _t < tf:
                taui = _tauv[iint]
                s = 2.0 / taui * (_t - t0) - 1.0
                return s, taui, iint

        if _t >= tis[-1]:
            return 1.0, _tauv[-1], _tauv.shape[0] - 1

        raise AssertionError('The program should not finish here.')

    def __call__(self, _t):

        # print(self.tis_)
        dim = self.dim_
        if np.isscalar(_t):
            _t = np.array([_t])

        nt = _t.shape[0]
        res = np.zeros((nt, dim))

        for i, ti in enumerate(_t):
            s, tau, iinter = self.__domain2window(ti)
            for j in range(dim):
                j0 = iinter * 6 * dim + 6 * j
                j1 = j0 + 6
                y = self.y_[j0:j1]
                B = self.basis_(s)
                res[i, j] = y.dot(B)

    def deriv(self, deg=1):
        y = self.y_.copy()
        dim = self.dim_
        for iinter, taui in enumerate(self.tau_):
            D = self.basis_.derivMatrixOnWindow(taui, deg).T
            for j in range(dim):
                j0 = iinter * 6 * dim + 6 * j
                j1 = j0 + 6
                y[j0:j1] = D.dot(self.y_[j0:j1])
        return cPiecewiseFunction_2(self.tau_, y, dim, self.basis_)

    def __intervals_union(self, other):
        ti1 = [np.sum(self.tau_[0:i]) for i in range(0, self.N_ + 1)]
        ti2 = [np.sum(other.tau_[0:i]) for i in range(0, other.N_ + 1)]
        tis = np.sort(np.concatenate((ti1, ti2)))
        indexes_to_remove = []
        i = 0
        for ti, tj in pairwise(tis):
            if abs(ti - tj) < 1.0e-5:
                indexes_to_remove.append(i)
            i += 1
        tis = np.delete(tis, indexes_to_remove)
        return tis

    def __add__(self, other):
        print(self.tis_)
        print(other.tis_)
        assert self.alpha_ - other.alpha_ < 1.0e-5
        tis = self.__intervals_union(other)
        print(tis)

        wp = self(tis) + other(tis)
        tauv = np.array([tr - tl for tl, tr in pairwise(tis)])
        from .c4b6constraint import cC4B6Constraint
        constraints = cC4B6Constraint(_q=wp)
        return constraints.solve(_tauv=tauv, _alpha=self.alpha_)

    def __sub__(self, other):
        assert self.alpha_ - other.alpha_ < 1.0e-5
        tis = self.__intervals_union(other)

        wp = self(tis) - other(tis)
        tauv = np.array([tr - tl for tl, tr in pairwise(tis)])
        from .c4b6constraint import cC4B6Constraint
        constraints = cC4B6Constraint(_q=wp)
        return constraints.solve(_tauv=tauv, _alpha=self.alpha_)
