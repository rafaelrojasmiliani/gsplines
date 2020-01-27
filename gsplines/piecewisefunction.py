import numpy as np
from matplotlib import pyplot as plt

from .piece import cPiece

import itertools

import sympy as sp

import copy as cp


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class cPiecewiseFunction(object):
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

        #        print('tis_')
        #        print(self.tis_)

        #        self.tis_[0] -= 1.0e-5
        #        self.tis_[-1] += 1.0e-5

        self.funcTab_ = []  # components of the curve
        self.Qbuff = np.zeros((6, 6))

        for idim in range(0, self.dim_):
            func_row = [
                cPiece(
                    self.y_[idim * 6 + i * 6 * self.dim_:
                            idim * 6 + i * 6 * self.dim_ + 6],
                    _domain=[self.tis_[i], self.tis_[i + 1]],
                    _basis=self.basis_) for i in range(0, self.N_)
            ]
            #            for f in func_row:
            #                print(f.domain_)
            #                print(f.coeff_)
            #                print(f.basis_)
            self.funcTab_.append(func_row)

#            for i in range(0, self.N_):
#                print(self.y_[idim * 6 + i * 6 * self.dim_:
#                              idim * 6 + i * 6 * self.dim_ + 6])
        self.wp_ = None

    def __call__(self, _t):

        # print(self.tis_)
        if hasattr(_t, '__len__'):
            pass
        else:
            _t = float(_t)

        cond_list = [
            np.logical_and(tl <= _t, _t <= tr)
            for tl, tr in pairwise(self.tis_)
        ]

        return np.vstack([
            np.piecewise(_t, cond_list, self.funcTab_[i])
            for i in range(0, self.dim_)
        ]).transpose()

    def deriv(self, m=1):
        result = cp.deepcopy(self)
        for i, pol_row in enumerate(self.funcTab_):
            for j, p in enumerate(pol_row):
                result.funcTab_[i][j] = p.deriv(m)

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

    def L2_norm(self):
        y = self.y_
        res = 0.0
        for iinter in range(0, self.N_):
            i0 = iinter * 6 * self.dim_
            compute_Q_block(self.tau_[iinter], self.alpha_, self.Qbuff)
            #            print('----  Jerk Q ----')
            #            print(self.Q3buff)
            for idim in range(0, self.dim_):
                j0 = i0 + idim * 6
                yi = y[j0:j0 + 6]
                res += yi.transpose().dot(self.Qbuff).dot(yi)


#                print(yi)
        return res


def example():
    """
        Example: Creates a piecewise function and
        plot it with its derivatives
    """
    pass


if __name__ == '__main__':
    example()
