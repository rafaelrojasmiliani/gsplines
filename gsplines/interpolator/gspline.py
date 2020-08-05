'''
    Classes and functions for construction of generalized splines from
    waypoints and time intervals.
'''
import sys
if sys.version_info >= (3, 0):
    from time import process_time
else:
    from time import time as process_time

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import copy as cp


class cSplineCalc(object):
    '''
        This is a class to compute the continuity and waypoints constraints
        required to compuyte a spline.  This class in intened to containt an
        allocated piece of memory and handle a fast computation of this
        constraints.
    '''

    def __init__(self, _dim, _N, _basis):
        """__init__: Initialize an instance of this class allocating the
        memory for the splines calculation process.
        :param _dim: int, dimension of the ambient space where the spline is
        :param _N: int, number of intervals of the spline
        :param _basistype: basis for the interpolation
        """
        self.dim_ = _dim
        self.N_ = _N
        self.basis_ = cp.deepcopy(_basis)
        if _basis.dim_ != 6:
            raise ValueError(
                "only implement for vector spaces with dimension 6.")

        self.basisdim_ = _basis.dim_
        #        if (not np.isfinite(x).all() or not np.isfinite(y).all()):
        #        raise ValueError("x and y array must not contain " "NaNs or
        #        infs.")

        self.N_ = _N

        self.dim_ = _dim

        self.pdof_ = self.basisdim_ * self.N_ * self.dim_

        self.nz_diags = 4 * self.dim_ + 4

        self.Aeval = None
        self.Adord = None

        dim = _dim

        self.cscAdata = np.zeros((((_N - 1) * dim + (3) * dim) * 4 * 3 +
                                  (3 - 1) * (_N - 1) * dim * 8 * 3, ))
        self.cscAindices = np.zeros(
            (((_N - 1) * dim + (3) * dim) * 4 * 3 + (3 - 1) *
             (_N - 1) * dim * 8 * 3, ),
            dtype=np.int16)
        self.cscAindptr = np.zeros((2 * 3 * _N * dim + 1, ), dtype=np.int16)

        self.b_ = np.zeros((2 * _N * 3 * dim, ))
        self.linsys_shape_ = 2 * _N * 3 * dim
        self.dbdwpi_ = np.zeros((2 * _N * 3 * dim, ))

        self.DiffResBuff = np.zeros((10 * self.dim_, ))

        self.dydtau_buff_ = np.zeros((self.linsys_shape_, self.N_))

        self.prfcpar_time_eval_A = 0
        self.prfcpar_times_eval_A_is_called = 0

    def printPerformace(self):
        res = self.prfcpar_time_eval_A / self.prfcpar_times_eval_A_is_called
        print("mean time to evaluate A: {:.4f}".format(res))

    def eval_A(self, tauv):
        """
        Alternative way to fill the Matrix A
        WARNING:  This work ok for _N<120
        """
        self.prfcpar_time_eval_A = process_time()

        dim = self.dim_
        N = self.N_

        basis = self.basis_

        Pl = [basis.evalDerivOnWindow(-1.0, tauv[0], i) for i in range(0, 5)]
        Pr = [basis.evalDerivOnWindow(1.0, tauv[0], i) for i in range(0, 5)]
        # Fill the content for the derivatives at boundaries
        Cpl = -np.vstack(Pl[1:5])
        Cpr = np.vstack(Pr[1:5])

        App1 = np.vstack([Pl[:3], Pr[0]])

        nnz = 0
        nptr = 0
        # --------------------------------------------
        for j in range(0, 6 * dim):
            self.cscAindptr[nptr] = nnz
            i0 = (j // 6) * 4
            for i in range(i0, i0 + 4):
                self.cscAdata[nnz] = App1[(i - i0), j % 6]
                self.cscAindices[nnz] = i
                nnz += 1
            i0 += 4 * dim
            for i in range(i0, i0 + 4):
                self.cscAdata[nnz] = Cpr[(i - i0), j % 6]
                self.cscAindices[nnz] = i
                nnz += 1
            nptr += 1
        # --------------------------------------------
        for iinter in range(1, N - 1):
            j0 = iinter * 6 * dim
            i0 = 4 * dim + 6 * dim * (iinter - 1)

            Pl = [
                basis.evalDerivOnWindow(-1.0, tauv[iinter], i)
                for i in range(0, 5)
            ]
            Pr = [
                basis.evalDerivOnWindow(1.0, tauv[iinter], i)
                for i in range(0, 5)
            ]
            Cpl = -np.vstack(Pl[1:5])
            Cpr = np.vstack(Pr[1:5])
            Appi = np.vstack([Pl[0], Pr[0]])
            # --------------------------------------------
            for j in range(j0, j0 + 6 * dim):
                self.cscAindptr[nptr] = nnz
                i1 = i0 + ((j - j0) // 6) * 4
                for i in range(i1, i1 + 4):
                    self.cscAdata[nnz] = Cpl[(i - i1), j % 6]
                    self.cscAindices[nnz] = i
                    nnz += 1
                i1 += (dim - (j - j0) // 6) * 4 + ((j - j0) // 6) * 2
                for i in range(i1, i1 + 2):
                    self.cscAdata[nnz] = Appi[i - i1, j % 6]
                    self.cscAindices[nnz] = i
                    nnz += 1
                i1 += (dim - (j - j0) // 6) * 2 + ((j - j0) // 6) * 4
                for i in range(i1, i1 + 4):
                    self.cscAdata[nnz] = Cpr[(i - i1), j % 6]
                    self.cscAindices[nnz] = i
                    nnz += 1
                nptr += 1
        # -------------------------------------
        #     Last column of the matrix A
        # ------------------------------------
        i0 = 4 * dim + 6 * dim * (N - 2)
        j0 = (N - 1) * 6 * dim
        Pl = [basis.evalDerivOnWindow(-1.0, tauv[-1], i) for i in range(0, 5)]
        Pr = [basis.evalDerivOnWindow(1.0, tauv[-1], i) for i in range(0, 5)]
        Cpl = -np.vstack(Pl[1:5])
        AppN = np.vstack([Pl[0], Pr[1:3], Pr[0]])
        # --------------------------------------------
        for j in range(j0, j0 + 6 * dim):
            self.cscAindptr[nptr] = nnz
            i1 = i0 + ((j - j0) // 6) * 4
            for i in range(i1, i1 + 4):
                self.cscAdata[nnz] = Cpl[(i - i1), j % 6]
                self.cscAindices[nnz] = i
                nnz += 1
            i1 += (dim - (j - j0) // 6) * 4 + ((j - j0) // 6) * 4
            for i in range(i1, i1 + 4):
                self.cscAdata[nnz] = AppN[(i - i1), j % 6]
                self.cscAindices[nnz] = i
                nnz += 1
            nptr += 1

        self.cscAindptr[nptr] = nnz
        res = csc_matrix(
            (self.cscAdata, self.cscAindices, self.cscAindptr),
            shape=2 * (2 * 3 * N * dim, ))
        self.prfcpar_time_eval_A = process_time() - self.prfcpar_time_eval_A
        self.prfcpar_times_eval_A_is_called += 1
        return res

    def eval_dAdtiy(self, x, idx, y):
        """
          Returns the left product of a matrix  dAdt_{idx} times the col vector
          y, where dAdti is the derivative of A w.r.t ti. The procedure is the
          following:
            1) given idx we compute
                - i0, j0: the upper left indices of the non-vanishing terms of
                  A
               of the block of A which is not zero

          Parameters
          ----------
            i: uint
              component of tau w.r.t. derivate
            y: np.array float,
              vector to multiplicate
          Returns:
          -------
            csc_matrix of dimension self.pdof, 1
              dAd/ti*y
        """
        res = self.DiffResBuff
        j0 = idx * 6 * self.dim_

        dim = self.dim_
        #        Pl = [self.basis_dtau_[i](-1.0, x[idx]) for i in range(0, 5)]
        #        Pr = [self.basis_dtau_[i](1.0, x[idx]) for i in range(0, 5)]
        Pl = [
            self.basis_.evalDerivWrtTauOnWindow(-1.0, x[idx], i)
            for i in range(0, 5)
        ]
        Pr = [
            self.basis_.evalDerivWrtTauOnWindow(1.0, x[idx], i)
            for i in range(0, 5)
        ]
        Cpl = -np.vstack(Pl[1:5])
        Cpr = np.vstack(Pr[1:5])

        if (idx == 0):
            App1 = np.vstack([Pl[:3], Pr[0]])
            # Compute the derivative of A w.r.t. x0
            i0 = 0  # upper limit of the matrix block
            i1 = 4 * self.dim_  # lower limit of the matrix block
            ir = 0  # Component of the result vector

            # Composition of the first rows of the result.
            # This are the rows associated to left boundary cofition
            # of position, velocity and acceleration and first waypoint.
            for i in range(0, 4 * dim):
                k0 = (i // 4) * 6 % (dim * 6)
                res[ir] = 0.0
                for k in range(k0, k0 + 6):
                    res[i0 + i] += App1[i % 4, k % 6] * y[j0 + k]
                ir += 1

            i0 = i1

            for i in range(0, 4 * dim):
                res[ir] = 0.0
                k0 = (i // 4) * 6 % (dim * 6)
                for k in range(k0, k0 + 6):
                    res[ir] += Cpr[i % 4, k % 6] * y[j0 + k]
                ir += 1

            nzi = [r for r in range(0, 8 * self.dim_)]

            return csc_matrix(
                (res[:ir], (nzi, ir * [0])), shape=(self.pdof_, 1))

        elif (idx > 0 and idx < self.N_ - 1):

            i0 = 4 * self.dim_ + 6 * self.dim_ * (idx - 1)
            i1 = i0 + 4 * self.dim_
            ir = 0
            for i in range(0, 4 * dim):
                k0 = (i // 4) * 6 % (dim * 6)
                res[ir] = 0.0
                for k in range(k0, k0 + 6):
                    res[ir] += Cpl[i % 4, k % 6] * y[j0 + k]
                ir += 1

            nzi1 = [r for r in range(i0, i1)]

            Appi = np.vstack([Pl[0], Pr[0]])
            i0 = i1
            i1 = i0 + 2 * self.dim_  # Api.shape[iinter]
            nzi2 = [r for r in range(i0, i1)]
            for i in range(0, 2 * dim):
                k0 = (i // 2) * 6 % (dim * 6)
                res[ir] = 0.0
                for k in range(k0, k0 + 6):
                    res[ir] += Appi[i % 2, k % 6] * y[j0 + k]
                ir += 1

            i0 = i1
            nzi3 = [r for r in range(i0, i0 + 4 * self.dim_)]

            for i in range(0, 4 * dim):
                res[ir] = 0.0
                k0 = (i // 4) * 6 % (dim * 6)
                for k in range(k0, k0 + 6):
                    res[ir] += Cpr[i % 4, k % 6] * y[j0 + k]
                ir += 1

            return csc_matrix(
                (res[:ir], (nzi1 + nzi2 + nzi3, ir * [0])),
                shape=(self.pdof_, 1))

        else:
            i0 = 4 * self.dim_ + 6 * self.dim_ * (self.N_ - 2)
            i1 = i0 + 4 * self.dim_
            nzi = [r for r in range(i0, i0 + 8 * self.dim_)]
            ir = 0
            for i in range(0, 4 * dim):
                k0 = (i // 4) * 6 % (dim * 6)
                res[ir] = 0.0
                for k in range(k0, k0 + 6):
                    res[ir] += Cpl[i % 4, k % 6] * y[j0 + k]
                ir += 1

            i0 = i1

            AppN = np.vstack([Pl[0], Pr[1:3], Pr[0]])

            for i in range(0, 4 * dim):
                k0 = (i // 4) * 6 % (dim * 6)
                res[ir] = 0.0
                for k in range(k0, k0 + 6):
                    res[ir] += AppN[i % 4, k % 6] * y[j0 + k]
                ir += 1

            return csc_matrix(
                (res[:ir], (nzi, ir * [0])), shape=(self.pdof_, 1))

    def eval_b(self, _wp):
        '''Construct the column vector with the boundary and waypoint
            constraints.'''
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_, '''
        _wp.shape[0] = {:d}
        _wp.shape[1] = {:d}
        self.N_      = {:d}
        self.dim_    = {:d}
        '''.format(_wp.shape[0], _wp.shape[1], self.N_, self.dim_)
        dim = self.dim_

        _dwp0 = np.zeros((dim, ))
        _ddwp0 = np.zeros((dim, ))
        _dwpT = np.zeros((dim, ))
        _ddwpT = np.zeros((dim, ))
        b = self.b_

        for i in range(dim):
            b[4 * i:4 * i + 4] = (_wp[0][i], _dwp0[i], _ddwp0[i], _wp[1][i])

        i0 = 8 * dim

        for idxwp, _ in enumerate(_wp[1:-2]):
            idxwp += 1
            for i in range(dim):
                b[i0 + 2 * i:i0 + 2 * i + 2] = (_wp[idxwp][i],
                                                _wp[idxwp + 1][i])
            i0 += 6 * dim

        for i in range(dim):
            b[i0 + 4 * i:i0 + 4 * i + 4] = (_wp[-2][i], _dwpT[i], _ddwpT[i],
                                            _wp[-1][i])
        return b

    def eval_dbdwpij(self, _wpidx, _i):
        ''' Evaluates the derivative of the vector b w.r.t. the ith
        component of the jwaypoint.'''
        dim = self.dim_

        b = self.dbdwpi_
        b.fill(0.0)

        if _wpidx == 0:
            i = _i
            b[4 * i:4 * i + 4] = (1.0, 0.0, 0.0, 0.0)
            return b

        if _wpidx == 1:
            i = _i
            b[4 * i:4 * i + 4] = (0.0, 0.0, 0.0, 1.0)
            i0 = 8 * dim
            if self.N_ == 2:
                b[i0 + 4 * i:i0 + 4 * i + 4] = (1.0, 0.0, 0.0, 0.0)
            else:
                b[i0 + 2 * i:i0 + 2 * i + 2] = (1.0, 0.0)
            return b

        i0 = 8 * dim

        for idxwp in range(2, self.N_ - 1):
            if idxwp == _wpidx:
                i = _i
                b[i0 + 2 * i:i0 + 2 * i + 2] = (0.0, 1.0)
                i0 += 6 * dim
                b[i0 + 2 * i:i0 + 2 * i + 2] = (1.0, 0.0)
                return b
            i0 += 6 * dim

        if _wpidx == self.N_ - 1:
            i = _i
            b[i0 + 2 * i:i0 + 2 * i + 2] = (0.0, 1.0)
            i0 += 6 * dim
            b[i0 + 4 * i:i0 + 4 * i + 4] = (1.0, 0.0, 0.0, 0.0)
            return b

        if (self.N_ > 2):
            i0 += 6 * dim

        if _wpidx == self.N_:
            i = _i
            b[i0 + 4 * i:i0 + 4 * i + 4] = (0.0, 0.0, 0.0, 1.0)

            return b

        raise ValueError('')

    def solveLinSys(self, _tauv, _wp):
        assert _tauv.shape[0] == self.N_ and len(_tauv.shape) == 1
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_
        b = self.eval_b(_wp)
        A = self.eval_A(_tauv)
        y = spsolve(A, b)
        return y

    def eval_y(self, _tauv, _wp):
        return self.solveLinSys(_tauv, _wp)

    def get_gspline(self, _tauv, _wp):
        assert _tauv.shape[0] == self.N_ and len(_tauv.shape) == 1
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_
        y = self.solveLinSys(_tauv, _wp)

        from ..piecewisefunction.piecewisefunction import cPiecewiseFunction
        res = cPiecewiseFunction(_tauv, y, self.dim_, self.basis_)
        res.wp_ = _wp.copy()
        return res

    def __call__(self, _tauv, _wp):
        return self.get_gspline(_tauv, _wp)

    def eval_dydtau(self, _tauv, _wp, _y=None):
        ''' Computs the derivatives of the vector y w.r.t. tau.  This retuns
        a matrix where the i-column is the deriviative of y w.r.t. tau_i.
        This returns a tuple where the first component is the derivatives
        matrix a the second is the y vector
        :param _tauv: np.array, tau vector
        :param _wp: np.array, waypoints matrix
        '''
        assert _tauv.shape[0] == self.N_ and len(_tauv.shape) == 1
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_
        b = self.eval_b(_wp)
        if _y is None:
            A = self.eval_A(_tauv)
            y = spsolve(A, b)
        else:
            A = self.eval_A(_tauv)
            y = _y

        for iinter, taui in enumerate(_tauv):
            dAdtauy = self.eval_dAdtiy(_tauv, iinter, y)
            self.dydtau_buff_[:, iinter] = -spsolve(A, dAdtauy)

        return self.dydtau_buff_, y

    def eval_dydu(self, _tauv, _wp, _indexes, _res, _y=None):
        ''' Computs the derivatives of the vector y w.r.t. the desired
        componens of the the desired waypoints.  This retuns a matrix where
        the i-column is the deriviative of y w.r.t. tau_i.  This returns a
        tuple where the first component is the derivatives matrix a the
        second is the y vector
        :param _tauv: np.array, tau vector
        :param _wp: np.array, waypoints matrix
        :param _indexes: array with tuples (waypioint index, compoent index)
        :param _res: buffer to store the resulr
        '''
        assert _tauv.shape[0] == self.N_ and len(_tauv.shape) == 1
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_
        if _y is None:
            b = self.eval_b(_wp)
            A = self.eval_A(_tauv)
            y = spsolve(A, b)
        else:
            A = self.eval_A(_tauv)
            y = _y

        for uidx, (wpidx, j) in enumerate(_indexes):
            dbdwpij = self.eval_dbdwpij(wpidx, j)
            res = spsolve(A, dbdwpij)
            _res[:, uidx] = res 

        return _res, y
