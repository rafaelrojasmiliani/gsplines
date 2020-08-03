'''
    Classes and functions for construction of generalized splines from
    waypoints and time intervals.
'''
from time import process_time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import copy as cp


class cSplineCalc_2(object):
    '''
        This is a class to compute the continuity and
        waypoints constraints required to compuyte a spline.
        This class in intened to containt an allocated piece
        of memory and handle a fast computation of this
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
        self.bdim_ = _basis.dim_
        self.basis_ = cp.deepcopy(_basis)

        # Compute vertical size of blocks
        self.App1N_vsize_ = self.bdim_ // 2 + 1
        self.Cpp_vsize_ = self.bdim_ - 2
        self.Appi_vsize_ = 2

        # Compute non zero elements
        #      Size of the boundary conditions
        nz = 2 * self.bdim_ * self.App1N_vsize_ * self.dim_
        #      Size of continuity conditions
        nz += (2 * self.Cpp_vsize_ + 2*(self.N_ - 2) *
               self.Cpp_vsize_) * self.dim_*self.bdim_
        #      Size of waypoint conditions
        nz += self.Appi_vsize_*(self.N_ - 2) * self.dim_ * self.bdim_
        self.nnz_ = nz

        self.y_vsize_ = self.bdim_ * self.N_ * self.dim_

        # Initialization of csc structures
        self.cscAdata_ = np.zeros((nz, ))
        self.cscAindices_ = np.zeros((nz, ), dtype=np.int16)
        self.cscAindptr_ = np.zeros(
            (self.bdim_ * self.N_ * self.dim_ + 1, ), dtype=np.int16)

        self.b_ = np.zeros((self.bdim_ * self.N_ * self.dim_, ))
        self.dbdwpi_ = np.zeros((self.bdim_ * self.N_ * self.dim_, ))

        self.DiffResBuff = np.zeros(
            ((self.App1N_vsize_ + 2*self.Cpp_vsize_) * self.dim_, ))

        self.dydtau_buff_ = np.zeros(
            (self.bdim_ * self.N_ * self.dim_, self.N_))

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
        bdim = self.bdim_

        basis = self.basis_

        App1N_vsize = self.App1N_vsize_
        Cpp_vsize = self.Cpp_vsize_
        Appi_vsize = self.Appi_vsize_

        Pl = [basis.evalDerivOnWindow(-1.0, tauv[0], i)
              for i in range(0, bdim - 1)]
        Pr = [basis.evalDerivOnWindow(1.0, tauv[0], i)
              for i in range(0, bdim - 1)]
        # Fill the content for the derivatives at boundaries
        if bdim - 1 > 1:
            Cpl = -np.vstack(Pl[1:bdim - 1])
            Cpr = np.vstack(Pr[1:bdim - 1])

        App1 = np.vstack([Pl[:bdim // 2], Pr[0]])
        nnz = 0
        nptr = 0
        # --------------------------------------------
        # Fill matrix for the first interval
        for j in range(0, bdim * dim):
            self.cscAindptr_[nptr] = nnz
            i0 = (j // bdim) * App1N_vsize
            for i in range(i0, i0 + App1N_vsize):
                self.cscAdata_[nnz] = App1[(i - i0), j % bdim]
                self.cscAindices_[nnz] = i
                nnz += 1
            i0 += App1N_vsize * dim
            for i in range(i0, i0 + Cpp_vsize):
                self.cscAdata_[nnz] = Cpr[(i - i0), j % bdim]
                self.cscAindices_[nnz] = i
                nnz += 1
            nptr += 1
        # --------------------------------------------
        # Matrix "middle" columns
        for iinter in range(1, N - 1):
            j0 = iinter * bdim * dim
            i0 = App1N_vsize * dim + bdim * dim * (iinter - 1)

            Pl = [
                basis.evalDerivOnWindow(-1.0, tauv[iinter], i)
                for i in range(0, bdim - 1)
            ]
            Pr = [
                basis.evalDerivOnWindow(1.0, tauv[iinter], i)
                for i in range(0, bdim - 1)
            ]
            if bdim - 1 > 1:
                Cpl = -np.vstack(Pl[1:bdim - 1])
                Cpr = np.vstack(Pr[1:bdim - 1])
            Appi = np.vstack([Pl[0], Pr[0]])
            # --------------------------------------------
            for j in range(j0, j0 + bdim * dim):
                self.cscAindptr_[nptr] = nnz
                i1 = i0 + ((j - j0) // bdim) * Cpp_vsize
                for i in range(i1, i1 + Cpp_vsize):
                    self.cscAdata_[nnz] = Cpl[(i - i1), j % bdim]
                    self.cscAindices_[nnz] = i
                    nnz += 1
                i1 += (dim - (j - j0) // bdim) * Cpp_vsize + \
                    ((j - j0) // bdim) * Appi_vsize
                for i in range(i1, i1 + Appi_vsize):
                    self.cscAdata_[nnz] = Appi[i - i1, j % bdim]
                    self.cscAindices_[nnz] = i
                    nnz += 1
                i1 += (dim - (j - j0) // bdim) * Appi_vsize + \
                    ((j - j0) // bdim) * Cpp_vsize
                for i in range(i1, i1 + Cpp_vsize):
                    self.cscAdata_[nnz] = Cpr[(i - i1), j % bdim]
                    self.cscAindices_[nnz] = i
                    nnz += 1
                nptr += 1
        # -------------------------------------
        #     Last column of the matrix A
        # ------------------------------------
        i0 = Cpp_vsize * dim + bdim * dim * (N - 2)
        j0 = (N - 1) * bdim * dim
        Pl = np.array([basis.evalDerivOnWindow(-1.0, tauv[-1], i)
                       for i in range(0, bdim - 1)])
        Pr = np.array([basis.evalDerivOnWindow(1.0, tauv[-1], i)
                       for i in range(0, bdim - 1)])
        if bdim - 1 > 1:
            Cpl = -np.vstack(Pl[1:bdim - 1])
        AppN = np.vstack([Pl[0], Pr[1:bdim // 2], Pr[0]])
        # --------------------------------------------
        for j in range(j0, j0 + bdim * dim):
            self.cscAindptr_[nptr] = nnz
            i1 = i0 + ((j - j0) // bdim) * App1N_vsize
            for i in range(i1, i1 + Cpp_vsize):
                self.cscAdata_[nnz] = Cpl[(i - i1), j % bdim]
                self.cscAindices_[nnz] = i
                nnz += 1
            i1 += (dim - (j - j0) // bdim) * App1N_vsize + \
                ((j - j0) // bdim) * App1N_vsize
            for i in range(i1, i1 + App1N_vsize):
                self.cscAdata_[nnz] = AppN[(i - i1), j % bdim]
                self.cscAindices_[nnz] = i
                nnz += 1
            nptr += 1

        self.cscAindptr_[nptr] = nnz
        res = csc_matrix(
            (self.cscAdata_, self.cscAindices_, self.cscAindptr_),
            shape=2 * (bdim * N * dim, ))
        self.prfcpar_time_eval_A = process_time() - self.prfcpar_time_eval_A
        self.prfcpar_times_eval_A_is_called += 1
        return res

    def eval_dAdtiy(self, x, idx, y):
        """
          Returns the left product of a matrix  dAdt_{idx}
          times the col vector y, where dAdti is the
          derivative of A w.r.t ti. The procedure is the
          following:
            1) given idx we compute
                - i0, j0: the upper left indices of the
                  non-vanishing terms of A of the block of A
                  which is not zero

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
        bdim = self.bdim_
        dim = self.dim_
        App1N_vsize = self.App1N_vsize_
        Cpp_vsize = self.Cpp_vsize_
        Appi_vsize = self.Appi_vsize_
        j0 = idx * bdim * self.dim_

        #        Pl = [self.basis_dtau_[i](-1.0, x[idx]) for i in range(0, 5)]
        #        Pr = [self.basis_dtau_[i](1.0, x[idx]) for i in range(0, 5)]
        Pl = [
            self.basis_.evalDerivWrtTauOnWindow(-1.0, x[idx], i)
            for i in range(0, bdim - 1)
        ]
        Pr = [
            self.basis_.evalDerivWrtTauOnWindow(1.0, x[idx], i)
            for i in range(0, bdim - 1)
        ]
        Cpl = -np.vstack(Pl[1:bdim - 1])
        Cpr = np.vstack(Pr[1:bdim - 1])

        if (idx == 0):
            App1 = np.vstack([Pl[:bdim // 2], Pr[0]])
            # Compute the derivative of A w.r.t. x0
            i0 = 0  # upper limit of the matrix block
            i1 = App1N_vsize * dim  # lower limit of the matrix block
            ir = 0  # Component of the result vector

            # Composition of the first rows of the result.
            # This are the rows associated to left boundary cofition
            # of position, velocity and acceleration and first waypoint.
            for i in range(0, App1N_vsize * dim):
                k0 = (i // App1N_vsize) * bdim % (dim * bdim)
                res[ir] = 0.0
                for k in range(k0, k0 + bdim):
                    res[i0 + i] += App1[i % App1N_vsize, k % bdim] * y[j0 + k]
                ir += 1

            i0 = i1

            for i in range(0, Cpp_vsize * dim):
                res[ir] = 0.0
                k0 = (i // 4) * bdim % (dim * bdim)
                for k in range(k0, k0 + bdim):
                    res[ir] += Cpr[i % 4, k % bdim] * y[j0 + k]
                ir += 1

            nzi = [r for r in range(0, 8 * self.dim_)]

            return csc_matrix(
                (res[:ir], (nzi, ir * [0])), shape=(self.y_vsize_, 1))

        elif (idx > 0 and idx < self.N_ - 1):

            i0 = 4 * self.dim_ + bdim * self.dim_ * (idx - 1)
            i1 = i0 + 4 * self.dim_
            ir = 0
            for i in range(0, 4 * dim):
                k0 = (i // 4) * bdim % (dim * bdim)
                res[ir] = 0.0
                for k in range(k0, k0 + bdim):
                    res[ir] += Cpl[i % 4, k % bdim] * y[j0 + k]
                ir += 1

            nzi1 = [r for r in range(i0, i1)]

            Appi = np.vstack([Pl[0], Pr[0]])
            i0 = i1
            i1 = i0 + 2 * self.dim_  # Api.shape[iinter]
            nzi2 = [r for r in range(i0, i1)]
            for i in range(0, 2 * dim):
                k0 = (i // 2) * bdim % (dim * bdim)
                res[ir] = 0.0
                for k in range(k0, k0 + bdim):
                    res[ir] += Appi[i % 2, k % bdim] * y[j0 + k]
                ir += 1

            i0 = i1
            nzi3 = [r for r in range(i0, i0 + 4 * self.dim_)]

            for i in range(0, 4 * dim):
                res[ir] = 0.0
                k0 = (i // 4) * bdim % (dim * bdim)
                for k in range(k0, k0 + bdim):
                    res[ir] += Cpr[i % 4, k % bdim] * y[j0 + k]
                ir += 1

            return csc_matrix(
                (res[:ir], (nzi1 + nzi2 + nzi3, ir * [0])),
                shape=(self.y_vsize_, 1))

        else:
            i0 = 4 * self.dim_ + bdim * self.dim_ * (self.N_ - 2)
            i1 = i0 + 4 * self.dim_
            nzi = [r for r in range(i0, i0 + 8 * self.dim_)]
            ir = 0
            for i in range(0, 4 * dim):
                k0 = (i // 4) * bdim % (dim * bdim)
                res[ir] = 0.0
                for k in range(k0, k0 + bdim):
                    res[ir] += Cpl[i % 4, k % bdim] * y[j0 + k]
                ir += 1

            i0 = i1

            AppN = np.vstack([Pl[0], Pr[1:3], Pr[0]])

            for i in range(0, 4 * dim):
                k0 = (i // 4) * bdim % (dim * bdim)
                res[ir] = 0.0
                for k in range(k0, k0 + bdim):
                    res[ir] += AppN[i % 4, k % bdim] * y[j0 + k]
                ir += 1

            return csc_matrix(
                (res[:ir], (nzi, ir * [0])), shape=(self.y_vsize_, 1))

    def eval_b(self, _wp):
        '''Construct the column vector with the boundary and
        waypoint constraints.'''
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_, '''
        Error in the shape of waypoints.
        _wp.shape[0] = {:d}
        _wp.shape[1] = {:d}
        self.N_      = {:d}
        self.dim_    = {:d}
        '''.format(_wp.shape[0], _wp.shape[1], self.N_, self.dim_)
        dim = self.dim_
        bdim = self.bdim_
        App1N_vsize = self.App1N_vsize_
        Cpp_vsize = self.Cpp_vsize_
        Appi_vsize = self.Appi_vsize_

        _dwp0 = np.zeros((dim, ))
        _ddwp0 = np.zeros((dim, ))
        _dwpT = np.zeros((dim, ))
        _ddwpT = np.zeros((dim, ))
        b = self.b_

        ss = App1N_vsize
        for i in range(dim):
            b[ss * i:ss * i +
                ss] = np.hstack([_wp[0, i], np.zeros((bdim // 2 - 1, )), _wp[1, i]])

        i0 = (App1N_vsize + Cpp_vsize) * dim

        ss = Appi_vsize
        for idxwp, _ in enumerate(_wp[1:-2, :]):
            idxwp += 1
            for i in range(dim):
                b[i0 + ss * i:i0 + ss * i + ss] = (_wp[idxwp, i],
                                                   _wp[idxwp + 1, i])
            i0 += (Appi_vsize + Cpp_vsize) * dim

        ss = App1N_vsize
        for i in range(dim):
            b[i0 + ss * i:i0 + ss * i +
                ss] = np.hstack([_wp[-2][i], np.zeros((bdim // 2 - 1, )), _wp[-1][i]])

        return b

    def eval_dbdwpij(self, _wpidx, _i):
        ''' Evaluates the derivative of the vector b w.r.t.
        the ith component of the jwaypoint.
        Parameters:
        ----------
            _wpidx: int
                Index of the waypoint
            _i: int
                Index of the waypoints component'''
        dim = self.dim_

        b = self.dbdwpi_
        b.fill(0.0)
        bdim = self.bdim_
        App1N_vsize = self.App1N_vsize_
        Cpp_vsize = self.Cpp_vsize_
        Appi_vsize = self.Appi_vsize_

        ss = App1N_vsize
        if _wpidx == 0:
            i = _i
            b[ss * i] = 1.0
            return b

        if _wpidx == 1:
            i = _i
            b[ss * i + ss - 1] = 1.0
            i0 = (App1N_vsize + Cpp_vsize) * dim
            if self.N_ == 2:
                ss = App1N_vsize
            else:
                ss = Appi_vsize
            b[i0 + ss * i] = 1.0

            return b

        i0 = (App1N_vsize + Cpp_vsize) * dim

        for idxwp in range(2, self.N_ - 1):
            ss = App1N_vsize
            if idxwp == _wpidx:
                i = _i
                b[i0 + ss * i + ss - 1] = 1.0
                i0 += (Appi_vsize + Cpp_vsize) * dim
                b[i0 + ss * i + ss - 1] = 1.0
                return b
            i0 += (Appi_vsize + Cpp_vsize) * dim

        if _wpidx == self.N_ - 1:
            i = _i
            ss = Appi_vsize
            b[i0 + ss * i + ss - 1] = 1.0
            i0 += (Appi_vsize + Cpp_vsize) * dim
            ss = App1N_vsize
            b[i0 + ss * i] = 1.0
            return b

        if (self.N_ > 2):
            i0 += (Appi_vsize + Cpp_vsize) * dim

        if _wpidx == self.N_:
            i = _i
            ss = App1N_vsize
            b[i0 + ss * i + ss - 1] = 1.0

            return b

        raise ValueError('')

    def waypoint_constraint(self, _tauv, _wp):
        assert _tauv.shape[0] == self.N_ and len(_tauv.shape) == 1
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_
        b = self.eval_b(_wp)
        A = self.eval_A(_tauv)
        y = spsolve(A, b)
        return y

    def eval_y(self, _tauv, _wp):
        return self.waypoint_constraint(_tauv, _wp)

    def get_gspline(self, _tauv, _wp):
        assert _tauv.shape[0] == self.N_ and len(_tauv.shape) == 1
        assert _wp.shape[0] == self.N_ + 1 and _wp.shape[1] == self.dim_
        y = self.waypoint_constraint(_tauv, _wp)

        from .piecewisefunction import cPiecewiseFunction
        res = cPiecewiseFunction(_tauv, y, self.dim_, self.basis_)
        res.wp_ = _wp.copy()
        return res

    def eval_dydtau(self, _tauv, _wp, _y=None):
        ''' Computs the derivatives of the vector y w.r.t.
        tau.  This retuns a matrix where the i-column is the
        deriviative of y w.r.t. tau_i.  This returns a tuple
        where the first component is the derivatives
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
        ''' Computs the derivatives of the vector y w.r.t.
        the desired componens of the the desired waypoints.
        This retuns a matrix where the i-column is the
        deriviative of y w.r.t. tau_i.  This returns a tuple
        where the first component is the derivatives matrix a
        the
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


def plt_show_gspline(_q, _dt=0.1):
    import matplotlib.pyplot as plt
    dim = _q.dim_
    fig, ax=plt.subplots(4, dim)
    t = np.arange(0.0, _q.T_, _dt)

    for i in range(0, 4):
        q = _q.deriv(i)
        qt = q(t)
        for j in range(0, dim):
            ax[i, j].plot(t, qt[:, j])
            ax[i, j].grid()
            ax[i, j].set_xticklabels(ax[i, j].get_xticks(), fontsize=5)
            ax[i, j].set_yticklabels(ax[i, j].get_yticks(), fontsize=5)
            if i == 0:
                ax[i, j].set_title('coordinate {:d}'.format(j+1), fontsize=8)

    plt.subplots_adjust(left=0.025, bottom=0.05, right=0.975, top=0.95, wspace=0.25, hspace=0.15)

    if dim == 2:
        q = _q
        qt = q(t)
        fig = plt.figure()
        plt.plot(qt[:, 0], qt[:, 1], 'b-')
    plt.show()
