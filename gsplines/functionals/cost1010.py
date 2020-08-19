from time import process_time
from scipy.sparse.linalg import spsolve
import numpy as np
from .problem1010utils import compute_Qd1_block
from .problem1010utils import compute_Qd1_dtau_block
from .problem1010utils import compute_Qd3_block
from .problem1010utils import compute_Qd3_dtau_block
from gsplines.basis.basis1010 import cBasis1010
from .functional import cFixedWaypointsFunctional
import copy as cp


class cCost1010(cFixedWaypointsFunctional):
    def __init__(self, _wp, _alpha):
        ''' Initialize an instance of this class given a set of via points.
        Such set must contain at least 3 points in R^n. The dimension of the
        space where the curve q(t) lies as well as the number of intervals is
        computed using the input.

          Parameters:
          ----------
            _wp: 2D float np.array
              Array which contains the via points to get the total L2 norm of
              the Jerk. The dimension of the space where the curve lies is
              computed as _q.shape[1] and the number of intervals is
              _q.shape[0]-1.
            _alpha: real
            _T: real'''

        self.alpha_ = _alpha
        self.dim_ = _wp.shape[1]
        self.N_ = _wp.shape[0] - 1
        self.wp_ = _wp.copy()
        self.basis_ = cBasis1010(self.alpha_)
        cFixedWaypointsFunctional.__init__(self, _wp, self.basis_)

        self.Q1buff = np.zeros((6, 6))
        self.Q2buff = np.zeros((6, 6))
        self.Q3buff = np.zeros((6, 6))


        self.P_ = np.eye(self.N_) - (1.0 / self.N_) * \
            np.ones((self.N_, self.N_))

        self.b_ = self.splcalc_.eval_b(self.wp_)


    def __call__(self, _tauv):
        """
          Evaluate the quadratic form y^T Q z,  where the Q matrix
          is defined in the cost function.

          Parameters:
          ----------
            _tauv: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar.
        """
        #        A = self.eval_A(_tauv, self.alpha_)
        #        y = spsolve(A, self.b_)
        y = self.waypoint_constraints(_tauv)
        res = 0.0
        for iinter in range(0, self.N_):
            i0 = iinter * 6 * self.dim_
            compute_Qd1_block(_tauv[iinter], self.alpha_, self.Q1buff)
            compute_Qd3_block(_tauv[iinter], self.alpha_, self.Q3buff)
            self.Q1buff[:, :] = (1.0-self.alpha_) * self.Q3buff + \
                self.alpha_ * self.Q1buff
            for idim in range(0, self.dim_):
                j0 = i0 + idim * 6
                yi = y[j0:j0 + 6]
                res += yi.transpose().dot(self.Q1buff).dot(yi)

        return res

    def gradient(self, _tauv, _result):
        """
          Evaluate the quadratic form y^T Q z,  where the Q matrix
          is defined in the cost function.

          Parameters:
          ----------
            _tauv: np.array float
              vector containing the time intervals.
          Returns:
          -------
            np.array.
        """
        y = self.waypoint_constraints(_tauv)
        A = self.splcalc_.eval_A(_tauv)

        for iinter in range(0, self.N_):
            _result[iinter] = 0.0

            z = self.splcalc_.eval_dAdtiy(_tauv, iinter, y)
            dydt = spsolve(A, z)
            # print(dydt)
            for iinter_2 in range(0, self.N_):
                i0 = iinter_2 * 6 * self.dim_
                compute_Qd1_block(_tauv[iinter_2], self.alpha_, self.Q1buff)
                compute_Qd3_block(_tauv[iinter_2], self.alpha_, self.Q3buff)
                self.Q2buff[:, :] = (1.0-self.alpha_) * self.Q3buff + \
                    self.alpha_ * self.Q1buff

                Q = self.Q2buff
                for idim in range(0, self.dim_):
                    j0 = i0 + idim * 6
                    yi = y[j0:j0 + 6]
                    zi = dydt[j0:j0 + 6]
                    _result[iinter] += -2.0 * yi.transpose().dot(Q).dot(zi)

            # print(iinter, _result[iinter])
            compute_Qd1_dtau_block(_tauv[iinter], self.alpha_, self.Q1buff)
            compute_Qd3_dtau_block(_tauv[iinter], self.alpha_, self.Q3buff)
            # self.Q1buff contains the derivative of Q
            self.Q2buff[:, :] = (1.0-self.alpha_) * self.Q3buff + \
                self.alpha_ * self.Q1buff

            Qd = self.Q2buff
            i0 = iinter * 6 * self.dim_
            # print(Qd)
            for idim in range(0, self.dim_):
                j0 = i0 + idim * 6
                yi = y[j0:j0 + 6]
                # print(yi)
                _result[iinter] += yi.transpose().dot(Qd).dot(yi)
            # print(iinter, _result[iinter])

        return _result
