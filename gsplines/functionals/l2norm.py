'''
Computes the L2 norm of the required derivative of a gspline as well as its gradient
'''
import numpy as np
import copy
from scipy.sparse.linalg import spsolve
from ..interpolator.gspline import cSplineCalc
from .functional import cFixedWaypointsFunctional
import abc


class cL2Norm(cFixedWaypointsFunctional):
    """
      This class implements the calculus of the L2 norm of the jerk for a
      gspline.  
    """

    def __init__(self, _wp, _basis, _deg):
        """
          Initialize an instance of this class given a set of via points.
          Such set must contain at least 3 points in R^n. The dimension of
          the space where the curve q(t) lies as well as the number of
          intervals is
          computed using the input.

          Parameters:
          ----------
        """
        cFixedWaypointsFunctional.__init__(self, _wp, _basis)


        self.Q_ = self.basis_.l2_norm_deriv_matrix(_deg)


    def first_guess(self):

        ds = np.ones((self.N_, ))
        ds = ds/np.sum(ds)

        return ds

    def eval_yTdQdiz(self, tau, idx, y, z):
        """
          Evaluate the quadratic form y^T dQdti z, where dQdti is the
          derivative of the Q matrix of the cost function with respect to t_i

          Parameters:
          ----------
            x: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar.
        """
        res = 0.0
        bdim = self.bdim_
        dim = self.dim_
        N = self.N_

        scal_multiplier = -5.0*0.5 * np.power(2.0 / tau[idx], 6)
        i0 = idx * bdim * self.dim_
        for idim in range(0, self.dim_):
            j0 = i0 + idim * bdim
            yi = y[j0:j0+bdim]
            zi = z[j0:j0+bdim]
            res += self.Q_.dot(yi).dot(zi) * scal_multiplier

        return res

    def eval_yTQz(self, tau, y, z):
        """
          Evaluate the quadratic form y^T Q z,  where the Q matrix
          is defined in the cost function.

          Parameters:
          ----------
            x: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar.
        """
        res = 0.0
        bdim = self.bdim_
        dim = self.dim_
        N = self.N_
        for iinter in range(0, N):
            scal_multiplier = np.power(2.0 / tau[iinter], 5.0)
            i0 = iinter * bdim * dim
            for idim in range(0, dim):
                j0 = i0 + idim * bdim
                j1 = j0 + bdim
                yi = y[j0:j1]
                zi = z[j0:j1]
                res += self.Q_.dot(z).dot(y)*scal_multiplier

        return res

    def __call__(self, _x):
        """
          Evaluate the total jerk for a set of intervals _tauv.

          Parameters:
          ----------
            _tauv: np.array float
              vector containing the time intervals.
          Returns:
          -------
            scalar:
              L2 norm of the Jerk
        """
        tauv = _x
        y = self.waypoint_constraints(tauv)
        res = self.eval_yTQz(tauv, y, y)

        return res

    def gradient(self, _x, _res):
        tauv = _x
        y = self.waypoint_constraints(tauv)

        A = self.splcalc_.eval_A(tauv)
        for i in range(0, self.N_):
            z = self.splcalc_.eval_dAdtiy(tauv, i, y)
            dydt = spsolve(A, z)
            _res[i] = -2.0*self.eval_yTQz(tauv, y, dydt) + \
                self.eval_yTdQdiz(tauv, i, y, y)

        return _res
