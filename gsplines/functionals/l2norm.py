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
      This class implements the functions required for the optimization of the
      L2 norm of some derivative in a gradient-based optimizers.  It implments
      the evaluation of the L2 norm given a sequence of waypooints and time
      intervals and its gradient w.r.t. the time intervals.
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
          Evaluate the quadratic form y^T dQdti z, where dQdti
          is the derivative of the Q matrix of the cost
          function with respect to t_i

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
                res += self.Q_.dot(zi).dot(yi)*scal_multiplier

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


class cJerkL2Norm(cL2Norm):
    def __init__(self, _wp, _basis):
        cL2Norm.__init__(self, _wp, _basis, 3)

class cConvexCombinationL2Norm(object):
    """
      This class implements the functions required for the optimization of the
      linear combination of the L2 norm of two derivatives in a gradient-based optimizers.  It implments
      the evaluation of the L2 norm given a sequence of waypooints and time
      intervals and its gradient w.r.t. the time intervals.
    """

    def __init__(self, _wp, _basis, _deg1, _deg2, _alpha):
        """
          Initialize an instance of this class given a set of via points.
          Such set must contain at least 3 points in R^n. The dimension of
          the space where the curve q(t) lies as well as the number of
          intervals is
          computed using the input.

          Parameters:
          ----------
            _wp: numpy array
                A matrix which rows are the waypoint
        """
        func1 = cL2Norm(_wp, _basis, _deg1)
        func2 = cL2Norm(_wp, _basis, _deg2)


    def first_guess(self):

        ds = np.ones((self.N_, ))
        ds = ds/np.sum(ds)

        return ds


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
        alpha = self.alpha_
        return alpha*self.func1(x)+(1-alpha)*self.func2(x)

    def gradient(self, _x, _res):
        self.func1.gradient(_x, _res)
        grad1 = _res.copy()
        self.func1.gradient(_x, _res)
        grad2 = _res
        alpha = self.alpha_
        _res[:] = alpha*grad1 + (1-alpha)*grad2

