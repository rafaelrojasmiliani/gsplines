
import abc
import numpy as np
from ..interpolator.gspline import cSplineCalc
from scipy.sparse.linalg import spsolve
import copy


class cFunctional(metaclass=abc.ABCMeta):
    ''' This is a class which represents a non linear function
    of  a spline. In other words this represents a map
        F : gpline --> R
        This class represents a relation between a gspline and a real number.
        As the gspline is defined by its base, waypoints and time intervals
        this is a base for functions which takes waypoiints and time intervals
        and return a real number.
        '''
    def __init__(self, _N, _dim, _basis):

        self.dim_ = _dim
        self.N_ = _N
        self.basis_ = copy.deepcopy(_basis)
        self.bdim_ = _basis.dim_

        self.splcalc_ = cSplineCalc(self.dim_, self.N_, self.basis_)


    def domain2window(self, _t, _tauv):
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


    def waypoint_constraints(self, _tauv, _u):
        '''Solves the waypoint constraints  
                y=A(_tau)b(_u)
        This Computes the coefficient of the basis of the piecewise
        functions which interpolates thw waypoints _u given the lenght of
        intervals _tau .'''
        b = self.splcalc_.eval_b(_u)
        A = self.splcalc_.eval_A(_tauv)
        y = spsolve(A, b)
        return y




    @abc.abstractmethod
    def __call__(self, _x):
        pass

    @abc.abstractmethod
    def gradient(self, _x, _res):
        pass


class cFixedWaypointsFunctional(cFunctional):
    ''' This is a class which represents a non linear function of  a spline.
    In other words this represents a map F : gspline --> R, where gspline
    has its waypoints fixed.  '''
    def __init__(self, _wp, _basis):

        N = _wp.shape[0] - 1
        dim = _wp.shape[1]



        cFunctional.__init__(self, N, dim, _basis)

        self.b_ = self.splcalc_.eval_b(_wp)

        self.wp_ = _wp.copy()

    def waypoint_constraints(self, _tauv):
        '''Solves the waypoint constraints  
                y=A(_tau) b
        This Computes the coefficient of the basis of the piecewise
        functions which interpolates thw waypoints _u given the lenght of
        intervals _tau .'''
        A = self.splcalc_.eval_A(_tauv)
        y = spsolve(A, self.b_)
        return y

    def get_first_guess(self):
        return np.ones((self.N_, ))
