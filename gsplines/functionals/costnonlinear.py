from time import process_time
import quadpy
import abc
import numpy as np
from splines.basis0010 import cBasis0010
from splines.piece1010 import cBasis1010
from splines.gspline import cSplineCalc
from scipy.sparse.linalg import spsolve
from constraints.constraints import cAffineBilateralConstraint


class cCostNonLinear(metaclass=abc.ABCMeta):
    ''' This is a class which represents a non linear function
    of  a spline. In other words this represents a map
        F : gpline --> R
        This class represents a relation between a gspline and a real number.
        As the gspline is defined by its base, waypoints and time intervals
        this is a base for functions which takes waypoiints and time intervals
        and return a real number.
        '''
    def __init__(self, _wp, _T, _Ni, _Ngl):

        #  Main instance data
        self.dim_ = _wp.shape[1]
        self.N0_ = _wp.shape[0] - 1
        self.wp0_ = _wp.copy()
        self.T_ = _T
        self.Ni_ = _Ni
        self.N_ = (_wp.shape[0] - 1) + _Ni * (_wp.shape[0] - 1) 
        assert self.N_ == self.N0_*(_Ni+1), '''
        N nom =  {:d}
        N test = {:d}
        N0     = {:d}
        Ni     = {:d}
        '''.format(self.N_, self.N0_*(_Ni+1), self.N0_, _Ni)
        self.number_virtual_points_ = self.N_ + 1 - _wp.shape[0]
        self.ushape_ = self.number_virtual_points_ * self.dim_

        # Spline related data
        self.basis_ = cBasis0010()
        self.splcalc_ = cSplineCalc(self.dim_, self.N_, self.basis_)
        self.y_ = None
        self.A_ = None
        lss = self.splcalc_.linsys_shape_
        self.dydu_buff_ = np.zeros((lss, self.ushape_))

        # Construction of waypoint array with fixed and "mobile" waypoints
        self.Fixedwp_ = _wp
        self.wp_ = np.zeros((self.N_ + 1, self.dim_))
        wp = self.wp0_
        self.uToWp_ = []
        for i, (v0, v1) in enumerate(zip(wp[:-1, :], wp[1:, :])):
            for j in range(0, _Ni + 2):
                self.wp_[(_Ni + 1) * i + j, :] = v0 + (v1 - v0) * j / (_Ni + 1)
                if j != 0 and j != _Ni + 1:
                    widx = (_Ni + 1) * i + j
                    for ui in range(0, self.dim_):
                        self.uToWp_.append((widx, ui))

        self.ushape_ = len(self.uToWp_)

        self.gradient_sum_buff = np.zeros((self.ushape_ + self.N_, ))

        # Integration scheme
        self.int_scheme_ = quadpy.line_segment.gauss_legendre(_Ngl)

        self.initPerfornaceIndicators()

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

    def initPerfornaceIndicators(self):
        self.wpfCalls = 0  # waypoint functions
        self.gradCalls = 0
        self.evalCalls = 0
        self.twpf = 0
        self.tcall = 0
        self.tgrad = 0
        self.tspsinv_ = 0
        self.tspscov_ = 0

    def printPerformanceIndicators(self):
        print('Number of cost function evaluations = {:d}'.format(
            self.evalCalls))
        print('Number of gradient evaluations = {:d}'.format(self.gradCalls))

        print('Number of waypoint constraints evaluations = {:d}'.format(
            self.wpfCalls))

        print('Mean time in cost function evalution = {:.4f}'.format(
            self.tcall / self.evalCalls))
        print('Mean time in gradient evalution = {:.4f}'.format(
            self.tgrad / self.gradCalls))
        print('Mean time in waypointConstraints evalution = {:.4f}'.format(
            self.twpf / self.wpfCalls))

        print('Total time in cost function evalution = {:.4f}'.format(
            self.tcall))
        print('Total time in gradient evalution = {:.4f}'.format(self.tgrad))
        print('Total time in waypoint constraint evalution= {:.4f}'.format(
            self.twpf))


#        print('Total time in sparse matrix evaluation = {:.4f}'.format(
#            self.tspscal_))

    def getFirstGuess(self):

        wp = self.wp_
        #        ds = np.linalg.norm(wp[:-1, :] - wp[1:, :], axis=1)
        u0 = np.zeros((self.ushape_, ))
        u0 = self.wp2u(u0)
        ds = np.ones((self.N_, ))
        ds = ds / np.sum(ds) * self.T_

        res = np.hstack([ds, u0])
        return res

    def waypointConstraints(self, _tauv, _u):
        '''Solves the waypoint constraints.  Computes the coefficient of the
        basis functions given the interval lenghts.'''
        t = process_time()
        wp = self.u2wp(_u)
        b = self.splcalc_.eval_b(wp)
        A = self.splcalc_.eval_A(_tauv)
        y = spsolve(A, b)
        self.wpfCalls += 1
        self.twpf += process_time() - t
        self.y_ = y

        return y

    def u2wp(self, _u):
        i0 = 0
        for i in range(0, self.N0_):
            for j in range(1, self.Ni_ + 1):
                self.wp_[(self.Ni_ + 1) * i + j, :] = _u[i0:i0 + self.dim_]
                i0 += self.dim_
        return self.wp_

    def wp2u(self, _u):
        i0 = 0
        for i in range(0, self.N0_):
            for j in range(1, self.Ni_ + 1):
                _u[i0:i0 + self.dim_] = self.wp_[(self.Ni_ + 1) * i + j, :]
                i0 += self.dim_
        assert _u.shape[0] == i0, '''
            i0          = {:d},
            u.shape[0]  = {:d}
            dim         = {:d}
            N0          = {:d}
            Ni          = {:d}
            '''.format(i0, _u.shape[0], self.dim_, self.N0_, self.Ni_)
        return _u

    def timeToCollocation(self, _t, _tauv):
        pass

    @abc.abstractmethod
    def runningCost(self, _t, _tauv, _u, _y=None, _inter=None):
        pass

    @abc.abstractmethod
    def runningCostGradient(self,
                            _t,
                            _tauv,
                            _u,
                            _y=None,
                            _inter=None,
                            _dydtau=None,
                            _dydu=None):
        pass

    def __call__(self, _x):

        _tauv = _x[:self.N_]
        _u = _x[self.N_:]

        result = 0.0
        t0 = 0.0
        y = self.waypointConstraints(_tauv, _u)
        mycost = np.vectorize(
            lambda t, inter: self.runningCost(t, _tauv, _u, y, inter))
        for iinter, taui in enumerate(_tauv):
            tf = t0 + taui
            result += self.int_scheme_.integrate(lambda t: mycost(t, iinter),
                                                 [t0, tf])
            t0 = tf
        return result

    def gradient(self, _x, grad_buff=None):
        _tauv = _x[:self.N_]
        _u = _x[self.N_:]
        y = self.waypointConstraints(_tauv, _u)
        dydtau, _ = self.splcalc_.eval_dydtau(_tauv, self.wp_, y)
        dydu, _ = self.splcalc_.eval_dydu(_tauv, self.wp_, self.uToWp_,
                                          self.dydu_buff_, y)
        if grad_buff is None:
            grad_buff = self.gradient_sum_buff

        result = grad_buff
        result.fill(0.0)

        mygradient = np.vectorize(
            lambda t, inter: self.runningCostGradient(t, _tauv, _u, y, inter, dydtau, dydu),
            signature='(),()->(n)')
        t0 = 0.0
        for iinter, taui in enumerate(_tauv):
            tf = t0 + taui
            result += self.int_scheme_.integrate(
                lambda _t: mygradient(_t, iinter), [t0, tf],
                dot=lambda a, b: a.T.dot(b))
            t0 = tf

        # print(result)
        return result

