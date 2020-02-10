'''
    This is a base class used to create a constraint distribyted along
    the spline.
'''
import copy as cp
import numpy as np
from gsplines.gspline import cSplineCalc
from gsplines.basis0010 import cBasis0010


class cDistFunction(object):
    '''
        As the spline is a map
        phi:R^N \times R^k -> Sobolev(R, R)
        this construct a map
        g:R^M \times R^k -> R^n
        s.t.
        g_i(tau, y) = g(q(t_i), \dot q(t_i), \ddot q(t_i))
    '''

    def __init__(self,
                 _scalarmap,
                 _N,
                 _basis=cBasis0010(),
                 _points=np.arange(-1, 1, 0.1),
                 _wpidx=[]):
        '''
            _splcalc: cSplineCalc,
                    Spline computation entity
            _points: array of double
                    points in the interval [-1, 1]
                    where the constraint is evaluated
        '''
        self.dim_ = _scalarmap.dim_
        _splcalc = cSplineCalc(self.dim_, _N, _basis)
        self.splcalc_ = _splcalc
        self.N_ = _N
        self.scalarmap_ = cp.deepcopy(_scalarmap)

        self.points_ = _points.copy()

        nt = self.N_ * _points.shape[0]
        self.timebuffer_ = np.zeros((nt, ))
        self.timebuffer2_ = np.zeros((nt, ))
        self.timebuffer3_ = np.zeros((nt, ), dtype=np.int32)
        self.timebuffer4_ = np.zeros((nt, ))
        self.valbuff_ = np.zeros((nt, ))

        self.dim_ = _splcalc.dim_
        self.qbuff1_ = np.zeros((self.dim_, ))
        self.qbuff2_ = np.zeros((self.dim_, ))

        self.jacbuff_ = np.zeros((nt, 100))

    def getTimes(self, tauv):

        tauv = tauv[:self.N_]
        n = self.points_.shape[0]
        t0 = 0.0
        for i, taui in enumerate(tauv):
            for j, si in enumerate(self.points_):
                self.timebuffer_[n * i + j] = 0.5 * (1.0 + si) * taui + t0
                self.timebuffer2_[n * i + j] = si
                self.timebuffer3_[n * i + j] = i
                self.timebuffer4_[n * i + j] = taui
            t0 += taui

        return self.timebuffer_, self.timebuffer2_, self.timebuffer3_, self.timebuffer4_

    def __call__(self, _tauv, _wp):
        '''
            x = (tau, u)
        '''

        tis, sis, iintrs, taus = self.getTimes(_tauv)
        q = self.splcalc_.getSpline(_tauv, _wp)
        qd = q.deriv()
        q = q(tis)
        qd = qd(tis)

        g = self.scalarmap_
        self.valbuff_[:] = [g(qi, qdi) for qi, qdi in zip(q, qd)]

        return self.valbuff_

    def jacobian(self, _tauv, _wp, _wpidx=[], _dydwbuff=None, _resbuff=None):
        ''' Returns the Jacobian of the function w.r.t. the taus and desired
        waypoints.
            d g(t_i)  d q_j       d g(t_i)  d qd_j    |
            --------  -------  +  --------  --------  |
            d q_j     d tau_k     d qd_j     d tau_k  |
            d g(t_i)  d q_j       d g(t_i)  d qd_j   
            --------  -------  +  --------  -------- 
            d q_j     d wp_k      d qd_j     d wp_k

        '''
        tis, sis, iintrs, taus = self.getTimes(_tauv)
        q = self.splcalc_.getSpline(_tauv, _wp)
        y = q.y_
        qd = q.deriv()

        ulen = len(_wpidx)
        if ulen > 0:
            if _dydwbuff is None:
                _dydwbuff = np.zeros((q.y_.shape[0], ulen))
            else:
                assert _dydwbuff.shape == (q.y_.shape[0], ulen)

        if _resbuff is None:
            _resbuff = np.zeros((tis.shape[0], self.N_ + ulen))
        else:
            assert _resbuff.shape == (tis.shape[0], self.N_ + ulen)

        g = self.scalarmap_

        dydtau, _ = self.splcalc_.eval_dydtau(_tauv, _wp, q.y_)

        dydwp, _ = self.splcalc_.eval_dydu(_tauv, _wp, _wpidx, _dydwbuff, q.y_)

        N = self.N_

        basis = self.splcalc_.basis_

        dim = self.dim_

        q = q(tis)
        qd = qd(tis)
        for i, (ti, si, jintr, tauj) in enumerate(zip(tis, sis, iintrs, taus)):
            dg_dq_ti = g.deriv_wrt_q(q[i, :], qd[i, :])
            dg_dqd_ti = g.deriv_wrt_qd(q[i, :], qd[i, :])

            dq_dtaui = self.qbuff1_
            dqd_dtaui = self.qbuff2_
            B = basis.evalDerivOnWindow(si, tauj, 0)
            Bd = basis.evalDerivOnWindow(si, tauj, 1)
            dB_dtau = basis.evalDerivWrtTauOnWindow(si, tauj, 0)
            dBd_dtau = basis.evalDerivWrtTauOnWindow(si, tauj, 1)

            for iintr, _ in enumerate(_tauv):
                for idim in range(dim):
                    i0 = jintr * 6 * dim + 6 * idim
                    i1 = i0 + 6
                    dyidtaui = dydtau[i0:i1, iintr]
                    yi = y[i0:i1]
                    dq_dtaui[idim] = dyidtaui.dot(B)

                    if iintr == jintr:
                        dq_dtaui[idim] += yi.dot(dB_dtau)

                    dqd_dtaui[idim] = dyidtaui.dot(Bd)

                    if iintr == jintr:
                        dqd_dtaui[idim] += yi.dot(dBd_dtau)

                _resbuff[i, iintr] = dg_dq_ti.dot(dq_dtaui) \
                    + dg_dqd_ti.dot(dqd_dtaui)

            dq_dui = self.qbuff1_
            dqd_dui = self.qbuff2_

            for i_u in range(ulen):
                for idim in range(self.dim_):
                    i0 = jintr * 6 * dim + 6 * idim
                    i1 = i0 + 6
                    dq_dui[idim] = dydwp[i0:i1, i_u].dot(B)

                    dqd_dui[idim] = dydwp[i0:i1, i_u].dot(Bd)

                _resbuff[i, i_u + N] = dg_dq_ti.dot(dq_dui) \
                    + dg_dqd_ti.dot(dqd_dui)

        return _resbuff
