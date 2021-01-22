from .gspline import cSplineCalc
import numpy as np


def interpolate(_tauv, _wp, _basis):
    N = _wp.shape[0] - 1
    dim = _wp.shape[1]
    gsplcalc = cSplineCalc(dim, N, _basis)

    result = gsplcalc(_tauv, _wp)

    del gsplcalc

    return result


def rand_interpolate(_N, _dim, _basis, _T=10.0, _bounding_box=1.0):
    N = _N
    dim = _dim
    wp = (2 * np.random.rand(N + 1, dim) - 1) * _bounding_box

    tauv = np.random.uniform(0.2, 1, _N)
    tauv = tauv/np.sum(tauv)*_T
    gsplcalc = cSplineCalc(dim, N, _basis)

    result = gsplcalc(tauv, wp)

    del gsplcalc

    return result
