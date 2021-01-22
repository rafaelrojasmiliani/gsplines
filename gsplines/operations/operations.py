
from copy import deepcopy
from scipy.special import binom
import numpy as np


class Function:
    def __iadd__(self, _f):
        f1 = deepcopy(self)
        return FunctionSum(f1, _f)

    def __imul__(self, _f):
        f1 = deepcopy(self)
        return FunctionScalarMul(f1, _f)

    def compose(self, _f):
        f1 = deepcopy(self)
        return FunctionComposition(f1, _f)

    def __add__(self, _f):
        f1 = deepcopy(self)
        return FunctionSum(f1, _f)


class ConstFunction(Function):
    def __init__(self, _val, _dim):
        self.dim_ = _dim
        self.val_ = _val

    def __call__(self, _t):
        return np.ones((self.dim_, 1))*self.val_

    def deriv(self, _deg=1):
        return ZeroFunction(self.dim_)


class ZeroFunction(ConstFunction):
    def __init__(self, _dim):
        ConstFunction.__init__(self, 0.0, _dim)


class OneFunction(ConstFunction):
    def __init__(self, _dim):
        ConstFunction.__init__(self, 1.0, _dim)


class FunctionSum(Function):
    def __init__(self, _f1, _f2):
        assert _f1.dim_ == _f2.dim_
        self.f1_ = deepcopy(_f1)
        self.f2_ = deepcopy(_f2)
        self.dim_ = _f1.dim_

    def __call__(self, _t):
        return self.f1_(_t) + self.f2_(_t)

    def deriv(self, _deg=1):
        f1 = self.f1_.deriv(_deg)
        f2 = self.f2_.deriv(_deg)
        return FunctionSum(f1, f2)


class FunctionScalarMul(Function):
    def __init__(self, _f1, _scalarfun):
        assert _scalarfun.dim_ == 1
        self.dim_ = _f1.dim_
        self.f1_ = deepcopy(_f1)
        self.scalarfun_ = deepcopy(_scalarfun)

    def __call__(self, _t):
        _t = np.atleast_1d(_t)
        f1 = self.f1_(_t)
        scalar = self.scalarfun_(_t)
        return np.multiply(scalar, f1)

    def deriv(self, _deg=1):
        result = ZeroFunction(self.dim_)
        fun1 = self.f1_
        sfun = self.scalarfun_
        for i in range(_deg):
            term1 = ConstFunction(binom(_deg, i), 1)
            term2 = FunctionScalarMul(fun1.deriv(_deg-i), sfun.deriv(i))
            term3 = FunctionScalarMul(term1, term2)
            result = FunctionSum(term3, result)

        return result


class FunctionComposition(Function):
    def __init__(self, _f1, _sfun):
        self.f1_ = _f1
        assert _sfun.dim_ == 1
        self.scalarfun_ = _sfun

    def __call__(self, _t):
        return self.f1_(self.scalarfun_(_t))

    def deriv(self, _deg=1):
        f1 = FunctionComposition(self.f1_.deriv(), self.scalarfun_)
        f2 = self.scalarfun_.deriv()
        return FunctionScalarMul(f1, f2)
