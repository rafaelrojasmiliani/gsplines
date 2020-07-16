'''
Test basis, which represents the basis of the Legendre Polynomials
'''
import numpy as np
import unittest
from gsplines.basis0010 import cBasis0010
from gsplines.basis1010 import cBasis1010
from gsplines.gspline import cSplineCalc
from gsplines.json import piecewise2json, json2piecewise
    
class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        ''' Initialize the symbolic expression of the Legendre Polynomials
        '''
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)
        self.dim_ = 6  # np.random.randint(2, 6)
        self.N_ = 50  # np.random.randint(3, 120)

        self.wp_ = np.random.rand(self.N_+1, self.dim_)
        self.tauv_ = np.random.rand(self.N_)
        self.T_ = np.sum(self.tauv_)

    def test(self):

        b1 = cBasis0010()
        b2 = cBasis1010(0.5)

        spc1 = cSplineCalc(self.dim_, self.N_, b1)
        spc2 = cSplineCalc(self.dim_, self.N_, b2)

        q1 = spc1.getSpline(self.tauv_, self.wp_) 
        q2 = spc2.getSpline(self.tauv_, self.wp_) 
        
        time = np.arange(0, self.T_, 0.001)
        for q in [q1, q2]:
            qt = q(time)
            json = piecewise2json(q)

            p = json2piecewise(json)
            pt = p(time)
            assert(np.linalg.norm(pt - qt)<1.0e-10)





def main():
    unittest.main()


if __name__ == '__main__':
    main()
