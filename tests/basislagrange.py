'''
Test basis, which represents the basis of the Legendre Polynomials
'''
import numpy as np
from scipy.special import factorial
import unittest
from gsplines.basis.basislagrange import cBasisLagrange


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        ''' Initialize the symbolic expression of the Legendre Polynomials
        '''
        super(cMyTest, self).__init__(*args, **kwargs)
        np.set_printoptions(linewidth=500, precision=4)

    def test_value(self):
        """ Test the approximation quality of the approximation of the cosine
        function on [-1, 1] by the lagrange intepolators using the Chebychev
        points"""

        for n in range(5, 10):
            nodes = np.array([np.cos((2*k-1)/2/n*np.pi)
                              for k in range(1, n+1)])
            basis = cBasisLagrange(nodes)
            omega = 2.4
            coeff = np.cos(omega*nodes)
            err_bound = 1.0/np.power(2, n-1)/factorial(n)*np.power(omega, n)
            der_err_bound = n/np.power(2, n-2) / \
                factorial(n)*np.power(omega, n+1)
            print("error bound", err_bound)
            print("error bound for deriv", der_err_bound)
            for _ in range(100):
                s = np.random.rand()*2 - 1
                approx = basis.evalOnWindow(s, 2).dot(coeff)

                error = np.linalg.norm(approx - np.cos(omega*s))
                assert error < err_bound

                der_approx = basis.evalDerivOnWindow(s, 2, 1).dot(coeff)
                error = np.linalg.norm(der_approx + omega*np.sin(omega*s))

                assert error < der_err_bound


def main():
    unittest.main()


if __name__ == '__main__':
    main()
