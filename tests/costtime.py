"""
    Test the cost function from the problem 1010
"""
import numpy as np
import sympy as sp
import quadpy
import unittest
from opttrj.costtime import cCostTime

from itertools import tee


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        self.wp_ = np.array([[0.8981, -1.4139, -1.7304, -3.0529, -1.239, -9.391],
                [0.1479, -1.6466, -1.8270, -2.523, -1.1025, -9.3893],
                [-0.539, -1.6466, -1.8270, -2.5237, -1.1091, -9.389],
                [-0.9361, -2.0998, -1.5694, -2.523, -0.769, -9.39756],
                [-1.0740, -2.6962, -0.6919, -2.5144, -0.9305, -9.722]])
        self.N_ = self.wp_.shape[0] - 1
        self.dim_ = 6

    def testRun(self):
        cost = cCostTime(self.wp_)

        tauv = np.random.rand(self.N_)

        res = np.zeros((self.N_, ))
        for i in range(5):
            cost(tauv)
            cost.gradient(tauv, res)


def main():
    unittest.main()


if __name__ == '__main__':
    main()

