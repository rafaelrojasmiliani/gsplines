"""
    Test functions in the space of solutions of the
    Euler Lagrange equations of

    \int_{-1}^{1} (2/tau) \alpha dq/ds + (2/tau)^5 (1-\alpha) d^3 q / ds^3 dt
"""
import numpy as np
import sympy as sp
from scipy.sparse import csc_matrix
import unittest
from gsplines.linspline import cSplineCalc_2, plt_show_gspline
from gsplines.basis1010 import cBasis1010
from gsplines.basis0010 import cBasis0010
from gsplines.basis1000 import cBasis1000
from gsplines.banded import diagonal_form
import matplotlib.pyplot as plt


class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        import sys
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)
        self.dim_ = 2
        self.N_ = 2
        self.tauv_ = 2.0 + np.random.rand(self.N_)
        self.wp_ = np.random.rand(self.N_ + 1, self.dim_)

    def test_constructor(self):

        sp = cSplineCalc_2(self.dim_, self.N_, cBasis1000())

        A = sp.eval_A(self.tauv_).todense()
#        print()
#        print(A)
        b = sp.eval_b(self.wp_)
#        print(self.wp_)
#        print(b)

        q = sp.get_gspline(self.tauv_, self.wp_)

        plt_show_gspline(q)

    def test_derivative_b(self):
        '''Test derivative of b w.r.t. waypoint components
            Here we rest the correctness of the numerical
            output of the basis class comparing it with
            its analitical form optained using sympy
        '''
        splcalc = cSplineCalc_2(self.dim_, self.N_, cBasis1000())

        dwp = 0.0005
        for i in range(self.N_ + 1):
            for j in range(self.dim_):
                wpidx = i

                wp_aux = self.wp_.copy()
                wp_aux[wpidx, j] += -dwp
                b1 = splcalc.eval_b(wp_aux).copy()
                wp_aux[wpidx, j] += 2 * dwp
                b2 = splcalc.eval_b(wp_aux).copy()

                dbdwpij_num = 0.5 * (b2 - b1) / dwp

                dbdwpij_nom = splcalc.eval_dbdwpij(wpidx, j)

                e = np.max(np.abs(dbdwpij_num - dbdwpij_nom))

                if e > 1.0e-8:
                    print('Erroe in db_dwpij:')
                    print('implementation:')
                    print(dbdwpij_nom)
                    print('(b1-b2)/dwp:')
                    print(dbdwpij_num)
                    print('component', i)
                    print('waypoint ', wpidx)
                    print('dimension ', self.dim_)
                    print('number of intervals ', self.N_)
                    raise AssertionError('Error of {:14.7e}'.format(e))


    def test_derivative_wp(self):
        ''' Compare the numerical derivate of y w.r.t
        waypoints with the nominal one'''
        for _ in range(4):
            tauv = 0.5 + np.random.rand(self.N_) * 2.0
            splcalc = cSplineCalc_2(self.dim_, self.N_, cBasis1000())
            y = splcalc.waypoint_constraint(tauv, self.wp_)

            err = 0.0
            errp = 0.0

            err = 0.0
            errp = 0.0
            dwp = 1.0e-5

            wpidx = [(i, j) for i in range(self.N_ + 1) for j in range(self.dim_)]
            dydwpNom = np.zeros((y.shape[0], len(wpidx)))
            dydwpNom, _ = splcalc.eval_dydu(tauv, self.wp_, wpidx, dydwpNom)

            for k, (i, j) in enumerate(wpidx):
                wp_aux = self.wp_.copy()
                wpidx = i
                wpcom = j

                wp_aux[wpidx, wpcom] += -3 * dwp
                y0 = splcalc.eval_y(tauv, wp_aux).copy() * (-1.0 / 60.0)
                wp_aux[wpidx, wpcom] += dwp
                y1 = splcalc.eval_y(tauv, wp_aux).copy() * (3.0 / 20.0)
                wp_aux[wpidx, wpcom] += dwp
                y2 = splcalc.eval_y(tauv, wp_aux).copy() * (-3.0 / 4.0)
                wp_aux[wpidx, wpcom] += 2 * dwp
                y3 = splcalc.eval_y(tauv, wp_aux).copy() * (3.0 / 4.0)
                wp_aux[wpidx, wpcom] += dwp
                y4 = splcalc.eval_y(tauv, wp_aux).copy() * (-3.0 / 20.0)
                wp_aux[wpidx, wpcom] += dwp
                y5 = splcalc.eval_y(tauv, wp_aux).copy() * (1.0 / 60.0)

                dydwpTest = (y0 + y1 + y2 + y3 + y4 + y5) / dwp

                ev = np.abs(dydwpNom[:, k] - dydwpTest)
                e = np.max(ev)
                eidx = np.argmax(ev)
#                print('{:14.7e} {:14.7e} {:14.7e}'.format(
#                    e, dydwpNom[eidx, k], dydwpTest[eidx]))

                ep = e / dydwpTest[eidx]

                if e > err:
                    err = e
                if ep > errp:
                    errp = ep

                if e > 1.0e-4:
                    assert ep < 1.0e-8, '''
                    Relative Error   = {:10.3e}
                    Absolute Error   = {:10.3e}
                    '''.format(ep, e)


#            print('Maximum Error for dy dwp           = {:14.7e}'.format(err))
#            print('Maximum Relative Error for dy dwp  = {:14.7e}'.format(errp))

#                assert e < 5.0e-2, 'error = {:14.7e}'.format(e)




def main():
    unittest.main()


if __name__ == '__main__':
    main()

