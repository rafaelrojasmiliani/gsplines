''' Example for generation of generalized splines
This library provide simple classes to build splines of any regular basis.
In other words, given the following ingredients:
    1) basis
    2) time inervals
    3) waypoints
the library can compute the piece-wise function which joints the required waypoints.
'''
from modules import cBasis0010  # fifth order polynomial basis
from modules import cBasis1010  # reg jerk basis
from modules import cSplineCalc # Spline calculator
from modules import plt_show_curve
import matplotlib.pyplot as plt  # plot library
import numpy as np  # matlab-like library

def main():
    plot_fifth_order_spline()

    plot_weighed_speed_jerk_spline()



def plot_fifth_order_spline():
    ''' Example where we show how to build a fifth-order spline passing through
    a sequence of waypoints.
    This requires two main steps:
    1) build the gspline constructor: This allocates the memory necessary to compute fast the "passing through waypoints constrain". This step requires the following data:
        1.1) Dimension of the ambient space. Dimension of the space where we
            are computing the curve. For example, to compute a minimum jerk
            trajectory in Cartesian space, the dimension of the ambient space
            is 3. To compute a minimum jerk motion of a 6-axis robot in the
            joint space the dimension of the ambient space is 6.
        1.2) Number of intervals (number of waypoints minus one)
        1.3) Basis of function to joint the waypoints

    2) Call the getSpline method, specifing the time intervals (tauv) and the waypoints.
    '''
    N = 2
    # Number of intervals is N=2
    tauv = np.array([1.0, 2.0])  # Time interval between waypoints

    dim = 3
    # Number of waypoints is N+1 = 3, dimenion of ambient space dim=3
    wp = np.array([  [0, 0, 0],  # waypoint 0
            [1, 1, 1],  # waypoint 1
            [2, -1, 0]]) # waypoint 2

    fifth_order_pol = cBasis0010()

    # Here we build the spline constructor
    spline_calculator = cSplineCalc(dim, N, fifth_order_pol)

    fifth_order_spline = spline_calculator.getSpline(tauv, wp)


    plt_show_curve(fifth_order_spline, _title='fifth order spline', _tauv=tauv, _wp=wp)

    print('The L2 norm of the velocity of the curve is {:+.5f}'.format(fifth_order_spline.l2_norm(1)))

    print('The L2 norm of the jerk of the curve is {:+.5f}'.format(fifth_order_spline.l2_norm(3)))



def plot_weighed_speed_jerk_spline():
    ''' Example where we show how to build a speed-jerk weighed spline passing through
    a sequence of waypoints.
    This requires two main steps:
    1) build the gspline constructor: This allocates the memory necessary to compute fast the "passing through waypoints constrain". This step requires the following data:
        1.1) Dimension of the ambient space. Dimension of the space where we
            are computing the curve. For example, to compute a minimum jerk
            trajectory in Cartesian space, the dimension of the ambient space
            is 3. To compute a minimum jerk motion of a 6-axis robot in the
            joint space the dimension of the ambient space is 6.
        1.2) Number of intervals (number of waypoints minus one)
        1.3) Basis of function to joint the waypoints

    2) Call the getSpline method, specifing the time intervals (tauv) and the waypoints.
    '''
    N = 5
    # Number of intervals is N=2
    tauv = np.random.rand(N)  # Time interval between waypoints

    dim = 3
    # Number of waypoints is N+1 = 3, dimenion of ambient space dim=3
    wp = np.random.rand(N+1, dim)

    alpha = 0.99
    weighed_speed_jerk_spline = cBasis1010(alpha)

    # Here we build the spline constructor
    spline_calculator = cSplineCalc(dim, N, weighed_speed_jerk_spline)

    fifth_order_spline = spline_calculator.getSpline(tauv, wp)


    plt_show_curve(fifth_order_spline, _title='weighed speed-jerk spline', _tauv=tauv, _wp=wp)

    print('The L2 norm of the velocity of the curve is {:+.5f}'.format(fifth_order_spline.l2_norm(1)))

    print('The L2 norm of the jerk of the curve is {:+.5f}'.format(fifth_order_spline.l2_norm(3)))


if __name__ == '__main__':
    main()
