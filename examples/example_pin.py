"""
 Example of pinocchio implementation.
 This example only shows how to create a simple pinoccio model
 from a URDF file and then it compute the derivatives of the inverse kinematics w.r.t q, qd, and qdd
"""
from sys import argv
from os.path import dirname, join, abspath
import pinocchio
import numpy as np
from modules import cBasis0010  # fifth order polynomial basis
from modules import cBasis1010  # reg jerk basis
from modules import cSplineCalc  # Spline calculator

# This path refers to Pinocchio source code but you can define your own
# directory here.


def main():

    q = create_gspline()

    model, data = create_pinocchio_model()

    time = np.arange(0.0, q.T_, 0.01)

    q_data = [q.deriv(i)(time) for i in range(3)]
    torque = np.array([
        pinocchio.rnea(model, data, _q, _qd, _qdd)
        for _q, _qd, _qdd in zip(q_data)
    ])

    q_data = q_data + torque

    for curve in q_data:
        for j in range(curve.shape[1]):
            ax[i, j].plot(time, curve[:, j], 'b')
            ax[i, j].grid()

    # plt.subplots_adjust(left=0.025, bottom=0.05, right=0.975, top=0.95, wspace=0.25, hspace=0.15)
    fig.suptitle(_title)
    plt.show()



def create_pinocchio_model():
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "example_pin_model/")

    urdf_filename = pinocchio_model_dir + 'model.urdf' 
    # Load the urdf model
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    print('model name: ' + model.name)
    # Create data required by the algorithms
    data = model.createData()

    return model, data


def create_gspline():
    """
    """
    N = 5
    # Number of intervals is N=2
    tauv = np.random.rand(N) * 2.0  # Time interval between waypoints

    dim = 2
    # Number of waypoints is N+1 = 3, dimension of ambient space dim=3
    wp = np.random.rand(N + 1, dim)

    fifth_order_pol = cBasis0010()

    # Here we build the spline constructor
    spline_calculator = cSplineCalc(dim, N, fifth_order_pol)

    fifth_order_spline = spline_calculator.getSpline(tauv, wp)

    return fifth_order_spline


if __name__ == '__main__':
    main()
