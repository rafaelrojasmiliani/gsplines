
import xmlrpc.client

from gsplines.gspline import cSplineCalc
from gsplines.basis1010 import cBasis1010
from gsplines.basis0010 import cBasis0010
from gsplines import piecewise2json, json2piecewise
import numpy as np
import json
import unittest
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp


import os
import unittest
import functools
import traceback
import sys
import pdb

u_last = np.zeros((6, ))
u_hist = []

def debug_on(*exceptions):
    ''' Decorator for entering in debug mode after exceptions '''
    if not exceptions:
        exceptions = (Exception, )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])
                sys.exit(1)

        return wrapper

    return decorator

class cMyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        import sys
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)

        dim = 6  # np.random.randint(2, 6)
        N = 5  # np.random.randint(3, 120)
        T = 10.0
        wp = (np.random.rand(N + 1, dim) - 0.5) * 2.0 * np.pi
        self.wp_ = wp
        args = {'unique_id': 0,
                'maximum_speed': 10,
                'maximum_acceleration': 100,
                'sampling_time': 0,
                'operator_vector': 0,  # ?
                'execution_time': T,  # total
                'regularization_factor': 0,  #
                'basis_type': 0,  # a keyword
                'waypoints': wp.tolist()}

        json_args = json.dumps(args)
        proxy = xmlrpc.client.ServerProxy("http://10.10.238.32:5000/mjt")
#
#        q_json = proxy.trajectory_generate(json_args)
#        self.q_ = json2piecewise(q_json)
#        proxy.trajectory_load(unique_id, q_json)

        self.proxy_ = proxy

    @debug_on()
    def test_mock_planner_proxy(self):

        proxy = self.proxy_
        unique_id = '04e7ef7b-e50d-43d8-92f7-03b18945ef91'
        T = proxy.trajectory_execution_time(unique_id)
        tspan = np.arange(0, T, 0.008)

        q_d = np.array([proxy.trajectory_get_desired_position(unique_id, float(t)) for t in tspan])
        qd_d = np.array([proxy.trajectory_get_desired_velocity(unique_id, float(t)) for t in tspan])

        fig, ax = plt.subplots(3, 2)
        ij = 0
        for i in range(3):
            for j in range(2):
                ax[i, j].plot(tspan, q_d[:, ij], 'b')
                #ax[i, j].plot(self.q_.tis_, self.wp_[:, ij], 'ro')
                ax2 = ax[i, j].twinx()
                ax2.set_ylabel('vel', color='magenta')
                ax2.plot(tspan, qd_d[:, ij], color='magenta')
                ax2.grid()

                ij += 1

        plt.show()






def main():
    unittest.main()


if __name__ == '__main__':
    main()

