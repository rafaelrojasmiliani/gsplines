
import xmlrpc.client

from gsplines.interpolator import rand_interpolate
from gsplines.basis.basis0010 import cBasis0010
from gsplines.services.gsplinesjson import piecewise2json, json2piecewise
from gsplines.services.xmlrpc import cGplineXMLRPCServer
import numpy as np
import json
import unittest
import matplotlib.pyplot as plt
import time
from threading import Thread





class cMyTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(cMyTest, self).__init__(*args, **kwargs)
        import sys
        np.set_printoptions(
            linewidth=5000000,
            formatter={'float': '{:+10.3e}'.format},
            threshold=sys.maxsize)

        self.path_ =  '/mjt'
        self.port_ = 8001
        self.server_ = None
        self.thread_ = None
        self.url_ = 'http://127.0.0.1:{}{}'.format(self.port_, self.path_)

    def setUp(self):
        server = cGplineXMLRPCServer(self.path_, self.port_)
        self.server_ = server

        self.thread_ = Thread(target=server.serve_forever)
        self.thread_.start()

    def tearDown(self):
        self.server_.shutdown()


    def test_container(self):


        dim = 6  # np.random.randint(2, 6)
        N = 5  # np.random.randint(3, 120)
        T = 10.0
        box = np.pi
        q0 = rand_interpolate(N, dim, cBasis0010(), T, box)


        server = xmlrpc.client.ServerProxy(self.url_)
        
        server.gspline_load('unique_id', piecewise2json(q0))
        

        res = server.gspline_get('u*')
        q1 = [json2piecewise(el) for el in res][0]



        timespan = np.arange(0, T, 0.01)

        assert np.linalg.norm(q0(timespan) - q1(timespan), ord=np.inf) < 1.0e-10

        server.gspline_load('unique_id_2', piecewise2json(q0))
        res = server.gspline_get('u*')

        assert len(res) == 2


    def test_values(self):


        server = xmlrpc.client.ServerProxy(self.url_)
        dim = 6  # np.random.randint(2, 6)
        N = 5  # np.random.randint(3, 120)
        T = 10.0
        box = np.pi
        unique_id = 'hola'
        q0 = rand_interpolate(N, dim, cBasis0010(), T, box)
        tspan = np.arange(0, T, 0.01)
        q0t = q0(tspan) 
        q0dt = q0.deriv()(tspan)

        server.gspline_load(unique_id, piecewise2json(q0))

        q1t = np.array([server.gspline_desired_position(unique_id, float(t)) for t in tspan])
        q1dt = np.array([server.gspline_desired_velocity(unique_id, float(t)) for t in tspan])


        assert np.linalg.norm(q0t - q1t, ord=np.inf) < 1.0e-10
        assert np.linalg.norm(q0dt - q1dt, ord=np.inf) < 1.0e-10

#        args = {'unique_id': 0,
#                'maximum_speed': 10,
#                'maximum_acceleration': 500,
#                'sampling_time': 0,
#                'operator_vector': 0,  # ?
#                'execution_time': T,  # total
#                'regularization_factor': 0,  #
#                'basis_type': 0,  # a keyword
#                'waypoints': wp.tolist()}
#
#        json_args = json.dumps(args)
#        ip = '10.10.238.1'
#        server = xmlrpc.client.ServerProxy('http://'+ip+':5000/mjt')
#        q_json = server.trajectory_generate(json_args)
#
#        q = json2piecewise(q_json)
#
#        assert abs(q.T_ - T) < 1.0e-10
#
#
#
#        plt.plot(time, qt[:, 0])
#        plt.show()
#
#        res = server.trajectory_load('unique-id', piecewise2json(q))
#        assert res
#
#        res = server.trajectory_execution_time('unique-id')
#
#        T = res
#
#        tarray = np.arange(0, T, T/20.0)
#        qt = q(tarray)
#
#        pt = np.array([server.trajectory_eval_time('unique-id', float(t)) for t in tarray])
#
#        assert abs(np.max(pt-qt)) < 1.0e-8
#
#        t0 = time.time()
#        q0=server.trajectory_eval_time('unique-id', 0.0)
#        t1 = time.time()
#
#        print(t1-t0)
#        t0 = time.time()
#        q0=q(0.0)
#        t1 = time.time()
#
#        print(t1-t0)
#
#





def main():
    unittest.main()


if __name__ == '__main__':
    main()

