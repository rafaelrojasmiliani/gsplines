#!/usr/bin/env python
from __future__ import print_function
from functools import wraps

try:  # Python 2.X
    from SimpleXMLRPCServer import SimpleXMLRPCServer
    from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
    from SocketServer import ThreadingMixIn
    import ConfigParser
    import StringIO
except ImportError:  # Python 3.X
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.server import SimpleXMLRPCRequestHandler
    from socketserver import ThreadingMixIn

import json
import numpy as np
import math
import os
import sys
import time
import traceback
import time

from .gsplinesjson import json2piecewise
from .gsplinesjson import piecewise2json


def debug_response(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb_list = traceback.extract_tb(exc_tb)
            for i, tb in enumerate(tb_list):
                if tb[0].find(__file__) < 0:
                    raise RuntimeError(' "{}" @ {}:{}'.format(
                        str(e), tb_list[i - 1][0], tb_list[i - 1][1]))
            raise RuntimeError(' "{}" @ {}:{}'.format(
                str(e), tb_list[-1][0], tb_list[-1][1]))

    return decorator


class cGplineXMLRPCServer(object):
    ''' XMLRPC service which stores gsplines and
        retrieve their values on request.
    '''

    def __init__(self, _service_path, _port):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = (_service_path, )

        server = SimpleXMLRPCServer(('0.0.0.0', _port),
            requestHandler = RequestHandler,
            logRequests = False,
            allow_none = True)
        # self.register_instance(self)
        server.register_instance(self)

        self.server_ = server
        self.trajectories={}
        self.trajectories_deriv_={}
        self.follower=None
        self.kp_=0.001
        self.kacc_=10000

        # https://gist.github.com/mcchae/280afebf7e8e4f491a66


    def serve_forever(self):
        self.server_.serve_forever()

    def shutdown(self):
        self.server_.shutdown()
        self.server_.socket.close()

    @debug_response
    def gspline_load(self, unique_id, jsonreq):
        if type(unique_id) is not str:
            return False
        q = json2piecewise(jsonreq)
        self.trajectories[unique_id] = q
        self.trajectories_deriv_[unique_id] = q.deriv()
        return True

    @debug_response
    def gspline_get(self, _regex):
        import re
        dic=self.trajectories
        result=[piecewise2json(dic[k]) for k in dic if re.match(_regex, k)]

        return result

    # proxy service: to evaluate a gspline and return the corresponding control command to follow it (joint velocities)


    @debug_response
    def gspline_control_parameters(self, _kp, _kacc):
        self.kp_=_kp
        self.kacc_=_kacc
        return True

    @debug_response
    def gspline_pff_control(self, unique_id, _t, q_now, qd_now, _scalling = 1.0):
        ''' Returns a control actioin in velocity to trak the trajectory'''
        if unique_id not in self.trajectories:
            raise RuntimeError(
                'gspline {} need to be loaded before evaluating it'.format(
                    unique_id))
        q_d=self.trajectories[unique_id](_t)[0]
        qd_d=self.trajectories_deriv_[unique_id](_t)[0] * _scalling

        err=q_d - np.array(q_now)

        err_d=qd_d - np.array(qd_now)

        acc=self.kacc_ * float(np.linalg.norm(err_d, ord=np.inf))

        u=(-self.kp_ * err + qd_d).tolist()

        result=u + [acc]

        return result

    @debug_response
    def gspline_desired_position(self, unique_id, _t):
        ''' Evaluates the gspline unique_id at time t and returns the
        desired joints' position.'''
        if unique_id not in self.trajectories:
            raise RuntimeError(
                'gspline {} need to be loaded before evaluating it'.format(
                    unique_id))
        q=self.trajectories[unique_id](_t)[0]
        return q.tolist()

    @debug_response
    def gspline_desired_velocity(self, unique_id, _t):
        ''' Evaluates the gspline unique_id at time t and returns the
        desired joints' position.'''
        if unique_id not in self.trajectories:
            raise RuntimeError(
                'gspline {} need to be loaded before evaluating it'.format(
                    unique_id))
        q=self.trajectories_deriv_[unique_id](_t)[0]
        return q.tolist()

    @debug_response
    def gspline_execution_time(self, unique_id):
        ''' Evaluates the gspline unique_id at time t and returns the
        desired joints' position.'''
        if unique_id not in self.trajectories:
            raise RuntimeError(
                'gspline {} need to be loaded before evaluating it'.format(
                    unique_id))
        return float(self.trajectories[unique_id].T_)
