from ..piecewisefunction.piecewisefunction import cPiecewiseFunction
import json
import numpy as np

import gsplines.basis


def piecewise2json(_pw):
    basis_name = _pw.basis_.__class__.__name__
    if hasattr(_pw.basis_, 'params_'):
        basis_params = _pw.basis_.params_
    else:
        basis_params = None

    basis = [basis_name, basis_params]
    result = [_pw.tau_.tolist(), _pw.y_.tolist(), _pw.dim_, basis]

    return json.dumps(result)


def json2piecewise(_data):
    array = json.loads(_data)
    for i, element in enumerate(array[:-2]):
        array[i] = np.array(element)
    basis_data = array[-1]
    class_ = getattr(gsplines.basis, basis_data[0])
    
    if basis_data[1] is not None:
        basis = class_(basis_data[1])
    else:
        basis = class_()

    array[-1] = basis

    result = cPiecewiseFunction(*array)

    return result
