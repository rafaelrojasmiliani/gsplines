
import numpy as np


def optimal_taus(_wp):
    assert _wp.ndim == 2

    ni = _wp.shape[0] - 1

    matrix = np.zeros((ni, ni))

    for i in range(0, ni-1):
        matrix[i, i] = np.linalg.norm(_wp[i+2] - _wp[i+1])
        matrix[i, i+1] = -np.linalg.norm(_wp[i+1] - _wp[i])
    matrix[ni-1, :] = 1

    vector = np.zeros((ni,))
    vector[-1] = 1

    result = np.linalg.solve(matrix, vector)
    return result
