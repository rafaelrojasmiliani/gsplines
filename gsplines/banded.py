import numpy as np


def diagonal_form(a, upper=1, lower=1):
    """
    a is a numpy square matrix this function converts a square matrix to
    diagonal ordered form returned matrix in ab shape which can be used
    directly for scipy.linalg.solve_banded
    """
    n = a.shape[1]
    assert (np.all(a.shape == (n, n)))

    ab = np.zeros((2 * n - 1, n))

    for i in range(n):
        ab[i, (n - 1) - i:] = np.diagonal(a, (n - 1) - i)

    for i in range(n - 1):
        ab[(2 * n - 2) - i, :i + 1] = np.diagonal(a, i - (n - 1))

    mid_row_inx = int(ab.shape[0] / 2)
    upper_rows = [mid_row_inx - i for i in range(1, upper + 1)]
    upper_rows.reverse()
    upper_rows.append(mid_row_inx)
    lower_rows = [mid_row_inx + i for i in range(1, lower + 1)]
    keep_rows = upper_rows + lower_rows
    ab = ab[keep_rows, :]

    return ab
