import sys
import os
import pathlib
import matplotlib.pyplot as plt  # plot library
import numpy as np

pwd = pathlib.Path(__file__).parent.absolute().parents[0]

modpath = pathlib.Path(pwd, '')
sys.path.append(str(modpath))

from gsplines import cSplineCalc
from gsplines import cBasis0010
from gsplines import cBasis1010

def plt_show_curve(_q, _dt=0.05, _title='', _tauv=None, _wp=None):
    def der_label(_der):
        if _der == 0:
            return 'Trajectory'
        else:
            return 'Derivative {:d}'.format(_der)
        
    dim = _q.dim_
    fig, ax=plt.subplots(4, dim)
    t = np.arange(0.0, _q.T_, _dt)

    if _tauv is not None:
        time_limits = np.array([np.sum(_tauv[:_n]) for _n in range(len(_tauv)+1)])

    for j in range(dim):
        ax[0, j].set_title('coordinate {:d}'.format(j+1), fontsize=8)
        if _wp is not None:
            for wpi in _wp:
                ax[0, j].axhline(wpi[j], *ax[0,j].get_xlim(), color='m', linestyle='--')
            if _tauv is not None:
                ax[0, j].plot(time_limits, _wp[:, j], 'ro')
                ax[0, j].plot([], [], 'ro', label='waypoints')


    for i in range(4):
        q = _q.deriv(i)
        qt = q(t)
        for j in range(dim):
            ax[i, j].plot(t, qt[:, j], 'b', label=der_label(i))
            ax[i, j].grid()
            ax[i, j].set_xticklabels(ax[i, j].get_xticks(), fontsize=5)
            ax[i, j].set_yticklabels(ax[i, j].get_yticks(), fontsize=5)
            if _tauv is not None:
                for t_i in time_limits:
                    ax[i, j].axvline(t_i, *ax[i,j].get_ylim(), color='m', linestyle='--')

            ax[i, j].plot([],[], 'm--', label='t_i')
            ax[i, j].legend()



    # plt.subplots_adjust(left=0.025, bottom=0.05, right=0.975, top=0.95, wspace=0.25, hspace=0.15)
    fig.suptitle(_title)
    plt.show()

