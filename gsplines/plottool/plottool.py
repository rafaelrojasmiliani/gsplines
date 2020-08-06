
import matplotlib.pyplot as plt
import numpy as np

def show_piecewisefunction(_q, _up_to_deriv=3, _dt=0.1):
    dim = _q.dim_
    fig, ax=plt.subplots(4, dim)
    t = np.arange(0.0, _q.T_, _dt)

    for i in range(0, _up_to_deriv+1):
        q = _q.deriv(i)
        qt = q(t)
        for j in range(0, dim):
            ax[i, j].plot(t, qt[:, j])
            ax[i, j].grid()
            if i == 0:
                ax[i, j].set_title('coordinate {:d}'.format(j+1), fontsize=8)

    plt.show()
