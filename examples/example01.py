''' Example for basis usage.
This library provide simple classes to represent the basis functions
These basis can provide the evaluation these basis at each normalized value in [-1,1]
In general the calls for evaluating the basis has at least two arguments.
    1) s: the normalized  time in [-1, 1]
    2) tau: the length of the time interval.
'''
from modules import cBasis0010  # fifth order polynomial basis
from modules import cBasis1010  # reg jerk basis
import matplotlib.pyplot as plt  # plot library
import numpy as np  # matlab-like library

def main():
    plot_minimum_jerk_basis()

    plot_reg_jerk_basis()


def plot_minimum_jerk_basis():
    ''' Example where we show the usages of the basis for the minimum jerk motions
    '''
    tau = 1  # time interval length
    # This class represents 6 polynomials stored in memory
    # The aim of this class is make easy to get the value of each of
    # these polynomials and their derivatives in the normalized 
    fifth_order_pol = cBasis0010()

    # Here we evaluate the basis at 0

    val = fifth_order_pol.evalOnWindow(0, tau)
    print('The value of the Polynomial base at s=0 is:')
    print(val)


    normalized_time_interval = np.arange(-1, 1.05, 0.05)  # written "s" in the paper

    # Here we use a python comprehension to create a matrix such that each row
    # contains the values of the basis at the same instant.
    res = np.array([fifth_order_pol.evalOnWindow(t, tau) for t in normalized_time_interval])

    # Here we plot the value of the basis
    for i in range(fifth_order_pol.dim_):
        plt.plot(normalized_time_interval, res[:, i], label='Pol base of degree {:d}'.format(i))
    plt.grid()
    plt.title('Polynomial basis for minimum jerk motions')  # plot title
    plt.xlabel('normalized time interval \"s\"') # Plot horizontal axis lable
    plt.legend()  # Comand to show the labels of each plot
    plt.show()  # show plot


    # Here we use a python comprehension to create a matrix such that each row
    # contains the values of the first derivative of the basis at the same instant.
    res = np.array([fifth_order_pol.evalDerivOnWindow(t, tau, 1) for t in normalized_time_interval])

    # Here we plot the value of the basis
    for i in range(fifth_order_pol.dim_):
        plt.plot(normalized_time_interval, res[:, i], label='Derivative of the pol base of degree {:d}'.format(i))
    plt.grid()
    plt.title('Derivative of the polynomial basis for minimum jerk motions')  # plot title
    plt.xlabel('normalized time interval \"s\"') # Plot horizontal axis lable
    plt.legend()  # Comand to show the labels of each plot
    plt.show()  # show plot


def plot_reg_jerk_basis():
    ''' Example where we show the usages of the basis for the regularized or weighed speed-jerk motions
    '''
    tau = 1  # time interval length
    alpha = 0.8  # Regularization value
    # This class represents 6 polynomials stored in memory
    # The aim of this class is make easy to get the value of each of
    # these polynomials and their derivatives in the normalized 
    reg_jerk_basis = cBasis1010(alpha)

    # Here we evaluate the basis at 0

    val = reg_jerk_basis.evalOnWindow(0, tau)
    print('The value of the weighed speed-jerk basis at s=0 is:')
    print(val)


    normalized_time_interval = np.arange(-1, 1.05, 0.05)  # written "s" in the paper

    # Here we use a python comprehension to create a matrix such that each row
    # contains the values of the basis at the same instant.
    res = np.array([reg_jerk_basis.evalOnWindow(t, tau) for t in normalized_time_interval])

    # Here we plot the value of the basis
    for i in range(reg_jerk_basis.dim_):
        plt.plot(normalized_time_interval, res[:, i], label='base of number {:d}'.format(i))
    plt.grid()
    plt.title('Basis for weighed speed-jerk motions')  # plot title
    plt.xlabel('normalized time interval \"s\"') # Plot horizontal axis lable
    plt.legend()  # Comand to show the labels of each plot
    plt.show()  # show plot


    # Here we use a python comprehension to create a matrix such that each row
    # contains the values of the first derivative of the basis at the same instant.
    res = np.array([reg_jerk_basis.evalDerivOnWindow(t, tau, 1) for t in normalized_time_interval])

    # Here we plot the value of the basis
    for i in range(reg_jerk_basis.dim_):
        plt.plot(normalized_time_interval, res[:, i], label='Derivative of base number {:d}'.format(i))
    plt.grid()
    plt.title('Derivative of the weighed speed-jerk basis')  # plot title
    plt.xlabel('normalized time interval \"s\"') # Plot horizontal axis lable
    plt.legend()  # Comand to show the labels of each plot
    plt.show()  # show plot

if __name__ == '__main__':
    main()
