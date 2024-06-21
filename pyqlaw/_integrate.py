"""
integration
"""

import numpy as np
from numba import njit

@njit
def rk4(rhs, t, h, y, p):
    """Perform single-step Runge-Kutta 4th order
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = rhs(t, y, p)
    k2 = rhs(t + 0.5 * h, y + 0.5 * k1, p)
    k3 = rhs(t + 0.5 * h, y + 0.5 * k2, p)
    k4 = rhs(t + h, y + k3, p)
    return y + (h / 6.0)*(k1 + 2*k2 + 2*k3 + k4)


@njit
def rkf45(rhs, t, h, y, p, tol=1e-6):
    """Perform single-step Runge-Kutta 4th order with 5th order step correction
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`
        tol (float): tolerance

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = h * rhs(t, y, p)
    k2 = h * rhs(t + 1/4 * h,   y + 1/4*h * k1, p)
    k3 = h * rhs(t + 3/8 * h,   y + 3/32*h*k1 + 9/32*h*k2, p)
    k4 = h * rhs(t + 12/13 * h, y + 1932/2197*h*k1 - 7200/2197*h*k2 + 7296/2197*h*k3, p)
    k5 = h * rhs(t + h,         y + 439/216*h*k1 - 8*h*k2 + 3680/513*h*k3 - 845/4104*h*k4, p)
    k6 = h * rhs(t + 1/2*h,     y - 8/27*h*k1 + 2*h*k2 - 3544/2565*h*k3 + 1859/4104*h*k4 - 11/40*h*k5, p)
    y_next = y + 25/216*k1 + 1408/2565*k3 + 2197/4101*k4 - 1/5*k5
    z_next = y + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6
    h_next = h*0.84*(tol/np.linalg.norm(z_next - y_next))**(1/4)
    return y_next, h_next