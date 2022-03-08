"""
integration
"""

import numpy as np


def eom_gauss(t, state, p):
    # unpack parameters
    mu, f, alpha, beta = p
    # unpack elements
    a,e,i,ra,om,ta = state
    psi = [
        # multiplies f_r
        [
            2*a**2/h*e*np.sin(ta),
            p/h*np.sin(ta),
            0.0,
            0.0,
            -p/(e*h)*np.cos(ta)
        ],
        # multiplies f_theta
        [
            2*a**2/h * p/r,
            ((p+r)*np.cos(ta) + r*e)/h,
            0.0,
            0.0,
            (p+r)*np.sin(ta)/(e*h)
        ],
        # multiplies f_h
        [
            0.0,
            0.0,
            r*np.cos(ta+om)/h,
            r*np.sin(ta+om)/(h*np.sin(i)),
            -r*np.sin(ta+om)*np.cos(i)/(h*np.sin(i)),
        ]
    ]
    return


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
    k1 = h * rhs(t, y, p)
    k2 = h * rhs(t + 0.5 * h, y + 0.5 * k1, p)
    k3 = h * rhs(t + 0.5 * h, y + 0.5 * k2, p)
    k4 = h * rhs(t + h, y + k3, p)
    return y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)