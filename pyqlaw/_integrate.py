"""
integration
"""

import numpy as np


def eom_gauss(t, state, p):
    """Equations of motino for gauss"""
    # unpack parameters
    mu, f, alpha, beta = p
    u = f*np.array([
        np.cos(beta)*np.sin(alpha),
        np.cos(beta)*np.cos(alpha),
        np.sin(beta),
    ])

    # unpack elements
    sma,ecc,inc,ra,om,ta = state

    p = sma*(1 - ecc**2)
    h = np.sqrt(sma*mu*(1-ecc**2))
    r = h**2/(mu*(1+ecc*np.cos(ta)))

    # Gauss perturbation
    psi = np.array([   
        [2*sma**2*ecc*np.sin(ta) / h, (2*sma**2*p) / (r*h), 0.0],
        [p*np.sin(ta) / h, ( (p + r)*np.cos(ta) + r*ecc ) / h, 0.0],
        [0, 0, r*np.cos(ta+om) / h],
        [0, 0, r*np.sin(ta+om)/ ( h*np.sin(inc) )],
        [-p*np.cos(ta) / ( ecc*h ), (p+r)*np.sin(ta) / ( ecc*h ),  -( r*np.sin(ta+om)*np.cos(inc) ) / ( h*np.sin(inc) )],
        [p*np.cos(ta) / ( ecc*h ), - (p+r)*np.sin(ta) / ( ecc*h ), 0],
    ])

    # combine
    doe = np.dot(psi,u) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, h/r**2])
    return doe


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