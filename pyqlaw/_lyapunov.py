"""
Lyapunov feedback control
"""

import numpy as np
import numpy.linalg as la
import dill

# load function from binary
#print(f"directory: {os.listdir()}")
fun_lyapunov_control = dill.load(open("fun_lyapunov_control", "rb"))  # FIXME


def lyapunov_control(
        mu, 
        f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, b_petro, k_petro, 
        wp, woe
    ):
    """Compute thrust angles from Lyapunov feedback control law"""
    # compute u direction
    us = fun_lyapunov_control(
        mu, f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, b_petro, k_petro, 
        wp, woe
    )
    # compute thrust angles
    u = f*np.array(us)/la.norm(np.array(us))

    alpha = np.arctan2(u[0],u[1])
    sin_b = u[2]/la.norm(u)
    if np.isnan(sin_b):
        sin_b = 0.0
    beta = np.arcsin(sin_b)
    return alpha, beta