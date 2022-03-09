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
    u_raw = fun_lyapunov_control(
        mu, f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, b_petro, k_petro, 
        wp, woe
    )
    
    # compute thrust angles
    u_float = np.array([float(el) for el in u_raw])
    u_float_norm = np.sqrt( u_float[0]**2 + u_float[1]**2 + u_float[2]**2 )
    u = u_float/u_float_norm

    alpha = np.arctan2(u[0],u[1])
    sin_b = u[2]/la.norm(u)
    if np.isnan(sin_b):
        sin_b = 0.0
    beta = np.arcsin(sin_b)
    return alpha, beta