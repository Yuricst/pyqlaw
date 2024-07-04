"""
Lyapunov feedback control
"""

import numpy as np
import numpy.linalg as la
from numba import njit

@njit
def _u_to_thrust_angles(u_float):
    """Convert unscaled vector u_float to unit-vector u & thrust angles alpha and beta"""
    u_float_norm = np.sqrt( u_float[0]**2 + u_float[1]**2 + u_float[2]**2 )
    u = u_float/u_float_norm
    alpha = np.arctan2(-u[0],-u[1])
    beta = np.arctan(-u[2]/np.sqrt(u[0]**2 + u[1]**2))
    return u, alpha, beta


def lyapunov_control_angles(
        fun_lyapunov_control,
        mu, 
        f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, b_petro, k_petro, 
        wp, 
        woe,
    ):
    """Compute thrust angles from Lyapunov feedback control law
    
    Args:
        fun_lyapunov_control (callable): symbolic-generated Lyapunov control law
        mu (float): gravitational parameter
        f (float): thrust-acceleration
        oe (np.array): current osculating elements
        oeT (np.array): target osculating elements
        rpmin (float): minimum periapsis
        k_petro (float): scalar factor k on minimum periapsis
        m_petro (float): scalar factor m to prevent non-convergence, default is 3.0
        n_petro (float): scalar factor n to prevent non-convergence, default is 4.0
        r_petro (float): scalar factor r to prevent non-convergence, default is 2.0
        b_petro (float): scalar factor b for dot(omega)_xx, default is 0.01
        wp (float): penalty scalar on minimum periapsis, default is 1.0
        woe (np.array): weight on each osculating element
    
    Returns:
        (tuple): alpha, beta, vector u, list of columns of psi
    """
    # compute u direction
    u_raw, psi, q, dqdoe = fun_lyapunov_control(
        mu, 
        f, 
        oe, 
        oeT, 
        rpmin, m_petro, n_petro, 
        r_petro, b_petro, k_petro, 
        wp, 
        woe
    )

    # compute thrust angles
    u_float = np.array([float(el) for el in u_raw])
    u, alpha, beta = _u_to_thrust_angles(u_float)

    # compute dqdt
    # print("np.array(dqdoe).shape = ", np.array(dqdoe).shape)
    # print("np.array(psi).shape = ", np.array(psi).shape)
    dqdt = np.array(dqdoe) @ np.array(psi).T @ (f * u)
    return alpha, beta, u, psi, q, dqdt