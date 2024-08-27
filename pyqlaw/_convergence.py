"""
Check convergence
"""

import copy
import numpy as np
from numba import njit

@njit
def check_convergence_keplerian(oe, oeT, woe, tol_oe):
    """Check convergence between oe and oeT for Keplerian elements"""
    check_array = np.zeros(len(tol_oe),)
    doe = np.abs(np.array([
        oeT[0] - oe[0],  # a
        oeT[1] - oe[1],  # e
        oeT[2] - oe[2],  # i
        np.arccos(np.cos(oeT[3] - oe[3])),  # RAAN
        np.arccos(np.cos(oeT[4] - oe[4])),  # omega
    ]))
    for idx in range(len(tol_oe)):
        if woe[idx] == 0.0:
            check_array[idx] = 1        # consider converged
        else:
            check_array[idx] = doe[idx] <= tol_oe[idx]

    if np.sum(check_array)==len(tol_oe):
        return True
    else:
        return False


@njit
def check_convergence_mee(oe, oeT, woe, tol_oe):
    """Check convergence between oe and oeT for MEE or MEE-a"""
    check_array = np.zeros(len(tol_oe),)
    doe = np.abs(oeT[0:5] - oe[0:5])
    for idx in range(len(tol_oe)):
        if woe[idx] == 0.0:
            check_array[idx] = 1        # consider converged
        else:
            check_array[idx] = doe[idx] <= tol_oe[idx]

    if np.sum(check_array)==len(tol_oe):
        return True
    else:
        return False


@njit
def elements_safety(oe, oe_min, oe_max):
    """Ensure osculating elements stays within bounds
    
    Args:
        oe (np.array): current osculating elements
        oe_min (np.array): minimum values for each osculating element
        oe_max (np.array): maximum values for each osculating element

    Returns:
        (np.array): "cleaned" osculating elements
    """
    oe_clean = oe
    for idx in range(5):
        oe_clean[idx] = min(max(oe_min[idx], oe[idx]), oe_max[idx])
    return oe_clean