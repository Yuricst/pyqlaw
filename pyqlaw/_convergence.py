"""
Check convergence
"""

import copy
import numpy as np
from numba import njit


@njit
def check_convergence(oe, oeT, woe, tol_oe):
    """Check convergence between oe and oeT"""
    check_array = np.zeros(5,) # np.array([1 if el == 0.0 else 0 for el in woe])
    doe = np.abs(oeT[0:5] - oe[0:5])  # FIXME is this ok? or should we be careful for angles?
    # check for each element
    for idx in range(5):
        if woe[idx] > 0.0:
            if doe[idx] < tol_oe[idx]:
                check_array[idx] = 1
        # if we don't care about the idx^th element, we just say it is converged
        else:
            check_array[idx] = 1

    if np.sum(check_array)==5:
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