"""
Check convergence
"""

import copy
import numpy as np


#@njit
def check_convergence(oe, oeT, woe, tol_oe):
    """Check convergence between oe and oeT"""
    check_array = np.array([1 if el == 0.0 else 0 for el in woe])
    doe = oeT[0:5] - oe[0:5]
    # check for each element
    for idx in range(5):
        if woe[idx] > 0.0:
            if doe[idx] < tol_oe[idx]:
                check_array[idx] = 1

    if sum(check_array)==5:
        return True
    else:
        return False

#@njit
def keplerian_safety(oe, oe_min, oe_max):
    oe_clean = copy.copy(oe)
    for idx in range(5):
        oe_clean[idx] = min(max(oe_min[idx], oe[idx]), oe_max[idx])
    return oe_clean