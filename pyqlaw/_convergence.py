"""
Check convergence
"""

import numpy as np


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
