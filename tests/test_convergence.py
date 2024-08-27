"""test for convergence checks"""

import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw


def test_check_convergence():
    oe = np.array([1.0, 0.1, 0.2, 0.3, 0.4])
    oeT = oe + 1e-6 * np.ones(5,)
    woe = np.ones(5,)
    tol_loose = 1e-3 * np.ones(5,)
    tol_tight = 1e-7 * np.ones(5,)

    assert pyqlaw.check_convergence_keplerian(oe, oeT, woe, tol_loose) == True
    assert pyqlaw.check_convergence_keplerian(oe, oeT, woe, tol_tight) == False

    assert pyqlaw.check_convergence_keplerian.py_func(oe, oeT, woe, tol_loose) == True
    assert pyqlaw.check_convergence_keplerian.py_func(oe, oeT, woe, tol_tight) == False

    assert pyqlaw.check_convergence_mee(oe, oeT, woe, tol_loose) == True
    assert pyqlaw.check_convergence_mee(oe, oeT, woe, tol_tight) == False

    assert pyqlaw.check_convergence_mee.py_func(oe, oeT, woe, tol_loose) == True
    assert pyqlaw.check_convergence_mee.py_func(oe, oeT, woe, tol_tight) == False

    oe_unsafe = np.array([1.0, 0.0, 0.2, 0.3, 0.4, 0.5])
    oe_min = np.array([0.1, 1e-3, 0.0, 0.0, 0.0, 0.0])
    oe_max = np.array([9.0, 0.9, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    oe_clean = pyqlaw.elements_safety.py_func(oe_unsafe, oe_min, oe_max)
    return

if __name__ == "__main__":
    test_check_convergence()
    print("Done!")