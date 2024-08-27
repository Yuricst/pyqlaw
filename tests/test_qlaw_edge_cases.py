"""
Edge cases handling with Qlaw
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw


def test_mass_depletion():
    # construct problem
    prob = pyqlaw.QLaw(
        elements_type="keplerian",
        integrator="rk4"
    )#, verbosity=2)
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = np.array([1.5, 0.2, 0.3, 1e-2, 1e-2, 1e-2])
    oeT = np.array([2.2, 0.3, 1.1, 0.3, 0.0])
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 0.1
    tmax = 1e-3
    mdot = 0.15
    tf_max = 3000.0
    t_step = 1.0
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    prob.solve()
    prob.pretty_results()
    assert prob.converge == False
    assert prob.exitcode == -1


# def test_angles_nan_depletion():
#     # construct problem
#     prob = pyqlaw.QLaw(
#         elements_type="keplerian",
#         integrator="rk4"
#     )
#     # initial and final elements: [a,e,i,RAAN,omega,ta]
#     oe0 = np.array([1.5, 0.2, 0.3, 1e-2, 1e-2, 1e-2])
#     oeT = np.array([2.2, 0.3, 1.1, 0.3, 0.0])
#     woe = [1.0, 1.0, 1.0, 1.0, 1.0]
#     # spacecraft parameters
#     mass0 = 1.0
#     tmax = 1e-1
#     mdot = 1e-3
#     tf_max = 3000.0
#     t_step = 1.0
#     # set problem
#     prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
#     prob.pretty()

#     # solve
#     prob.solve()
#     prob.pretty_results()
#     assert prob.converge == False
#     assert prob.exitcode == -3


if __name__=="__main__":
    test_mass_depletion()
    # test_angles_nan_depletion()
    print("Done!")