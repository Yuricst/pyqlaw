"""
Test for constructing QLaw object
"""


import numpy as np

import sys
sys.path.append("../")
import pyqlaw


def test_object():
    # construct problem
    prob = pyqlaw.QLaw()
    # initial and final elements
    oe0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oeT = np.array([1.1, 0.05, 0.0, 0.0, 0.0])
    woe = [1,1,0,0,0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 1e-3
    isp = 1e-3
    tf_max = 10.0
    t_step = 0.05
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, isp, tf_max, t_step, woe=woe)

    prob.pretty()
    return


if __name__=="__main__":
    test_object()
    print("Done!")