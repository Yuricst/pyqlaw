"""
Test for constructing QLaw object
"""


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import pyqlaw


def test_object():
    # construct problem
    prob = pyqlaw.QLaw()
    # initial and final elements
    oe0 = np.array([1.0, 1e-2, 1e-4, 1e-4, 1e-4, 1e-4])
    oeT = np.array([1.1, 0.05, 0.01, 0.0, 0.0])
    woe = [1,1,0,0,0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 1e-3
    mdot = 1e-4
    tf_max = 100.0
    t_step = 0.1
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    prob.solve()

    prob.pretty_results()

    # plot
    fig1, ax1 = prob.plot_elements_history()
    fig2, ax2 = prob.plot_trajectory_2d()
    return fig1, fig2


if __name__=="__main__":
    fig1, fig2 = test_object()
    plt.show()
    print("Done!")