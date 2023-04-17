"""
Test for constructing QLaw object
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import pyqlaw


def test_object():
    # construct problem
    prob = pyqlaw.QLaw(
        elements_type="mee_with_a",
        integrator="rk4",
        verbosity=2,
        print_frequency=3000,
    )#, verbosity=2)
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = pyqlaw.kep2mee_with_a(np.array([ 1.0975227343994982,
        0.001,
        1.0471975511965976,
        1.7453292519943295,
        2.4,
        0.5]))
    oeT = pyqlaw.kep2mee_with_a(np.array([1.136719974913766,
        0.0011,
        1.2217304763960306,
        1.7802358370342162,
        2.2,
        0.3]))
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 8.16434301979195e-5
    mdot = 2.6317150035742462e-5
    tf_max = 10000.0
    t_step = 1.0
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    prob.solve()
    prob.pretty_results()

    # plot
    fig1, ax1 = prob.plot_elements_history()
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    return fig1, fig2


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")