"""
Test for constructing QLaw object
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw


def test_solve_kep(close_figures=True):
    use_sundman_options = [False,True]
    for use_sundman in use_sundman_options:
        # construct problem
        prob = pyqlaw.QLaw(
            elements_type="keplerian",
            integrator="rk4",
            use_sundman = use_sundman,
        )#, verbosity=2)
        # initial and final elements: [a,e,i,RAAN,omega,ta]
        oe0 = np.array([1.5, 0.2, 0.3, 1e-2, 1e-2, 1e-2])
        oeT = np.array([2.2, 0.3, 1.1, 0.3, 0.0])
        woe = [1.0, 1.0, 1.0, 1.0, 1.0]
        # spacecraft parameters
        mass0 = 1.0
        tmax = 1e-3
        mdot = 1e-4
        tf_max = 3000.0
        t_step = 1.0
        # set problem
        prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
        prob.pretty()

        # solve
        prob.solve(eta_a=0.2)
        prob.pretty_results()

        # plot
        fig1, ax1 = prob.plot_elements_history(to_keplerian=False)
        fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
        fig3, ax3 = prob.plot_controls()
        fig4, ax4 = prob.plot_efficiency()
        fig5, ax5 = prob.plot_Q()
        fig6, ax6 = prob.plot_trajectory_2d()
        if close_figures:
            plt.close('all')
    assert prob.converge == True


if __name__=="__main__":
    test_solve_kep(close_figures=False)
    plt.show()
    print("Done!")