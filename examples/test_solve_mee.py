"""
Test for constructing QLaw object
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import time 

import sys
sys.path.append("../")
import pyqlaw

import faulthandler
faulthandler.enable()


def test_object():
    tstart = time.time()

    # construct problem
    prob = pyqlaw.QLaw(
        integrator="rk4", 
        elements_type="mee_with_a",
        verbosity=2,
        print_frequency=3000,
    )
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-4, 0.0, 1e-2, 1e-2, 1e-3]))
    oeT = pyqlaw.kep2mee_with_a(np.array([1.8, 0.2, np.pi/6, 1e-2, 1e-2, 1e-3]))
    #oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-4, 0.0, 0.3, 0.4, 0.1]))
    #oeT = pyqlaw.kep2mee_with_a(np.array([1.52, 0.01, 0.02, 2.3, 0.4, 0.1]))
    print(f"oe0: {oe0}")
    print(f"oeT: {oeT}")
    
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 1e-3
    mdot = 1e-3
    tf_max = 10000.0
    t_step = -1.0
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    prob.solve(eta_a=0.0, eta_r=0.0)
    prob.pretty_results()
    tend = time.time()
    print(f"Simulation took {tend-tstart:4.4f} seconds")

    # plot
    fig1, ax1 = prob.plot_elements_history(to_keplerian=True)
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    fig3, ax3 = prob.plot_controls()
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")