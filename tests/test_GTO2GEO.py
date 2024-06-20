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
    KEP0 = [0.578004933118300,0.730089040252759,0.401425727958696,1.745329251994330,4.712388980384690,0.401425727958696]
    KEPF = [1,0,0,1.780235837034216,0.593411945678072,0.087266462599716]
    oe0 = pyqlaw.kep2mee_with_a(np.array(KEP0))
    oeT = pyqlaw.kep2mee_with_a(np.array(KEPF))
    print(f"oe0: {oe0}")
    print(f"oeT: {oeT}")
    
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 0.0149
    mdot = 0.0031
    tf_max = 10000.0
    t_step = 0.05

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

    # export state history as initial guess for ICLOCS2
    prob.save_to_dict('initial_guess_GTO2GEO.json')
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")