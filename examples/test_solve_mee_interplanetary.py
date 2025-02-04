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
    tol_oe = np.array([1e-4,1e-2,1e-2,1e-2,1e-2])
    prob = pyqlaw.QLaw(
        integrator="rk4", 
        elements_type="mee_with_a",
        verbosity=2,
        use_sundman = False,
        print_frequency = 1000,
        relaxed_tol_factor = 1,
        tol_oe = tol_oe,
    )
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-4, 0.0, 0.3, 0.4, 0.1]))
    oeT = pyqlaw.kep2mee_with_a(np.array([1.52, 0.01, 0.02, 2.3, 0.4, 0.1]))
    print(f"oe0: {oe0}")
    print(f"oeT: {oeT}")
    
    woe = [100.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 1e-2
    mdot = 1e-3
    tf_max = 35.0
    t_step = 0.01
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    prob.solve(eta_a=0.1, eta_r=0.0)
    prob.pretty_results()
    tend = time.time()
    print(f"Simulation took {tend-tstart:4.4f} seconds")

    # plot
    fig1, ax1 = prob.plot_elements_history(to_keplerian=False)
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    fig3, ax3 = prob.plot_controls()
    fig4, ax4 = prob.plot_efficiency()
    fig5, ax5 = prob.plot_Q()

    # fig, ax = plt.subplots(1,1,figsize=(6,4))
    # ax.step(prob.times[1:-1], -(np.array(prob.Qs[1:]) - np.array(prob.Qs[:-1])), marker='o', markersize=2)
    # ax.step(prob.times[1:], np.array(prob.dQdts) * (np.array(prob.times[1:]) - np.array(prob.times[:-1])),
    #         marker='o', markersize=2)
    # ax.set_yscale('log')
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")