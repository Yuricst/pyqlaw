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
        use_sundman = True,
    )
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    # KEP0 = [1,0.5,np.deg2rad(20),1e-2,1e-2,0]
    # KEPF = [1,1e-2,np.deg2rad(5),1e-3,1e-3,0]
    # oe0 = np.array(KEP0)  #pyqlaw.kep2mee_with_a(np.array(KEP0))
    # oeT = np.array(KEPF)  #pyqlaw.kep2mee_with_a(np.array(KEPF))
    oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-4, 0.0, 1e-2, 1e-2, 1e-3]))
    oeT = pyqlaw.kep2mee_with_a(np.array([1.8, 0.2, np.pi/6, 1e-2, 1e-2, 1e-3]))
    print(f"oe0: {oe0}")
    print(f"oeT: {oeT}")
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]

    # spacecraft parameters
    mass0 = 1.0
    tmax = 0.0149 * 1
    mdot = 0.0031 * 1
    tf_max = 10000.0
    t_step = np.deg2rad(5)

    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()
    
    # solve
    prob.solve(eta_a=0.1, eta_r=0.2)
    prob.pretty_results()
    tend = time.time()
    print(f"Simulation took {tend-tstart:4.4f} seconds")

    # plot
    fig1, ax1 = prob.plot_elements_history(to_keplerian=False)
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    fig3, ax3 = prob.plot_controls()
    fig4, ax4 = prob.plot_efficiency()
    fig5, ax5 = prob.plot_Q()
    #fig6, ax6 = prob.plot_battery_history()

    # export state history as initial guess for ICLOCS2
    #prob.save_to_dict('initial_guess_GTO2GEO.json')
    return fig1, fig2, fig3, fig4, fig5


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")