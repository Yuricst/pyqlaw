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

    # initial and final elements: [a,e,i,RAAN,omega,ta]
    LU = 42164.0
    GM_EARTH = 398600.44
    VU = np.sqrt(GM_EARTH/LU)
    TU = LU/VU
    
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
    t_step = np.deg2rad(5)

    # battery levels
    battery_initial = 3000*3600/TU            # Wh --> Ws --> W.TU
    battery_capacity = (500,battery_initial)
    charge_rate = 1500
    discharge_rate = 500
    battery_charge_discharge_rate = (charge_rate, discharge_rate)

    # construct problem
    prob = pyqlaw.QLaw(
        integrator="rk4", 
        elements_type="mee_with_a",
        verbosity=2,
        print_frequency=500,
        use_sundman = True,
        battery_initial = battery_initial,
        battery_capacity = battery_capacity,
        battery_charge_discharge_rate = battery_charge_discharge_rate,
    )

    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()

    # solve
    tstart_solve = time.time()
    prob.solve(eta_a=0.0, eta_r=0.0)
    prob.pretty_results()
    tend = time.time()
    print(f"Simulation took {tend-tstart:4.4f} seconds")
    print(f"prob.solve took {tend-tstart_solve:4.4f} seconds")

    # plot
    fig1, ax1 = prob.plot_elements_history(to_keplerian=True, TU=TU/86400, time_unit_name="day")
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    fig3, ax3 = prob.plot_controls()
    fig4, ax4 = prob.plot_battery_history(TU=TU/86400, BU=TU/3600, time_unit_name="day")
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")