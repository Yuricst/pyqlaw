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

    rp_gto = 200 + 6378
    ra_gto = 35786 + 6378
    sma_gto = (rp_gto + ra_gto)/(2*LU)
    ecc_gto = (ra_gto - rp_gto)/(ra_gto + rp_gto)
    KEP0 = [sma_gto,ecc_gto,np.deg2rad(23),0,0,0]
    KEPF = [1,0,np.deg2rad(3),0,0,0]
    oe0 = pyqlaw.kep2mee_with_a(np.array(KEP0))
    oeT = pyqlaw.kep2mee_with_a(np.array(KEPF))
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]

    # spacecraft parameters
    MU = 1500
    tmax_si = 10 * 0.1   # 100 mN
    isp_si  = 1455   # seconds
    mdot_si = tmax_si/(isp_si*9.81)  # kg/s

    # non-dimensional quantities
    mass0 = 1.0
    tmax = tmax_si * (1/MU)*(TU**2/(1e3*LU))
    mdot = np.abs(mdot_si) *(TU/MU)
    tf_max = 10000.0
    t_step = np.deg2rad(15)
    
    # battery levels
    battery_initial = 3000*3600/TU            # Wh --> Ws --> W.TU
    battery_dod = 500*3600/TU
    battery_capacity = (battery_dod,battery_initial)
    charge_rate = 1500
    discharge_rate = 500
    battery_charge_discharge_rate = (charge_rate, discharge_rate)
    require_full_recharge = True

    # construct problem
    prob = pyqlaw.QLaw(
        integrator="rk4", 
        elements_type="mee_with_a",
        verbosity=2,
        print_frequency=500,
        use_sundman = True,
    )

    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step,
        battery_initial = battery_initial,
        battery_capacity = battery_capacity,
        battery_charge_discharge_rate = battery_charge_discharge_rate,
        require_full_recharge = require_full_recharge,
        woe = woe)
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
    fig4, ax4 = prob.plot_battery_history(TU=TU/86400, BU=TU/3600,
        time_unit_name="day", battery_unit_name="Wh")
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")