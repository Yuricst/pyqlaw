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
    
    use_keplerian = False
    if use_keplerian:
        elements_type = "keplerian"
    else:
        elements_type = "mee_with_a"

    # initial and final elements: [a,e,i,RAAN,omega,ta]
    LU = 42164.0
    GM_EARTH = 398600.44
    VU = np.sqrt(GM_EARTH/LU)
    TU = LU/VU

    rp_gto = 200 + 6378
    ra_gto = 35786 + 6378
    sma_gto = (rp_gto + ra_gto)/(2*LU)
    ecc_gto = (ra_gto - rp_gto)/(ra_gto + rp_gto)
    KEP0 = [sma_gto,ecc_gto,np.deg2rad(6),0,0,0]
    KEPF = [1,0,np.deg2rad(3),0,0,0]

    if use_keplerian:
        element_min = 1e-2
        oe0 = np.clip(np.array(KEP0), element_min, 100.0)
        oeT = np.clip(np.array(KEPF), element_min, 100.0)
        woe = [1.0, 1.0, 1.0, 1e-2, 1e-2]
    else:
        oe0 = pyqlaw.kep2mee_with_a(np.array(KEP0))
        oeT = pyqlaw.kep2mee_with_a(np.array(KEPF))
        woe = [3.0, 1.0, 1.0, 1.0, 1.0]

    # duty cycles
    duty_cycle = (0.85*86400/TU, 0.15*86400/TU)
    print(f"duty_cycle = {duty_cycle}")

    # construct problem
    prob = pyqlaw.QLaw(
        rpmin = 6578/LU,
        integrator="rk4", 
        elements_type=elements_type,
        verbosity=2,
        print_frequency=3000,
        duty_cycle = duty_cycle,
    )

    # spacecraft parameters
    MU = 1500
    tmax_si = 0.1   # 100 mN
    isp_si  = 1455   # seconds
    mdot_si = tmax_si/(isp_si*9.81)  # kg/s

    # non-dimensional quantities
    mass0 = 1.0
    tmax = tmax_si * (1/MU)*(TU**2/(1e3*LU))
    mdot = np.abs(mdot_si) *(TU/MU)
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
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=6378/LU, lw=0.1, interpolate=False)
    fig3, ax3 = prob.plot_controls()

    # export state history as initial guess for ICLOCS2

    
    print(f"oe0 = {oe0}")
    print(f"oeT = {oeT}")
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")