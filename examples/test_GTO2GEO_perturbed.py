"""
Test for constructing QLaw object
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import time
import spiceypy as spice

import sys
sys.path.append("../")
import pyqlaw

import os
spice.furnsh(os.path.join(os.getenv("SPICE"), "lsk", "naif0012.tls"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "spk", "de440.bsp"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "gm_de440.tpc"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "pck", "earth_200101_990825_predict.bpc"))
spice.furnsh(os.path.join(os.getenv("SPICE"), "fk", "earth_assoc_itrf93.tf"))


def test_object():
    tstart = time.time()
    # reference quantities
    MU_EARTH = spice.bodvrd("399", "GM", 1)[1][0]
    LU = 42164.0
    VU = np.sqrt(MU_EARTH / LU)
    TU = LU / VU

    # initialize object for perturbations
    et_ref = spice.str2et("2028-01-01T00:00:00")
    perturbations = pyqlaw.SpicePerturbations(
        et_ref, LU, TU,
        use_J2=False,
    )
    print(f"   perturbations.third_bodies_gms = {perturbations.third_bodies_gms}")

    # construct problem
    prob = pyqlaw.QLaw(
        integrator="rk4", 
        elements_type="mee_with_a",
        verbosity=2,
        print_frequency=3000,
        use_sundman = True,
        perturbations = perturbations,
    )
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    KEP0 = np.array([
        (6578+42164)/2/LU,
        (42164-6578)/(42164+6578),
        np.deg2rad(28.5),
        np.deg2rad(10), np.deg2rad(270), np.deg2rad(50)
    ])
    KEPF = np.array([
        42164/LU,
        1e-3,
        np.deg2rad(1e-3),
        np.deg2rad(10), np.deg2rad(30), np.deg2rad(50)
    ])
    oe0 = pyqlaw.kep2mee_with_a(np.array(KEP0))
    oeT = pyqlaw.kep2mee_with_a(np.array(KEPF))
    print(f"oe0: {oe0}")
    print(f"oeT: {oeT}")
    woe = [3.0, 1.0, 1.0, 1.0, 1.0]

    # spacecraft parameters
    mass0 = 1.0
    tmax = 0.0149
    mdot = 0.0031
    tf_max = 10000.0
    t_step = np.deg2rad(5)

    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    prob.pretty()
    
    # solve
    prob.solve(eta_a=0.2, eta_r=0.3)
    prob.pretty_results()
    tend = time.time()
    print(f"Simulation took {tend-tstart:4.4f} seconds")

    # plot
    fig1, ax1 = prob.plot_elements_history(to_keplerian=False)
    fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
    fig3, ax3 = prob.plot_controls()
    fig4, ax4 = prob.plot_efficiency()
    fig5, ax5 = prob.plot_Q()

    # export state history as initial guess for ICLOCS2
    #prob.save_to_dict('initial_guess_GTO2GEO.json')
    return fig1, fig2, fig3


if __name__=="__main__":
    figs = test_object()
    plt.show()
    print("Done!")