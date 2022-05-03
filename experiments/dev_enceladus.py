"""
Dev for dimensions with Enceladus
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


MU_SATURN =  3.793120749865224E+07
MU_ENCELADUS = 7.211292085479989E+00
SMA_ENCELADUS = 238.02e3
R_ENCELADUS = 250.0  # KM
V_ENCELADUS = np.sqrt(MU_ENCELADUS/R_ENCELADUS)

r_SOI = SMA_ENCELADUS * (MU_ENCELADUS/MU_SATURN)**(2/5)
print(f"Surface radius:   {R_ENCELADUS:f} [km/s]")
print(f"Surface velocity: {V_ENCELADUS:f} [km/s]")
print(f"SOI: {r_SOI:f} [km] ({r_SOI/R_ENCELADUS} * Enceladus radii)")

# set non-dimensional parameters
lstar = R_ENCELADUS
vstar = V_ENCELADUS
tstar = lstar/vstar
mstar = 500.0   # kg
g0 = 9.80665

print(tstar/86400)

def run():
    tstart = time.time()

    # construct problem
    prob = pyqlaw.QLaw(
        integrator="rkf45", 
        elements_type="mee_with_a",
        verbosity=2,
    )
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    r0 = 1.4
    #oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-4, 0.0, 1e-2, 1e-2, 1e-3]))
    #oeT = pyqlaw.kep2mee_with_a(np.array([1.8, 0.2, np.pi/6, 1e-2, 1e-2, 1e-3]))
    oe0 = pyqlaw.kep2mee_with_a(np.array([r0, 1e-4, np.pi/2, 0.3, 0.4, 0.1]))
    oeT = pyqlaw.kep2mee_with_a(np.array([r0, 1e-4, np.pi/2, 0.35, 0.4, 0.1]))
    print(f"oe0: {oe0}")
    print(f"oeT: {oeT}")
    
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax_si = 15e-3  # N
    isp_si = 2500.0  # sec
    mdot_si = tmax_si/(isp_si*g0)

    tmax = tmax_si * (1/mstar)*(tstar**2/(1e3*lstar))
    mdot = mdot_si *(tstar/mstar)
    print(f"tmax: {tmax:1.4e}, mdot: {mdot:1.4e}")
    
    tf_max = 200*86400/tstar  # day -> sec -> canonical
    t_step = 0.1
    print(f"tf_max: {tf_max}")
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
    figs = run()
    plt.show()
    print("Done!")