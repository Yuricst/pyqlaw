"""
Test for constructing QLaw object
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import pyqlaw


def test_gauss_intergrate():
    # construct problem
    prob = pyqlaw.QLaw()
    # initial and final elements
    mu = 1.0
    oe0 = np.array([1.0, 1e-6, 1e-6, 0.0, 0.0, 1e-8])
    mass_iter = 1.0
    oe_iter = copy.deepcopy(oe0)
    t_iter = 0.0
    h = 1e-2

    tmax = 1.e-2
    mdot = 1.e-4

    n_steps = 6000
    oes = np.zeros((6,n_steps+1))
    oes[:,0] = oe0
    cart_coord = np.zeros((6,n_steps+1))
    cart_coord[:,0] = oe_iter
    ts = np.zeros(n_steps+1,)

    for idx in range(n_steps):
        # break clause
        if oe_iter[0] < 0.1:
            break
        if oe_iter[1] > 0.95:
            break
        htest = np.sqrt(oe_iter[0]*mu*(1-oe_iter[1]**2))
        if np.isnan(htest) == True:
            break
        if np.isnan(oe_iter[0]*mu*(1-oe_iter[1]**2))==True:
            break

        f = tmax/mass_iter
        p = (mu, f, np.pi/2, 0.0)
        oe_next = pyqlaw.rk4(pyqlaw.eom_gauss, t_iter, h, oe_iter, p)
        # store
        oes[:,idx+1] = oe_next
        cart_coord[:,idx+1] = pyqlaw.kep2sv(oe_next, mu)
        ts[idx+1] = t_iter+h
        # update
        oe_iter = oe_next
        mass_iter -= mdot
        t_iter += h

    # create plot
    fig, axs = plt.subplots(2,1,figsize=(6,8))
    axs[0].plot(cart_coord[0,:], cart_coord[1,:], c="navy")
    axs[0].set_aspect('equal')
    axs[0].set(xlabel="x", ylabel="y")

    axs[1].plot(ts, oes[0,:], label="a")
    axs[1].plot(ts, oes[1,:], label="e")
    axs[1].plot(ts, oes[2,:], label="i")
    axs[1].plot(ts, oes[5,:] % (2*np.pi), label="ta")
    axs[1].set(xlabel="t")
    axs[1].legend(loc='lower center')
    return fig, axs


if __name__=="__main__":
    fig, axs = test_gauss_intergrate()
    plt.show()
    print("Done!")