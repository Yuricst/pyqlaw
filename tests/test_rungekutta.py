"""Test for runge-kutta algos"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw


def test_rk4_rk45():
    # construct problem
    prob = pyqlaw.QLaw(
        elements_type="keplerian",
        integrator="rk4"
    )#, verbosity=2)
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = np.array([1.5, 0.2, 0.3, 1e-2, 1e-2, 1e-2])
    oeT = np.array([2.2, 0.3, 1.1, 0.3, 0.0])
    woe = [1.0, 1.0, 1.0, 1.0, 1.0]
    # spacecraft parameters
    mass0 = 1.0
    tmax = 1e-3
    mdot = 1e-4
    tf_max = 3000.0
    t_step = 1.0
    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    
    # query psi
    accel_thrust = tmax/mass0
    _, _, _, psi, _, _ = pyqlaw.lyapunov_control_angles(
        fun_lyapunov_control=prob.lyap_fun,
        mu=prob.mu, 
        f=accel_thrust, 
        oe=oe0, 
        oeT=prob.oeT, 
        rpmin=prob.rpmin, 
        m_petro=prob.m_petro, 
        n_petro=prob.n_petro, 
        r_petro=prob.r_petro, 
        b_petro=prob.b_petro, 
        k_petro=prob.k_petro, 
        wp=prob.wp, 
        woe=prob.woe,
    )
    u = np.array([0.1,0.4,0.2])
    u *= accel_thrust/np.linalg.norm(u)
    ode_params = (prob.mu, u, psi[0], psi[1], psi[2], np.zeros(3,))

    # known check case
    doe_check = np.array([
        0.00392952589873677, 0.00209742303058534, 
        0.0004363521324628918, 2.9535057990649755e-05, 
        -0.001241443160063603, 0.8345326718595107])
    
    h = 0.01    # step-size

    # evaluate eom (jit-ed)
    ynext_rk4 = pyqlaw.rk4(pyqlaw.eom_kep_gauss, 0.0, h, oe0, ode_params)
    ynext_rk45, h_next = pyqlaw.rkf45(pyqlaw.eom_kep_gauss, 0.0, h, oe0, ode_params)
    assert all(np.abs(ynext_rk4[0:4] - ynext_rk45[0:4]) <= 1e-6)
    assert np.abs(ynext_rk4[5] - ynext_rk45[5]) <= 1e-3

    # evaluate eom (pure python)
    ynext_rk4 = pyqlaw.rk4.py_func(pyqlaw.eom_kep_gauss, 0.0, h, oe0, ode_params)
    ynext_rk45, h_next = pyqlaw.rkf45.py_func(pyqlaw.eom_kep_gauss, 0.0, h, oe0, ode_params)
    assert all(np.abs(ynext_rk4[0:4] - ynext_rk45[0:4]) <= 1e-6)
    assert np.abs(ynext_rk4[5] - ynext_rk45[5]) <= 1e-3
    return 


if __name__=="__main__":
    test_rk4_rk45()
    print("Done!")