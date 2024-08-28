"""Test for eoms"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw


def test_u_vector_to_thrust_angles():
    thrust_vector = np.array([0.2,0.4,0.67])
    umag,alpha,beta = pyqlaw._u_to_thrust_angles.py_func(thrust_vector)
    return


def test_eom_kep():
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
    
    # evaluate eom (jit-ed)
    doe = pyqlaw.eom_kep_gauss(0.0, oe0, ode_params)
    assert all(np.abs(doe - doe_check)) <= 1e-12

    # evaluate eom (pure python)
    doe = pyqlaw.eom_kep_gauss.py_func(0.0, oe0, ode_params)
    assert all(np.abs(doe - doe_check)) <= 1e-12


def test_eom_mee_with_a():
    # construct problem
    prob = pyqlaw.QLaw(
        elements_type="mee_with_a",
        integrator="rk4"
    )#, verbosity=2)
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-4, 0.0, 1e-2, 1e-2, 1e-3]))
    oeT = pyqlaw.kep2mee_with_a(np.array([1.8, 0.2, np.pi/6, 1e-2, 1e-2, 1e-3]))
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
        0.0017459177484860002, 0.001749940431641845, -0.00018151195024652812,
        0.0002181479590836209, 4.581780682318711e-06, 1.0002000249029905])
    
    # evaluate eom (jit-ed)
    doe = pyqlaw.eom_mee_with_a_gauss(0.0, oe0, ode_params)
    #print(f"doe_check = {list(doe)}")
    assert all(np.abs(doe - doe_check)) <= 1e-12

    # evaluate eom (pure python)
    doe = pyqlaw.eom_mee_with_a_gauss.py_func(0.0, oe0, ode_params)
    assert all(np.abs(doe - doe_check)) <= 1e-12


if __name__=="__main__":
    test_u_vector_to_thrust_angles()
    test_eom_kep()
    test_eom_mee_with_a()
    print("Done!")