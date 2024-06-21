"""
Run with: `python -m kernprof -lvr profile_iterate.py`
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import time 

import sys
sys.path.append("../")
import pyqlaw

from line_profiler import profile


def step_no_profile(prob):
    # efficiency thresholds
    prob.eta_a = 0.2
    prob.eta_r = 0.1

    # initialize values for propagation
    t_iter = 0.0
    oe_iter = prob.oe0
    mass_iter = prob.mass0

    # initialize storage
    prob.times = [t_iter,]
    prob.states = [oe_iter,]
    prob.masses = [mass_iter,]
    prob.controls = []
    n_relaxed_cleared = 0
    n_nan_angles = 0

    # place-holder for handling duty cycle
    duty = True
    t_last_ON  = 0.0
    t_last_OFF = 0.0

    # ensure numerical stabilty
    if prob.elements_type=="keplerian":
        oe_iter = elements_safety(oe_iter, prob.oe_min, prob.oe_max)

    # if using Sundman transformation, choose new time-step
    if prob.use_sundman:
        # compute eccentric anomaly
        if prob.elements_type=="keplerian":
            ecc_iter = oe_iter[1]
            E0 = pyqlaw.ta2ea(oe_iter[5], oe_iter[1])
            period = 2*np.pi*np.sqrt(oe_iter[0]**3/prob.mu)
        elif prob.elements_type=="mee_with_a":
            ecc_iter = np.sqrt(oe_iter[1]**2 + oe_iter[2]**2)
            E0 = pyqlaw.mee2ea(oe_iter)
            period = 2*np.pi*np.sqrt(oe_iter[0]**3/prob.mu)
        E1 = E0 + prob.t_step

        # compute mean anomaly
        M0 = E0 - ecc_iter*np.sin(E0)
        M1 = E1 - ecc_iter*np.sin(E1)
        if M1 > M0:
            t_step_local = (M1/(2*np.pi) - M0/(2*np.pi)) * period
        else:
            t_step_local = (M1/(2*np.pi) + 1 - M0/(2*np.pi)) * period
    else:
        t_step_local = prob.t_step

    # compute instantaneous acceleration magnitude due to thrust
    accel_thrust = np.sign(t_step_local) * prob.tmax/mass_iter

    # evaluate duty cycle
    if ((t_iter - t_last_ON) > prob.duty_cycle[0]) and (duty is True):
        duty = False            # turn off duty cycle
        t_last_OFF = t_iter     # latest time when we turn off
    
    if ((t_iter - t_last_OFF) > prob.duty_cycle[1]) and (duty is False):
        duty = True             # turn on duty cycle
        t_last_ON = t_iter      # latest time when we turn on

    if duty:
        # evaluate Lyapunov function
        alpha, beta, _, psi = pyqlaw.lyapunov_control_angles(
            fun_lyapunov_control=prob.lyap_fun,
            mu=prob.mu, 
            f=accel_thrust, 
            oe=oe_iter, 
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

        # ensure angles are not nan and otherwise compute thrust vector
        throttle = 1   # initialize
        if np.isnan(alpha) == True or np.isnan(beta) == True:
            alpha, beta = 0.0, 0.0
            throttle = 0  # turn off
            u = np.array([0.0,0.0,0.0])
            n_nan_angles += 1
            if n_nan_angles > prob.nan_angles_threshold:
                if prob.verbosity > 0:
                    print("Breaking as angles are nan")
                prob.exitcode = -3
                #break
        else:
            u = accel_thrust*np.array([
                np.cos(beta)*np.sin(alpha),
                np.cos(beta)*np.cos(alpha),
                np.sin(beta),
            ])

            # check effectivity to decide whether to thrust or coast
            if prob.eta_r > 0 or prob.eta_a > 0:
                qdot_current = prob.dqdt_fun(
                    prob.mu, 
                    accel_thrust, 
                    oe_iter, 
                    prob.oeT, 
                    prob.rpmin, prob.m_petro, prob.n_petro, 
                    prob.r_petro, prob.b_petro, prob.k_petro, 
                    prob.wp, prob.woe
                )
                qdot_min, qdot_max = prob.evaluate_osculating_qdot(
                    oe_iter, accel_thrust
                )
                val_eta_a = qdot_current/qdot_min
                val_eta_r = (qdot_current - qdot_max)/(qdot_min - qdot_max)
                # turn thrust off if below threshold
                if val_eta_a < prob.eta_a or val_eta_r < prob.eta_r:
                    throttle = 0  # turn off
                    u = np.zeros((3,))
    else:
        u = np.zeros((3,))
        throttle = 0  # turn off
        # evaluate Lyapunov function just for psi (FIXME)
        _, _, _, psi = pyqlaw.lyapunov_control_angles(
            fun_lyapunov_control=prob.lyap_fun,
            mu=prob.mu, 
            f=accel_thrust, 
            oe=oe_iter, 
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
        #print(f"throttle = {throttle}")

    # ODE parameters
    ode_params = (prob.mu, u, psi[0], psi[1], psi[2])
    if prob.integrator == "rk4":
        oe_next = pyqlaw.rk4(
            prob.eom, 
            t_iter,
            t_step_local,
            oe_iter,
            ode_params,
        )
        t_iter += t_step_local  # update time
        if throttle == 1:
            mass_iter -= prob.mdot*t_step_local  # update mass
        oe_iter = oe_next

    elif prob.integrator == "rkf45":
        oe_next, h_next = pyqlaw.rkf45(
            prob.eom, 
            t_iter,
            t_step_local,
            oe_iter,
            ode_params,
            prob.ode_tol,
        )
        t_iter += t_step_local  # update time
        if throttle == 1:
            mass_iter -= prob.mdot*t_step_local  # update mass
        oe_iter = oe_next
        #print(f"h_nect: {h_next}")
        t_step_local = max(prob.step_min, min(prob.step_max,h_next))
    else:
        raise ValueError("integrator name invalid!")
        
    # check convergence
    if pyqlaw.check_convergence(oe_next, prob.oeT, prob.woe, prob.tol_oe) == True:
        prob.exitcode = 1
        prob.converge = True
        #break

    # check relaxed condition
    if pyqlaw.check_convergence(oe_next, prob.oeT, prob.woe, prob.tol_oe_relaxed) == True:
        n_relaxed_cleared += 1
        if n_relaxed_cleared >= prob.exit_at_relaxed:
            prob.exitcode = 2
            prob.converge = True
            #break
    return



@profile
def step(prob):
    # efficiency thresholds
    prob.eta_a = 0.2
    prob.eta_r = 0.1

    # initialize values for propagation
    t_iter = 0.0
    oe_iter = prob.oe0
    mass_iter = prob.mass0

    # initialize storage
    prob.times = [t_iter,]
    prob.states = [oe_iter,]
    prob.masses = [mass_iter,]
    prob.controls = []
    n_relaxed_cleared = 0
    n_nan_angles = 0

    # place-holder for handling duty cycle
    duty = True
    t_last_ON  = 0.0
    t_last_OFF = 0.0

    # ensure numerical stabilty
    if prob.elements_type=="keplerian":
        oe_iter = elements_safety(oe_iter, prob.oe_min, prob.oe_max)

    # if using Sundman transformation, choose new time-step
    if prob.use_sundman:
        # compute eccentric anomaly
        if prob.elements_type=="keplerian":
            ecc_iter = oe_iter[1]
            E0 = pyqlaw.ta2ea(oe_iter[5], oe_iter[1])
            period = 2*np.pi*np.sqrt(oe_iter[0]**3/prob.mu)
        elif prob.elements_type=="mee_with_a":
            ecc_iter = np.sqrt(oe_iter[1]**2 + oe_iter[2]**2)
            E0 = pyqlaw.mee2ea(oe_iter)
            period = 2*np.pi*np.sqrt(oe_iter[0]**3/prob.mu)
        E1 = E0 + prob.t_step

        # compute mean anomaly
        M0 = E0 - ecc_iter*np.sin(E0)
        M1 = E1 - ecc_iter*np.sin(E1)
        if M1 > M0:
            t_step_local = (M1/(2*np.pi) - M0/(2*np.pi)) * period
        else:
            t_step_local = (M1/(2*np.pi) + 1 - M0/(2*np.pi)) * period
    else:
        t_step_local = prob.t_step

    # compute instantaneous acceleration magnitude due to thrust
    accel_thrust = np.sign(t_step_local) * prob.tmax/mass_iter

    # evaluate duty cycle
    if ((t_iter - t_last_ON) > prob.duty_cycle[0]) and (duty is True):
        duty = False            # turn off duty cycle
        t_last_OFF = t_iter     # latest time when we turn off
    
    if ((t_iter - t_last_OFF) > prob.duty_cycle[1]) and (duty is False):
        duty = True             # turn on duty cycle
        t_last_ON = t_iter      # latest time when we turn on

    if duty:
        # evaluate Lyapunov function
        alpha, beta, _, psi = pyqlaw.lyapunov_control_angles(
            fun_lyapunov_control=prob.lyap_fun,
            mu=prob.mu, 
            f=accel_thrust, 
            oe=oe_iter, 
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

        # ensure angles are not nan and otherwise compute thrust vector
        throttle = 1   # initialize
        if np.isnan(alpha) == True or np.isnan(beta) == True:
            alpha, beta = 0.0, 0.0
            throttle = 0  # turn off
            u = np.array([0.0,0.0,0.0])
            n_nan_angles += 1
            if n_nan_angles > prob.nan_angles_threshold:
                if prob.verbosity > 0:
                    print("Breaking as angles are nan")
                prob.exitcode = -3
                #break
        else:
            u = accel_thrust*np.array([
                np.cos(beta)*np.sin(alpha),
                np.cos(beta)*np.cos(alpha),
                np.sin(beta),
            ])

            # check effectivity to decide whether to thrust or coast
            if prob.eta_r > 0 or prob.eta_a > 0:
                qdot_current = prob.dqdt_fun(
                    prob.mu, 
                    accel_thrust, 
                    oe_iter, 
                    prob.oeT, 
                    prob.rpmin, prob.m_petro, prob.n_petro, 
                    prob.r_petro, prob.b_petro, prob.k_petro, 
                    prob.wp, prob.woe
                )
                qdot_min, qdot_max = prob.evaluate_osculating_qdot(
                    oe_iter, accel_thrust
                )
                val_eta_a = qdot_current/qdot_min
                val_eta_r = (qdot_current - qdot_max)/(qdot_min - qdot_max)
                # turn thrust off if below threshold
                if val_eta_a < prob.eta_a or val_eta_r < prob.eta_r:
                    throttle = 0  # turn off
                    u = np.zeros((3,))
    else:
        u = np.zeros((3,))
        throttle = 0  # turn off
        # evaluate Lyapunov function just for psi (FIXME)
        _, _, _, psi = pyqlaw.lyapunov_control_angles(
            fun_lyapunov_control=prob.lyap_fun,
            mu=prob.mu, 
            f=accel_thrust, 
            oe=oe_iter, 
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
        #print(f"throttle = {throttle}")

    # ODE parameters
    ode_params = (prob.mu, u, psi[0], psi[1], psi[2])
    if prob.integrator == "rk4":
        oe_next = pyqlaw.rk4(
            prob.eom, 
            t_iter,
            t_step_local,
            oe_iter,
            ode_params,
        )
        t_iter += t_step_local  # update time
        if throttle == 1:
            mass_iter -= prob.mdot*t_step_local  # update mass
        oe_iter = oe_next

    elif prob.integrator == "rkf45":
        oe_next, h_next = pyqlaw.rkf45(
            prob.eom, 
            t_iter,
            t_step_local,
            oe_iter,
            ode_params,
            prob.ode_tol,
        )
        t_iter += t_step_local  # update time
        if throttle == 1:
            mass_iter -= prob.mdot*t_step_local  # update mass
        oe_iter = oe_next
        #print(f"h_nect: {h_next}")
        t_step_local = max(prob.step_min, min(prob.step_max,h_next))
    else:
        raise ValueError("integrator name invalid!")
        
    # check convergence
    if pyqlaw.check_convergence(oe_next, prob.oeT, prob.woe, prob.tol_oe) == True:
        prob.exitcode = 1
        prob.converge = True
        #break

    # check relaxed condition
    if pyqlaw.check_convergence(oe_next, prob.oeT, prob.woe, prob.tol_oe_relaxed) == True:
        n_relaxed_cleared += 1
        if n_relaxed_cleared >= prob.exit_at_relaxed:
            prob.exitcode = 2
            prob.converge = True
            #break
    return


def main():
    print('start calculating')
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
    KEP0 = [sma_gto,ecc_gto,np.deg2rad(23),0,0,0]
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
        use_sundman = True,
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
    t_step = np.deg2rad(15)
    tf_max = t_step * 2

    # set problem
    prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
    step_no_profile(prob)   # run once to make sure numba func are compiled
    
    # go for a single step
    step(prob)
    return


if __name__ == '__main__':
    main()