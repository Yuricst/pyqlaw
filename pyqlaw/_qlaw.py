"""
main functions for Q-Law transfers
"""

import copy
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ._symbolic import symbolic_qlaw_keplerian, symbolic_qlaw_mee_with_a
from ._lyapunov import lyapunov_control_angles
from ._eom import eom_kep_gauss, eom_mee_gauss, eom_mee_with_a_gauss
from ._integrate import rk4, rkf45
from ._convergence import check_convergence, elements_safety
from ._elements import (
    kep2sv,
    mee_with_a2sv,
    get_orbit_coordinates, 
    mee_with_a2mee,
    mee2mee_with_a,
    mee_with_a2kep,
    ta2ea,
    mee2ea
)
from ._plot_helper import plot_sphere_wireframe, set_equal_axis

import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class QLaw:
    """Object for Q-law based transfer problem.
    The overall procedure for using this class is:
    1. Create object via `prob = QLaw()`
    2. Set problem parameters via `prob.set_problem()`
    3. solve problem via `prob.solve()`

    Exitcodes:
    0 : initial value (problem not yet attempted)
    1 : solved within tolerance
    2 : solved within relaxed tolerance
    -1 : mass is below threshold
    -2 : target elements could not be reached within allocated time
    -3 : thrust angles from feedback control law is nan

    Args:
        mu (float): gravitational parameter, default is 1.0
        rpmin (float): minimum periapsis
        k_petro (float): scalar factor k on minimum periapsis
        m_petro (float): scalar factor m to prevent non-convergence, default is 3.0
        n_petro (float): scalar factor n to prevent non-convergence, default is 4.0
        r_petro (float): scalar factor r to prevent non-convergence, default is 2.0
        b_petro (float): scalar factor b for dot(omega)_xx, default is 0.01
        wp (float): penalty scalar on minimum periapsis, default is 1.0
        elements_type (str): type of elements to define Q-law
        integrator (str): "rk4" or "rkf45"
        verbosity (int): verbosity level for Q-law
        anomaly_grid_size (int): number of evaluation point along orbit for Q-law effectivity
        disable_tqdm (bool): whether to disable progress bar
        tol_oe (np.array or None): tolerance on 5 elements targeted
        oe_min (np.array): minimum values of elements for safe-guarding
        oe_max (np.array): minimum values of elements for safe-guarding
        nan_angles_threshold (int): number of times to ignore `nan` thrust angles
        print_frequency (int): if verbosity >= 2, prints at this frequency
        use_sundman (bool): whether to use Sundman transformation for propagation

    Attributes:
        print_frequency (int): if verbosity >= 2, prints at this frequency
    """
    def __init__(
        self, 
        mu=1.0,
        rpmin=0.1, 
        k_petro=1.0, 
        m_petro=3.0, 
        n_petro=4.0, 
        r_petro=2.0, 
        b_petro=0.01,
        wp=1.0,
        elements_type="mee_with_a",
        integrator="rk4",
        verbosity=1,
        anomaly_grid_size=5,
        tol_oe=None,
        oe_min=None,
        oe_max=None,
        nan_angles_threshold=10,
        print_frequency=200,
        use_sundman=False,
    ):
        """Construct QLaw object"""
        # assertions on inputs
        assert elements_type in ["keplerian", "mee_with_a"], "elements_type must be either 'keplerian' or 'mee_with_a'"
        assert integrator in ["rk4", "rkf45"], "integrator must be either 'rk4' or 'rkf45'"

        # dynamics
        self.mu = mu

        # Q-law parameters
        self.rpmin = rpmin
        self.k_petro = k_petro
        self.m_petro = m_petro
        self.n_petro = n_petro 
        self.r_petro = r_petro
        self.b_petro = b_petro
        self.wp = wp
        self.elements_type = elements_type
        self.integrator = integrator
        self.anomaly_grid_size = anomaly_grid_size

        # settings
        self.verbosity = verbosity

        # tolerance for convergence
        if tol_oe is None:
            self.tol_oe = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        else:
            assert len(tol_oe)==5, "tol_oe must have 5 components"
            self.tol_oe = np.array(tol_oe)
        self.tol_oe_relaxed = 10*self.tol_oe  # relaxed tolerance
        self.exit_at_relaxed = 25

        # orbital elements bounds
        if oe_min is None:
            if self.elements_type == "keplerian":
                self.oe_min = np.array([0.05,1e-4,1e-4,1e-4,1e-4,1e-8])
            elif self.elements_type == "mee_with_a":
                self.oe_min = np.array([0.05, -np.inf, -np.inf, -np.inf, -np.inf])
        else:
            self.oe_min = np.array(oe_min)

        # bounds on elements
        if oe_max is None:
            if self.elements_type == "keplerian":
                self.oe_max = np.array([1e2,0.95,np.pi,np.inf,np.inf,np.inf])
            elif self.elements_type == "mee_with_a":
                self.oe_max = np.array([1e2, 10.0, 10.0, np.inf, np.inf, np.inf])
        else:
            self.oe_max = np.array(oe_max)

        # number of times to accept nan angles
        self.nan_angles_threshold = nan_angles_threshold

        # construct element names, depending on setting
        if self.elements_type == "keplerian":
            self.element_names = ["a", "e", "i", "raan", "omega", "ta"]
            self.eom = eom_kep_gauss
            fun_lyapunov_control, _, _, fun_eval_dqdt = symbolic_qlaw_keplerian()
            self.lyap_fun = fun_lyapunov_control
            self.dqdt_fun = fun_eval_dqdt

        elif self.elements_type == "mee_with_a":
            self.element_names = ["a", "f", "g", "h", "k", "L"]
            self.eom = eom_mee_with_a_gauss
            fun_lyapunov_control, _, _, fun_eval_dqdt = symbolic_qlaw_mee_with_a()
            self.lyap_fun = fun_lyapunov_control
            self.dqdt_fun = fun_eval_dqdt

        # max and min step sizes used with adaptive step integrators
        self.step_min = 1e-4
        self.step_max = 2.0
        self.ode_tol = 1.e-5
        self.use_sundman = use_sundman

        # # duty cycles
        # self.duty_cycle = duty_cycle

        # # battery parameters
        # self.battery_initial = battery_initial
        # self.battery_capacity = battery_capacity
        # self.battery_charge_discharge_rate = battery_charge_discharge_rate
        # self.require_full_recharge = require_full_recharge

        # print frequency
        self.print_frequency = print_frequency  # print at (# of iteration) % self.print_frequency

        # checks
        self.ready = False
        self.converge = False
        self.exitcode = 0
        return
    

    def set_problem(
        self, 
        oe0, 
        oeT, 
        mass0, 
        tmax, 
        mdot, 
        tf_max=100000.0, 
        t_step=0.1,
        mass_min=0.1,
        woe=None,
        duty_cycle=(1e16,0.0),
        battery_initial=1.0,
        battery_capacity=(0.2,1.0),
        battery_charge_discharge_rate=(0.2,0.01),
        require_full_recharge=False,
    ):
        """Set transfer problem
        
        Args:
            oe0 (np.array): initial state, in Keplerian elements (6 components)
            oeT (np.array): final target Keplerian elements (5 components)
            mass0 (float): initial mass
            tmax (float): max thrust
            mdot (float): mass-flow rate
            tf_max (float): max time allocated to transfer
            t_step (float): initial time-step size to be used for integration
            mass_min (float): minimum mass
            woe (np.array): weight on each osculating element
            duty_cycle (tuple): ON and OFF times for duty cycle, default is (1e16, 0.0)
            use_sundman (bool): whether to use Sundman transformation for propagation
            battery_capacity (tuple): min and max battery capacity (min should be DOD)
            battery_charge_discharge_rate (tuple): charge and discharge rate (both positive values)
            require_full_recharge (bool): whether full recharge is required once DOD is reached
        """
        assert len(oe0)==6, "oe6 must have 6 components"
        assert mass_min >= 1e-2, "mass should be above 0.01 to avoid numerical difficulties"
        #assert len(oeT)==5, "oeT must have 5 components"
        # weight parameters
        if woe is None:
            self.woe = np.array([1.0,1.0,1.0,1.0,1.0])
        else:
            assert len(woe)==5, "woe must have 5 components"
            self.woe = np.array(woe)

        # time parameters
        self.tf_max = tf_max
        self.t_step = t_step
        # spacecraft parameters
        self.oe0   = oe0
        if len(oeT) == 6:
            self.oeT = oeT[0:5]
        else:
            self.oeT = oeT
        self.mass0 = mass0
        self.tmax  = tmax
        self.mdot  = mdot
        self.mass_min = mass_min
        self.ready = True               # toggle for solving

        # duty cycles
        self.duty_cycle = duty_cycle

        # battery parameters
        self.battery_initial = battery_initial
        self.battery_capacity = battery_capacity
        self.battery_charge_discharge_rate = battery_charge_discharge_rate
        self.require_full_recharge = require_full_recharge
        return


    def solve(self, eta_a=0.0, eta_r=0.0):
        """Propagate and solve control problem
        
        Args:
            eta_a (float): min absolute effectivity to thrust, `0.0 <= eta_a <= 1.0`
            eta_r (float): relative effectivity, `0.0 <= eta_r <= 1.0`
        """
        assert self.ready == True, "Please first call `set_problem()`"

        # efficiency thresholds
        self.eta_a = eta_a
        self.eta_r = eta_r

        # initialize values for propagation
        t_iter = 0.0
        oe_iter = self.oe0
        mass_iter = self.mass0
        battery_iter = self.battery_initial

        # initialize storage
        self.times = [t_iter,]
        self.states = [oe_iter,]
        self.masses = [mass_iter,]
        self.controls = []
        self.etas = []
        self.battery = [self.battery_initial,]
        n_relaxed_cleared = 0
        n_nan_angles = 0

        if self.verbosity >= 2:
            self.disable_tqdm = True
            header = " iter   |  time      |  del1       |  del2       |  del3       |  del4       |  del5       |  el6        |"

        # place-holder for handling duty cycle and battery level
        duty = True
        t_last_ON  = 0.0
        t_last_OFF = 0.0
        charging = False

        # iterate until nmax
        idx = 0
        while abs(self.times[-1]) < abs(self.tf_max):  #for idx in tqdm(range(nmax), disable=self.disable_tqdm, desc="qlaw"):
            # ensure numerical stabilty
            if self.elements_type=="keplerian":
                oe_iter = elements_safety(oe_iter, self.oe_min, self.oe_max)

            # if using Sundman transformation, choose new time-step
            if self.use_sundman:
                # compute eccentric anomaly
                if self.elements_type=="keplerian":
                    ecc_iter = oe_iter[1]
                    E0 = ta2ea(oe_iter[5], oe_iter[1])
                    period = 2*np.pi*np.sqrt(oe_iter[0]**3/self.mu)
                elif self.elements_type=="mee_with_a":
                    ecc_iter = np.sqrt(oe_iter[1]**2 + oe_iter[2]**2)
                    E0 = mee2ea(oe_iter)
                    period = 2*np.pi*np.sqrt(oe_iter[0]**3/self.mu)
                E1 = E0 + self.t_step

                # compute mean anomaly
                M0 = E0 - ecc_iter*np.sin(E0)
                M1 = E1 - ecc_iter*np.sin(E1)
                if M1 > M0:
                    t_step_local = (M1/(2*np.pi) - M0/(2*np.pi)) * period
                else:
                    t_step_local = (M1/(2*np.pi) + 1 - M0/(2*np.pi)) * period
            else:
                t_step_local = self.t_step

            # compute instantaneous acceleration magnitude due to thrust
            accel_thrust = np.sign(t_step_local) * self.tmax/mass_iter

            # evaluate duty cycle
            if ((t_iter - t_last_ON) > self.duty_cycle[0]) and (duty is True):
                duty = False            # turn off duty cycle
                t_last_OFF = t_iter     # latest time when we turn off
            
            if ((t_iter - t_last_OFF) > self.duty_cycle[1]) and (duty is False):
                duty = True             # turn on duty cycle
                t_last_ON = t_iter      # latest time when we turn on

            # check battery state
            if battery_iter - self.battery_charge_discharge_rate[1]*t_step_local < self.battery_capacity[0]:
                duty = False                # turn off
                t_last_OFF = t_iter         # latest time when we turn off
                charging = True

            # check if battery is full
            if (self.require_full_recharge is True) and (charging is True) and\
                (battery_iter < self.battery_capacity[1]):
                duty = False                # turn off

            # initialize efficiency parameters for storage
            val_eta_a, val_eta_r = np.nan, np.nan
            if duty:
                # evaluate Lyapunov function
                alpha, beta, _, psi = lyapunov_control_angles(
                    fun_lyapunov_control=self.lyap_fun,
                    mu=self.mu, 
                    f=accel_thrust, 
                    oe=oe_iter, 
                    oeT=self.oeT, 
                    rpmin=self.rpmin, 
                    m_petro=self.m_petro, 
                    n_petro=self.n_petro, 
                    r_petro=self.r_petro, 
                    b_petro=self.b_petro, 
                    k_petro=self.k_petro, 
                    wp=self.wp, 
                    woe=self.woe,
                )

                # ensure angles are not nan and otherwise compute thrust vector
                throttle = 1   # initialize
                if np.isnan(alpha) == True or np.isnan(beta) == True:
                    alpha, beta = 0.0, 0.0
                    throttle = 0  # turn off
                    u = np.array([0.0,0.0,0.0])
                    n_nan_angles += 1
                    if n_nan_angles > self.nan_angles_threshold:
                        if self.verbosity > 0:
                            print("Breaking as angles are nan")
                        self.exitcode = -3
                        break
                else:
                    u = accel_thrust*np.array([
                        np.cos(beta)*np.sin(alpha),
                        np.cos(beta)*np.cos(alpha),
                        np.sin(beta),
                    ])

                    # check effectivity to decide whether to thrust or coast
                    if self.eta_r > 0 or self.eta_a > 0:
                        qdot_current = self.dqdt_fun(
                            self.mu, 
                            accel_thrust, 
                            oe_iter, 
                            self.oeT, 
                            self.rpmin, self.m_petro, self.n_petro, 
                            self.r_petro, self.b_petro, self.k_petro, 
                            self.wp, self.woe
                        )
                        qdot_min, qdot_max = self.evaluate_osculating_qdot(
                            oe_iter, accel_thrust
                        )
                        val_eta_a = qdot_current/qdot_min
                        val_eta_r = (qdot_current - qdot_max)/(qdot_min - qdot_max)
                        # turn thrust off if below threshold
                        if val_eta_a < self.eta_a or val_eta_r < self.eta_r:
                            throttle = 0  # turn off
                            u = np.zeros((3,))
            else:
                u = np.zeros((3,))
                throttle = 0  # turn off
                # evaluate Lyapunov function just for psi (FIXME)
                _, _, _, psi = lyapunov_control_angles(
                    fun_lyapunov_control=self.lyap_fun,
                    mu=self.mu, 
                    f=accel_thrust, 
                    oe=oe_iter, 
                    oeT=self.oeT, 
                    rpmin=self.rpmin, 
                    m_petro=self.m_petro, 
                    n_petro=self.n_petro, 
                    r_petro=self.r_petro, 
                    b_petro=self.b_petro, 
                    k_petro=self.k_petro, 
                    wp=self.wp, 
                    woe=self.woe,
                )
                #print(f"throttle = {throttle}")

            # ODE parameters
            ode_params = (self.mu, u, psi[0], psi[1], psi[2])  # control fixed in RTN for step
            if self.integrator == "rk4":
                oe_next = rk4(
                    self.eom, 
                    t_iter,
                    t_step_local,
                    oe_iter,
                    ode_params,
                )
                t_iter += t_step_local  # update time
                if throttle == 1:
                    mass_iter -= self.mdot*t_step_local  # update mass
                oe_iter = oe_next

            elif self.integrator == "rkf45":
                oe_next, h_next = rkf45(
                    self.eom, 
                    t_iter,
                    t_step_local,
                    oe_iter,
                    ode_params,
                    self.ode_tol,
                )
                t_iter += t_step_local  # update time
                if throttle == 1:
                    mass_iter -= self.mdot*t_step_local  # update mass
                oe_iter = oe_next
                #print(f"h_nect: {h_next}")
                t_step_local = max(self.step_min, min(self.step_max,h_next))
            else:
                raise ValueError("integrator name invalid!")
                
            # check convergence
            if check_convergence(oe_next, self.oeT, self.woe, self.tol_oe) == True:
                self.exitcode = 1
                self.converge = True
                break

            # check relaxed condition
            if check_convergence(oe_next, self.oeT, self.woe, self.tol_oe_relaxed) == True:
                n_relaxed_cleared += 1
                if n_relaxed_cleared >= self.exit_at_relaxed:
                    self.exitcode = 2
                    self.converge = True
                    break
            
            # print progress
            if self.verbosity >= 2 and np.mod(idx,self.print_frequency)==0:
                if np.mod(idx, 20*self.print_frequency) == 0:
                    print("\n" + header)
                #t_fraction = t_iter/self.tf_max
                print(f" {idx:6.0f} | {t_iter: 1.3e} | {oe_next[0]-self.oeT[0]: 1.4e} | {oe_next[1]-self.oeT[1]: 1.4e} | {oe_next[2]-self.oeT[2]: 1.4e} | {oe_next[3]-self.oeT[3]: 1.4e} | {oe_next[4]-self.oeT[4]: 1.4e} | {oe_next[5]: 1.4e} |")

            # check if mass is below threshold
            if mass_iter <= self.mass_min:
                if self.verbosity > 0:
                    print("Breaking as mass is now under mass_min")
                self.exitcode = -1
                break

            # store
            self.times.append(t_iter)
            self.states.append(oe_iter)
            self.masses.append(mass_iter)
            self.controls.append([alpha, beta, throttle])
            self.etas.append([val_eta_a, val_eta_r])

            # update battery
            if duty:
                battery_iter = np.clip(battery_iter-self.battery_charge_discharge_rate[1]*t_step_local,
                                       self.battery_capacity[0], self.battery_capacity[1])
            else:
                battery_iter = np.clip(battery_iter+self.battery_charge_discharge_rate[0]*t_step_local,
                                       self.battery_capacity[0], self.battery_capacity[1])
                if battery_iter == self.battery_capacity[1]:
                    charging = False        # turn OFF charging mode
            self.battery.append(battery_iter)

            # index update
            idx += 1

        if self.converge == False:
            if self.verbosity > 0:
                print("Could not arrive to target elements within time")
            self.exitcode = -2
        else:
            if self.verbosity > 0:
                print("Target elements successfully reached!")
        return

    
    def evaluate_osculating_qdot(self, oe, accel_thrust):
        """Evaluate Qdot over the entire orbit
        
        Args:
            oe (np.array): current osculating elements
            accel_thrust (float): acceleration magnitude, tmax/mass

        Returns:
            (tuple): min and max Qdot
        """
        # evaluate qdot at current as well as for all anomalies
        eval_pts = np.linspace(oe[5], oe[5]+2*np.pi, self.anomaly_grid_size+1)[1:]
        # storage FIXME
        qdot_list = []
        for anomaly in eval_pts:
            # construct element
            oe_test = np.array([oe[0], oe[1], oe[2], oe[3], oe[4], anomaly])
            # # evaluate Lyapunov function
            # alpha, beta, _, psi_test = lyapunov_control_angles(
            #     fun_lyapunov_control=self.lyap_fun,
            #     mu=self.mu, 
            #     f=accel_thrust, 
            #     oe=oe_test, 
            #     oeT=self.oeT, 
            #     rpmin=self.rpmin, 
            #     m_petro=self.m_petro, 
            #     n_petro=self.n_petro, 
            #     r_petro=self.r_petro, 
            #     b_petro=self.b_petro, 
            #     k_petro=self.k_petro, 
            #     wp=self.wp, 
            #     woe=self.woe,
            # )
            # u_test = accel_thrust*np.array([
            #     np.cos(beta)*np.sin(alpha),
            #     np.cos(beta)*np.cos(alpha),
            #     np.sin(beta),
            # ])

            # evaluate qdot
            qdot_list.append(
                self.dqdt_fun(
                    self.mu, 
                    accel_thrust, 
                    oe_test, 
                    self.oeT, 
                    self.rpmin, self.m_petro, self.n_petro, 
                    self.r_petro, self.b_petro, self.k_petro, 
                    self.wp, self.woe
                )
            )
            #eval_qdot(psi_test, u_test)
        return min(qdot_list), max(qdot_list)


    def plot_elements_history(
        self,
        figsize = (10,6),
        to_keplerian = False,
        TU = 1.0,
        time_unit_name = "TU",
        plot_mass = True,
        plot_periapsis = False,
    ):
        """Plot elements time history
        
        Args:
            figsize (tuple): figure size
            to_keplerian (bool): whether to convert to Keplerian elements
            TU (float): time unit
            time_unit_name (str): name of time unit
            plot_mass (bool): whether to plot mass or anomaly
            plot_periapsis (bool): whether to overlay periapsis history on top of first plot
        
        Returns:
            (tuple): figure and axis objects
        """
        oes = np.zeros((6,len(self.times)))
        fig, axs = plt.subplots(2,3,figsize=figsize)
        for idx in range(len(self.times)):
            if to_keplerian == True and self.elements_type=="mee_with_a":
                oes[:,idx] = mee_with_a2kep(self.states[idx])
                labels = ["a", "e", "i", "raan", "om", "ta"]
                multipliers = [1,1,180/np.pi,180/np.pi,180/np.pi,180/np.pi]
                oe_T = mee_with_a2kep(np.concatenate((self.oeT,[0.0])))[0:5]
                show_band = False
            else:
                oes[:,idx] = self.states[idx]
                labels = self.element_names
                multipliers = [1,1,1,1,1,1]
                oe_T = self.oeT
                show_band = True

        for idx,ax in enumerate(axs.flatten()[:5]):
            # targeted elements
            if show_band:
                ax.fill_between(
                    np.array(self.times)*TU, 
                    (oe_T[idx]-self.tol_oe[idx])*multipliers[idx], 
                    (oe_T[idx]+self.tol_oe[idx])*multipliers[idx], 
                    color='red', alpha=0.25
                )
            else:
                ax.axhline(oe_T[idx]*multipliers[idx], color='r', linestyle='--')

            # state history
            ax.plot(np.array(self.times)*TU, oes[idx,:]*multipliers[idx], label=labels[idx])
            ax.set(xlabel=f"Time, {time_unit_name}", ylabel=labels[idx])

        # overlay periapsis
        if plot_periapsis:
            axs[0,0].plot(np.array(self.times)*TU, oes[0,:]*(1 - oes[1,:]))

        # plot mass or TA
        if plot_mass:
            axs[1,2].plot(np.array(self.times)*TU, self.masses)
            axs[1,2].set(xlabel=f"Time, {time_unit_name}", ylabel="Mass")
        else:
            axs[1,2].plot(np.array(self.times)*TU, (oes[5,:] % (2*np.pi))*180/np.pi, label=labels[5])
            axs[1,2].set(xlabel=f"Time, {time_unit_name}", ylabel=labels[5])
        plt.tight_layout()
        return fig, ax


    def plot_battery_history(
        self,
        figsize=(9,5),
        TU=1.0,
        BU=1.0,
        time_unit_name="TU",
        battery_unit_name="BU"
    ):
        """Plot battery history
        
        Args:
            figsize (tuple): figure size
            TU (float): time unit
            BU (float): battery unit
            time_unit_name (str): name of time unit
            battery_unit_name (str): name of battery unit
        
        Returns:
            (tuple): figure and axis objects
        """
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(np.array(self.times)*TU, np.array(self.battery)*BU, color='k')
        ax.axhline(np.array(self.battery_capacity[0])*BU, color='r', linestyle='--', label="min capacity")
        ax.axhline(np.array(self.battery_capacity[1])*BU, color='g', linestyle='--', label="max capacity")
        ax.set(xlabel=f"Time, {time_unit_name}", ylabel=f"Battery, {battery_unit_name}")
        return fig, ax


    def plot_controls(self, figsize=(9,5), TU=1.0, time_unit_name="TU"):
        """Plot control time history
        
        Args:
            figsize (tuple): figure size
            TU (float): time unit
            time_unit_name (str): name of time unit
        
        Returns:
            (tuple): figure and axis objects
        """
        alphas, betas, throttles = [], [], []
        for control in self.controls:
            alphas.append(control[0])
            betas.append(control[1])
            throttles.append(control[2])
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.step(np.array(self.times[0:-1])*TU, np.array(alphas)*180/np.pi, where='pre', marker='o', markersize=2, label="alpha")
        ax.step(np.array(self.times[0:-1])*TU, np.array(betas)*180/np.pi, where='pre', marker='o', markersize=2, label="beta")
        ax.step(np.array(self.times[0:-1])*TU, np.array(throttles)*100, where='pre', marker='o', markersize=2, label="throttle, %")
        ax.set(xlabel=f"Time, {time_unit_name}", ylabel="Control angles and throttle")
        ax.legend()
        return fig, ax


    def interpolate_states(self, kind="quadratic"):
        """Create interpolation states"""
        # prepare states matrix
        state_matrix = np.zeros((6,len(self.states)))
        for idx,state in enumerate(self.states):
            state_matrix[:,idx] = state
        f_a = interp1d(self.times, state_matrix[0,:], kind=kind)
        f_e = interp1d(self.times, state_matrix[1,:], kind=kind)
        f_i = interp1d(self.times, state_matrix[2,:], kind=kind)
        f_r = interp1d(self.times, state_matrix[3,:], kind=kind)
        f_o = interp1d(self.times, state_matrix[4,:], kind=kind)
        f_t = interp1d(self.times, state_matrix[5,:], kind=kind)
        return (f_a, f_e, f_i, f_r, f_o, f_t)


    def get_cartesian_history(self, interpolate=False, steps=None):
        """Get Cartesian history of states"""
        if interpolate:
            # interpolate orbital elements
            f_a, f_e, f_i, f_r, f_o, f_t = self.interpolate_states()
            if steps is None:
                steps = min(8000, abs(int(round(self.times[-1]/0.1))))
                print(f"Using {steps} steps for evaluation")
            t_evals = np.linspace(self.times[0], self.times[-1], steps)
            cart = np.zeros((6,steps))
            for idx, t in enumerate(t_evals):
                if self.elements_type=="keplerian":
                    cart[:,idx] = kep2sv([f_a(t), f_e(t), f_i(t), f_r(t), f_o(t), f_t(t)], self.mu)
                elif self.elements_type=="mee_with_a":
                    cart[:,idx] = mee_with_a2sv([f_a(t), f_e(t), f_i(t), f_r(t), f_o(t), f_t(t)], self.mu)
        else:
            cart = np.zeros((6,len(self.times)))
            for idx in range(len(self.times)):
                if self.elements_type=="keplerian":
                    cart[:,idx] = kep2sv(self.states[idx], self.mu)
                elif self.elements_type=="mee_with_a":
                    cart[:,idx] = mee_with_a2sv(np.array(self.states[idx]), self.mu)
        return cart


    def plot_trajectory_2d(
        self, 
        figsize=(6,6),
        interpolate=False, 
        steps=None, 
    ):
        """Plot trajectory in xy-plane"""
        # get cartesian history
        cart = self.get_cartesian_history(interpolate, steps)

        fig, ax = plt.subplots(1,1,figsize=figsize)
        # plot initial and final orbit
        if self.elements_type == "keplerian":
            coord_orb0 = get_orbit_coordinates(self.oe0, self.mu)
            coord_orbT = get_orbit_coordinates(np.concatenate((self.oeT,[0.0])), self.mu)
        elif self.elements_type == "mee_with_a":
            coord_orb0 = get_orbit_coordinates(mee_with_a2kep(self.oe0), self.mu)
            coord_orbT = get_orbit_coordinates(np.concatenate((mee_with_a2kep(self.oeT)[0:5],[0.0])), self.mu)
        ax.plot(coord_orb0[0,:], coord_orb0[1,:], label="Initial", c="darkblue")
        ax.plot(coord_orbT[0,:], coord_orbT[1,:], label="Final", c="forestgreen")

        # plot transfer
        ax.plot(cart[0,:], cart[1,:], label="transfer", c="crimson", lw=0.4)
        ax.scatter(cart[0,0], cart[1,0], label=None, c="crimson", marker="x")
        ax.scatter(cart[0,-1], cart[1,-1], label=None, c="crimson", marker="o")
        ax.set_aspect('equal')
        return fig, ax


    def plot_trajectory_3d(
        self, 
        figsize=(6,6), 
        interpolate=True, 
        steps=None, 
        plot_sphere=True,
        sphere_radius=0.35,
        scale=1.02,
        lw = 0.4
    ):
        """Plot trajectory in xyz"""
        # get cartesian history
        cart = self.get_cartesian_history(interpolate, steps)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        # plot initial and final orbit
        if self.elements_type == "keplerian":
            coord_orb0 = get_orbit_coordinates(self.oe0, self.mu)
            coord_orbT = get_orbit_coordinates(np.concatenate((self.oeT,[0.0])), self.mu)
        elif self.elements_type == "mee_with_a":
            coord_orb0 = get_orbit_coordinates(mee_with_a2kep(self.oe0), self.mu)
            coord_orbT = get_orbit_coordinates(
                mee_with_a2kep(np.concatenate((self.oeT, [0.0]))), 
                self.mu
            )

        # plot transfer
        ax.plot(cart[0,:], cart[1,:], cart[2,:], label="transfer", c="crimson", lw=lw)
        ax.scatter(cart[0,0], cart[1,0], cart[2,0], label=None, c="crimson", marker="x")
        ax.scatter(cart[0,-1], cart[1,-1], cart[2,-1], label=None, c="crimson", marker="o")

        ax.plot(coord_orb0[0,:], coord_orb0[1,:], coord_orb0[2,:], label="Initial", c="darkblue")
        ax.plot(coord_orbT[0,:], coord_orbT[1,:], coord_orbT[2,:], label="Final", c="forestgreen")

        # plot center body
        if plot_sphere:
            plot_sphere_wireframe(ax, sphere_radius)
            xlims = [min(cart[0,:]), max(cart[0,:])]
            ylims = [min(cart[1,:]), max(cart[1,:])]
            zlims = [min(cart[2,:]), max(cart[2,:])]
            set_equal_axis(ax, xlims, ylims, zlims, scale=scale)

        #ax.set_aspect('equal')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        return fig, ax


    def plot_efficiency(self, figsize=(6,6), TU=1.0, time_unit_name="TU"):
        """Plot efficiency
        
        Args:
            figsize (tuple): figure size
            TU (float): time unit
            time_unit_name (str): name of time unit
        
        Returns:
            (tuple): figure and axis objects
        """
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(np.array(self.times[0:-1])*TU, np.array(self.etas)[:,0], label="eta_a")
        ax.plot(np.array(self.times[0:-1])*TU, np.array(self.etas)[:,1], label="eta_r")
        ax.set(xlabel=f"Time, {time_unit_name}", ylabel="Efficiency")
        ax.legend()
        return fig, ax


    def pretty(self):
        """Pretty print"""
        print(f"Transfer:")
        print(f"  {self.element_names[0]}  : {self.oe0[0]:1.4e} -> {self.oeT[0]:1.4e} (weight: {self.woe[0]:2.2f})")
        print(f"  {self.element_names[1]}  : {self.oe0[1]:1.4e} -> {self.oeT[1]:1.4e} (weight: {self.woe[1]:2.2f})")
        print(f"  {self.element_names[2]}  : {self.oe0[2]:1.4e} -> {self.oeT[2]:1.4e} (weight: {self.woe[2]:2.2f})")
        print(f"  {self.element_names[3]}  : {self.oe0[3]:1.4e} -> {self.oeT[3]:1.4e} (weight: {self.woe[3]:2.2f})")
        print(f"  {self.element_names[4]}  : {self.oe0[4]:1.4e} -> {self.oeT[4]:1.4e} (weight: {self.woe[4]:2.2f})")
        return


    def pretty_results(self):
        """Pretty print results"""
        print(f"Exit code : {self.exitcode}")
        print(f"Converge  : {self.converge}")
        print(f"Final state:")
        print(f"  {self.element_names[0]}  : {self.states[-1][0]:1.4e} (error: {abs(self.states[-1][0]-self.oeT[0]):1.4e})")
        print(f"  {self.element_names[1]}  : {self.states[-1][1]:1.4e} (error: {abs(self.states[-1][1]-self.oeT[1]):1.4e})")
        print(f"  {self.element_names[2]}  : {self.states[-1][2]:1.4e} (error: {abs(self.states[-1][2]-self.oeT[2]):1.4e})")
        print(f"  {self.element_names[3]}  : {self.states[-1][3]:1.4e} (error: {abs(self.states[-1][3]-self.oeT[3]):1.4e})")
        print(f"  {self.element_names[4]}  : {self.states[-1][4]:1.4e} (error: {abs(self.states[-1][4]-self.oeT[4]):1.4e})")
        print(f"Transfer time : {self.times[-1]}")
        print(f"Final mass    : {self.masses[-1]}")
        return


    def pretty_settings(self):
        """Pretty print settings"""
        print(f"Element type  : {self.elements_type}")
        print(f"Element names : {self.element_names}")
        print(f"Integrator    : {self.integrator}")
        print(f"Tolerance     : {self.tol_oe}")
        print(f"Relaxed tolerance : {self.tol_oe_relaxed}")
        print(f"Exit at relaxed   : {self.exit_at_relaxed}")
        return

    def save_to_dict(self, filepath, canonical_units=None, save_control_angles = False):
        """Export result into a dictionary, saved as json if filepath is provided
        
        Args:
            filepath (str): filepath to json filename to save the dictionary
        """
        initial_guess = {
            "t0": 0.0,
            "tf": self.times[-1],
            "times": self.times,
            "canonical_units": canonical_units,
            "states": [list(mee_with_a2mee(oe))+[m] for (oe,m) in zip(self.states, self.masses)],
        }
        _controls = copy.deepcopy(self.controls)
        _controls.append([0,0,0])   # final step to match time-steps
        _controls = np.array(_controls)
        if save_control_angles:
            initial_guess["controls"] = _controls
        else:
            # convert to RTN unit vectors
            alphas    = _controls[:,0]
            betas     = _controls[:,1]
            taus      = _controls[:,2]
            uR = np.cos(betas) * np.sin(alphas)
            uT = np.cos(betas) * np.cos(alphas)
            uN = np.sin(betas)
            initial_guess["controls"] = np.stack((uR,uT,uN,taus)).T

        if filepath is not None:
            #dumped = json.dumps(initial_guess, cls=NumpyEncoder)
            with open(filepath, 'w') as f:
                json.dump(initial_guess, f, cls=NumpyEncoder, indent=4)
        return initial_guess
