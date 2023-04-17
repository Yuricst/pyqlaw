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
from ._integrate import eom_kep_gauss, eom_mee_gauss, eom_mee_with_a_gauss, rk4, rkf45
from ._convergence import check_convergence, elements_safety
from ._elements import (
    kep2sv, mee_with_a2sv, get_orbit_coordinates, 
    mee_with_a2mee, mee2mee_with_a, mee_with_a2kep
)
from ._plot_helper import plot_sphere_wireframe, set_equal_axis


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

    Attributes:
        print_frequency (int): if verbosity >= 2, prints at this frequency
    """
    def __init__(
        self, 
        mu=1.0,
        rpmin=0.5, 
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
    ):
        """Construct QLaw object"""
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
            fun_lyapunov_control, _, _ = symbolic_qlaw_keplerian()
            self.lyap_fun = fun_lyapunov_control

        elif self.elements_type == "mee_with_a":
            self.element_names = ["a", "f", "g", "h", "k", "L"]
            self.eom = eom_mee_with_a_gauss
            fun_lyapunov_control, _, _ = symbolic_qlaw_mee_with_a()
            self.lyap_fun = fun_lyapunov_control

        # max and min step sizes used with adaptive step integrators
        self.step_min = 1e-4
        self.step_max = 2.0
        self.ode_tol = 1.e-5

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
        tf_max, 
        t_step=0.1,
        mass_min=0.1,
        woe=None,
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
            tol_oe (np.array): tolerances on osculating elements to check convergence
            woe (np.array): weight on each osculating element
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
        self.ready = True  # toggle
        return


    def solve(self, eta_a=0.0, eta_r=0.0):
        """Propagate and solve control problem
        
        Args:
            eta_a (float): min absolute effectivity to thrust, `0.0 <= eta_a <= 1.0`
            eta_r (float): relative effectivity, `0.0 <= eta_r <= 1.0`
        """
        assert self.ready == True, "Please first call `set_problem()`"
        # get max number of steps
        #nmax = int(round(self.tf_max / self.t_step))

        # efficiency thresholds
        self.eta_a = eta_a
        self.eta_r = eta_r

        # initialize values for propagation
        t_iter = 0.0
        oe_iter = self.oe0
        mass_iter = self.mass0

        # initialize storage
        self.times = [t_iter,]
        self.states = [oe_iter,]
        self.masses = [mass_iter,]
        self.controls = []
        n_relaxed_cleared = 0
        n_nan_angles = 0

        if self.verbosity >= 2:
            self.disable_tqdm = True
            header = " iter   |  time      |  del1       |  del2       |  del3       |  del4       |  del5       |  el6        |"
    
        # iterate until nmax
        idx = 0
        while self.times[-1] < self.tf_max:  #for idx in tqdm(range(nmax), disable=self.disable_tqdm, desc="qlaw"):
            # ensure numerical stabilty
            if self.elements_type=="keplerian":
                oe_iter = elements_safety(oe_iter, self.oe_min, self.oe_max)

            # compute instantaneous acceleration magnitude due to thrust
            accel_thrust = self.tmax/mass_iter

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
                    qdot_current = eval_qdot(psi, u)    # FIXME!!
                    qdot_min, qdot_max = self.evaluate_osculating_qdot(
                        oe_iter, accel_thrust
                    )
                    val_eta_a = qdot_current/qdot_min
                    val_eta_r = (qdot_current - qdot_max)/(qdot_min - qdot_max)
                    # turn thrust off if below threshold
                    if val_eta_a < self.eta_a or val_eta_r < self.eta_r:
                        throttle = 0  # turn off
                        u = np.zeros((3,))

            # ODE parameters
            ode_params = (self.mu, u, psi[0], psi[1], psi[2])
            if self.integrator == "rk4":
                oe_next = rk4(
                    self.eom, 
                    t_iter,
                    self.t_step,
                    oe_iter,
                    ode_params,
                )
                t_iter += self.t_step  # update time
                if throttle == 1:
                    mass_iter -= self.mdot*self.t_step  # update mass
                oe_iter = oe_next

            elif self.integrator == "rkf45":
                oe_next, h_next = rkf45(
                    self.eom, 
                    t_iter,
                    self.t_step,
                    oe_iter,
                    ode_params,
                    self.ode_tol,
                )
                t_iter += self.t_step  # update time
                if throttle == 1:
                    mass_iter -= self.mdot*self.t_step  # update mass
                oe_iter = oe_next
                #print(f"h_nect: {h_next}")
                self.t_step = max(self.step_min, min(self.step_max,h_next))
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
                t_fraction = t_iter/self.tf_max
                print(f" {idx:6.0f} | {t_fraction: 1.3e} | {oe_next[0]-self.oeT[0]: 1.4e} | {oe_next[1]-self.oeT[1]: 1.4e} | {oe_next[2]-self.oeT[2]: 1.4e} | {oe_next[3]-self.oeT[3]: 1.4e} | {oe_next[4]-self.oeT[4]: 1.4e} | {oe_next[5]: 1.4e} |")

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

            # evaluate Lyapunov function
            alpha, beta, _, psi_test = lyapunov_control_angles(
                fun_lyapunov_control=self.lyap_fun,
                mu=self.mu, 
                f=accel_thrust, 
                oe=oe_test, 
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
            u_test = accel_thrust*np.array([
                np.cos(beta)*np.sin(alpha),
                np.cos(beta)*np.cos(alpha),
                np.sin(beta),
            ])

            # evaluate qdot
            qdot_test = eval_qdot(psi_test, u_test)
            qdot_list.append(qdot_test)
        return min(qdot_list), max(qdot_list)


    def plot_elements_history(self, figsize=(6,4), loc='lower center', to_keplerian=False):
        """Plot elements time history"""
        oes = np.zeros((6,len(self.times)))
        for idx in range(len(self.times)):
            if to_keplerian == True and self.elements_type=="mee_with_a":
                oes[:,idx] = mee_with_a2kep(self.states[idx])
                labels = ["a", "e", "i", "raan", "om", "ta"]
            else:
                oes[:,idx] = self.states[idx]
                labels = self.element_names

        fig, ax = plt.subplots(1,1,figsize=figsize)
        for idx in range(5):
            ax.plot(self.times, oes[idx,:], label=labels[idx])
        ax.plot(self.times, oes[5,:] % (2*np.pi), label=labels[5])
        ax.legend(loc=loc)
        ax.set(xlabel="Time", ylabel="Elements")
        return fig, ax


    def plot_controls(self, figsize=(9,6)):
        """Plot control time history"""
        alphas, betas, throttles = [], [], []
        for control in self.controls:
            alphas.append(control[0])
            betas.append(control[1])
            throttles.append(control[2])
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(self.times[0:-1], np.array(alphas)*180/np.pi, marker='o', markersize=2, label="alpha")
        ax.plot(self.times[0:-1], np.array(betas)*180/np.pi,  marker='o', markersize=2, label="beta")
        ax.plot(self.times[0:-1], np.array(throttles)*100,  marker='o', markersize=2, label="throttle, %")
        ax.set(xlabel="Time", ylabel="Control angles and throttle")
        ax.legend()
        return fig, ax


    def interpolate_states(self):
        """Create interpolation states"""
        # prepare states matrix
        state_matrix = np.zeros((6,len(self.states)))
        for idx,state in enumerate(self.states):
            state_matrix[:,idx] = state
        f_a = interp1d(self.times, state_matrix[0,:])
        f_e = interp1d(self.times, state_matrix[1,:])
        f_i = interp1d(self.times, state_matrix[2,:])
        f_r = interp1d(self.times, state_matrix[3,:])
        f_o = interp1d(self.times, state_matrix[4,:])
        f_t = interp1d(self.times, state_matrix[5,:])
        return (f_a, f_e, f_i, f_r, f_o, f_t)


    def get_cartesian_history(self, interpolate=True, steps=None):
        """Get Cartesian history of states"""
        if interpolate:
            # interpolate orbital elements
            f_a, f_e, f_i, f_r, f_o, f_t = self.interpolate_states()
            if steps is None:
                steps = min(8000, int(round(self.times[-1]/0.1)))
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
                    cart[:,idx] = mee_with_a2sv(self.states[idx], self.mu)
        return cart


    def plot_trajectory_2d(
        self, 
        figsize=(6,6),
        interpolate=True, 
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
        scale=1.02
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
        ax.plot(coord_orb0[0,:], coord_orb0[1,:], coord_orb0[2,:], label="Initial", c="darkblue")
        ax.plot(coord_orbT[0,:], coord_orbT[1,:], coord_orbT[2,:], label="Final", c="forestgreen")

        # plot center body
        if plot_sphere:
            plot_sphere_wireframe(ax, sphere_radius)
            xlims = [min(cart[0,:]), max(cart[0,:])]
            ylims = [min(cart[1,:]), max(cart[1,:])]
            zlims = [min(cart[2,:]), max(cart[2,:])]
            set_equal_axis(ax, xlims, ylims, zlims, scale=scale)

        # plot transfer
        ax.plot(cart[0,:], cart[1,:], cart[2,:], label="transfer", c="crimson", lw=0.4)
        ax.scatter(cart[0,0], cart[1,0], cart[2,0], label=None, c="crimson", marker="x")
        ax.scatter(cart[0,-1], cart[1,-1], cart[2,-1], label=None, c="crimson", marker="o")
        #ax.set_aspect('equal')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
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


def eval_qdot(psi, u):
    return psi[0][0]*u[0] + psi[1][0]*u[1] + psi[2][0]*u[2] +\
           psi[0][1]*u[0] + psi[1][1]*u[1] + psi[2][1]*u[2] +\
           psi[0][2]*u[0] + psi[1][2]*u[1] + psi[2][2]*u[2] +\
           psi[0][3]*u[0] + psi[1][3]*u[1] + psi[2][3]*u[2] +\
           psi[0][4]*u[0] + psi[1][4]*u[1] + psi[2][4]*u[2]