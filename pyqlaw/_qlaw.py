"""
main functions for Q-Law transfers
"""

import copy
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from ._lyapunov import lyapunov_control
from ._integrate import eom_gauss, rk4
from ._convergence import check_convergence, keplerian_safety
from ._elements import kep2sv, get_orbit_coordinates


class QLaw:
    """Object for Q-law based transfer problem.

    Exitcodes:
         0 : initial value (problem not yet attempted)
         1 : solve succeed
        -1 : mass is below threshold
        -2 : target elements could not be reached within allocated time
        -3 : thrust angles from feedback control law is nan
    """
    def __init__(
        self, 
        mu=1.0,
        rpmin=0.8, 
        k_petro=1.0, 
        m_petro=3.0, 
        n_petro=4.0, 
        r_petro=2.0, 
        b_petro=0.01,
        wp=1.0,
        verbosity=1,
        disable_tqdm=False,
        oe_min=None,
        oe_max=None,
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
        # settings
        self.verbosity = verbosity
        self.disable_tqdm = disable_tqdm
        # orbital elements bounds
        if oe_min is None:
            self.oe_min = [0.05,1e-6,1e-6,0.0,0.0,0.0]
        else:
            self.oe_min = oe_min
        if oe_max is None:
            self.oe_max = [1e3,0.95,np.pi,np.inf,np.inf,np.inf]
        else:
            self.oe_max = oe_max
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
        t_step,
        mass_min=0.1,
        tol_oe=None,
        woe=None,
    ):
        """Set transfer problem
        
        Args:
            oe0 (np.array): initial state, in Keplerian elements
            oeT (np.array): final state, in Keplerian elements
            f (float): max thrust
        """
        assert len(oe0)==6, "oe6 must have 6 components"
        assert len(oeT)==5, "oeT must have 5 components"
        # weight parameters
        if tol_oe is None:
            self.tol_oe = [1e-4,1e-6,1e-4,1e-4,1e-4]
        else:
            assert len(tol_oe)==5, "tol_oe must have 5 components"
            self.tol_oe = tol_oe
        if woe is None:
            self.woe = [1.0,1.0,1.0,1.0,1.0]
        else:
            assert len(woe)==5, "woe must have 5 components"
            self.woe = woe

        # time parameters
        self.tf_max = tf_max
        self.t_step = t_step
        # spacecraft parameters
        self.oe0   = oe0
        self.oeT   = oeT
        self.mass0 = mass0
        self.tmax  = tmax
        self.mdot  = mdot
        self.mass_min = mass_min
        self.ready = True  # toggle
        return


    def solve(self):
        """Propagate and solve control problem"""
        assert self.ready == True, "Please first call `set_problem()`"
        # get max number of steps
        nmax = int(round(self.tf_max / self.t_step))

        # initialize values for propagation
        t_iter = 0.0
        oe_iter = self.oe0
        mass_iter = self.mass0

        # initialize storage
        self.times = [t_iter,]
        self.states = [oe_iter,]
        self.masses = [mass_iter,]
        self.controls = []

        # iterate until nmax
        for idx in tqdm(range(nmax), disable=self.disable_tqdm, desc="qlaw"):
            # ensure numerical stabilty
            oe_iter = keplerian_safety(oe_iter, self.oe_min, self.oe_max)

            # compute instantaneous acceleration magnitude due to thrust
            accel_thrust = self.tmax/mass_iter

            # get control angles
            alpha, beta = lyapunov_control(
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
                woe=self.woe
            )
            if np.isnan(alpha) == True or np.isnan(beta) == True:
                if self.verbosity > 0:
                    print("Breaking as angles are nan")
                self.exitcode = -3
                break

            # update state
            oe_next = rk4(
                eom_gauss, 
                t_iter,
                self.t_step,
                oe_iter,
                (self.mu, accel_thrust, alpha, beta),
            )

            # check convergence/break conditions
            if check_convergence(oe_next, self.oeT, self.woe, self.tol_oe) == True:
                self.exitcode = 1
                self.converge = True
                break

            if mass_iter <= self.mass_min:
                if self.verbosity > 0:
                    print("Breaking as mass is now under mass_min")
                self.exitcode = -1
                break

            # update
            t_iter += self.t_step
            mass_iter -= self.mdot*self.t_step
            oe_iter = oe_next

            # store
            self.times.append(t_iter)
            self.states.append(oe_iter)
            self.controls.append([alpha,beta])
        if self.converge == False:
            if self.verbosity > 0:
                print("Could not arrive to target elements within time")
            self.exitcode = -2
        else:
            if self.verbosity > 0:
                print("Target elements successfully reached!")
        return


    def plot_elements_history(self, figsize=(6,4), loc='lower center'):
        """Plot elements"""
        oes = np.zeros((6,len(self.times)))
        for idx in range(len(self.times)):
            oes[:,idx] = self.states[idx]

        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(self.times, oes[0,:], label="a")
        ax.plot(self.times, oes[1,:], label="e")
        ax.plot(self.times, oes[2,:], label="i")
        ax.plot(self.times, oes[3,:], label="raan")
        ax.plot(self.times, oes[4,:], label="om")
        ax.plot(self.times, oes[5,:] % (2*np.pi), label="ta")
        ax.legend(loc=loc)
        ax.set(xlabel="Time", ylabel="Elements")
        return fig, ax


    def plot_trajectory_2d(self, figsize=(6,6)):
        """Plot trajectory in xy-plane"""
        cart = np.zeros((6,len(self.times)))
        for idx in range(len(self.times)):
            cart[:,idx] = kep2sv(self.states[idx], self.mu)

        fig, ax = plt.subplots(1,1,figsize=figsize)
        # plot initial and final orbit
        coord_orb0 = get_orbit_coordinates(self.oe0, self.mu)
        coord_orbT = get_orbit_coordinates(np.concatenate((self.oeT,[0.0])), self.mu)
        ax.plot(coord_orb0[0,:], coord_orb0[1,:], label="Initial", c="darkblue")
        ax.plot(coord_orbT[0,:], coord_orbT[1,:], label="Final", c="forestgreen")

        # plot transfer
        ax.plot(cart[0,:], cart[1,:], label="transfer", c="crimson")
        ax.set_aspect('equal')
        return fig, ax


    def pretty(self):
        """Pretty print"""
        print(f"Transfer:")
        print(f"  sma  : {self.oe0[0]:1.4e} -> {self.oeT[0]:1.4e} (weight: {self.woe[0]:2.2f})")
        print(f"  ecc  : {self.oe0[1]:1.4e} -> {self.oeT[1]:1.4e} (weight: {self.woe[1]:2.2f})")
        print(f"  inc  : {self.oe0[2]:1.4e} -> {self.oeT[2]:1.4e} (weight: {self.woe[2]:2.2f})")
        print(f"  raan : {self.oe0[3]:1.4e} -> {self.oeT[3]:1.4e} (weight: {self.woe[3]:2.2f})")
        print(f"  aop  : {self.oe0[4]:1.4e} -> {self.oeT[4]:1.4e} (weight: {self.woe[4]:2.2f})")
        return


    def pretty_results(self):
        """Pretty print"""
        print(f"Transfer:")
        print(f"  sma  : {self.states[-1][0]:1.4e} -> {self.oeT[0]:1.4e} (weight: {self.woe[0]:2.2f})")
        print(f"  ecc  : {self.states[-1][1]:1.4e} -> {self.oeT[1]:1.4e} (weight: {self.woe[1]:2.2f})")
        print(f"  inc  : {self.states[-1][2]:1.4e} -> {self.oeT[2]:1.4e} (weight: {self.woe[2]:2.2f})")
        print(f"  raan : {self.states[-1][3]:1.4e} -> {self.oeT[3]:1.4e} (weight: {self.woe[3]:2.2f})")
        print(f"  aop  : {self.states[-1][4]:1.4e} -> {self.oeT[4]:1.4e} (weight: {self.woe[4]:2.2f})")
        return

