"""
Plotting methods for QLaw class
"""

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


from ._elements import (
    get_orbit_coordinates, 
    mee_with_a2kep,
)
from ._plot_helper import plot_sphere_wireframe, set_equal_axis
from ._transformations import wrap


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
    for ax in axs.flatten():
        ax.grid(True, alpha=0.3)
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

# 
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


def plot_controls(self, figsize=(9,5), TU=1.0, time_unit_name="TU", show_anomaly=True):
    """Plot control time history
    
    Args:
        figsize (tuple): figure size
        TU (float): time unit
        time_unit_name (str): name of time unit
        show_anomaly (bool): whether to overlay anomaly
    
    Returns:
        (tuple): figure and axis objects
    """
    alphas, betas, throttles = [], [], []
    for control in self.controls:
        alphas.append(control[0])
        betas.append(control[1])
        throttles.append(control[2])
    fig, ax = plt.subplots(1,1,figsize=figsize)
    if show_anomaly:
        ax.plot(np.array(self.times)*TU, wrap(np.array(self.states)[:,5])*180/np.pi,
                color='grey', label="Anomaly", lw=0.35)
    ax.step(np.array(self.times[0:-1])*TU, np.array(alphas)*180/np.pi, where='pre', marker='o', markersize=2, label="alpha")
    ax.step(np.array(self.times[0:-1])*TU, np.array(betas)*180/np.pi, where='pre', marker='o', markersize=2, label="beta")
    ax.step(np.array(self.times[0:-1])*TU, np.array(throttles)*100, where='pre', marker='o', markersize=2, label="throttle, %")
    
    radii = np.array(self.states)[:,0]*(1 - np.array(self.states)[:,1])

    ax.set(xlabel=f"Time, {time_unit_name}", ylabel="Control angles and throttle")
    ax.legend()
    return fig, ax


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
        coord_orbT = get_orbit_coordinates(
            mee_with_a2kep(np.concatenate((self.oeT[0:5],[0.0]))), self.mu)
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


def plot_efficiency(self, figsize=(9,6), TU=1.0, time_unit_name="TU"):
    """Plot efficiency
    
    Args:
        figsize (tuple): figure size
        TU (float): time unit
        time_unit_name (str): name of time unit
    
    Returns:
        (tuple): figure and axis objects
    """
    fig, axs = plt.subplots(2,1,figsize=figsize)
    axs[0].plot(np.array(self.times[0:-1])*TU, np.array(self.etas)[:,0], label="eta_a")
    axs[0].plot(np.array(self.times[0:-1])*TU, np.array(self.etas_bounds)[:,0], label="eta_a_min", linestyle='-', color='red')
    axs[0].set(xlabel=f"Time, {time_unit_name}", ylabel="Absolute efficiency")

    axs[1].plot(np.array(self.times[0:-1])*TU, np.array(self.etas)[:,1], label="eta_r")
    axs[1].plot(np.array(self.times[0:-1])*TU, np.array(self.etas_bounds)[:,1], label="eta_r_min", linestyle='-', color='red')
    axs[1].set(xlabel=f"Time, {time_unit_name}", ylabel="Relative efficiency")
    for ax in axs:
        ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_Q(self, figsize=(9,6), TU=1.0, time_unit_name="TU"):
    """Plot quotient history
    
    Args:
        figsize (tuple): figure size
        TU (float): time unit
        time_unit_name (str): name of time unit
    
    Returns:
        (tuple): figure and axis objects
    """
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.step(np.array(self.times[0:-1])*TU, np.array(self.Qs), label="Q")
    ax.step(np.array(self.times[0:-1])*TU, np.array(self.dQdts), label="|dQ/dt|, [1/TU]")
    ax.grid(True, alpha=0.3)
    ax.set(yscale="log")
    ax.legend()
    plt.tight_layout()
    return fig, ax