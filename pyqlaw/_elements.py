"""
Useful expressions in Keplerian dynamics
"""

import numpy as np
import numpy.linalg as la
from numba import njit

from ._transformations import rotmat1, rotmat3

@njit
def get_inclination(state):
    """Function computes inclination in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
    Returns:
        (float): inclination in radians
    """

    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # inclination
    inc = np.arccos(h[2] / la.norm(h))
    return inc


@njit
def get_raan(state):
    """Function computes RAAN in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
    Returns:
        (float): RAAN in radians
    """
    
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    zdir = np.array([0, 0, 1])
    ndir = np.cross(zdir, h)
    # compute RAAN
    raan = np.arctan2(ndir[1], ndir[0])
    return raan
    
    
@njit
def get_eccentricity(state, mu):
    """Function computes eccentricity vector from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter
    Returns:
        (np.arr): eccentricity vector
    """
    
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    #zdir = np.array([0, 0, 1])
    #ndir = np.cross(zdir, h)
    # compute eccentricity vector
    ecc = (1/mu) * np.cross(v,h) - r/la.norm(r)
    return ecc


@njit
def get_omega(state, mu):
    """Function computes argument of periapsis in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter
    Returns:
        (float): argument of periapsis in radians
    """
    
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    zdir = np.array([0, 0, 1])
    ndir = np.cross(zdir, h)
    # compute eccentricity vector
    ecc = (1/mu) * np.cross(v,h) - r/la.norm(r)

    # compute argument of periapsis
    if la.norm(ndir)*la.norm(ecc) != 0.0:
        omega = np.arccos( np.dot(ndir,ecc) / (la.norm(ndir)*la.norm(ecc)) )
        if ecc[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = 0.0
    return omega


@njit
def get_trueanom(state, mu):
    """Function computes argument of periapsis in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter
    Returns:
        (float): true anomaly in radians
    """
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = la.norm( np.cross(r,v) )
    # radial velocity
    vr = np.dot(v,r)/la.norm(r)
    theta = np.arctan2(h*vr, h**2/la.norm(r) - mu)
    return theta


@njit
def get_semiMajorAxis(state, mu):
    """Function computes semi major axis of keplrian orbit
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter
    Returns:
        (float): semi-major axis
    """
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = la.norm( np.cross(r,v) )
    # eccentricity
    e = la.norm( get_eccentricity(state, mu) )
    # semi-major axis
    a = h**2 / (mu*(1 - e**2))
    return a


@njit
def get_period(state, mu):
    """Function computes period of keplerian orbit
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter
        
    Returns:
        (float): period
    """
    a = get_semiMajorAxis(state, mu)
    if a < 0:
        period = -1
    else:
        period = 2*np.pi*np.sqrt(a**3/mu)
    return period


@njit
def sv2kep(state, mu):
    """Convert Cartesian states to Keplerian elements
    Args:
        state (np.array): cartesian states
        mu (float): two-body mass parameter
    
    Returns:
        (np.array): elements, in order [sma, ecc, inc, raan, aop, ta]
    """
    # convert to orbital elements
    sma = get_semiMajorAxis(state, mu)
    ecc = la.norm(get_eccentricity(state, mu))
    inc = get_inclination(state)
    raan = get_raan(state)
    aop = get_omega(state, mu)
    ta = get_trueanom(state, mu)
    return np.array([sma, ecc, inc, raan, aop, ta])


def kep2sv(kep, mu):
    """Convert Keplerian elements to Cartesian states
    Args:
        kep (np.array): elements, in order [sma, ecc, inc, raan, aop, ta]
        mu (float): two-body mass parameter
    
    Returns:
        (np.array): cartesian states
    """
    # unpack elements
    sma, ecc, inc, raan, aop, ta = kep
    # construct state in perifocal frame
    h = np.sqrt(sma*mu*(1 - ecc**2))
    rPF = h**2 / (mu*(1 + ecc*np.cos(ta))) * np.array([np.cos(ta), np.sin(ta), 0.0])
    vPF = mu/h * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])
    # convert to inertial frame
    if (aop != 0.0):
        r1 = np.dot(rotmat3(-aop), rPF)
        v1 = np.dot(rotmat3(-aop), vPF)
    else:
        r1 = rPF
        v1 = vPF

    if (inc != 0.0):
        r2 = np.dot(rotmat1(-inc), r1)
        v2 = np.dot(rotmat1(-inc), v1)
    else:
        r2 = r1
        v2 = v1
    
    if (raan != 0.0):
        rI = np.dot(rotmat3(-raan), r2)
        vI = np.dot(rotmat3(-raan), v2)
    else:
        rI = r2
        vI = v2
    return np.concatenate((rI, vI))


def get_orbit_coordinates(oe_kep,mu,steps=200):
    # unpack elements
    sma, ecc, inc, raan, aop, _ = oe_kep
    tas = np.linspace(0.0, 2*np.pi, steps)
    coord = np.zeros((6,steps))
    for idx in range(steps):
        coord[:,idx] = kep2sv(np.array([sma, ecc, inc, raan, aop, tas[idx]]), mu)
    return coord

