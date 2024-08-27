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
    if np.abs(la.norm(ndir)*la.norm(ecc)) >= 1e-12:
        omega = np.arccos( np.dot(ndir,ecc) / (la.norm(ndir)*la.norm(ecc)) )
        if ecc[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = np.arctan2(ecc[1], ecc[0])
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


@njit
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


@njit
def get_orbit_coordinates(oe_kep,mu,steps=200):
    # unpack elements
    sma, ecc, inc, raan, aop, _ = oe_kep
    tas = np.linspace(0.0, 2*np.pi, steps)
    coord = np.zeros((6,steps))
    for idx in range(steps):
        coord[:,idx] = kep2sv(np.array([sma, ecc, inc, raan, aop, tas[idx]]), mu)
    return coord


@njit
def kep2mee(oe_kep):
    """Convert Keplerian elements to MEE"""
    # unpack
    a,e,i,raan,om,ta = oe_kep
    # compute MEEs
    p = a*(1-e**2)
    f = e*np.cos(raan+om)
    g = e*np.sin(raan+om)
    h = np.tan(i/2)*np.cos(raan)
    k = np.tan(i/2)*np.sin(raan)
    l = raan + om + ta
    return np.array([p,f,g,h,k,l])


@njit
def mee2kep(oe_mee):
    """Convert MEE to Keplerian elements"""
    # unpack 
    p,f,g,h,k,l = oe_mee
    # compute Keplerian elements
    a = p/(1-f**2-g**2)
    e = np.sqrt(f**2 + g**2)
    i = np.arctan2(2*np.sqrt(h**2+k**2), 1-h**2-k**2)
    raan = np.arctan2(k,h)
    om = np.arctan2(g*h-f*k, f*h+g*k)
    ta = l - raan - om
    return np.array([a,e,i,raan,om,ta])


@njit
def mee_with_a2mee(oe_mee_with_a):
    # unpack 
    a,f,g,h,k,l = oe_mee_with_a
    p = a * (1-f**2-g**2)
    return np.array([p, f,g,h,k,l])


@njit
def mee_with_a2kep(oe_mee_with_a):
    return mee2kep(mee_with_a2mee(oe_mee_with_a))


@njit
def mee2mee_with_a(oe_mee):
    """Convert MEE to MEE with SMA"""
    # unpack 
    p,f,g,h,k,l = oe_mee
    a = p/(1-f**2-g**2)
    return np.array([a,f,g,h,k,l])


@njit
def kep2mee_with_a(oe_kep):
    """Get targeting element set used by Q-law when using MEE"""
    # unpack
    a,e,i,raan,om,ta = oe_kep
    # compute MEEs
    #p = a*(1-e**2)
    f = e*np.cos(raan+om)
    g = e*np.sin(raan+om)
    h = np.tan(i/2)*np.cos(raan)
    k = np.tan(i/2)*np.sin(raan)
    l = raan + om + ta
    return np.array([a,f,g,h,k,l])


@njit
def mee_with_a2sv(mee_with_a, mu):
    """Convert MEE with SMA to Cartesian states"""
    a,f,g,h,k,l = mee_with_a
    mee = np.array([a * (1 - f**2 - g**2), f,g,h,k,l])
    kep = mee2kep(mee)
    return kep2sv(kep, mu)


@njit
def ta2ea(ta,ecc):
    """Convert true anomaly to eccentric anomaly"""
    ea = 2*np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(ta/2))
    return ea


@njit
def mee2ea(mee):
    ecc = np.sqrt(mee[1]**2 + mee[2]**2)
    ta = mee[5] - np.arctan2(mee[2],mee[1])
    return ta2ea(ta,ecc)