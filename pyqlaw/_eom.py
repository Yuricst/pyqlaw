"""
equations of motion
"""

import numpy as np
from numba import njit


@njit
def eom_kep_gauss(t, state, param):
    """Equations of motion for gauss with keplerian elements"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2, ptrb_RTN = param

    # unpack elements
    sma,ecc,inc,ra,om,ta = state

    p = sma*(1 - ecc**2)
    h = np.sqrt(sma*mu*(1-ecc**2))
    r = h**2/(mu*(1+ecc*np.cos(ta)))

    # Gauss perturbation
    psi = np.array([   
        #[2*sma**2*ecc*np.sin(ta) / h, (2*sma**2*p) / (r*h), 0.0],
        #[p*np.sin(ta) / h, ( (p + r)*np.cos(ta) + r*ecc ) / h, 0.0],
        #[0, 0, r*np.cos(ta+om) / h],
        #[0, 0, r*np.sin(ta+om)/ ( h*np.sin(inc) )],
        #[-p*np.cos(ta) / ( ecc*h ), (p+r)*np.sin(ta) / ( ecc*h ),  -( r*np.sin(ta+om)*np.cos(inc) ) / ( h*np.sin(inc) )],
        [psi_c0[0], psi_c1[0], psi_c2[0]],
        [psi_c0[1], psi_c1[1], psi_c2[1]],
        [psi_c0[2], psi_c1[2], psi_c2[2]],
        [psi_c0[3], psi_c1[3], psi_c2[3]],
        [psi_c0[4], psi_c1[4], psi_c2[4]],
        [p*np.cos(ta) / ( ecc*h ), - (p+r)*np.sin(ta) / ( ecc*h ), 0.0],
    ])

    # combine
    doe = np.dot(psi, u + ptrb_RTN) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, h/r**2])
    return doe


@njit
def eom_mee_gauss(t, state, param):
    """Equations of motion for gauss with MEE"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2, ptrb_RTN = param
    # u = accel*np.array([
    #     np.cos(beta)*np.sin(alpha),
    #     np.cos(beta)*np.cos(alpha),
    #     np.sin(beta),
    # ])

    # unpack elements
    p,f,g,h,k,l = state

    # Gauss perturbation
    sinL = np.sin(l)
    cosL = np.cos(l)
    w = 1 + f*np.cos(l) + g*np.sin(l)
    sqrt_pmu = np.sqrt(p/mu)
    psi = np.array([   
        [0.0, 1/w*sqrt_pmu* 2*p, 0.0],
        [ sqrt_pmu*sinL, sqrt_pmu/w*((w+1)*cosL + f), -g/w*sqrt_pmu*(h*sinL - k*cosL)],
        [-sqrt_pmu*cosL, sqrt_pmu/w*((w+1)*sinL + g),  f/w*sqrt_pmu*(h*sinL - k*cosL)],
        [0.0, 0.0, sqrt_pmu/w * 0.5*(1 + h**2 + k**2)*cosL],
        [0.0, 0.0, sqrt_pmu/w * 0.5*(1 + h**2 + k**2)*sinL],
        [0.0, 0.0, sqrt_pmu/w * (h*sinL - k*cosL)]
    ])

    # combine
    doe = np.dot(psi, u + ptrb_RTN) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu*p)*(w/p)**2])
    return doe


@njit
def eom_mee_with_a_gauss(t, state, param):
    """Equations of motion for gauss with MEE"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2, ptrb_RTN = param

    # unpack elements
    sma,f,g,h,k,l = state

    # compute additional parameters
    ecc = np.sqrt(f**2 + g**2)
    ang_mom = np.sqrt(sma*mu*(1-ecc**2))
    ta = l - np.arctan(g/f)
    r = ang_mom**2/(mu*(1+ecc*np.cos(ta)))
    p = sma*(1 - ecc**2)

    # Gauss perturbation
    sinL = np.sin(l)
    cosL = np.cos(l)
    w = 1 + f*cosL + g*sinL
    sqrt_pmu = np.sqrt(p/mu)
    psi = np.array([   
        # [2*sma**2*ecc*np.sin(ta) / ang_mom, (2*sma**2*p) / (r*ang_mom), 0.0],
        # [ sqrt_pmu*sinL, sqrt_pmu/w*((w+1)*cosL + f), -g/w*sqrt_pmu*(h*sinL - k*cosL)],
        # [-sqrt_pmu*cosL, sqrt_pmu/w*((w+1)*sinL + g),  f/w*sqrt_pmu*(h*sinL - k*cosL)],
        # [0.0, 0.0, sqrt_pmu/w * 0.5*(1 + h**2 + k**2)*cosL],
        # [0.0, 0.0, sqrt_pmu/w * 0.5*(1 + h**2 + k**2)*sinL],
        [psi_c0[0], psi_c1[0], psi_c2[0]],
        [psi_c0[1], psi_c1[1], psi_c2[1]],
        [psi_c0[2], psi_c1[2], psi_c2[2]],
        [psi_c0[3], psi_c1[3], psi_c2[3]],
        [psi_c0[4], psi_c1[4], psi_c2[4]],
        [0.0, 0.0, sqrt_pmu/w * (h*sinL - k*cosL)]
    ])

    # combine
    doe = np.dot(psi, u + ptrb_RTN) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu*p)*(w/p)**2])
    return doe
