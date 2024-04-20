"""
integration
"""

import numpy as np
from numba import njit


@njit
def eom_kep_gauss(t, state, param):
    """Equations of motion for gauss with keplerian elements"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2 = param

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
    doe = np.dot(psi,u) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, h/r**2])
    return doe


@njit
def eom_mee_gauss(t, state, param):
    """Equations of motion for gauss with MEE"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2 = param
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
    doe = np.dot(psi,u) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu*p)*(w/p)**2])
    return doe



@njit
def eom_mee_with_a_gauss(t, state, param):
    """Equations of motion for gauss with MEE"""
    # unpack parameters
    mu, u, psi_c0, psi_c1, psi_c2 = param

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
    doe = np.dot(psi,u) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu*p)*(w/p)**2])
    return doe



# @njit
# def eom_gauss_nonspherical_gravity(t, state, p):
#     """Equations of motion for gauss with gravitational perturbation"""
#     # unpack parameters
#     mu, f, alpha, beta = p
#     u = f*np.array([
#         np.cos(beta)*np.sin(alpha),
#         np.cos(beta)*np.cos(alpha),
#         np.sin(beta),
#     ])

#     # unpack elements
#     sma,ecc,inc,ra,om,ta = state

#     p = sma*(1 - ecc**2)
#     h = np.sqrt(sma*mu*(1-ecc**2))
#     r = h**2/(mu*(1+ecc*np.cos(ta)))

#     # Gauss perturbation
#     psi = np.array([   
#         [2*sma**2*ecc*np.sin(ta) / h, (2*sma**2*p) / (r*h), 0.0],
#         [p*np.sin(ta) / h, ( (p + r)*np.cos(ta) + r*ecc ) / h, 0.0],
#         [0, 0, r*np.cos(ta+om) / h],
#         [0, 0, r*np.sin(ta+om)/ ( h*np.sin(inc) )],
#         [-p*np.cos(ta) / ( ecc*h ), (p+r)*np.sin(ta) / ( ecc*h ),  -( r*np.sin(ta+om)*np.cos(inc) ) / ( h*np.sin(inc) )],
#         [p*np.cos(ta) / ( ecc*h ), - (p+r)*np.sin(ta) / ( ecc*h ), 0],
#     ])

#     # combine
#     doe = np.dot(psi,u) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, h/r**2])
#     return doe


@njit
def rk4(rhs, t, h, y, p):
    """Perform single-step Runge-Kutta 4th order
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = rhs(t, y, p)
    k2 = rhs(t + 0.5 * h, y + 0.5 * k1, p)
    k3 = rhs(t + 0.5 * h, y + 0.5 * k2, p)
    k4 = rhs(t + h, y + k3, p)
    return y + (h / 6.0)*(k1 + 2*k2 + 2*k3 + k4)


@njit
def rkf45(rhs, t, h, y, p, tol=1e-6):
    """Perform single-step Runge-Kutta 4th order with 5th order step correction
    
    Args:
        rhs (callable): ODE right-hand side expressions
        t (float): current time
        h (float): time-step
        y (np.array): current state-vector
        p (Real or tuple): additional parameters passed to `rhs`
        tol (float): tolerance

    Returns:
        (np.array); state-vector at time t+h
    """
    k1 = h * rhs(t, y, p)
    k2 = h * rhs(t + 1/4 * h,   y + 1/4*h * k1, p)
    k3 = h * rhs(t + 3/8 * h,   y + 3/32*h*k1 + 9/32*h*k2, p)
    k4 = h * rhs(t + 12/13 * h, y + 1932/2197*h*k1 - 7200/2197*h*k2 + 7296/2197*h*k3, p)
    k5 = h * rhs(t + h,         y + 439/216*h*k1 - 8*h*k2 + 3680/513*h*k3 - 845/4104*h*k4, p)
    k6 = h * rhs(t + 1/2*h,     y - 8/27*h*k1 + 2*h*k2 - 3544/2565*h*k3 + 1859/4104*h*k4 - 11/40*h*k5, p)
    y_next = y + 25/216*k1 + 1408/2565*k3 + 2197/4101*k4 - 1/5*k5
    z_next = y + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6
    h_next = h*0.84*(tol/np.linalg.norm(z_next - y_next))**(1/4)
    return y_next, h_next