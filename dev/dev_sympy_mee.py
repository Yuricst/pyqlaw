"""
sympy dev for MEE
"""

import numpy as np
import sympy as sym
import math
from sympy import lambdify
import dill

# define parameters
rpmin, k_petro, m_petro, n_petro, r_petro, b_petro = sym.symbols("rpmin k_petro m_petro n_petro r_petro b_petro")
wp = sym.symbols("Wp")
mu = sym.symbols("mu")
accel = sym.symbols("accel")

# orbital elements
a, f, g, h, k, l = sym.symbols("a f g h k l")
oe = [a,f,g,h,k,l]

# targeted orbital elements
aT, fT, gT, hT, kT, lT = sym.symbols("a_T f_T g_T h_T k_T l_T")
oeT = [aT, fT, gT, hT, kT]

# weights on orbital elements
wa, wf, wg, wh, wk = sym.symbols("w_a w_f w_g w_h w_k")
woe = [wa, wf, wg, wh, wk]


def angle_difference(phi1, phi2):
    return sym.acos(sym.cos(phi1 - phi2))


def quotient(mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe):
    # unpack elements
    a, f, g, h, k, l = oe
    aT, fT, gT, hT, kT = oeT

    # compute semi-parameter p, periapsis rp
    p = a * (1 - f**2 - g**2)
    e = sym.sqrt(f**2 + g**2)
    ang_mom = sym.sqrt(a*mu*(1-e**2))
    rp = a*(1 - e)
    ta = l - sym.atan(g/f)
    r = ang_mom**2/(mu*(1+e*sym.cos(ta)))
    s_squared = 1 + h**2 + k**2

    # -------- compute quotient Q -------- #
    doe = [
        a-aT, 
        f-fT, 
        g-gT, 
        h-hT, 
        k-kT,
    ]

    # compute oedot
    sqrt_pmu = sym.sqrt(p/mu)
    adot_xx = 2*accel*a*sym.sqrt(a/mu) * sym.sqrt((1 + sym.sqrt(f**2 + g**2)) / (1 - sym.sqrt(f**2 + g**2)))
    fdot_xx = 2*accel*sqrt_pmu
    gdot_xx = 2*accel*sqrt_pmu
    hdot_xx = 0.5*accel*sqrt_pmu * s_squared/(sym.sqrt(1 - g**2) + f)
    kdot_xx = 0.5*accel*sqrt_pmu * s_squared/(sym.sqrt(1 - f**2) + g)
    oedot = [
        adot_xx,
        fdot_xx,
        gdot_xx,
        hdot_xx,
        kdot_xx,
    ]
    
    # compute periapsis radius constraint P
    penalty_rp = sym.exp(k_petro*(1.0 - rp/rpmin))
    # compute scaling for each element Soe
    soe = [
        (1 + (sym.sqrt((a-aT)**2)/(m_petro*aT))**n_petro)**(1/r_petro), 
        1.0, 1.0, 1.0, 1.0
    ]
    # compute quotient Q
    q = (1 + wp*penalty_rp) * (
        woe[0]*soe[0]*(doe[0]/oedot[0])**2 +
        woe[1]*soe[1]*(doe[1]/oedot[1])**2 + 
        woe[2]*soe[2]*(doe[2]/oedot[2])**2 +
        woe[3]*soe[3]*(doe[3]/oedot[3])**2 +
        woe[4]*soe[4]*(doe[4]/oedot[4])**2 
    )

    # -------- compute Gauss differential equation terms -------- #
    cosL = sym.cos(l)
    sinL = sym.sin(l)
    w = 1 + f*cosL + g*sinL
    # let psi be column major!
    psi = [
        # multiplies f_r
        [
            2*a**2/ang_mom*e*sym.sin(ta), # this is for a!! #0.0,
             sqrt_pmu* sinL,
            -sqrt_pmu* cosL,
            0.0,
            0.0,
        ],
        # multiplies f_theta
        [
            2*a**2/ang_mom * p/r,  # this is for a!!! #sqrt_pmu*2*p/w,
            sqrt_pmu/w * ((w+1)*cosL + f),
            sqrt_pmu/w * ((w+1)*sinL + g),
            0.0,
            0.0,
        ],
        # multiplies f_h
        [
            0.0,  # this is for a!!!
            sqrt_pmu/w* (-g*(h*sinL - k*cosL)),
            sqrt_pmu/w* ( f*(h*sinL - k*cosL)),
            sqrt_pmu/w* 0.5*(1 + h**2 + k**2)*cosL,
            sqrt_pmu/w* 0.5*(1 + h**2 + k**2)*sinL,
        ]
    ]

    # -------- Apply Lyapunov descent direction -------- #
    # dqdoe = [
    #     sym.diff(q, oe[0]),
    #     sym.diff(q, oe[1]),
    #     sym.diff(q, oe[2]),
    #     sym.diff(q, oe[3]),
    #     sym.diff(q, oe[4]),
    # ]
    dqdoe0 = sym.diff(q, oe[0])   # a
    dqdoe1 = sym.diff(q, oe[1])   # f
    dqdoe2 = sym.diff(q, oe[2])   # g 
    dqdoe3 = sym.diff(q, oe[3])   # h
    dqdoe4 = sym.diff(q, oe[4])   # k

    # compute thrust vector components
    u_unscaled = [
        psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4,
        psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4,
        psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4
    ]
    u_unscaled_norm = sym.sqrt(u_unscaled[0]**2 + u_unscaled[1]**2 + u_unscaled[2]**2)
    ux = u_unscaled[0]/u_unscaled_norm
    uy = u_unscaled[0]/u_unscaled_norm
    uz = u_unscaled[0]/u_unscaled_norm

    # compute thrust angles
    alpha = sym.atan2(-ux,-uy)
    beta = sym.atan(-uz/sym.sqrt(ux**2 + uy**2))

    # compute thrust vector
    thrust_vec = [
        accel* sym.cos(beta)*sym.sin(alpha),
        accel* sym.cos(beta)*sym.cos(alpha),
        accel* sym.sin(beta),
    ]

    # compute qdot
    qdot = 0

    # ux = (psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4)
    # uy = (psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4)
    # uz = (psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4)


    fun_lyapunov_control = lambdify(
        [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
        [thrust_vec, psi, qdot], 
        "numpy",
    )

    # PREVIOUS FUNCTION
    # fun_lyapunov_control = lambdify(
    #     [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
    #     [ux,uy,uz], 
    #     "numpy",
    # )

    fun_eval_psi = lambdify(
        [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
        psi, 
        "sympy",
    )

    fun_eval_dqdoe = lambdify(
        [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
        [dqdoe0,dqdoe1,dqdoe2,dqdoe3,dqdoe4], 
        "sympy",
    )
    return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe


# create function
print("Generating lyapunov control funcion with sympy")
fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe = quotient(
    mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe
)

#print(f"sym.diff(q, oe[2]): \n{hoge0}")
#print(f"sym.diff(q, oe[4]): \n{hoge1}")

# assign numerical values
# a_n, e_n, i_n, ra_n, om_n, ta_n       = 1.0,0.02, 0.2,0.3,0.4,0.5
# aT_n, eT_n, iT_n, raT_n, omT_n, taT_n = 1.4,0.1,  0.7,0.8,0.9,0.1
a_n, e_n, i_n, ra_n, om_n, ta_n  = 1.0, 1e-2, 1e-4, 1e-4, 1e-4, 1e-4
aT_n, eT_n, iT_n, raT_n, omT_n   = 1.2, 0.05, 1e-4, 0.0, 0.0
oe_n = [a_n, e_n, i_n, ra_n, om_n, ta_n]
oeT_n = [aT_n, eT_n, iT_n, raT_n, omT_n]
mu_n = 1.0
f_n = 1.e-5
rpmin_n = 0.8
m_petro_n = 3.0
n_petro_n = 4.0
r_petro_n = 2.0
b_petro_n = 0.01
k_petro_n = 1.0
wp_n = 1.0
woe_n = [1.0, 1.0, 1.0, 1.0, 1.0]

# run the function as a test
psi = fun_eval_psi(
    mu_n, f_n, oe_n, oeT_n, rpmin_n, m_petro_n, n_petro_n, r_petro_n, b_petro_n, k_petro_n, wp_n, woe_n
)
print(f"psi: \n{psi}")

dqdoe = fun_eval_dqdoe(
    mu_n, f_n, oe_n, oeT_n, rpmin_n, m_petro_n, n_petro_n, r_petro_n, b_petro_n, k_petro_n, wp_n, woe_n
)
print(f"dqdoe: \n{dqdoe}")


us, psi_n, qdot_n = fun_lyapunov_control(
    mu_n, f_n, oe_n, oeT_n, rpmin_n, m_petro_n, n_petro_n, r_petro_n, b_petro_n, k_petro_n, wp_n, woe_n
)
print(f"us: \n{us}")
# dill.dump(fun_lyapunov_control, open("../tests/fun_lyapunov_control_mee", "wb"))
# print("Success!")