"""
sympy dev
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
f = sym.symbols("f")

# orbital elements
a, e, i, ra, om, ta = sym.symbols("a e i Omega omega theta")
oe = [a,e,i,ra,om,ta]

# targeted orbital elements
aT, eT, iT, raT, omT = sym.symbols("a_T e_T i_T Omega_T omega_T")
oeT = [aT, eT, iT, raT, omT]

# weights on orbital elements
wa, we, wi, wra, wom = sym.symbols("w_a w_e w_i w_Omega w_omega")
woe = [wa, we, wi, wra, wom]


def quotient(mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe):
    # unpack elements
    a,e,i,ra,om,ta = oe
    aT, eT, iT, raT, omT = oeT
    rp = a*(1-e)
    p = a*(1 - e**2)
    h = sym.sqrt(a*mu*(1-e**2))

    # sin and cos terms
    cosTA = sym.cos(ta)
    sinTA = sym.sin(ta)
    cosi  = sym.cos(i)
    sini  = sym.sin(i)
    cosom = sym.cos(om)
    sinom = sym.sin(om)

    # -------- compute quotient Q -------- #
    doe = [
        a-aT, e-eT, i-iT, 
        sym.acos(sym.cos(ra - raT)),
        sym.acos(sym.cos(om - omT)),
        #sym.acos(sym.cos(ta - taT)),
    ]

    # compute RAdot and omdot
    radot = p*f/(h*sini*(sym.sqrt(1 - e**2*cosom**2) - e*sym.sqrt((sinom**2))))  # FIXME
    cos_ta_xx_sqrt = sym.sqrt((1/4)*((1-e**2)/e**3)**2 + 1/27)
    ta_xx_term1 = ( (1-e**2) / (2*e**3) + cos_ta_xx_sqrt)**(1/3)
    ta_xx_term2 = (-(1-e**2) / (2*e**3) + cos_ta_xx_sqrt)**(1/3)
    cos_ta_xx = ta_xx_term1 - ta_xx_term2 - 1/e
    r_xx = p / (1 + e*cos_ta_xx)
    omdot_i = f/(e*h)*sym.sqrt(
       p**2*cos_ta_xx**2 + (p+r_xx)**2*(1-cos_ta_xx**2)
    )
    omdot_o = radot*sym.sqrt(cosi**2)
    omdot   = (omdot_i + b_petro*omdot_o)/(1+b_petro)

    # compute oedot
    oedot = [
        2*f*sym.sqrt(a**3 * (1+e)/(mu*(1-e))),  # adot
        2*p*f/h,                                # edot
        p*f/(h*(sym.sqrt(1 - e**2*sinom**2) - e*sym.sqrt(cosom**2))),  # idot
        radot,
        omdot,
    ]
    
    # compute periapsis radius constraint P
    p_rp = sym.exp(k_petro*(1.0 - rp/rpmin))
    # compute scaling for each element Soe
    soe = [(1 + ((a-aT)/(m_petro*aT))**n_petro)**(1/r_petro), 1.0, 1.0, 1.0, 1.0, 1.0]
    # compute quotient Q
    #sum_term = 0
    #for idx in range(5):
    #    sum_term += woe[idx]*soe[idx]*(doe[idx]/oedot[idx])**2
    q = (1 + wp*p_rp) * (
        woe[0]*soe[0]*(doe[0]/oedot[0])**2 +
        woe[1]*soe[1]*(doe[1]/oedot[1])**2 + 
        woe[2]*soe[2]*(doe[2]/oedot[2])**2 +
        woe[3]*soe[3]*(doe[3]/oedot[3])**2 +
        woe[4]*soe[4]*(doe[4]/oedot[4])**2 
    )

    # -------- compute Gauss differential equation terms -------- #
    r = h**2/(mu*(1+e*cosTA))
    # let psi be column major!
    psi = [
        # multiplies f_r
        [
            2*a**2/h*e*sinTA,
            p/h*sinTA,
            0.0,
            0.0,
            -p/(e*h)*cosTA
        ],
        # multiplies f_theta
        [
            2*a**2/h * p/r,
            ((p+r)*cosTA + r*e)/h,
            0.0,
            0.0,
            (p+r)*sinTA/(e*h)
        ],
        # multiplies f_h
        [
            0.0,
            0.0,
            r*sym.cos(ta+om)/h,
            r*sym.sin(ta+om)/(h*sini),
            -r*sym.sin(ta+om)*cosi/(h*sini),
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
    dqdoe1 = sym.diff(q, oe[1])   # e
    dqdoe2 = sym.diff(q, oe[2])   # i 
    dqdoe3 = sym.diff(q, oe[3])   # RA
    dqdoe4 = sym.diff(q, oe[4])   # om  --- FIXME

    ux = (psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4)
    uy = (psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4)
    uz = (psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4)

    fun_lyapunov_control = lambdify(
        [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
        [ux,uy,uz], 
        "sympy",
    )

    fun_eval_psi = lambdify(
        [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
        psi, 
        "sympy",
    )

    fun_eval_dqdoe = lambdify(
        [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
        [dqdoe0,dqdoe1,dqdoe2,dqdoe3,dqdoe4], 
        "sympy",
    )
    #sym.diff(q, oe[2]), sym.diff(q, oe[4])
    return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe


# create function
print("Generating lyapunov control funcion with sympy")
fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe = quotient(mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe)

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


us = fun_lyapunov_control(
    mu_n, f_n, oe_n, oeT_n, rpmin_n, m_petro_n, n_petro_n, r_petro_n, b_petro_n, k_petro_n, wp_n, woe_n
)
print(f"us: \n{us}")
dill.dump(fun_lyapunov_control, open("../tests/fun_lyapunov_control", "wb"))
print("Success!")