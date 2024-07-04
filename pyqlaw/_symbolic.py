"""
Symbolic derivation for feedback control law
"""

"""
sympy dev
"""

import numpy as np
import sympy as sym
from sympy import lambdify


def symbolic_qlaw_keplerian(cse = True):
    """Generate symbolic function for Keplerian Q-Law"""
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
        dqdoe = [dqdoe0, dqdoe1, dqdoe2, dqdoe3, dqdoe4]

        ux = (psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4)
        uy = (psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4)
        uz = (psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4)

        umag = sym.sqrt(ux**2 + uy**2 + uz**2)
        ux_unit = ux/umag
        uy_unit = uy/umag
        uz_unit = uz/umag

        dqdt = f * (
            dqdoe0*(psi[0][0]*ux_unit + psi[1][0]*uy_unit + psi[2][0]*uz_unit) +\
            dqdoe1*(psi[0][1]*ux_unit + psi[1][1]*uy_unit + psi[2][1]*uz_unit) +\
            dqdoe2*(psi[0][2]*ux_unit + psi[1][2]*uy_unit + psi[2][2]*uz_unit) +\
            dqdoe3*(psi[0][3]*ux_unit + psi[1][3]*uy_unit + psi[2][3]*uz_unit) +\
            dqdoe4*(psi[0][4]*ux_unit + psi[1][4]*uy_unit + psi[2][4]*uz_unit))

        fun_lyapunov_control = lambdify(
            [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            [[ux,uy,uz], psi, q, dqdoe], 
            "numpy",
            cse = cse,
        )

        fun_eval_psi = lambdify(
            [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            psi, 
            "numpy",
            cse = cse,
        )

        fun_eval_dqdoe = lambdify(
            [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            [dqdoe0,dqdoe1,dqdoe2,dqdoe3,dqdoe4], 
            "numpy",
            cse = cse,
        )

        fun_eval_dqdt = lambdify(
            [mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            dqdt, 
            "numpy",
            cse = cse,
        )
        #sym.diff(q, oe[2]), sym.diff(q, oe[4])
        return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_dqdt

    # create function
    #print("Generating Keplerian lyapunov control funcion with sympy")
    fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_dqdt = quotient(mu, f, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe)
    return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_dqdt



def symbolic_qlaw_mee_with_a(cse = True):
    """Generate symbolic function for Keplerian Q-Law"""
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
        dqdoe = [dqdoe0, dqdoe1, dqdoe2, dqdoe3, dqdoe4]

        # # compute thrust vector components
        # u_unscaled = [
        #     psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4,
        #     psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4,
        #     psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4
        # ]
        # u_unscaled_norm = sym.sqrt(u_unscaled[0]**2 + u_unscaled[1]**2 + u_unscaled[2]**2)
        # ux = u_unscaled[0]/u_unscaled_norm
        # uy = u_unscaled[0]/u_unscaled_norm
        # uz = u_unscaled[0]/u_unscaled_norm

        # # compute thrust angles
        # alpha = sym.atan2(-ux,-uy)
        # beta = sym.atan(-uz/sym.sqrt(ux**2 + uy**2))

        # # compute thrust vector
        # # thrust_vec = [
        # #     accel* sym.cos(beta)*sym.sin(alpha),
        # #     accel* sym.cos(beta)*sym.cos(alpha),
        # #     accel* sym.sin(beta),
        # # ]

        # fun_lyapunov_control = lambdify(
        #     [
        #         mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, 
        #         b_petro, k_petro, wp, woe
        #     ], 
        #     [psi, qdot, alpha, beta], 
        #     "numpy",
        # )

        ux = (psi[0][0]*dqdoe0 + psi[0][1]*dqdoe1 + psi[0][2]*dqdoe2 + psi[0][3]*dqdoe3 + psi[0][4]*dqdoe4)
        uy = (psi[1][0]*dqdoe0 + psi[1][1]*dqdoe1 + psi[1][2]*dqdoe2 + psi[1][3]*dqdoe3 + psi[1][4]*dqdoe4)
        uz = (psi[2][0]*dqdoe0 + psi[2][1]*dqdoe1 + psi[2][2]*dqdoe2 + psi[2][3]*dqdoe3 + psi[2][4]*dqdoe4)
        
        umag = sym.sqrt(ux**2 + uy**2 + uz**2)
        ux_unit = ux/umag
        uy_unit = uy/umag
        uz_unit = uz/umag

        dqdt = accel * (
            dqdoe0*(psi[0][0]*ux_unit + psi[1][0]*uy_unit + psi[2][0]*uz_unit) +\
            dqdoe1*(psi[0][1]*ux_unit + psi[1][1]*uy_unit + psi[2][1]*uz_unit) +\
            dqdoe2*(psi[0][2]*ux_unit + psi[1][2]*uy_unit + psi[2][2]*uz_unit) +\
            dqdoe3*(psi[0][3]*ux_unit + psi[1][3]*uy_unit + psi[2][3]*uz_unit) +\
            dqdoe4*(psi[0][4]*ux_unit + psi[1][4]*uy_unit + psi[2][4]*uz_unit))

        fun_lyapunov_control = lambdify(
            [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            [[ux, uy, uz], psi, q, dqdoe], 
            "numpy",
            cse = cse,
        )

        fun_eval_psi = lambdify(
            [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            psi, 
            "numpy",
            cse = cse,
        )

        fun_eval_dqdoe = lambdify(
            [
                mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, 
                b_petro, k_petro, wp, woe
            ], 
            [dqdoe0,dqdoe1,dqdoe2,dqdoe3,dqdoe4], 
            "numpy",
            cse = cse,
        )

        fun_eval_dqdt = lambdify(
            [mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe], 
            dqdt, 
            "numpy",
            cse = cse,
        )
        return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_dqdt

    # create function
    #print("Generating MEE-SMA lyapunov control funcion with sympy")
    fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_dqdt = quotient(mu, accel, oe, oeT, rpmin, m_petro, n_petro, r_petro, b_petro, k_petro, wp, woe)
    return fun_lyapunov_control, fun_eval_psi, fun_eval_dqdoe, fun_eval_dqdt
