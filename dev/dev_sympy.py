"""testing development with sympy"""

import sympy as sym
from sympy import lambdify


def prepare_func():
    # define parameters
    rpmin, k_petro, m_petro, n_petro, r_petro, b_petro = sym.symbols("rpmin k_petro m_petro n_petro r_petro b_petro")
    wp = sym.symbols("Wp")
    mu = sym.symbols("mu")
    f = sym.symbols("f")
    # orbital elements
    a, e, i, ra, om, ta = sym.symbols("a e i Omega omega theta")

    # targeted orbital elements
    aT, eT, iT, raT, omT, taT = sym.symbols("a_T e_T i_T Omega_T omega_T theta_T")

    # weights on orbital elements
    wa, we, wi, wra, wom, wta = sym.symbols("w_a w_e w_i w_Omega w_omega w_theta")

    oe = [a, e, i, ra, om, ta]
    oeT = [aT, eT, iT, raT, omT, taT]
    woe = [wa, we, wi, wra, wom, wta]
    pparams = [rpmin, k_petro, m_petro, n_petro, r_petro, b_petro, wp]

    def get_thrustfunc(mu, f, oe, oeT, woe, pparams):
        # unpack elements
        a,e,i,ra,om,ta = oe
        aT, eT, iT, raT, omT, taT = oeT
        wa, we, wi, wra, wom, wta = woe
        rpmin, k_petro, m_petro, n_petro, r_petro, b_petro, wp = pparams

        doe = [a-aT, e-eT, i-iT, 
            sym.acos(sym.cos(ra - raT)),
            sym.acos(sym.cos(om - omT)),
            sym.acos(sym.cos(ta - taT)),
        ]

        rp = a*(1-e)
        p = a*(1 - e**2)
        h = sym.sqrt(a*mu*(1-e**2))
        r = h**2/(mu*(1+e*sym.cos(ta)))

        # system matrix
        psi = [
                [
                    2*a**2/h*e*sym.sin(ta),
                    p/h*sym.sin(ta),
                    0.0,
                    0.0,
                    -p/(e*h)*sym.cos(ta)
                ], 
                [
                    2*a**2/h * p/r,
                    (p+r)/h*sym.cos(ta) + r*e,
                    0.0,
                    0.0,
                    (p+r)/(e*h)*sym.sin(ta)
                ], 
                [
                    0.0,
                    0.0,
                    r*sym.cos(ta+om)/h,
                    r*sym.sin(ta+om)/(h*sym.sin(i)),
                    -r*sym.sin(ta+om)*sym.cos(i)/(h*sym.sin(i)),
                ]
        ]

        # compute omdot
        radot = p*f/(h*sym.sin(i)*(sym.sqrt(1 - e**2*sym.cos(om)**2) - e*abs(sym.sin(om))))
        cos_ta_xx = ((1-e**2)/(2*e**3)+sym.sqrt(0.25*((1-e)**2/e**3)**2 + 1/27))**(1/3) \
                    - (-(1-e**2)/(2*e**3)+sym.sqrt(0.25*((1-e)**2/e**3)**2 + 1/27))**(1/3) - 1/e
        r_xx = p/(1+e*cos_ta_xx)
        omdot_i = f/(e*h)*sym.sqrt(p**2*cos_ta_xx**2 + (p+r_xx)**2*(1-cos_ta_xx**2))
        omdot_o = radot*abs(sym.cos(i))
        omdot = (omdot_i + b_petro*omdot_o)/(1+b_petro)

        # compute oedot
        oedot = [
            2*f*sym.sqrt(a**3 * (1+e)/(mu*(1-e))),
            2*p*f/h,
            p*f/(h*(sym.sqrt(1 - e**2*sym.sin(om)**2) - e*abs(sym.cos(om)))),
            radot,
            omdot,
        ]
        
        # compute periapsis radius constraint P
        p_rp = sym.exp(k_petro*(1.0 - rp/rpmin))
        # compute scaling for each element Soe
        soe = [(1 + ((a-aT)/(m_petro*aT))**n_petro)**(1/r_petro), 1.0, 1.0, 1.0, 1.0, 1.0]
        # compute quotient Q
        q = (1 + wp*p_rp)
        for idx in range(5):
            q = q * woe[idx]*soe[idx]*(doe[idx]/oedot[idx])**2
        
        # thrust direction
        dqdoe = []
        for idx in range(5):
            dqdoe.append(sym.diff(q, oe[idx]))
        # multiply
        ux = -(psi[0][0]*dqdoe[0] + psi[0][1]*dqdoe[1] + psi[0][2]*dqdoe[2] + psi[0][3]*dqdoe[3] + psi[0][4]*dqdoe[4])
        uy = -(psi[1][0]*dqdoe[0] + psi[1][1]*dqdoe[1] + psi[1][2]*dqdoe[2] + psi[1][3]*dqdoe[3] + psi[1][4]*dqdoe[4])
        uz = -(psi[2][0]*dqdoe[0] + psi[2][1]*dqdoe[1] + psi[2][2]*dqdoe[2] + psi[2][3]*dqdoe[3] + psi[2][4]*dqdoe[4])
        fun_ux = lambdify([q,oe], ux)
        fun_uy = lambdify([q,oe], uy)
        fun_uz = lambdify([q,oe], uz)
        return [fun_ux, fun_uy, fun_uz]
    return get_thrustfunc


if __name__=="__main__":
    print("Generating function!")
    get_thrustfunc = prepare_func()

    # define
    oe  = [1.0, 0.2, 0.1,  0.0, 0.0, 0.0]
    oeT = [1.5, 0.0, 0.05, 0.0, 0.0, 0.0]
    woe = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    rpmin = 0.5
    k_petro = 1
    m_petro = 3
    n_petro = 4
    r_petro = 2
    b_petro = 0.01
    wp = 1.0
    pparams = [rpmin, k_petro, m_petro, n_petro, r_petro, b_petro, wp]

    mu = 1.0
    f  = 1.e-5

    print(get_thrustfunc(mu, f, oe, oeT, woe, pparams))

    print("Done!")