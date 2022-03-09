import dill


if __name__=="__main__":
    # assign numerical values
    # a_n, e_n, i_n, ra_n, om_n, ta_n = 1.0,0.02, 0.2,0.3,0.4,0.5
    # aT_n, eT_n, iT_n, raT_n, omT_n  = 1.4,0.1,0.7,0.8,0.9
    a_n, e_n, i_n, ra_n, om_n, ta_n  = 1.0, 1e-2, 1e-4, 1e-4, 1e-4, 1e-4
    aT_n, eT_n, iT_n, raT_n, omT_n   = 1.1, 0.05, 1e-4, 0.0, 0.0
    oe_n = [a_n, e_n, i_n, ra_n, om_n, ta_n]
    oeT_n = [aT_n, eT_n, iT_n, raT_n, omT_n]
    mu_n = 1.0
    f_n = 0.001
    rpmin_n = 0.8
    m_petro_n = 3.0
    n_petro_n = 4.0
    r_petro_n = 2.0
    b_petro_n = 0.01
    k_petro_n = 1.0
    wp_n = 1.0
    woe_n = [1.0, 1.0, 0.0, 0.0, 0.0]

    fun_lyapunov_control = dill.load(open("fun_lyapunov_control", "rb"))
    # run the function as a test
    us = fun_lyapunov_control(
        mu_n, f_n, oe_n, oeT_n, rpmin_n, m_petro_n, n_petro_n, 
        r_petro_n, b_petro_n, k_petro_n, wp_n, woe_n
    )
    print(f"res: \n{us}")
    print("Success!")