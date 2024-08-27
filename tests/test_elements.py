"""Tests for elements helper functions"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw


def diff_kep(KEP0, KEP1):
    return np.array([
        KEP1[0] - KEP0[0],  # a
        KEP1[1] - KEP0[1],  # e
        KEP1[2] - KEP0[2],  # i
        np.arccos(np.cos(KEP1[3] - KEP0[3])),  # RAAN
        np.arccos(np.cos(KEP1[4] - KEP0[4])),  # omega
        np.arccos(np.cos(KEP1[5] - KEP0[5])),  # ta
    ])


def test_kep_sv():
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    LU = 42164.0
    GM_EARTH = 398600.44
    VU = np.sqrt(GM_EARTH/LU)
    TU = LU/VU

    rp_gto = 200 + 6378
    ra_gto = 35786 + 6378
    sma_gto = (rp_gto + ra_gto)/(2*LU)
    ecc_gto = (ra_gto - rp_gto)/(ra_gto + rp_gto)

    KEP0s = [
        np.array([sma_gto,1e-3,np.deg2rad(23),0.3,np.deg2rad(100),3.06]),
        np.array([sma_gto,ecc_gto,np.deg2rad(23),0.3,np.deg2rad(100),3.06]),
        np.array([sma_gto,ecc_gto,np.deg2rad(23),0.3,np.deg2rad(200),3.06]),
    ]
    
    for KEP0 in KEP0s:
        # convert to SV and back
        sv = pyqlaw.kep2sv(KEP0, GM_EARTH)
        kep_converted_back = pyqlaw.sv2kep(sv, GM_EARTH)

        # jit-compiled ver
        assert np.abs(pyqlaw.get_semiMajorAxis(sv, GM_EARTH) - KEP0[0]) <= 1e-12
        assert np.abs(np.linalg.norm(pyqlaw.get_eccentricity(sv, GM_EARTH)) - KEP0[1]) <= 1e-12
        assert np.abs(pyqlaw.get_inclination(sv)             - KEP0[2]) <= 1e-12
        assert np.abs(pyqlaw.get_raan(sv)                    - KEP0[3]) <= 1e-12
        assert np.abs(pyqlaw.get_omega(sv, GM_EARTH)         - KEP0[4]) <= 1e-12
        assert np.abs(pyqlaw.get_trueanom(sv, GM_EARTH)      - KEP0[5]) <= 1e-12
        assert np.abs(pyqlaw.get_period(sv, GM_EARTH) - 2*np.pi*np.sqrt(KEP0[0]**3/GM_EARTH)) <= 1e-12
        
        # pure python ver for coverage purposes
        assert np.abs(pyqlaw.get_semiMajorAxis.py_func(sv, GM_EARTH) - KEP0[0]) <= 1e-12
        assert np.abs(np.linalg.norm(pyqlaw.get_eccentricity.py_func(sv, GM_EARTH)) - KEP0[1]) <= 1e-12
        assert np.abs(pyqlaw.get_inclination.py_func(sv)             - KEP0[2]) <= 1e-12
        assert np.abs(pyqlaw.get_raan.py_func(sv)                    - KEP0[3]) <= 1e-12
        assert np.abs(pyqlaw.get_omega.py_func(sv, GM_EARTH)         - KEP0[4]) <= 1e-12
        assert np.abs(pyqlaw.get_trueanom.py_func(sv, GM_EARTH)      - KEP0[5]) <= 1e-12
        assert np.abs(pyqlaw.get_period.py_func(sv, GM_EARTH) - 2*np.pi*np.sqrt(KEP0[0]**3/GM_EARTH)) <= 1e-12
        
        # jit-compiled ver
        sv = pyqlaw.kep2sv(KEP0, GM_EARTH)
        kep_converted_back = pyqlaw.sv2kep(sv, GM_EARTH)
        assert all(np.abs(diff_kep(kep_converted_back, KEP0)) <= 1e-12)
        
        # pure python ver for coverage purposes
        sv = pyqlaw.kep2sv.py_func(KEP0, GM_EARTH)
        kep_converted_back = pyqlaw.sv2kep.py_func(sv, GM_EARTH)
        assert all(np.abs(diff_kep(kep_converted_back, KEP0)) <= 1e-12)

        # conversion to MEE
        MEE0 = pyqlaw.kep2mee(KEP0)
        MEE0_nojit = pyqlaw.kep2mee.py_func(KEP0)
        assert all(np.abs(MEE0 - MEE0_nojit) <= 1e-15)

        KEP0_back = pyqlaw.mee2kep(MEE0)
        assert all(np.abs(diff_kep(KEP0_back, KEP0)) <= 1e-12)

        KEP0_back_nojit = pyqlaw.mee2kep.py_func(MEE0)
        assert all(np.abs(diff_kep(KEP0_back_nojit, KEP0)) <= 1e-12)

        # anomaly conversions
        EA0_direct_nojit = pyqlaw.ta2ea.py_func(KEP0[5], KEP0[1])
        EA0_nojit = pyqlaw.mee2ea.py_func(MEE0)
        assert np.abs(EA0_direct_nojit - EA0_nojit) <= 1e-15
    return
    

def test_kep_mee_a():
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    LU = 42164.0
    GM_EARTH = 398600.44
    VU = np.sqrt(GM_EARTH/LU)
    TU = LU/VU
    MU = 1.0

    rp_gto = 200 + 6378
    ra_gto = 35786 + 6378
    sma_gto = (rp_gto + ra_gto)/(2*LU)
    ecc_gto = (ra_gto - rp_gto)/(ra_gto + rp_gto)
    KEP0 = np.array([sma_gto,ecc_gto,np.deg2rad(23),0.3,2.1,0.0])
    MEEA0 = pyqlaw.kep2mee_with_a(KEP0)
    KEP_BACK = pyqlaw.mee_with_a2kep(MEEA0)
    assert all(np.abs(diff_kep(KEP_BACK, KEP0)) <= 1e-12)

    MEEA0_thru_MEE = pyqlaw.mee2mee_with_a.py_func(pyqlaw.kep2mee(KEP0))
    assert all(np.abs(MEEA0_thru_MEE - MEEA0) <= 1e-12)

    # jit-compiled ver
    kep_converted_back = pyqlaw.mee_with_a2kep(MEEA0)
    assert all(np.abs(diff_kep(kep_converted_back, KEP0)) <= 1e-12)

    # pure python ver for coverage purposes
    kep_converted_back = pyqlaw.mee_with_a2kep.py_func(pyqlaw.kep2mee_with_a.py_func(KEP0))
    assert all(np.abs(diff_kep(kep_converted_back, KEP0)) <= 1e-12)

    # get orbital states in Cartesian state over one rev
    coord_jit = pyqlaw.get_orbit_coordinates(KEP0,MU,15)
    coord     = pyqlaw.get_orbit_coordinates.py_func(KEP0,MU,15)
    assert all(np.abs(coord[:,0] - coord_jit[:,0]) <= 1e-12)

    RV0 = pyqlaw.mee_with_a2sv(MEEA0, MU)
    assert all(np.abs(coord[:,0] - RV0) <= 1e-15)

    # pure python ver for coverage purposes
    RV0_nojit = pyqlaw.mee_with_a2sv.py_func(MEEA0, MU)
    assert all(np.abs(coord[:,0] - RV0_nojit) <= 1e-15)


if __name__ == "__main__":
    test_kep_sv()
    test_kep_mee_a()
    print("Done!")