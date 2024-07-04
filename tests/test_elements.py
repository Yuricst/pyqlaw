"""Tests for elements helper functions"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pyqlaw

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
    KEP0 = np.array([sma_gto,ecc_gto,np.deg2rad(23),0.3,2.1,3.06])

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
    assert all(np.abs(kep_converted_back - KEP0) <= 1e-12)
    
    # pure python ver for coverage purposes
    sv = pyqlaw.kep2sv.py_func(KEP0, GM_EARTH)
    kep_converted_back = pyqlaw.sv2kep.py_func(sv, GM_EARTH)
    assert all(np.abs(kep_converted_back - KEP0) <= 1e-12)


def test_kep_mee_a():
    # initial and final elements: [a,e,i,RAAN,omega,ta]
    LU = 42164.0
    GM_EARTH = 398600.44
    VU = np.sqrt(GM_EARTH/LU)
    TU = LU/VU

    rp_gto = 200 + 6378
    ra_gto = 35786 + 6378
    sma_gto = (rp_gto + ra_gto)/(2*LU)
    ecc_gto = (ra_gto - rp_gto)/(ra_gto + rp_gto)
    KEP0 = np.array([sma_gto,ecc_gto,np.deg2rad(23),0.3,2.1,3.06])
    kep_converted_back = pyqlaw.mee_with_a2kep(pyqlaw.kep2mee_with_a(KEP0))
    assert all(np.abs(kep_converted_back - KEP0) <= 1e-12)


if __name__ == "__main__":
    test_kep_sv()
    test_kep_mee_a()
    print("Done!")