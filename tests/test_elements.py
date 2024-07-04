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
    kep_converted_back = pyqlaw.sv2kep(pyqlaw.kep2sv(KEP0, GM_EARTH), GM_EARTH)
    print(f"np.abs(kep_converted_back - KEP0) = {np.abs(kep_converted_back - KEP0)}")
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

    # convert to SV and back
    kep_converted_back = pyqlaw.mee_with_a2kep(pyqlaw.kep2mee_with_a(KEP0))
    print(f"np.abs(kep_converted_back - KEP0) = {np.abs(kep_converted_back - KEP0)}")
    assert all(np.abs(kep_converted_back - KEP0) <= 1e-12)

if __name__ == "__main__":
    test_kep_sv()
    test_kep_mee_a()
    print("Done!")