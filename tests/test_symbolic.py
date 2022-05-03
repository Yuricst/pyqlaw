"""
Test for constructing QLaw object
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import pyqlaw


def test_symbolic():
    pyqlaw.symbolic_qlaw_keplerian()
    pyqlaw.symbolic_qlaw_mee_with_a()
    return


if __name__=="__main__":
    test_symbolic()
    print("Done!")