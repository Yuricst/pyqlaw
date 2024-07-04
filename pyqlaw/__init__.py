"""
init file for module
"""

__copyright__    = 'Copyright (C) 2023 Yuri Shimane'
__version__      = '0.1.1'
__license__      = 'MIT License'
__author__       = 'Yuri Shimane'
__author_email__ = 'yuri.shimane@gatech.edu'
__url__          = 'https://github.com/Yuricst/pyqlaw'


from ._symbolic import (
    symbolic_qlaw_keplerian,
    symbolic_qlaw_mee_with_a,
)
from ._eom import (
    eom_kep_gauss, 
    eom_mee_with_a_gauss,
)
from ._integrate import (
    rk4, 
    rkf45,
)
from ._lyapunov import lyapunov_control_angles
from ._elements import (
    sv2kep, 
    kep2sv,
    kep2mee,
    mee2kep,
    kep2mee_with_a,
    mee_with_a2kep,
    mee_with_a2sv,
    ta2ea,
    mee2ea,
)
from ._convergence import (
    check_convergence_keplerian,
    check_convergence_mee,
    elements_safety
)
from ._qlaw import QLaw
from ._plot_helper import (
    get_sphere_coordinates,
    plot_sphere_wireframe,
    get_ellipsoid_coordinates,
    plot_ellipsoid_wireframe,
    set_equal_axis,
)