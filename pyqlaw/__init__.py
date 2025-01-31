"""
init file for module
"""

__copyright__    = 'Copyright (C) 2025 Yuri Shimane'
__version__      = '0.2.1'
__license__      = 'MIT License'
__author__       = 'Yuri Shimane'
__author_email__ = 'yuri.shimane@gatech.edu'
__url__          = 'https://github.com/Yuricst/pyqlaw'


# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("numpy", "matplotlib", "scipy", "numba", "sympy", "tqdm", "spiceypy")
_missing_dependencies = []
for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies


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
from ._lyapunov import (
    _u_to_thrust_angles,
    lyapunov_control_angles
)
from ._elements import (
    get_semiMajorAxis,
    get_eccentricity,
    get_inclination,
    get_raan,
    get_omega,
    get_trueanom,
    get_period,
    sv2kep, 
    kep2sv,
    get_orbit_coordinates,
    kep2mee,
    mee2kep,
    mee2mee_with_a,
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
from ._plot_helper import (
    get_sphere_coordinates,
    plot_sphere_wireframe,
    get_ellipsoid_coordinates,
    plot_ellipsoid_wireframe,
    set_equal_axis,
)
from ._plot_methods import (
    plot_elements_history,
)
from ._perturbations import (
    perturbation_third_body_battin,
    perturbation_J2,
    pxformECI2RTN,
    SpicePerturbations,
)
from ._qlaw import QLaw