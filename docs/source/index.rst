.. trajplotlib documentation master file, created by
   sphinx-quickstart on Tue Jul 20 18:46:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyqlaw's documentation!
=======================================

``pyqlaw`` is a python implementation of the Q-law feedback control for low-thrust orbital transfers. 

Capabilities
------------

- Q-law formulated in Keplerian & SMA-MEE (MEE with semilatus rectum replaced by semimajor axis)
- Coasting capabilities with efficiency parameters [3]
- Thrust duty cycles
- Battery level tracking

Installation
------------

To install, run::

   pip install pyqlaw

and to uninstall::
   
   pip uninstall pyqlaw
   

Overview of Q-law
-----------------

Q-law is very sensitive to the problem (initial & final orbital elements, choice of orbital elements, thruster specs = control authority) as well as its various hyperparamters, which must be chosen carefully. 
In general, the following should be kept in mind:

- For numerical stability, always work with canonical scales.
- Be very careful with initial/final orbits not to contain singular elements (e.g. inclination ~ 0 deg in Keplerian elements representation).
- Q-law is not suitable for high control authority applications (e.g. interplanetary transfer with 0~very few revolutions).
- Taking larger integration time steps `t_step` (or angle steps, if `use_sundman = True`) makes the algorithm "faster" (less time until reaching the targeted elements), but may also lead to instability/high jitter once the spacecraft is close to the target; an appropriate value must be found on a problem-to-problem basis.


For more discussions, see for example: 

- Petropoulos, A. E. (2004). Low-thrust orbit transfers using candidate Lyapunov functions with a mechanism for coasting. Collection of Technical Papers - AIAA/AAS Astrodynamics Specialist Conference, 2(August), 748–762. https://doi.org/10.2514/6.2004-5089
- Petropoulos, A. E. (2005). Refinements to the Q-law for low-thrust orbit transfers. AAS/AIAA Space Flight Mechanics Meeting.
- Hatten, N. (2012). A Critical Evaluation of Modern Low-Thrust, Feedback-Driven Spacecraft Control Laws.


Some References
---------------

[1] Petropoulos, A. E. (2003). Simple Control Laws for Low-Thrust Orbit Transfers. AAS Astrodynamics Specialists Conference.

[2] Petropoulos, A. E. (2004). Low-thrust orbit transfers using candidate Lyapunov functions with a mechanism for coasting. AIAA/AAS Astrodynamics Specialist Conference, August. https://doi.org/10.2514/6.2004-5089

[3] Petropoulos, A. E. (2005). Refinements to the Q-law for low-thrust orbit transfers. Advances in the Astronautical Sciences, 120(I), 963–982.

[4] Shannon, J. L., Ozimek, M. T., Atchison, J. A., & Hartzell, C. M. (2020). Q-law aided direct trajectory optimization of many-revolution low-thrust transfers. Journal of Spacecraft and Rockets, 57(4), 672–682. https://doi.org/10.2514/1.A34586

[5] Leomanni, M., Bianchini, G., Garulli, A., Quartullo, R., & Scortecci, F. (2021). Optimal Low-Thrust Orbit Transfers Made Easy: A Direct Approach. Journal of Spacecraft and Rockets, 1–11. https://doi.org/10.2514/1.a34949

[6] [Modified Equinoctial Elements (careful with typos in this document!)](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf)

[7] Hatten, N. (2012). A Critical Evaluation of Modern Low-Thrust, Feedback-Driven Spacecraft Control Laws.



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   theory
   api

.. toctree::
   :maxdepth: 1
   :caption: Basic Examples:

   examples/ex_GTO2GEO.ipynb
   examples/ex_LEO_transfer.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced Examples:

   examples/ex_eta_functions.ipynb
   examples/ex_thruster_dutycycle.ipynb
   examples/ex_gto_manifold.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`