---
title: 'pyqlaw: low-thrust trajectory design with Lyapunov controller in Python'
tags:
  - Python
  - astrodynamics
  - spacecraft trajectory design
authors:
  - name: Yuri Shimane
    orcid: 0000-0001-7728-3780
    affiliation: 1
  - name: Koki Ho
    orcid: 0000-XXXX-XXXX-XXXX
    affiliation: 1
affiliations:
 - name: School of Aerospace Engineering, Georgia Institute of Technology
   index: 1
date: 15 February 2025
bibliography: paper.bib
---

# Summary

With emerging space applications such as on-orbit servicing and manufacturing and active debris removal, in-orbit mobility is becoming increasingly important. 
A key enabler for mobility is low-thrust propulsion technology, which enables fuel-efficient, long duration orbit transfers to reposition the location of a satellite. 
Spacecraft transfer design is a challenging problem due to the underlying nonlinearity of the dynamics as well as the non-convexity of the resulting optimal control problem (OCP); there is a long standing literature that applies indirect, direct, or other approaches such as sequential convex programming or differential dynamic programming for tackling this challenge. 
Many of these methods benefit greatly from a strong, feasible initial guess. 
In the context of many-revolution transfers, Lyapunov controllers is a powerful approach for constructing feasible and near-optimal solutions. 
The so-called Q-law [@Petropoulos2003; @Petropoulos2004; @Petropoulos2005] consists of using a Lyapunov function defined in terms of orbital elements, and has been used extensively over the last two decades. 
Q-law can generate both the state and control history of the spacecraft, which may be used not only as initial guess for higher fidelity OCP solvers, but also to esimtate for the transfer cost and time between two orbits [@Jagannatha2020; @Shimane2023; @Apa2023], or to conduct large-scale trade-studies for parameters such as spacecraft mass or engine properties [@Lee2005; @Varga2016; @Shimane2023c]. 


# Statement of Need

`pyqlaw` is a python package of Q-law for constructing feasible low-thrust spacecraft trajectories in predominantly Keplerian environments. 
The module is implemented with standard, largely compatible python modules, lending itself well to being used in trajectory design workflows involving other softwares that includes a python interface. 
Making use of `sympy`'s common subexpression elimination [@sympy] and just-in-time compilation through `numba` [@numba], the module is able to construct many revolutions transfer, involving 10s to 100s of revolutions, on the other of seconds on a regular laptop with no parallelization. 
The `pyqlaw` module includes feedback controllers in two popular orbital elements representations: Keplerian elements and modified equinoctial elements (MEE). 

![Example transfer trajectory from GTO to GEO.\label{fig:traj}](example_3D_trajectory.png){ width=70% }

![Example state history from GTO to GEO.\label{fig:statehist}](example_state_history.png){ width=90% }

![Example control history from GTO to GEO.\label{fig:controlhist}](example_control_history.png){ width=90% }


# Acknowledgements

We acknowledge helpful comments from Umberto di Capua while developing tutorials for this package. 

# References