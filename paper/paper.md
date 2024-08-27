---
title: 'pyqlaw: Low-Thrust Trajectory Design with Lyapunov Controller in Python'
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
date: 27 August 2024
bibliography: paper.bib
---

# Summary

This paper describes the `pyqlaw` module, an open-source implementation of a Lyapunov controller for constructing feasible low-thrust spacecraft trajectories in predominantly Keplerian environments. 
The module is implemented with standard, largely compatible python modules, lending itself well to being used in trajectory design workflows involving other softwares that includes a python interface. 
Making use of `sympy`'s common subexpression elimination [@sympy] and just-in-time compilation through `numba` [@numba], the module is able to construct many revolutions transfer, involving 10s to 100s of revolutions, on the other of seconds on a regular laptop with no parallelization. 


# Statement of Need

With emerging space applications such as on-orbit servicing and manufacturing and active debris removal, in-orbit mobility is becoming increasingly important. 
A key enabler for mobility is low-thrust propulsion technology, which enables fuel-efficient, long duration orbit transfers to reposition the location of a satellite. 
Spacecraft transfer design is a challenging problem due to the underlying nonlinearity of the dynamics as well as the non-convexity of the resulting optimal control problem; there is a long standing literature that applies indirect, direct, or other approaches such as sequential convex programming or differential dynamic programming for tackling this challenge. 
Many of these methods benefit greatly from a strong, feasible initial guess. 
In the context of many-revolution transfers, Lyapunov controllers is a powerful approach for constructing feasible and near-optimal solutions. 
The so-called Q-law [@Petropoulos2003] [@Petropoulos2004] [@Petropoulos2005] consists of using a Lyapunov function defined in terms of orbital elements, and has been used extensively over the last two decades. 

The `pyqlaw` module implements the Q-law scheme in two popular orbital elements representations: Keplerian elements and modified equinoctial elements (MEE). 

Examples are provided in the `pyqlaw` repository. 

![](https://raw.githubusercontent.com/yuricst/pyqlaw/main/paper/scan_lines.png)
**Figure 1.** SONIC can be used to quickly perform a scan and do analysis on an image.

# Acknowledgements

We acknowledge helpful comments from Umberto di Capua while making use of this package. 

# References