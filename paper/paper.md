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

With emerging space applications such as on-orbit servicing and manufacturing and active debris removal, in-orbit mobility is becoming increasingly important. 
A key enabler for mobility is low-thrust propulsion technology, which enables fuel-efficient, long duration orbit transfers to reposition the location of a satellite. 
Spacecraft transfer design is a challenging problem due to the underlying nonlinearity of the dynamics as well as the non-convexity of the resulting optimal control problem; there is a long standing literature that applies indirect, direct, or other approaches such as sequential convex programming or differential dynamic programming for tackling this challenge. 
Many of these methods benefit greatly from a strong, feasible initial guess. 


In recent years, low-thrust propulsion technology has matured to the point that there is a paradigm shift in space mission design. 


Q-law [Petropoulos2003],[@Petropoulos2004],[Petropoulos2005]


# Acknowledgements

We acknowledge helpful comments from Umberto di Capua while making use of this package. 

# References