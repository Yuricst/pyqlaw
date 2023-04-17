"""
Tristan test
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
import pyqlaw

# initial and final elements (always in order: [SMA, ECC, INC, RAAN, AOP, TA])
oe0 = pyqlaw.kep2mee_with_a(np.array([1.0, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3]))
oeT = pyqlaw.kep2mee_with_a(np.array([1.1, 5e-3, 0.2, 0.0, 0.0, 0.0])) # -> add target's initial true anomaly
woe = [1.0, 1.0, 1.0, 1.0, 1.0]

# spacecraft parameters
mass0 = 1.0
tmax = 1e-3
mdot = 1e-4
tf_max = 300.0
t_step = 0.1

# Construct the problem object
prob = pyqlaw.QLaw(integrator="rkf45")
prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
prob.pretty()

# Solve the problem
prob.solve()
prob.pretty_results()   # print info

# Plots
fig1, ax1 = prob.plot_elements_history()
fig2, ax2 = prob.plot_trajectory_3d()
plt.show()