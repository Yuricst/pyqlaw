# Providing dynamic efficiency threshold as function


## Idea

We can give the efficiency threshold as a function with the signature:

```python
def eta_r(t,oe,mass,battery):
    # ... compute eta as a function of the inputs ...
    return eta_r
```

For example, this can be a function based on the battery level:

```python
def eta_r(t,oe,mass,battery):
    return min(400/(4*battery - battery_dod), 1)
```

## Full code

We first do our imports and define the initial/final state and weights

```python
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import time
import pyqlaw

tstart = time.time()

# initial and final elements: [a,e,i,RAAN,omega,ta]
LU = 42164.0
GM_EARTH = 398600.44
VU = np.sqrt(GM_EARTH/LU)
TU = LU/VU

rp_gto = 200 + 6378
ra_gto = 35786 + 6378
sma_gto = (rp_gto + ra_gto)/(2*LU)
ecc_gto = (ra_gto - rp_gto)/(ra_gto + rp_gto)
KEP0 = [sma_gto,ecc_gto,np.deg2rad(23),0,0,0]
KEPF = [1,0,np.deg2rad(3),0,0,0]
oe0 = pyqlaw.kep2mee_with_a(np.array(KEP0))
oeT = pyqlaw.kep2mee_with_a(np.array(KEPF))
woe = [3.0, 1.0, 1.0, 1.0, 1.0]
# spacecraft parameters
MU = 1500
tmax_si = 1.0    # 1 N
isp_si  = 1500   # seconds
mdot_si = tmax_si/(isp_si*9.81)  # kg/s

# non-dimensional quantities
mass0 = 1.0
tmax = tmax_si * (1/MU)*(TU**2/(1e3*LU))
mdot = np.abs(mdot_si) *(TU/MU)
tf_max = 10000.0
t_step = np.deg2rad(15)
```

Suppose we want to define eta as a function of battery; we define some battery related quantities

```python
# battery levels
battery_initial = 3000*3600/TU            # Wh --> Ws --> W.TU
battery_dod = 500*3600/TU
battery_capacity = (battery_dod,battery_initial)
charge_rate = 1500          # W
discharge_rate = 500        # W
battery_charge_discharge_rate = (charge_rate, discharge_rate)
require_full_recharge = True
```

We now construct the `QLaw` object, and set the problem with information about the battery. 

```python
# construct problem
prob = pyqlaw.QLaw(
    integrator="rk4", 
    elements_type="mee_with_a",
    verbosity=2,
    print_frequency=500,
    use_sundman = True,
)

# set problem
prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step,
    battery_initial = battery_initial,
    battery_capacity = battery_capacity,
    battery_charge_discharge_rate = battery_charge_discharge_rate,
    require_full_recharge = require_full_recharge,
    woe = woe)
prob.pretty()
```

We now define the functions `eta_a()` and `eta_r()` that will be called at each step of the integration. Here, we make use of the relative efficiency only, so the `eta_a()` function is merely a placeholder (we might as well pass `eta_a=0.0` later), but is still defined as a function for the sake of demonstration. 

```python
# efficiency functiond
def eta_a(t,oe,mass,battery):
    return 0.0    # not using absolute efficiency threshold

def eta_r(t,oe,mass,battery):
    return min(400/(4*battery - battery_dod), 1)
```

Finally, we call the `solve()` method of `prob`, passing the functions `eta_a` and `eta_r` as arguments

```python
# solve
tstart_solve = time.time()
prob.solve(eta_a=eta_a, eta_r=eta_r)
prob.pretty_results()
tend = time.time()
print(f"Simulation took {tend-tstart:4.4f} seconds")
print(f"prob.solve took {tend-tstart_solve:4.4f} seconds")
```

and we can visualize the results

```python
# plot
fig1, ax1 = prob.plot_elements_history(to_keplerian=True, TU=TU/86400, time_unit_name="day")
fig2, ax2 = prob.plot_trajectory_3d(sphere_radius=0.1)
fig3, ax3 = prob.plot_controls()
fig4, ax4 = prob.plot_battery_history(TU=TU/86400, BU=TU/3600,
    time_unit_name="day", battery_unit_name="Wh")
fig5, ax5 = prob.plot_efficiency(TU=TU/86400, time_unit_name="day")
```