# pyqlaw
Q-law feedback control for low-thrust orbital transfer in Python

### Dependencies

- `sympy`, `numpy`, `tqdm`, `matplotilb`, `numba`, `dill`

### Basic usage

Test files are included in `./tests/`. Here, we present a basic example. Before starting, please note a couple of things:

- Dynamics & spacecraft parameters are given in non-dimensional quantities, scaling `GM = 1.0` (which may be modified, but it is numerically desirable to use this scaling). 
- All angles are defined in radians.
- Due to instability of Gauss's equation and the Lyapunov feedback control law, some of the elements should not be smaller in magnitude than a certain safe-guarding threshold value. This is why some of the starting elements in the following example are not set to 0, but a mild value (e.g. `1e-2`, `1e-3`). 

We start by importing the module

```python
#import sys
#sys.path.append("../")  # make sure the pyqlaw folder is exposed
import pyqlaw
```

Construct initial and final Keplrian elements to target, along with weighting

```python
# initial and final elements (always in order: [SMA, ECC, INC, RAAN, AOP, TA])
oe0 = np.array([1.0, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
oeT = np.array([1.1, 5e-3, 0.2, 0.0, 0.0])
woe = [1.0, 1.0, 1.0, 1.0, 0.0]
```

Provide spacecraft parameters (max thrust and mass-flow rate), max transfer time, and time-step (note that integration is done using fixed-steps):

```python
# spacecraft parameters
mass0 = 1.0
tmax = 1e-3
mdot = 1e-4
tf_max = 300.0
t_step = 0.1
```

Construct the problem object, then set the problem parameters

```python
prob = pyqlaw.QLaw()
prob.set_problem(oe0, oeT, mass0, tmax, mdot, tf_max, t_step, woe=woe)
prob.pretty()  # print info
```

```
Transfer:
  sma  : 1.0000e+00 -> 1.1000e+00 (weight: 1.00)
  ecc  : 1.0000e-02 -> 5.0000e-03 (weight: 1.00)
  inc  : 1.0000e-02 -> 2.0000e-01 (weight: 1.00)
  raan : 1.0000e-03 -> 0.0000e+00 (weight: 1.00)
  aop  : 1.0000e-03 -> 0.0000e+00 (weight: 0.00)
```

solve the problem

```python
prob.solve()
prob.pretty_results()   # print info
```

```
qlaw:  99%|█████████████████████████████████████████████████████████████████▏| 2970/3000 [03:03<00:01, 16.20it/s]
Target elements successfully reached!
Final state:
  sma  : 1.1000e+00 (error: 1.8297e-05)
  ecc  : 5.0141e-03 (error: 1.4138e-05)
  inc  : 1.9994e-01 (error: 5.7253e-05)
  raan : 4.5828e-04 (error: 4.5828e-04)
  aop  : 1.2206e-01 (error: 1.2206e-01)
```

Some conveninence methods for plotting:

```python
fig1, ax1 = prob.plot_elements_history()
fig2, ax2 = prob.plot_trajectory_3d()
```

<p align="center">
  <img src="./plots//transfer_eg_3dtraj.png" width="400" title="transfer">
</p>


### To-dos
- [x] Effectivity for coasting
- [x] Version using MEE (See [4-6])
- [ ] Robustify numerical unstability


### Some things to be careful!

- Gauss's equation in terms of Keplerian elements is particularly unstable when eccentricity (e) and inclination (i) approach 0


### References

[1] Petropoulos, A. E. (2003). Simple Control Laws for Low-Thrust Orbit Transfers. AAS Astrodynamics Specialists Conference.

[2] Petropoulos, A. E. (2004). Low-thrust orbit transfers using candidate Lyapunov functions with a mechanism for coasting. AIAA/AAS Astrodynamics Specialist Conference, August. https://doi.org/10.2514/6.2004-5089

[3] Petropoulos, A. E. (2005). Refinements to the Q-law for low-thrust orbit transfers. Advances in the Astronautical Sciences, 120(I), 963–982.

[4] Shannon, J. L., Ozimek, M. T., Atchison, J. A., & Hartzell, C. M. (2020). Q-law aided direct trajectory optimization of many-revolution low-thrust transfers. Journal of Spacecraft and Rockets, 57(4), 672–682. https://doi.org/10.2514/1.A34586

[5] Leomanni, M., Bianchini, G., Garulli, A., Quartullo, R., & Scortecci, F. (2021). Optimal Low-Thrust Orbit Transfers Made Easy: A Direct Approach. Journal of Spacecraft and Rockets, 1–11. https://doi.org/10.2514/1.a34949

[6] [Modified Equinoctial Elements](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf)
