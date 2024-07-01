Theory behind Q-law
===================

(Extracted from: Y. Shimane, N. Gollins, and K. Ho, “Orbital Facility Location Problem for Satellite Constellation Servicing Depots,” J. Spacecr. Rockets, vol. 61, no. 3, pp. 808–825, May 2024, doi: [10.2514/1.A35691].(https://arc.aiaa.org/doi/10.2514/1.A35691))

Background
--------------------------------------------

Q-law is a Lyapunov-function-based feedback control law first. 
Despite its sub-optimal nature, its ability to rapidly compute multi-revolution low-thrust transfers has made it a popular tool, particularly for large-scale preliminary transfer designs. 
The feedback law consists of determining the optimal thrust direction given the current and target orbital elements. Among the target elements, only the five slow variables are used for the transfer, as the considered problem does not necessitate rendez-vous. 


Dynamics 
--------------------------------------------


The dynamics of the satellite may be expressed in terms of orbital elements via Gauss' variational equations, or the Variation of Parameters (VOP) equations. Let the state be given by $\boldsymbol{x} \in \mathbb{R}^6$, and consider $\boldsymbol{B} \in \mathbb{R}^{6 \times 3}$ and $\boldsymbol{D} \in \mathbb{R}^6$, such that

.. math::
    \dot{\boldsymbol{x}} = \boldsymbol{B}(\boldsymbol{x})\boldsymbol{F} + \boldsymbol{D} (\boldsymbol{x})


where $\boldsymbol{F} \in \mathbb{R}^3$ is the perturbing force expressed in terms of radial, tangential, and normal components

.. math::

    \boldsymbol{F} = [F_r, F_{\theta}, F_n]^{\mathrm{T}}

The set of elements $\boldsymbol{x}$ may be Keplerian elements $[a,e,i,\Omega,\omega,\theta]$ or other alternative sets of elements.
While the original Q-law has been developed in terms of Keplerian elements, the well-known singularities at $i=0$ and $e=0$ are problematic as these are typical orbital elements in which a spacecraft may be placed. For this reason, the use of alternative element sets, such as the modified equinoctial elements (MEE), given in terms of Keplerian elements as 

.. math::
    \begin{aligned}
        & p = a(1-e^2)   \\
        & f = e \cos{(\Omega + \omega)}  \\
        & g = e \sin{(\Omega + \omega)}\\
        & h = \tan{\left(\frac{i}{2}\right)} \cos \Omega  \\
        & k = \tan{\left(\frac{i}{2}\right)} \sin \Omega  \\
        & L = \Omega + \omega + \theta
    \end{aligned}


is beneficial as the singularity is moved to $i = \pi$. 
Furthermore, the use of the MEE with the semi-parameter $p$ replaced by the semimajor axis $a$ has been previously reported to yield convergence benefits \cite{Varga2016}, and is employed in this work as well. 
For this set of elements $\boldsymbol{x} = [a,f,g,h,k,L]$, the VOP are given by

.. math::

    \boldsymbol{B}(\boldsymbol{x}) = \left[\begin{array}{ccc}
        \dfrac{2 a^{2}}{h} e \sin \theta & \dfrac{2 a^{2}p}{hr} & 0 
        \\[0.1em]
        \sqrt{\dfrac{p}{\mu}} \sin L & \sqrt{\dfrac{p}{\mu}} \dfrac{1}{w}[(w+1) \cos L+f] & -\sqrt{\dfrac{p}{\mu}} \dfrac{g}{w}[h \sin L-k \cos L] 
        \\[0.1em]
        -\sqrt{\dfrac{p}{\mu}} \cos L & \sqrt{\dfrac{p}{\mu}} \dfrac{1}{w}[(w+1) \sin L+g] & \sqrt{\dfrac{p}{\mu}} \dfrac{f}{w}[h \sin L-k \cos L] 
        \\[0.1em]
        0 & 0 & \sqrt{\dfrac{p}{\mu}} \dfrac{s^{2}}{2 w}  \cos L
        \\[0.1em]
        0  & 0 & \sqrt{\dfrac{p}{\mu}} \dfrac{s^{2}}{2 w} \sin L
        \\[0.1em]
        0 & 0  & \sqrt{\dfrac{p}{\mu}} \dfrac{1}{w} [h \sin L-k \cos L]
    \end{array}\right]

and

.. math::

    \boldsymbol{D}(\boldsymbol{x}) =\left[\begin{array}{llllll}
    0 & 0 & 0 & 0 & 0 & \sqrt{\mu p}\left(\dfrac{w}{p}\right)^{2}
    \end{array}\right]^{\mathrm{T}}


.. math::

    \begin{aligned}
        w &= 1 + f \cos L + g \sin L
        \\
        s^2 &= 1 + h^2 + k^2
    \end{aligned}


Note that $\boldsymbol{F}$ can be due to any form of perturbing acceleration, such as propulsive force, atmospheric drag, third-body attraction, or non-spherical gravity. 
The acceleration due to the propulsive force, which is to be determined to guide the spacecraft to its target orbit, is given by

.. math::

    \boldsymbol{F} = \dfrac{T}{m}\left[
        \cos \alpha \cos \beta , \,
        \sin \alpha \cos \beta , \,
        \sin \beta
    \right]^{\mathrm{T}}


where :math:`\alpha` and :math:`\beta` are the in-plane and out-of-plane angles, respectively. 


Control Lyapunov function
--------------------------------------------

Denoting the osculating element :math:`\boldsymbol{\alpha} \in [a,f,g,h,k]$` and the targeted elements as :math:`\text{\boldsymbol{\alpha}}_T`, the Lyapunov function is given by

.. math::

    Q = 
    \left(1+W_{p} P\right) \sum_{\text{\boldsymbol{\alpha}}}
    S_{\text{\boldsymbol{\alpha}}} W_{\text{\boldsymbol{\alpha}}}
    \left(
        \dfrac{\text{\boldsymbol{\alpha}} - \text{\boldsymbol{\alpha}}_T}{\dot{\text{\boldsymbol{\alpha}}}_{x x}}
    \right)^{2}
    \, , 
    \quad \text{\boldsymbol{\alpha}} = a, f, g, h, k


In essence, :math:`Q` penalizes the difference between :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\alpha}_{T}` through the subtraction term in the summation. 
:math:`S_{\boldsymbol{\alpha}}` is a scaling factor given by

.. math::

    S_{\boldsymbol{\alpha}} 
    =
    \begin{cases}
        \left[1+\left(\dfrac{\left|a-a_{T}\right|}{\sigma a_{T}}\right)^{\nu}\right]^{1 / \zeta},
        & \boldsymbol{\alpha} = a
        \\
        1, & \text{ otherwise }
    \end{cases}


where :math:`\sigma`, :math:`\nu` and :math:`\zeta` are scalar coefficients, which prevents non-convergence of $a \to \infty$. This is necessary as when :math:`a \to \infty`, the :math:`\dot{\boldsymbol{\alpha}}_{x x}` terms also tend to :math:`\infty` and thus :math:`Q` is reduced, however, this is not a physically useful solution. :math:`W_{\boldsymbol{\alpha}}` are scalar weights that may be assigned to different elements, if one is to be favored for targeting over another. :math:`P` is a penalty term on the periapsis radius, given by

.. math::

    P =\exp \left[k_{r_p}\left(1-\frac{r_{p}}{r_{p \min }}\right)\right]


where :math:`r_p` is the current orbit's periapsis radius given by 

.. math::

    r_p = a (1 - e)


and :math:`r_{p \min }` is a user-defined threshold. Here, :math:`k_{r_p}` is also a pre-defined constant on this penalty term and represents the gradient of this exponential barrier function near :math:`r_{p \min }`. :math:`W_p` is a scalar weight to be placed on the periapsis penalty term. 
The :math:`\dot{\boldsymbol{\alpha}}_{x x}` terms represent the maximum rates of change of a given orbital element with respect to the thrust direction and true anomaly along the osculating orbit and are given in the Appendix.

Derivation of Control Law
--------------------------------------------

Through the application of Lyapunov control theory, the Q-law strategy consists of choosing the control angles :math:`\alpha` and :math:`\beta` such that the time-rate of change of $Q$ is minimized at each time-step

.. math::

    \min_{\alpha, \beta} \dot{Q}


where $\dot{Q}$ can be expressed using the chain rule as 

.. math::

    \begin{aligned}
        \dot{Q} &= \sum_{\boldsymbol{\alpha} } \dfrac{\partial Q}{\partial \boldsymbol{\alpha}} \dot{\boldsymbol{\alpha}}
        % first term
        = D_1 \cos \beta \cos \alpha 
        % second term
        + D_2 \cos \beta \sin \alpha 
        % third term
        + D_3 \sin \beta
        % elements
        \,, \quad \boldsymbol{\alpha}=a, f, g, h, k
    \end{aligned}


where

.. math::

    \begin{aligned}
        D_1 &= \sum_{\boldsymbol{\alpha}} \dfrac{\partial Q}{\partial \boldsymbol{\alpha}} \dfrac{\partial \dot{\boldsymbol{\alpha}}}{\partial F_{\theta}}
        \\
        D_2 &= \sum_{\boldsymbol{\alpha}} \dfrac{\partial Q}{\partial \boldsymbol{\alpha}} \dfrac{\partial \dot{\boldsymbol{\alpha}}}{\partial F_{r}}
        \\
        D_3 &= \sum_{\boldsymbol{\alpha}} \dfrac{\partial Q}{\partial \boldsymbol{\alpha}} \dfrac{\partial \dot{\boldsymbol{\alpha}}}{\partial F_{n}}
    \end{aligned}


The choice of $\alpha$ and $\beta$ based on condition \eqref{eq:minQ_def}, given by

.. math::

    \begin{aligned}
        \alpha^* &= \arctan(-D_2, -D_1)
        \\
        \beta^* &= \arctan\left( \dfrac{-D_3}{\sqrt{D_1^2 + D_2^2}} \right)
    \end{aligned}


ensures the fastest possible decrease of $Q$, thereby providing the best immediate action for the spacecraft to take to arrive at $\boldsymbol{\alpha}_T$. 
Note that while $\dot{\boldsymbol{\alpha}}$ consists simply of the first 5 rows of the VOP given in expression \eqref{eq:vop_Amatrix}, the expression for $\frac{\partial Q}{\partial \boldsymbol{\alpha}}$ is cumbersome to derive analytically. Instead, a symbolic toolbox is used to obtain these expressions. 