# The algorithm

spheropack implements the Lubachevsky-Stillinger (LS) algorithm
{cite:p}`lubachevsky1990`: an event-driven molecular dynamics simulation of hard
spheres whose radii grow in time. This page describes the model as implemented, so
that the parameters and the results can be interpreted.

## Model

Sphere $i$ has radius $a_i\,s(t)$, where the relative radii $a_i$ are fixed and the
scale factor $s$ grows linearly in time, $\mathrm{d}s/\mathrm{d}t = \dot s$. Between
collisions the spheres move ballistically. All spheres have the same mass $m = 1$,
independent of their size, and the temperature is $k_BT = 1$. Each velocity component
therefore has standard deviation $v_\mathrm{th} = \sqrt{k_BT/m} = 1$; this is the unit
of speed. The unit of length is the unit of the container and the radii you supply.

The run starts with points ($s = 0$) at uniformly random positions and with
Maxwell-Boltzmann velocities of zero total momentum. Because the spheres start as
points, the initial configuration never overlaps.

## Collisions

Two spheres touch when their distance equals the contact distance
$\sigma_{ij}(t) = (a_i + a_j)\,s(t)$, which grows at speed
$\dot\sigma_{ij} = (a_i + a_j)\,\dot s$. Let $u_n$ be the normal component of the
relative velocity (positive when the spheres separate). At contact $u_n < \dot\sigma_{ij}$,
otherwise the spheres would not have met. The collision rule sets

$$
u_n' = \max\left(-u_n,\; \dot\sigma_{ij} + \delta\right),
$$

and changes both velocities by equal and opposite amounts along the line of centres,
so momentum is conserved. The first argument is the elastic rule. The second ensures
that the spheres separate faster than their contact distance grows, so they do not
overlap immediately afterwards; $\delta = 10^{-9}\,v_\mathrm{th}$ is a margin that
keeps the gap opening in floating point arithmetic. Collisions with flat walls follow
the same rule with $\dot\sigma$ replaced by the growth speed of the radius.

This rule is inherited from the legacy code and differs from the rule of Donev,
Torquato and Stillinger {cite:p}`donev2005a`, which is elastic in the frame of the
growing surfaces. Their rule feeds energy into the system at every collision; this
one keeps the kinetic energy bounded. Final densities reported in the literature for a
given growth rate therefore need not carry over to spheropack.

When the spheres stop growing ($\dot s = 0$) the rule is exactly elastic and the
simulation is ordinary hard-sphere molecular dynamics, which conserves energy.

## Temperature

Collisions with $-u_n < \dot\sigma_{ij}$ change the kinetic energy. While the spheres
grow, all velocities are therefore rescaled to $k_BT = 1$ at regular intervals (every
10 collisions per sphere by default). The rescaling does not change the collision
sequence qualitatively; it keeps the growth rate meaningful relative to the thermal
speed.

## Event-driven simulation

Nothing is integrated in time steps. For every sphere the next event is predicted
exactly: a contact with another sphere, a contact with a wall, or the crossing of a
cell boundary of the neighbour grid. A contact is the first root of the quadratic

$$
f(\tau) = |\mathbf r + \mathbf u\tau|^2 - (\sigma + \dot\sigma\tau)^2
= A\tau^2 + 2B\tau + C,
$$

with $A = |\mathbf u|^2 - \dot\sigma^2$, $B = \mathbf r\cdot\mathbf u - \sigma\dot\sigma$
and $C = |\mathbf r|^2 - \sigma^2$. Growth makes $A < 0$ possible: then even
separating spheres meet again. The roots are computed in a form free of catastrophic
cancellation.

Each sphere keeps one predicted event in a priority queue. When two spheres collide,
the predictions of everything that involves them become invalid. Instead of searching
the queue, each sphere carries a collision counter; a predicted pair event is
discarded when it reaches the top of the queue and the partner's counter has changed
since the prediction. Positions are updated lazily: a sphere stores its position at
the time of its last event.

Periodic boundaries are handled with integer image counters, so positions stay inside
the box and pair vectors are exact. Boxes smaller than three cells along an axis are
handled correctly by checking all periodic images.

## Pressure

The reduced pressure $Z = PV/(Nk_BT)$ follows from the collision virial,

$$
Z = 1 + \frac{1}{D N k_BT\,\Delta t}\sum_\text{collisions} \sigma_{ij}\,J_{ij},
$$

where $J_{ij}$ is the impulse of a collision, $D$ the dimension and $\Delta t$ the
measurement window. Near jamming the collision rate, and with it $Z$, diverges, which
is what the stopping criteria use (see {doc}`stopping`).

## Final overlap check

When a run stops, all spheres are moved to the same time and the smallest gap over all
pairs and walls is computed exactly. Round-off can leave overlaps of order
$10^{-15}$ relative to the diameter; they are removed by shrinking all radii by the
same factor, which is reported as `shrink_factor`.

## References

```{bibliography}
:filter: docname in docnames
```
