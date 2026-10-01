# Rejection-free Monte Carlo

`spheropack.rejection_free` implements the event-driven Monte Carlo method of
{cite:t}`peters2012`. It samples the canonical distribution $\exp(-U/kT)$ of particles
with a pair potential, without ever rejecting a move.

## The method

All particles move along straight lines, $\mathrm{d}\mathbf x_i/\mathrm{d}s =
\mathbf v_i$, where $s$ plays the role of time. For every pair potential $U_\alpha$
separately, the increase of $U_\alpha$ along the path is accumulated; decreases are
free. The probability that no reflection due to $U_\alpha$ has happened yet is

$$
P_\mathrm{no\,refl}(s) = \exp\left[-\beta\int_{s_0}^{s}\max\left(\frac{\mathrm{d}U_\alpha}{\mathrm{d}s'}, 0\right)\mathrm{d}s'\right],
$$

so a pair reflects where the accumulated increase equals $-kT\ln u$ with $u$ uniform in
$(0, 1]$. A reflection is an elastic collision of two equal masses along the line of
centres, which leaves $\sum_i |\mathbf v_i|^2$ unchanged. Positions sampled at
equidistant times $s_n = n\,\Delta s$ follow the canonical distribution. The velocities
are not physical momenta: they only set the directions and the pace of the moves.

For an isotropic pair potential the separation along a straight path first decreases to
the closest approach and then increases. With at most one minimum $r_m$ of $U(r)$, the
uphill parts are the inward path inside $r_m$ (repulsion) and the outward path between
$r_m$ and the cutoff (attraction), so the reflection point follows from inverting
$U(r)$ on one branch and solving a quadratic for the time.

As in the packing engine, every particle keeps one predicted event in a priority queue.
Because a pair can be predicted by either particle, and again later, the random number
$u$ of a pair is a counter-based hash of the two particle indices, the number of
velocity changes of each, and the periodic image, and the accumulated increase is
counted from the start of the pair's current straight segment. All predictions of the
same pair therefore agree; independent draws would double the reflection rate.

## Usage

```python
import spheropack as sp
from spheropack import rejection_free as rf

system = rf.System(n=1000, density=0.317, potential=rf.LennardJones(cutoff=2.5), kT=1.085, seed=1)
system.run(100.0)  # equilibrate
g = rf.radial_distribution(system, n_samples=300, interval=2.0, r_max=5.0)
for x in system.samples(100, interval=1.0):  # positions every 1.0
    ...
system.snapshot().to_xyz("lj.xyz")  # analysis and output as for packings
```

Potentials: {func}`~spheropack.rejection_free.LennardJones` (truncated and shifted),
{func}`~spheropack.rejection_free.WCA`, {func}`~spheropack.rejection_free.DPD`,
{func}`~spheropack.rejection_free.SoftSpheres` (harmonic, Hertzian) and
{func}`~spheropack.rejection_free.HardSpheres`. The box must be periodic with edges of
at least twice the cutoff. For strongly repulsive potentials start from a packing,
`rf.System(sp.pack(...).positions, box=..., ...)`, rather than from random positions.

## Validation

- **Two particles.** The distance distribution of two particles in a periodic box
  is $r^{D-1}\exp(-U/kT)$; the tests compare the sampled histogram and the bound
  fraction for every potential in 2D and 3D, and check that a 20% error in the
  temperature would be detected.
- **Lennard-Jones fluid** at $\rho = 0.317$, $T = 1.085$, $r_c = 2.5\sigma$ (the state
  point of the paper): $g(r)$ agrees with the paper's long Metropolis runs within
  0.002 on average (see the notebook).
- **DPD liquid** at $\rho = 3$, $a = 25\,kT$: $g(r)$ agrees with the paper's data.

## Not (yet) included

The straight event-chain variant of the paper (one particle moves at a time) and
potentials with several minima or tabulated potentials.
