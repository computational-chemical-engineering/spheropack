# Stopping criteria

A run ends when the spheres reach the target density, or when the first of the
stopping criteria passed as `stop=` fires. The result reports what happened in
`status` (`"target"` or the name of the criterion), the criterion object in
`stopped_by`, and whether the run did what was asked in `success`.

```python
import spheropack as sp

p = sp.pack(n=1000, density=0.6)                       # target density
q = sp.pack(n=1000, density="max")                      # until jammed
r = sp.pack(n=1000, density="max",
            stop=[sp.stop.Jammed(pressure=1e10), sp.stop.Timeout(120)])
```

## Targets

`density` sets the goal of the run:

| `density` | the run succeeds when | default `stop` |
|---|---|---|
| `None` | the spheres have the `radii` you gave | `[Pressure(1e6)]` |
| a number | the volume (area) fraction reaches it | `[Pressure(1e6)]` |
| `"max"` | an *arrest* criterion fires | `[Jammed()]` |

With a target, the default `Pressure(1e6)` stops a run whose target lies beyond the
jamming density instead of letting it run forever. Without success `pack` warns
({class}`~spheropack.PackingWarning`), or raises {class}`~spheropack.PackingError`
with `strict=True`; the partial packing is available in both cases.

## Arrest criteria

Arrest criteria detect that the spheres can no longer grow.

{class}`~spheropack.stop.Jammed`
: Compresses quasi-statically to a jammed packing. The spheres grow at the requested
  growth rate until the median per-sphere reduced pressure reaches `start_pressure`
  ($10^3$). Then the radii alternate between relaxation at fixed size and short
  growth steps. Near jamming the reduced pressure diverges as
  $Z \approx D/(1 - \phi/\phi_J)$, so the remaining relative growth is about $1/Z$; each
  step closes the fraction `step` of it. The run stops when the median pressure after
  relaxation exceeds `pressure` ($10^9$). The median ignores rattlers and the few
  spheres that may collide at very high rates.

: Slower protocols (more relaxation windows, smaller steps) come closer to an
  isostatic contact network at a higher cost. For 1000 equal spheres at growth rate
  0.02, the contacts between force-bearing spheres reach about 94% of the isostatic
  number with `relax_windows=2, step=0.5`, 97% with the defaults
  (`relax_windows=4, step=0.2`, 1.75 times the run time) and 98% with
  `relax_windows=4, step=0.1` (3 times). The density changes only in the fourth
  decimal. Pressures above about $10^{11}$ bring the gaps between touching spheres
  close to floating-point round-off.

{class}`~spheropack.stop.Pressure`
: Stops when the mean reduced pressure over a measurement window exceeds `value`,
  while the spheres keep growing at the requested rate. Cheaper than `Jammed`, but the
  final approach is not quasi-static, so the contact network is not yet fully formed.
  Suitable when only the density and the overall structure matter.

{class}`~spheropack.stop.Stall`
: Stops when the radii grew by less than the relative amount `tol` during the last
  `window` collisions per sphere. It measures progress directly and needs no pressure.

## Limit criteria

Limit criteria bound the cost of a run. When one fires before the target or an
arrest, the run counts as failed.

{class}`~spheropack.stop.Collisions`
: A budget of `total` collisions or `per_particle` collisions per sphere, exact.

{class}`~spheropack.stop.Timeout`
: Wall-clock seconds. Unlike all other criteria it makes the result depend on the
  speed of the machine.

Pressing Ctrl-C in Python stops a run within a fraction of a second and raises
`KeyboardInterrupt`.

## How pressure is measured

The reduced pressure $Z = PV/(Nk_BT)$ is computed from the collision virial over
windows of 10 collisions per sphere,

$$
Z = 1 + \frac{\sum_\text{collisions} \sigma_{ij} J_{ij}}{2\int E_\text{kin}\,\mathrm{d}t},
$$

with $J_{ij}$ the impulse of a collision and the kinetic energy integrated over the
window. Each sphere receives half of the virial of its collisions, which gives a
per-sphere pressure (`Packing.sphere_pressure`); its mean is $Z$ and its median is
`Packing.median_pressure`. In a jammed packing the per-sphere pressure is the
time-averaged contact force on the sphere, so spheres with a value near 1 carry no
force: they are rattlers ({func}`spheropack.analysis.rattlers`). `Packing.history`
holds the pressures, temperature and radius scale of every window, which shows the
course of the compression.

## Combining criteria

Criteria of different types combine: the first that fires ends the run. Of several
criteria of the same type the strictest applies (the lowest pressure, the smallest
budget, the shortest timeout). At most one `Jammed` and one `Stall` may be given.

## Reproducibility

With the same seed, the same version and the same platform a run is reproducible
exactly, except when it is ended by `Timeout` or Ctrl-C. Between platforms results
agree statistically but not sphere by sphere: event-driven dynamics is chaotic and
amplifies round-off differences of the mathematical library.
