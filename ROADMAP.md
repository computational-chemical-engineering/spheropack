# spheropack roadmap

spheropack turns the legacy event-driven Lubachevsky-Stillinger (LS) packing code
(`~/Code/packing`, ancestor `~/Code/event`) into a Boost-free C++20 core with a
nanobind Python interface, released on GitHub and PyPI under the MIT license. The
rejection-free Monte Carlo method of Peters & de With, Phys. Rev. E 85, 026703
(2012), becomes the module `spheropack.rejection_free`.

## Design decisions

1. **Core.** A rewrite, not a refactor: the legacy formulas are ported, the structure
   is not. C++20, header-only core in `include/spheropack/`. Particles are stored as an
   array of structs `{x[D], v[D], t_local, a_i, counter}` with lazily updated positions,
   stored wrapped into the box with integer image counters. Every ~N collisions, all
   particles are synchronised, the time origin is reset and the queue is rebuilt.
2. **Event queue.** An indexed min-heap of size N holding each particle's earliest
   event. An event is invalid when the partner's collision counter has changed since
   prediction. Cell crossings are stored separately, so a crossing only scans the new
   layer of cells. This replaces the legacy all-events heap with back-pointers.
3. **Termination.** The API promises "target density or arrest, whichever comes
   first", and the result always reports which. Radii are `a_i * s(t)`, velocities are
   rescaled to kT = 1 at each synchronisation. A run stops at the first of: target
   scale reached; reduced pressure (collision virial over windows of ~10N collisions)
   above `max_pressure`; collision budget or wall-clock limit; Ctrl-C. Afterwards the
   exact minimum gap is computed (walls included); any overlap is removed by a uniform
   shrink, which is recorded. The result has `status` (`target_reached`,
   `max_pressure` (jammed, or crystallised at slow growth), `collision_budget`,
   `time_limit`, `timeout`, `interrupted`), `density`, `reduced_pressure`, `min_gap`, `n_collisions`
   and `wall_time`.
4. **Random numbers.** Own xoshiro256++ seeded through SplitMix64, own 53-bit uniform,
   Box-Muller normals, exponentials as -log(u). Built with `-ffp-contract=off`. Same
   seed, same platform and wheel: identical results. Across platforms: statistically
   equal only, because the dynamics is chaotic and libm differs in the last bit.
5. **Geometry.** `D` (2 or 3) and the interaction are template parameters. The
   container is a runtime `std::variant`: `Box` (periodic or walled per axis) and
   `BallWall` (a ball over a subset of axes: slab, cylinder, sphere, disk). The cell
   grid is non-periodic along walled axes. Polydisperse radii come from Python as an
   array of relative radii.
6. **Rejection-free module.** Same engine with `RejectionFree<Potential>`: hard
   spheres, DPD, truncated and shifted LJ (WCA included), kT as a parameter. To keep
   duplicate predictions of the same pair consistent, the random number for a pair is
   a counter-based hash of (seed, i, j, c_i, c_j). The sequential variant
   (`collision_sec.hpp`) and tabulated user potentials come later.
7. **Python API.** `spheropack.pack(radii, container, density=None | float | "max",
   *, n=None, seed=None, growth_rate=..., max_pressure=..., timeout=None, ...)`
   returning a `Packing` (positions and radii as numpy copies, box, status, stats,
   `to_csv`, `to_xyz`, `periodic_images()`). Containers: `PeriodicBox`, `Box`, `Slab`,
   `Cylinder`, `SphereContainer`, `Disk`. Lower level: `HardSphereSystem`,
   `rejection_free.System`. A `spheropack` CLI keeps the legacy option names. The GIL
   is released during runs; signals are checked every ~1e5 events.
8. **Build and release.** scikit-build-core + nanobind, stable ABI wheels for Python
   3.12+ and regular wheels for 3.10 and 3.11, built with cibuildwheel for Linux
   (x86_64, aarch64), macOS (arm64, x86_64) and Windows. C++ kernel tests with doctest,
   everything else with pytest (a `slow` marker for statistical tests, run nightly).
   Releases via PyPI trusted publishing.

## Phases

| phase | content | acceptance check |
|---|---|---|
| 0 | Baselines from the legacy code: N=500 and N=27 arrest densities (20 seeds), collisions/s at N=1e4, tube D/d=7 radial profile (10 seeds) | `tests/baselines/*.json` plus the script that regenerates them |
| 1 | Package skeleton, RNG, CI on Linux, macOS, Windows | clean venv on each OS installs and passes the smoke tests |
| 2 | Periodic 3D packing: particles, grid, queue, predictor, stopping rules, `pack()`, GIL release, brute-force oracle | fast suite green; N=500 arrest density over 20 seeds inside the baseline band |
| 3 | Containers, polydispersity, 2D, CLI parity | no overlaps per container; tube profile within baseline spread; CLI CSV matches legacy format |
| 4 | Release 0.1.0: docs, notebooks, wheels, PyPI | `pip install spheropack==0.1.0` runs the README example on 3 OSes |
| 5 | Rejection-free module, release 0.2.0 | two-particle chi-square p > 1e-3; U and g(r) match Metropolis (z < 4) for DPD and LJ |
| 6 | Performance: profiling, cell-sorted storage, neighbour lists for polydisperse systems | at least 2x legacy collisions/s at N=1e4 near arrest, statistical suite unchanged |

## Status (2026-10-02)

- 0.1.0 released on PyPI (2026-10-01); repository public in the organisation.
- Since 0.1.0 (unreleased): output formats (`spheropack.io`: XYZ, LAMMPS dump and data,
  VTP, STL, POV-Ray, npz), the rejection-free Monte Carlo module (phase 5, validated
  against exact two-particle distributions and the paper's DPD and LJ data, notebook
  05), performance work (phase 6), Colab and download badges on the notebooks.
- Speed (one core): packing N = 1e4 at 4.25e5 collisions/s (0.1.0: 2.65e5, legacy
  2.6e5), N = 1e5 at 2.6e5 from Python; rejection-free LJ 7.1e4 reflections/s (first version: 2.0e4; DPD about
  1.4e5). Gains: no size optimisation of the
  extension (nanobind NOMINSIZE), cell-sorted storage, crossing re-prediction over the
  new cell layer only, geometric pre-checks in the rejection-free prediction, cells of
  half the cutoff with a 5^D stencil for the rejection-free engine, cheaper pair hash.
- Event-chain variant of the rejection-free method (`method="event_chain"`,
  irreversible or reversible); at T = 2 all methods agree with an independent
  Metropolis run on U/N within 0.05%.
- Next: release 0.2.0 (after user approval); further performance (the stencil loop over mostly empty cells now costs
  about a quarter of the rejection-free time; precomputed cell neighbour lists would
  remove it).

## Robustness rules for event prediction

- Write the gap as a quadratic in contact-distance units and use the stable root form
  (q = -(b + sign(b) sqrt(D)); roots c/q and q/a). With growth, a < 0 is legal.
- One contact tie rule: if |c| <= eps * sigma^2 and the pair approaches, collide at
  time 0. No other absolute epsilons.
- The collision rule guarantees a separation speed of at least the growth speed plus
  a relative margin. Port the legacy rule first (it determines arrest densities).
- Debug builds assert that predicted times are non-negative and that the gap grows
  after every collision. A brute-force O(N^2) engine for N <= 50 serves as test oracle.

## Legacy issues noted (not fixed in the legacy repo)

- The legacy CI test (`packing.cpp`, N=27, target 0.62) is a coin flip: for N=27 the
  arrest density ranges 0.610 to 0.629 across seeds.
- `collision.hpp:619`: `CollisionsDPD` passes `L[dim]` (out of bounds, wrong type).
- `collision.hpp:348`: non-inline `operator<` in a header (ODR violation with several
  translation units).
- `nbrlist.hpp:564,578,592`: `size_t` compared with -1.
- The tube cell grid is periodic in x and y, so particles on the far side of the
  tube are image neighbours.
