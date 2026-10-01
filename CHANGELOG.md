# Changelog

## 0.1.0 (2026-10-01)

First release. A rewrite of the legacy event-driven Lubachevsky-Stillinger packing code
as a header-only C++20 core with a Python interface.

- Packing with `spheropack.pack`: target density, given radii, or `density="max"`
  (until jammed). Dimensionless growth rate relative to the thermal speed.
- Containers: periodic and walled boxes, `Cylinder` (periodic or capped),
  `SphereContainer`, `Disk`; 2D and 3D; polydisperse radii.
- Collision rules: `elastic_growing` (default, elastic in the frame of the growing
  surfaces) and `legacy` (reproduces the original code).
- Stopping criteria (`spheropack.stop`): `Jammed` (quasi-static jamming protocol with
  median per-sphere pressure), `Pressure`, `Stall`, `Collisions`, `Timeout`.
- Results with positions, radii, status, pressures (mean, median, per sphere) and the
  compression history; CSV export, periodic images.
- Analysis (`spheropack.analysis`): neighbour pairs, g(r), contacts, rattlers from
  collision forces, isostaticity, Steinhardt bond order, crystallinity, density
  profiles next to walls, radial porosity profiles.
- Command line: `spheropack pack`, and `legacy-periodic` and `legacy-tube` as drop-in
  replacements of the legacy tools.
- Documentation: user guide, four example notebooks, Python and C++ API reference.
- Typed package (`py.typed`, generated stub for the C++ extension); wheels for Linux,
  macOS and Windows, Python 3.10 and newer.
