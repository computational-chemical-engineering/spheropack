# Changelog

## Unreleased

- Rejection-free Monte Carlo (`spheropack.rejection_free`), the method of Peters and
  de With, Phys. Rev. E 85, 026703 (2012): `System`, potentials `LennardJones`, `WCA`,
  `DPD`, `SoftSpheres`, `HardSpheres`, `radial_distribution`. Validated against exact
  two-particle distributions and the data of the paper; notebook reproducing its
  Lennard-Jones case. Both implementations of the paper: all particles moving
  (`method="collisions"`) and straight event chains (`method="event_chain"`,
  irreversible or reversible).
- Performance: packing about 1.5 times faster than 0.1.0 (extension no longer
  optimised for size, cell-sorted particle storage, cheaper re-prediction after cell
  crossings); the rejection-free engine uses cells of half the cutoff with a 5^D
  stencil. The cell grid supports any stencil radius.
- Notebooks have "Open in Colab" and "Download notebook" badges.
- Output formats (`spheropack.io`, methods on `Packing`, CLI `--format`): extended XYZ
  (OVITO, ASE), LAMMPS dump and data file (atom_style sphere), VTK PolyData and
  container outlines (ParaView), binary STL meshes, POV-Ray scenes, and `.npz` save
  and `spheropack.load` for an exact round trip.

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
