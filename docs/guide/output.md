# Output formats

A {class}`~spheropack.Packing` can be written in the formats of common visualisation
and simulation tools. All writers are in {mod}`spheropack.io` and available as methods
of the packing:

| method | format | for |
|---|---|---|
| `to_xyz(path)` | extended XYZ (with cell, periodic flags and radii) | OVITO, ASE, VMD |
| `to_lammps_dump(path)` | LAMMPS text dump with a `radius` column | OVITO, LAMMPS `read_dump` |
| `to_lammps_data(path, density=...)` | LAMMPS data file, `atom_style sphere` | DEM in LAMMPS or LIGGGHTS |
| `to_vtp(path)` | VTK XML PolyData with `radius` and `diameter` | ParaView |
| `spheropack.io.write_container_vtp(p, path)` | outline of the container as VTK lines | ParaView |
| `to_stl(path)` | binary STL surface mesh of all spheres | CFD meshing, CAD, 3D printing |
| `to_pov(path)` | POV-Ray scene | ray-traced images |
| `to_csv(path)` | CSV `x,y,z,r` | spreadsheets, the legacy tools |
| `save(path)` / `spheropack.load(path)` | NumPy `.npz`, exact round trip | Python |

2D packings are written with $z = 0$. With `periodic_images=True` the writers add the
images of spheres that cut periodic faces, which completes the spheres at the box
boundary; this is the default for STL.

```python
import spheropack as sp

p = sp.pack(n=2000, density="max", container=sp.Cylinder(8.0, 12.0), radii=0.5, seed=1)
p.to_xyz("packing.xyz")  # OVITO
p.to_vtp("packing.vtp")  # ParaView
sp.io.write_container_vtp(p, "tube.vtp")
p.to_stl("packing.stl")  # CFD meshing
p.save("packing.npz")  # exact round trip: sp.load("packing.npz")
```

The command line writes the same formats: `spheropack pack ... --format xyz -o packing.xyz`
(also `lammps`, `lammps-data`, `vtp`, `stl`, `pov`, `npz`).

## OVITO

Open the `.xyz` (or the LAMMPS `.dump`) file. OVITO takes the simulation cell and the
periodic flags from the file and the particle radii from the `radius` column, so the
spheres appear with their true sizes. Per-sphere values written as `species` (for
example small and large spheres of a mixture) become particle types for colouring.
In OVITO's Python module:

```python
from ovito.io import import_file

pipeline = import_file("packing.xyz")
```

## ParaView

Open the `.vtp` file and apply a **Glyph** filter: glyph type *Sphere*, *Scale Array*
`diameter`, scale factor 1, glyph mode *All Points*. The default sphere glyph has radius
0.5, so scaling by the diameter gives the true sizes. For very large packings
the *Point Gaussian* representation with the *Sphere* shader preset, scaled by the
`radius` array, renders faster than glyphs. Open the container
outline written by `write_container_vtp` alongside. Further per-sphere arrays can be
added with `to_vtp(path, point_data={"contacts": analysis.contact_numbers(p)})`; the
per-sphere pressure is included automatically.

## LAMMPS and LIGGGHTS

`to_lammps_data` writes `id type diameter density x y z` for `atom_style sphere`; read
it with `read_data`. The boundary conditions are not part of a data file: use
`boundary p p p` for a periodic box, `p p f` for a slab, and add the walls of a tube
or a spherical container in the input script (for example `fix wall/gran` with a
`zcylinder`).

## CFD meshing

`to_stl` triangulates every sphere as an icosphere (320 triangles by default;
`subdivisions=3` gives 1280). For the pore space of a packed bed, use the STL file as
the geometry in a mesher such as OpenFOAM's snappyHexMesh. Touching spheres share
single contact points, which meshers cannot resolve; the usual remedies are to shrink
the spheres by a small factor before export (`sp.Packing` radii can be scaled, e.g.
`dataclasses.replace(p, radii=0.99 * p.radii)`) or to bridge the contacts.
