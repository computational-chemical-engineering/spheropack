"""Reading and writing packings in formats of common visualisation and simulation tools.

=====================  ==========================  ==========================================
function               format                      typical use
=====================  ==========================  ==========================================
:func:`write_xyz`      extended XYZ                OVITO, ASE, VMD (positions and radii)
:func:`write_lammps_dump`  LAMMPS dump (text)      OVITO, LAMMPS ``read_dump``
:func:`write_lammps_data`  LAMMPS data file        LAMMPS / LIGGGHTS ``read_data`` (atom_style sphere)
:func:`write_vtp`      VTK XML PolyData            ParaView (glyph spheres scaled by radius)
:func:`write_container_vtp`  VTK XML PolyData      ParaView outline of the container
:func:`write_stl`      binary STL surface mesh     CFD meshing (snappyHexMesh, ...), CAD, 3D printing
:func:`write_pov`      POV-Ray scene               ray-traced images
:func:`save` / :func:`load`  NumPy ``.npz``        exact round trip of a packing with its container
:meth:`Packing.to_csv`  CSV                        spreadsheets, the legacy tools
=====================  ==========================  ==========================================

2D packings are written with ``z = 0``. All writers accept ``periodic_images=True``
to add the images of spheres that cut periodic faces, which completes the spheres at
the box boundary for visualisation and meshing.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np

from .containers import Ball, Box
from .packing import Packing

__all__ = [
    "load",
    "save",
    "write_container_vtp",
    "write_lammps_data",
    "write_lammps_dump",
    "write_pov",
    "write_stl",
    "write_vtp",
    "write_xyz",
]

_FMT = "%.17g"  # round-trip exact


def _prepare(packing: Packing, periodic_images: bool) -> tuple[np.ndarray, np.ndarray]:
    p = packing.periodic_images() if periodic_images else packing
    x = p.positions
    if p.dim == 2:
        x = np.column_stack([x, np.zeros(len(x))])
    return x, p.radii


def _lengths3(packing: Packing) -> tuple[list[float], list[bool]]:
    c = packing.container
    lengths = list(c.lengths) + ([1.0] if c.dim == 2 else [])
    periodic = list(c.periodic) + ([False] if c.dim == 2 else [])
    return lengths, periodic


# ----------------------------------------------------------------- XYZ / LAMMPS


def write_xyz(
    packing: Packing,
    path: str | Path,
    *,
    species: str | np.ndarray = "X",
    periodic_images: bool = False,
    comment: str = "",
) -> None:
    """Writes an extended XYZ file (``Lattice``, ``pbc`` and a ``radius`` column).

    OVITO and ASE read the box, the periodic boundary conditions and the radii
    directly. ``species`` is one name for all spheres or one per sphere (used as the
    particle type, e.g. to colour a bidisperse packing). The default ``"X"`` is ASE's
    placeholder element; ASE accepts only chemical symbols, OVITO any name.
    """
    x, r = _prepare(packing, periodic_images)
    n = len(r)
    names = np.full(n, species, dtype=object) if np.ndim(species) == 0 else np.asarray(species, dtype=object)
    if len(names) != n:
        if periodic_images:
            raise ValueError("per-sphere species cannot be combined with periodic_images")
        raise ValueError("species needs one entry per sphere")
    lengths, periodic = _lengths3(packing)
    lattice = " ".join(_FMT % v for v in np.diag(lengths).ravel())
    pbc = " ".join("T" if p else "F" for p in periodic)
    header = f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:radius:R:1 pbc="{pbc}" Origin="0 0 0"'
    if comment:
        header += f' comment="{comment}"'
    with open(path, "w") as f:
        f.write(f"{n}\n{header}\n")
        for name, xi, ri in zip(names, x, r, strict=True):
            f.write(f"{name} {_FMT % xi[0]} {_FMT % xi[1]} {_FMT % xi[2]} {_FMT % ri}\n")


def _lammps_bounds(packing: Packing) -> tuple[str, list[tuple[float, float]]]:
    lengths, periodic = _lengths3(packing)
    flags = " ".join("pp" if p else "ff" for p in periodic)
    bounds = [(0.0, L) for L in lengths]
    if packing.dim == 2:
        bounds[2] = (-0.5, 0.5)
    return flags, bounds


def write_lammps_dump(
    packing: Packing, path: str | Path, *, types: np.ndarray | None = None, periodic_images: bool = False
) -> None:
    """Writes a LAMMPS text dump (``id type x y z radius``), one frame.

    OVITO recognises the ``radius`` column; LAMMPS can read it with ``read_dump``.
    ``types`` are integer particle types (default 1).
    """
    x, r = _prepare(packing, periodic_images)
    n = len(r)
    t = np.ones(n, dtype=int) if types is None else np.asarray(types, dtype=int)
    if len(t) != n:
        raise ValueError("types needs one entry per sphere")
    flags, bounds = _lammps_bounds(packing)
    with open(path, "w") as f:
        f.write(f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{n}\nITEM: BOX BOUNDS {flags}\n")
        for lo, hi in bounds:
            f.write(f"{_FMT % lo} {_FMT % hi}\n")
        f.write("ITEM: ATOMS id type x y z radius\n")
        for i in range(n):
            f.write(f"{i + 1} {t[i]} {_FMT % x[i, 0]} {_FMT % x[i, 1]} {_FMT % x[i, 2]} {_FMT % r[i]}\n")


def write_lammps_data(
    packing: Packing,
    path: str | Path,
    *,
    density: float = 1.0,
    types: np.ndarray | None = None,
) -> None:
    """Writes a LAMMPS data file for ``atom_style sphere`` (``id type diameter density x y z``).

    Use it to start DEM simulations (LAMMPS granular package, LIGGGHTS) from a packing.
    ``density`` is the mass density of the spheres. The box flags follow the
    container: use ``boundary p p f`` etc. in the LAMMPS input accordingly; curved
    walls must be added in LAMMPS (e.g. ``fix wall/gran ... zcylinder``).
    """
    x, r = _prepare(packing, False)
    n = len(r)
    t = np.ones(n, dtype=int) if types is None else np.asarray(types, dtype=int)
    if len(t) != n:
        raise ValueError("types needs one entry per sphere")
    _, bounds = _lammps_bounds(packing)
    with open(path, "w") as f:
        f.write(f"LAMMPS data file written by spheropack, density {packing.density:.6g}\n\n")
        f.write(f"{n} atoms\n{int(t.max())} atom types\n\n")
        for (lo, hi), ax in zip(bounds, "xyz", strict=True):
            f.write(f"{_FMT % lo} {_FMT % hi} {ax}lo {ax}hi\n")
        f.write("\nAtoms # sphere\n\n")
        for i in range(n):
            f.write(
                f"{i + 1} {t[i]} {_FMT % (2 * r[i])} {_FMT % density} "
                f"{_FMT % x[i, 0]} {_FMT % x[i, 1]} {_FMT % x[i, 2]}\n"
            )


# --------------------------------------------------------------------------- VTK


def _vtp(path, points: np.ndarray, point_data: dict[str, np.ndarray], lines: list[list[int]] | None = None):
    n = len(points)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        n_lines = 0 if lines is None else len(lines)
        n_verts = n if lines is None else 0
        f.write(
            f'<PolyData>\n<Piece NumberOfPoints="{n}" NumberOfVerts="{n_verts}" NumberOfLines="{n_lines}" '
            'NumberOfStrips="0" NumberOfPolys="0">\n'
        )
        f.write('<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, points, fmt=_FMT)
        f.write("</DataArray>\n</Points>\n")
        if point_data:
            first = next(iter(point_data))
            f.write(f'<PointData Scalars="{first}">\n')
            for name, values in point_data.items():
                values = np.asarray(values)
                vtype = "Int64" if np.issubdtype(values.dtype, np.integer) else "Float64"
                fmt = "%d" if vtype == "Int64" else _FMT
                f.write(f'<DataArray type="{vtype}" Name="{name}" format="ascii">\n')
                np.savetxt(f, values.reshape(-1, 1), fmt=fmt)
                f.write("</DataArray>\n")
            f.write("</PointData>\n")
        if lines is None:
            f.write('<Verts>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
            np.savetxt(f, np.arange(n).reshape(-1, 1), fmt="%d")
            f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
            np.savetxt(f, np.arange(1, n + 1).reshape(-1, 1), fmt="%d")
            f.write("</DataArray>\n</Verts>\n")
        else:
            conn = np.concatenate([np.asarray(line) for line in lines])
            offsets = np.cumsum([len(line) for line in lines])
            f.write('<Lines>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
            np.savetxt(f, conn.reshape(-1, 1), fmt="%d")
            f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
            np.savetxt(f, offsets.reshape(-1, 1), fmt="%d")
            f.write("</DataArray>\n</Lines>\n")
        f.write("</Piece>\n</PolyData>\n</VTKFile>\n")


def write_vtp(
    packing: Packing, path: str | Path, *, periodic_images: bool = False, point_data: dict | None = None
) -> None:
    """Writes the sphere centres as VTK XML PolyData with ``radius`` and ``diameter``.

    In ParaView: open the file, add a *Glyph* filter with glyph type *Sphere*,
    scale array ``diameter``, scale factor 1 and glyph mode *All Points* (the default
    sphere glyph has radius 0.5, so scaling by the diameter gives true sizes).
    ``point_data`` adds further per-sphere arrays (e.g. contact numbers).
    """
    x, r = _prepare(packing, periodic_images)
    data = {"radius": r, "diameter": 2 * r}
    if not periodic_images and packing.sphere_pressure is not None:
        data["sphere_pressure"] = packing.sphere_pressure
    for name, values in (point_data or {}).items():
        values = np.asarray(values)
        if len(values) != len(r):
            raise ValueError(f"point_data[{name!r}] needs one entry per sphere")
        data[name] = values
    _vtp(path, x, data)


def _circle(center: np.ndarray, radius: float, a: int, b: int, n: int = 128) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    pts = np.tile(center, (n, 1)).astype(float)
    pts[:, a] += radius * np.cos(t)
    pts[:, b] += radius * np.sin(t)
    return pts


def write_container_vtp(packing_or_container: Packing | Box, path: str | Path) -> None:
    """Writes the outline of the container (box edges, or circles and generators of a
    cylinder, or great circles of a sphere) as VTK lines, to show with the spheres."""
    c = packing_or_container.container if isinstance(packing_or_container, Packing) else packing_or_container
    L = np.array(list(c.lengths) + ([0.0] if c.dim == 2 else []), dtype=float)
    pts: list[np.ndarray] = []
    lines: list[list[int]] = []

    def add(polyline: np.ndarray, closed: bool) -> None:
        start = sum(len(p) for p in pts)
        pts.append(polyline)
        idx = list(range(start, start + len(polyline)))
        lines.append([*idx, start] if closed else idx)

    if c.ball is None:
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float) * L
        for a in range(8):
            for b in range(a + 1, 8):
                if np.count_nonzero(corners[a] != corners[b]) == 1:
                    add(np.array([corners[a], corners[b]]), False)
    else:
        axes = [k for k in range(c.dim) if c.ball.axes[k]]
        center = np.array(list(c.ball.center) + ([0.0] if c.dim == 2 else []))
        R = c.ball.radius
        if len(axes) == 2 and c.dim == 3:  # cylinder: end circles and four generators
            axis = next(k for k in range(3) if k not in axes)
            for h in (0.0, L[axis]):
                ctr = center.copy()
                ctr[axis] = h
                add(_circle(ctr, R, axes[0], axes[1]), True)
            for t in np.linspace(0, 2 * math.pi, 4, endpoint=False):
                p0 = center.copy()
                p0[axes[0]] += R * math.cos(t)
                p0[axes[1]] += R * math.sin(t)
                p1 = p0.copy()
                p1[axis] = L[axis]
                add(np.array([p0, p1]), False)
        elif c.dim == 3:  # sphere: three great circles
            for a, b in ((0, 1), (0, 2), (1, 2)):
                add(_circle(center, R, a, b), True)
        else:  # disk
            add(_circle(center, R, 0, 1), True)
    _vtp(path, np.concatenate(pts), {}, lines)


# --------------------------------------------------------------------------- STL


def _subdivide(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits every triangle into four, projecting the new vertices onto the unit sphere."""
    verts = list(v)
    cache: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = verts[a] + verts[b]
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]

    faces = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(verts), np.array(faces)


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5.0**0.5) / 2.0
    v = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=float,
    )
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )
    v /= np.linalg.norm(v, axis=1)[:, None]
    for _ in range(subdivisions):
        v, f = _subdivide(v, f)
    return v, f


def write_stl(packing: Packing, path: str | Path, *, subdivisions: int = 2, periodic_images: bool = True) -> None:
    """Writes the sphere surfaces as a binary STL mesh (icospheres, 20 * 4**subdivisions
    triangles per sphere; 320 for the default).

    Meant for meshing the pore space for CFD (e.g. OpenFOAM snappyHexMesh), CAD and 3D
    printing. With ``periodic_images`` (default) spheres cut by periodic faces are
    completed by their images. 3D packings only.
    """
    if packing.dim != 3:
        raise ValueError("STL output needs a 3D packing")
    x, r = _prepare(packing, periodic_images)
    v, f = _icosphere(subdivisions)
    tri = v[f]  # (nf, 3, 3) on the unit sphere
    normals = tri.mean(axis=1)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    nf = len(f)
    record = np.zeros(len(r) * nf, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["n"] = np.tile(normals, (len(r), 1))
    record["v"] = (x[:, None, None, :] + r[:, None, None, None] * tri[None]).reshape(-1, 3, 3)
    with open(path, "wb") as fh:
        fh.write(b"spheropack sphere packing".ljust(80, b" "))
        fh.write(struct.pack("<I", len(record)))
        fh.write(record.tobytes())


# ------------------------------------------------------------------------ POV-Ray


def write_pov(packing: Packing, path: str | Path, *, periodic_images: bool = False, color=(0.2, 0.45, 0.8)) -> None:
    """Writes a POV-Ray scene with camera, light and one ``sphere`` per particle."""
    x, r = _prepare(packing, periodic_images)
    L = np.array(_lengths3(packing)[0])
    center = 0.5 * L
    eye = center + np.array([1.6, 1.2, 2.0]) * L.max()
    with open(path, "w") as f:
        f.write("// written by spheropack\n#version 3.7;\nglobal_settings { assumed_gamma 1.0 }\n")
        f.write("background { color rgb <1, 1, 1> }\n")
        f.write(
            f"camera {{ location <{eye[0]:.6g}, {eye[1]:.6g}, {eye[2]:.6g}> "
            f"look_at <{center[0]:.6g}, {center[1]:.6g}, {center[2]:.6g}> angle 35 sky <0,0,1> }}\n"
        )
        f.write(f"light_source {{ <{eye[0]:.6g}, {-eye[1]:.6g}, {eye[2] * 2:.6g}> color rgb <1,1,1> }}\n")
        rgb = ", ".join(f"{c:g}" for c in color)
        f.write(
            f"#declare SphereTexture = texture {{ pigment {{ color rgb <{rgb}> }}"
            " finish { phong 0.5 ambient 0.15 } }\n"
        )
        f.write("union {\n")
        for xi, ri in zip(x, r, strict=True):
            f.write(f"  sphere {{ <{_FMT % xi[0]}, {_FMT % xi[1]}, {_FMT % xi[2]}>, {_FMT % ri} }}\n")
        f.write("  texture { SphereTexture }\n}\n")


# ---------------------------------------------------------------------- NPZ round trip

_SCALARS = (
    "status",
    "density",
    "reduced_pressure",
    "median_pressure",
    "n_collisions",
    "n_events",
    "sim_time",
    "wall_time",
    "min_gap",
    "shrink_factor",
    "seed",
    "growth_rate",
    "max_contact_error",
    "collision_rule",
    "success",
)


def save(packing: Packing, path: str | Path) -> None:
    """Saves a packing, its container and run statistics to a NumPy ``.npz`` file."""
    c = packing.container
    meta = {k: getattr(packing, k) for k in _SCALARS}
    meta = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in meta.items()}
    container = {
        "lengths": list(c.lengths),
        "periodic": list(c.periodic),
        "ball": None
        if c.ball is None
        else {"axes": list(c.ball.axes), "center": list(c.ball.center), "radius": c.ball.radius},
    }
    arrays = {"positions": packing.positions, "radii": packing.radii}
    if packing.sphere_pressure is not None:
        arrays["sphere_pressure"] = packing.sphere_pressure
    for k, v in (packing.history or {}).items():
        arrays[f"history_{k}"] = np.asarray(v)
    arrays["meta"] = np.array(json.dumps({"format": "spheropack-1", "container": container, **meta}))
    np.savez_compressed(path, **cast(dict[str, Any], arrays))


def load(path: str | Path) -> Packing:
    """Loads a packing written by :func:`save`."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.pop("format", None) != "spheropack-1":
            raise ValueError("not a spheropack .npz file")
        cd = meta.pop("container")
        ball = (
            None
            if cd["ball"] is None
            else Ball(tuple(cd["ball"]["axes"]), tuple(cd["ball"]["center"]), cd["ball"]["radius"])
        )
        container = Box(cd["lengths"], periodic=cd["periodic"], ball=ball)
        history = {k[len("history_") :]: z[k] for k in z.files if k.startswith("history_")} or None
        return Packing(
            positions=z["positions"],
            radii=z["radii"],
            container=container,
            sphere_pressure=z["sphere_pressure"] if "sphere_pressure" in z.files else None,
            history=history,
            stop=(),
            stopped_by=None,
            **meta,
        )
