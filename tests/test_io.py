"""Output formats. Reader tests (OVITO, ASE, VTK) run when those packages are installed."""

import struct

import numpy as np
import pytest

import spheropack as sp

CONTAINERS = {
    "slab": sp.Box([1.0, 1.0, 1.5], periodic=[True, True, False]),
    "tube": sp.Cylinder(1.0, 1.5),
    "disk2d": sp.PeriodicBox(1.0, dim=2),
}


@pytest.fixture(scope="module", params=list(CONTAINERS))
def packing(request):
    c = CONTAINERS[request.param]
    return sp.pack(n=120, density=0.5 if c.dim == 3 else 0.7, container=c, seed=1)


def _xyz3(p):
    return p.positions if p.dim == 3 else np.column_stack([p.positions, np.zeros(p.n)])


def test_xyz(packing, tmp_path):
    f = tmp_path / "p.xyz"
    packing.to_xyz(f)
    lines = f.read_text().splitlines()
    assert int(lines[0]) == packing.n
    assert "Properties=species:S:1:pos:R:3:radius:R:1" in lines[1] and "Lattice=" in lines[1]
    data = np.array([line.split()[1:] for line in lines[2:]], dtype=float)
    np.testing.assert_array_equal(data[:, :3], _xyz3(packing))
    np.testing.assert_array_equal(data[:, 3], packing.radii)


def test_lammps_dump(packing, tmp_path):
    f = tmp_path / "p.dump"
    packing.to_lammps_dump(f)
    lines = f.read_text().splitlines()
    i = lines.index("ITEM: ATOMS id type x y z radius")
    data = np.array([line.split() for line in lines[i + 1 :]], dtype=float)
    assert len(data) == packing.n
    np.testing.assert_array_equal(data[:, 2:5], _xyz3(packing))
    np.testing.assert_array_equal(data[:, 5], packing.radii)


def test_lammps_data(packing, tmp_path):
    f = tmp_path / "p.data"
    packing.to_lammps_data(f, density=2500.0)
    text = f.read_text()
    assert f"{packing.n} atoms" in text and "Atoms # sphere" in text
    rows = np.array([line.split() for line in text.split("Atoms # sphere")[1].strip().splitlines()], dtype=float)
    np.testing.assert_array_equal(rows[:, 2], 2 * packing.radii)
    assert np.all(rows[:, 3] == 2500.0)


def test_vtp_is_valid_xml(packing, tmp_path):
    import xml.etree.ElementTree as ET

    f = tmp_path / "p.vtp"
    packing.to_vtp(f, point_data={"z": np.arange(packing.n)})
    root = ET.parse(f).getroot()
    piece = root.find("PolyData/Piece")
    assert int(piece.get("NumberOfPoints")) == packing.n
    names = [a.get("Name") for a in piece.find("PointData")]
    assert {"radius", "diameter", "z"} <= set(names)
    sp.io.write_container_vtp(packing, tmp_path / "c.vtp")
    lines = ET.parse(tmp_path / "c.vtp").getroot().find("PolyData/Piece")
    assert int(lines.get("NumberOfLines")) > 0


def test_stl(tmp_path):
    p = sp.pack(n=30, density=0.4, seed=2)
    f = tmp_path / "p.stl"
    p.to_stl(f, subdivisions=1, periodic_images=False)
    raw = f.read_bytes()
    n_tri = struct.unpack("<I", raw[80:84])[0]
    assert n_tri == 30 * 80 and len(raw) == 84 + 50 * n_tri
    verts = np.frombuffer(raw[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"]
    # every vertex lies on its sphere surface
    centres = np.repeat(p.positions, 80, axis=0)[:, None, :]
    dist = np.linalg.norm(verts - centres, axis=2)
    np.testing.assert_allclose(dist, np.repeat(p.radii, 80)[:, None] * np.ones(3), rtol=1e-6)
    with pytest.raises(ValueError):
        sp.pack(n=10, density=0.4, container=sp.PeriodicBox(1.0, dim=2)).to_stl(f)


def test_pov(packing, tmp_path):
    f = tmp_path / "p.pov"
    packing.to_pov(f)
    assert f.read_text().count("sphere {") == packing.n


def test_npz_round_trip(packing, tmp_path):
    f = tmp_path / "p.npz"
    packing.save(f)
    q = sp.load(f)
    np.testing.assert_array_equal(q.positions, packing.positions)
    np.testing.assert_array_equal(q.radii, packing.radii)
    assert q.container == packing.container
    assert (q.status, q.density, q.seed, q.success) == (packing.status, packing.density, packing.seed, packing.success)
    np.testing.assert_array_equal(q.history["scale"], packing.history["scale"])


def test_periodic_images_option(tmp_path):
    p = sp.pack(n=100, density=0.5, seed=3)
    p.to_xyz(tmp_path / "a.xyz", periodic_images=True)
    assert int((tmp_path / "a.xyz").read_text().split("\n", 1)[0]) == p.periodic_images().n


# ------------------------------------------------------------ real readers


def test_ovito_reads_xyz_dump_and_data(packing, tmp_path):
    ovito_io = pytest.importorskip("ovito.io")
    packing.to_xyz(tmp_path / "p.xyz")
    packing.to_lammps_dump(tmp_path / "p.dump")
    packing.to_lammps_data(tmp_path / "p.data")
    for name, kwargs in (("p.xyz", {}), ("p.dump", {}), ("p.data", {"atom_style": "sphere"})):
        data = ovito_io.import_file(str(tmp_path / name), **kwargs).compute()
        assert data.particles.count == packing.n
        np.testing.assert_allclose(np.sort(np.asarray(data.particles["Radius"])), np.sort(packing.radii))


def test_ase_reads_xyz(packing, tmp_path):
    ase_io = pytest.importorskip("ase.io")
    packing.to_xyz(tmp_path / "p.xyz")
    atoms = ase_io.read(tmp_path / "p.xyz")
    np.testing.assert_allclose(atoms.positions, _xyz3(packing))
    np.testing.assert_allclose(atoms.arrays["radius"], packing.radii)
    assert list(atoms.pbc) == list(packing.container.periodic) + ([False] if packing.dim == 2 else [])


def test_vtk_reads_vtp_and_stl(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    p = sp.pack(n=50, density=0.5, container=sp.Cylinder(1.0, 1.0), seed=4)
    p.to_vtp(tmp_path / "p.vtp")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(tmp_path / "p.vtp"))
    reader.Update()
    out = reader.GetOutput()
    np.testing.assert_allclose(vtk_to_numpy(out.GetPoints().GetData()), p.positions)
    np.testing.assert_allclose(vtk_to_numpy(out.GetPointData().GetArray("radius")), p.radii)
    p.to_stl(tmp_path / "p.stl")
    stl = vtk.vtkSTLReader()
    stl.SetFileName(str(tmp_path / "p.stl"))
    stl.Update()
    assert stl.GetOutput().GetNumberOfCells() == p.periodic_images().n * 320
