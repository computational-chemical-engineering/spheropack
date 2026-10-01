import math

import numpy as np
import pytest

import spheropack as sp
from spheropack import analysis
from spheropack.packing import Packing


def _packing(positions, radii, container):
    positions = np.asarray(positions, float)
    radii = np.full(len(positions), radii) if np.ndim(radii) == 0 else np.asarray(radii)
    return Packing(positions, radii, container, "target_reached", 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0, 0.0)


def _fcc(cells):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    g = np.array([[i, j, k] for i in range(cells) for j in range(cells) for k in range(cells)])
    x = (g[:, None, :] + base[None]).reshape(-1, 3) + 0.25
    return _packing(x, math.sqrt(2) / 4, sp.PeriodicBox(float(cells)))


def _hcp(nx, ny, nz):
    a = 1.0
    pts = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = i * a + 0.5 * a * (j % 2) + (0.5 * a if k % 2 else 0.0)
                y = j * a * math.sqrt(3) / 2 + (a / (2 * math.sqrt(3)) if k % 2 else 0.0)
                pts.append([x + 0.1, y + 0.1, k * a * math.sqrt(2 / 3) + 0.1])
    box = sp.PeriodicBox([nx * a, ny * a * math.sqrt(3) / 2, nz * a * math.sqrt(2 / 3)])
    return _packing(pts, 0.5, box)


def test_legendre_matches_scipy():
    special = pytest.importorskip("scipy.special")
    if not hasattr(special, "sph_harm_y"):
        pytest.skip("scipy too old")
    theta = np.linspace(0.1, 3.0, 7)
    for l in (4, 6, 8):
        plm = analysis._legendre_normalised(l, np.cos(theta))
        for m in range(l + 1):
            np.testing.assert_allclose(plm[m], special.sph_harm_y(l, m, theta, 0.0).real, atol=1e-12)


@pytest.mark.parametrize("l, expected", [(4, 0.19094), (6, 0.57452)])
def test_fcc_bond_order(l, expected):
    q = analysis.bond_order(_fcc(4), l=l, cutoff=1.2)
    assert np.all(q.n_bonds == 12)
    assert q.global_value == pytest.approx(expected, abs=1e-4)
    np.testing.assert_allclose(q.local, expected, atol=1e-4)


@pytest.mark.parametrize("l, expected", [(4, 0.09722), (6, 0.48476)])
def test_hcp_bond_order(l, expected):
    q = analysis.bond_order(_hcp(6, 6, 4), l=l, cutoff=1.1)
    assert np.all(q.n_bonds == 12)
    np.testing.assert_allclose(q.local, expected, atol=1e-4)


def test_hexagonal_psi6_is_one():
    nx, ny = 8, 8
    pts = [[i + 0.5 * (j % 2) + 0.1, j * math.sqrt(3) / 2 + 0.1] for j in range(ny) for i in range(nx)]
    p = _packing(pts, 0.5, sp.PeriodicBox([nx, ny * math.sqrt(3) / 2]))
    q = analysis.bond_order(p, l=6, cutoff=1.1)
    assert np.all(q.n_bonds == 6)
    assert q.global_value == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(q.local, 1.0, atol=1e-12)


def test_random_packing_has_little_global_order():
    p = sp.pack(n=2000, density=0.5, seed=1)
    assert analysis.bond_order(p).global_value < 0.1


def test_ideal_gas_rdf_is_one():
    rng = np.random.default_rng(0)
    p = _packing(rng.uniform(0, 1, (4000, 3)), 0.01, sp.PeriodicBox(1.0))
    _, g = analysis.radial_distribution(p, r_max=0.3, bins=30)
    assert abs(g[5:].mean() - 1.0) < 0.02


def test_dense_rdf_has_contact_peak_and_no_overlap():
    p = sp.pack(n=1000, density=0.6, seed=2)
    rdf = analysis.radial_distribution(p, bins=250)
    d = p.diameters[0]
    assert np.all(rdf.g[rdf.r < 0.99 * d] == 0)
    assert rdf.r[np.argmax(rdf.g)] == pytest.approx(d, rel=0.02)


def test_jammed_packing_is_nearly_isostatic():
    p = sp.pack(n=1000, density="max", seed=3)
    rat = analysis.rattlers(p)
    assert 0.0 < rat.mean() < 0.1
    # Contact gaps at Z = 1e9 reach a few 1e-9, so the plateau starts near 1e-7.
    ratios = [analysis.isostaticity(p, tol) for tol in (1e-7, 1e-6, 1e-5)]
    assert max(ratios) - min(ratios) < 0.015  # contact plateau
    assert 0.95 < ratios[0] <= 1.0


def test_contact_based_rattlers_agree_roughly_with_force_based():
    p = sp.pack(n=500, density="max", seed=4)
    force = analysis.rattlers(p, method="force")
    count = analysis.rattlers(p, tol=1e-7, method="contacts")
    assert (force & ~count).sum() <= 3  # force-free spheres almost always lack contacts


@pytest.mark.parametrize(
    "container",
    [sp.PeriodicBox(1.0), sp.Box([1.0, 1.0, 1.2], periodic=[True, True, False]), sp.PeriodicBox(1.0, dim=2)],
)
def test_density_profile_integrates_to_density(container):
    p = sp.pack(n=400, density=0.5 if container.dim == 3 else 0.7, container=container, seed=4)
    _, phi = analysis.density_profile(p, axis=-1, bins=97)
    assert phi.mean() == pytest.approx(p.density, rel=1e-10)


def test_crystalline_detects_fcc_hcp_and_hexagonal():
    assert analysis.crystalline(_fcc(4), cutoff=1.2).all()
    assert analysis.crystalline(_hcp(6, 6, 4), cutoff=1.1).all()
    nx, ny = 8, 8
    pts = [[i + 0.5 * (j % 2) + 0.1, j * math.sqrt(3) / 2 + 0.1] for j in range(ny) for i in range(nx)]
    p = _packing(pts, 0.5, sp.PeriodicBox([nx, ny * math.sqrt(3) / 2]))
    assert analysis.crystalline(p, cutoff=1.1).all()


def test_fast_random_packing_is_not_crystalline():
    p = sp.pack(n=2000, density=0.6, growth_rate=0.1, seed=5)
    assert analysis.crystalline(p).mean() < 0.02
