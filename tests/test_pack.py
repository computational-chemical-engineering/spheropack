import math
import warnings

import numpy as np
import pytest

import spheropack as sp
from spheropack import _core
from helpers import min_gap_ratio

INF = float("inf")


@pytest.mark.parametrize(
    "container",
    [
        sp.PeriodicBox(1.0),
        sp.PeriodicBox(1.0, dim=2),
        sp.Box([1.0, 1.0, 1.5], periodic=[True, True, False]),
        sp.Box([1.0, 1.0, 1.0], periodic=False),
        sp.Box([1.0, 2.0], periodic=[False, True]),
    ],
    ids=["periodic3d", "periodic2d", "slab", "closed3d", "channel2d"],
)
@pytest.mark.parametrize("density", [0.3, 0.55])
def test_target_density_without_overlap(container, density):
    if container.dim == 2:
        density += 0.2
    p = sp.pack(n=300, density=density, container=container, seed=3)
    assert p.status == "target"
    assert p.density == pytest.approx(density, rel=1e-12)
    assert np.all(p.positions >= 0) and np.all(p.positions < np.array(container.lengths))
    assert min_gap_ratio(p.positions, p.radii, container.lengths, container.periodic) > -1e-12


def test_polydisperse_radii_keep_their_ratios():
    rng = np.random.default_rng(0)
    radii = rng.uniform(0.5, 1.5, 400)
    p = sp.pack(radii=radii, density=0.55, seed=1)
    assert p.status == "target"
    np.testing.assert_allclose(p.radii / radii, p.radii[0] / radii[0], rtol=1e-12)
    assert min_gap_ratio(p.positions, p.radii, p.container.lengths, p.container.periodic) > -1e-12


def test_absolute_radii_without_density():
    p = sp.pack(n=100, radii=0.05, seed=2)
    assert p.status == "target" and p.success
    np.testing.assert_allclose(p.radii, 0.05)
    assert p.density == pytest.approx(100 * 4 / 3 * math.pi * 0.05**3)


def test_same_seed_same_packing():
    a = sp.pack(n=200, density=0.5, seed=11)
    b = sp.pack(n=200, density=0.5, seed=11)
    c = sp.pack(n=200, density=0.5, seed=12)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, c.positions)


def test_momentum_is_conserved_in_periodic_box():
    p = sp.pack(n=300, density=0.5, seed=4)
    assert np.abs(p._velocities.sum(axis=0)).max() < 1e-10


def test_fixed_radii_conserve_energy_and_momentum():
    start = sp.pack(n=300, density=0.45, seed=5)
    rng = np.random.default_rng(1)
    v = rng.normal(size=(300, 3))
    v -= v.mean(axis=0)
    out = _core._ls_pack3(
        start.radii, [1.0] * 3, [True] * 3, INF, 0.0, INF, INF, 0.0, 100.0, 0, 20.0, INF, 10.0, 0,
        "elastic_growing", start.positions, 1.0, v,
    )
    assert out["status"] == "time_limit"
    assert out["n_collisions"] > 50_000
    e0, e1 = (v**2).sum(), (out["velocities"] ** 2).sum()
    assert abs(e1 - e0) / e0 < 1e-12
    assert np.abs(out["velocities"].sum(axis=0)).max() < 1e-10
    assert min_gap_ratio(out["positions"], out["radii"], [1.0] * 3, [True] * 3) > -1e-12


def test_tiny_box_that_jams_below_target_returns_quickly():
    # The legacy CI case: 27 spheres to 0.62 jams for many seeds and used to hang.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.PackingWarning)
        for seed in range(5):
            p = sp.pack(n=27, density=0.62, growth_rate=0.16, seed=seed, stop=[sp.stop.Pressure(1e6), sp.stop.Timeout(10)])
            assert p.status in ("target", "pressure")
            assert p.wall_time < 10
            assert min_gap_ratio(p.positions, p.radii, [1.0] * 3, [True] * 3) > -1e-12


def test_unreached_target_warns_or_raises():
    with pytest.warns(sp.PackingWarning):
        sp.pack(n=27, density=0.7, seed=1)
    with pytest.raises(sp.PackingError) as err:
        sp.pack(n=27, density=0.7, seed=1, strict=True)
    assert err.value.packing.status == "pressure"


def test_max_density_defaults_to_jammed():
    p = sp.pack(n=200, density="max", seed=6)
    assert p.status == "jammed" and p.success and isinstance(p.stopped_by, sp.stop.Jammed)
    assert p.median_pressure > 1e9
    assert 0.6 < p.density < 0.75


def test_pressure_criterion_with_max_density():
    p = sp.pack(n=200, density="max", stop=sp.stop.Pressure(1e6), seed=6)
    assert p.status == "pressure" and p.reduced_pressure > 1e6


def test_legacy_rule_rejects_jammed():
    with pytest.raises(ValueError):
        sp.pack(n=50, density="max", collision_rule="legacy")


def test_periodic_images_complete_the_spheres():
    p = sp.pack(n=200, density=0.5, seed=7)
    q = p.periodic_images()
    assert q.n > p.n
    lo = (q.positions - q.radii[:, None] < 0).any(axis=1)
    assert lo.sum() > 0


def test_csv_roundtrip(tmp_path):
    p = sp.pack(n=50, density=0.4, seed=8)
    f = tmp_path / "p.csv"
    p.to_csv(f)
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, :3], p.positions, rtol=1e-12)
    p.to_csv(f, legacy_order=True)
    data = np.loadtxt(f, delimiter=",")
    np.testing.assert_allclose(data[:, 0], p.positions[:, 2], rtol=1e-12)


@pytest.mark.parametrize("bad", [dict(n=10, radii=-1.0), dict(n=10, density=1.2), dict(radii=0.5)])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        sp.pack(**bad)


@pytest.mark.parametrize("rule", ["elastic_growing", "legacy"])
def test_both_collision_rules_pack_without_overlap(rule):
    p = sp.pack(n=300, density=0.6, seed=9, collision_rule=rule)
    assert p.status == "target" and p.collision_rule == rule
    assert min_gap_ratio(p.positions, p.radii, [1.0] * 3, [True] * 3) > -1e-12


def test_collision_budget_is_exact():
    with pytest.warns(sp.PackingWarning):
        p = sp.pack(n=200, density=0.6, stop=sp.stop.Collisions(total=12345), seed=1)
    assert p.status == "collisions" and p.n_collisions == 12345
    assert p.stopped_by == sp.stop.Collisions(total=12345) and not p.success


def test_strictest_criterion_of_a_type_wins():
    with pytest.warns(sp.PackingWarning):
        p = sp.pack(n=100, density=0.6, stop=[sp.stop.Collisions(per_particle=50), sp.stop.Collisions(total=1000)], seed=1)
    assert p.n_collisions == 1000 and p.stopped_by.total == 1000


def test_stall_detects_arrest():
    p = sp.pack(n=200, density="max", stop=sp.stop.Stall(tol=1e-5, window=50), seed=2)
    assert p.status == "stall" and p.success
    assert 0.6 < p.density < 0.75


def test_max_density_needs_an_arrest_criterion():
    with pytest.raises(ValueError):
        sp.pack(n=50, density="max", stop=sp.stop.Collisions(total=10))
    with pytest.raises(ValueError):
        sp.pack(n=50, density=0.5, stop=[])


@pytest.mark.parametrize(
    "bad",
    [lambda: sp.stop.Pressure(0.5), lambda: sp.stop.Collisions(), lambda: sp.stop.Collisions(total=1, per_particle=1),
     lambda: sp.stop.Stall(tol=2.0), lambda: sp.stop.Timeout(0)],
)
def test_invalid_criteria(bad):
    with pytest.raises(ValueError):
        bad()
