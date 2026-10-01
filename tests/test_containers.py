import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import spheropack as sp
from spheropack import analysis
from helpers import min_gap_ratio


def _ball_gap(p):
    """min over spheres of (R - distance from axis/centre) / r - 1."""
    ball = p.container.ball
    ax = np.flatnonzero(ball.axes)
    q = p.positions[:, ax] - np.asarray(ball.center)[ax]
    return ((ball.radius - np.sqrt((q**2).sum(axis=1))) / p.radii).min() - 1.0


CONTAINERS = [
    sp.Cylinder(diameter=1.0, length=2.0),
    sp.Cylinder(diameter=1.0, length=1.0, capped=True),
    sp.Cylinder(diameter=1.0, length=1.5, axis=0),
    sp.SphereContainer(1.0),
    sp.Disk(1.0),
]
IDS = ["tube", "capped", "tube_x", "sphere", "disk"]


@pytest.mark.parametrize("container", CONTAINERS, ids=IDS)
@pytest.mark.parametrize("rule", ["elastic_growing", "legacy"])
def test_curved_containers_keep_spheres_inside(container, rule):
    density = 0.5 if container.dim == 3 else 0.7
    p = sp.pack(n=300, density=density, container=container, seed=1, collision_rule=rule)
    assert p.status == "target"
    assert p.density == pytest.approx(density, rel=1e-12)
    assert _ball_gap(p) > -1e-12
    # pairs (the bounding box is non-periodic along the ball axes)
    assert min_gap_ratio(p.positions, p.radii, container.lengths, container.periodic) > -1e-12
    assert p.max_contact_error < 1e-9


@pytest.mark.parametrize("container", CONTAINERS, ids=IDS)
def test_curved_containers_jam(container):
    p = sp.pack(n=300, density="max", container=container, seed=2, stop=[sp.stop.Jammed(1e7), sp.stop.Timeout(60)])
    assert p.status == "jammed"
    assert _ball_gap(p) > -1e-12
    assert 0.5 < p.density < 0.9


def test_container_volumes():
    assert sp.Cylinder(2.0, 3.0).volume == pytest.approx(math.pi * 3.0)
    assert sp.SphereContainer(2.0).volume == pytest.approx(4 / 3 * math.pi)
    assert sp.Disk(2.0).volume == pytest.approx(math.pi)


@pytest.mark.parametrize("container", [sp.Cylinder(1.0, 2.0), sp.SphereContainer(1.0), sp.Disk(1.0)], ids=["tube", "sphere", "disk"])
def test_radial_profile_integrates_to_density(container):
    p = sp.pack(n=300, density=0.5 if container.dim == 3 else 0.7, container=container, seed=3)
    r, phi = analysis.radial_profile(p, bins=400)
    R = container.ball.radius
    m = sum(container.ball.axes)
    weight = r ** (m - 1)  # shell measure ~ r^(m-1) dr
    assert (phi * weight).sum() / weight.sum() == pytest.approx(p.density, rel=2e-3)
    assert phi[-1] < 0.05  # the wall touches spheres in points (lines) only


def test_sphere_larger_than_container_is_rejected():
    with pytest.raises(ValueError):
        sp.pack(n=1, density=0.9, container=sp.Cylinder(1.0, 0.1))


def test_ball_must_fit_box():
    with pytest.raises(ValueError):
        sp.Box([1.0, 1.0, 1.0], periodic=False, ball=sp.Ball((True, True, False), (0.5, 0.5, 0.0), 0.6))


@pytest.mark.slow
def test_tube_profile_matches_legacy():
    # Legacy baseline: 725 spheres d = 3 mm in a tube D = 21 mm, density 0.54, legacy rule.
    base = json.loads((Path(__file__).parent / "baselines" / "tube_D7.json").read_text())
    d, D = 3e-3, 21e-3
    length = 725 * (math.pi / 6) * d**3 / 0.54 / (0.25 * math.pi * D**2)
    hists = []
    for seed in range(101, 111):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sp.PackingWarning)  # occasionally stuck just below 0.54
            p = sp.pack(n=725, radii=0.5 * d, container=sp.Cylinder(D, length), density=0.54, growth_rate=0.16,
                        seed=seed, collision_rule="legacy", stop=sp.stop.Collisions(per_particle=1e4))
        q = p.positions[:, :2] - 0.5 * D
        rc = np.sqrt((q**2).sum(axis=1))
        h, edges = np.histogram(rc, bins=40, range=(0, 0.5 * D - 0.5 * d))
        uniform = len(rc) * np.diff(edges**2) / edges[-1] ** 2
        hists.append(h / uniform)
    mean = np.mean(hists, axis=0)
    ref = np.asarray(base["summary"]["radial_histogram_mean"])
    ref_std = np.asarray(base["summary"]["radial_histogram_std"])
    err = np.hypot(np.std(hists, axis=0, ddof=1), ref_std) / np.sqrt(10)
    z = (mean - ref) / np.maximum(err, 1e-3)
    assert np.mean(z**2) < 2.0, z.round(1)


@pytest.mark.parametrize("ratio", [7.0, 20.0])
def test_cylinder_radial_profile_quadrature_converged(ratio):
    p = sp.pack(n=400, density=0.5, container=sp.Cylinder(ratio * 0.1, 1.0), radii=0.05, seed=4)
    _, coarse = analysis.radial_profile(p, bins=120)
    _, fine = analysis.radial_profile(p, bins=120, n_theta=4096)
    assert np.abs(coarse - fine).max() < 1e-3


def test_density_profile_rejects_curved_axes():
    p = sp.pack(n=100, density=0.4, container=sp.Cylinder(1.0, 1.0), seed=1)
    analysis.density_profile(p, axis=2)  # along the tube axis: fine
    with pytest.raises(ValueError):
        analysis.density_profile(p, axis=0)


def test_ball_radius_must_be_positive():
    with pytest.raises(ValueError):
        sp.Box([1.0, 1.0, 1.0], periodic=False, ball=sp.Ball((True, True, False), (0.5, 0.5, 0.0), -0.5))


@pytest.mark.parametrize("container", [sp.Cylinder(1.0, 0.3, capped=True), sp.Box([1.0, 1.0, 0.3], periodic=[True, True, False])],
                         ids=["short_capped_tube", "thin_slab"])
def test_spheres_limited_by_flat_walls_stop(container):
    p = sp.pack(n=3, density="max", container=container, seed=1, stop=[sp.stop.Jammed(), sp.stop.Timeout(10)])
    assert p.status in ("box_limit", "jammed") and p.wall_time < 2
    assert 2 * p.radii.max() <= 0.3 * (1 + 1e-12)
    with pytest.raises(ValueError):
        sp.pack(n=1, density=0.5, container=container)
