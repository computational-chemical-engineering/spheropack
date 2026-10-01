"""Tests of the event-driven dynamics: exact collisions, equation of state, stress."""

import math

import numpy as np
import pytest

import spheropack as sp
from spheropack import _core
from helpers import min_gap_ratio

INF = float("inf")


def _run(radii, lengths, periodic, *, growth_rate, positions, scale, velocities, max_time=INF,
         rule="elastic_growing", target=INF, max_collisions=0, jammed=INF):
    return _core._ls_pack3(
        np.asarray(radii, float), list(lengths), list(periodic), target, growth_rate, INF, jammed,
        0.0, 100.0, max_collisions, max_time, INF, 10.0, 0, rule,
        np.ascontiguousarray(positions, float), scale, np.ascontiguousarray(velocities, float),
    )


@pytest.mark.parametrize("rule", ["elastic_growing", "legacy"])
def test_head_on_collision_of_growing_spheres(rule):
    # Two spheres of radius 0.5 on the x axis, gap 1, approaching at relative speed
    # 1.2 (before the run rescales velocities to kT = 1).
    gamma = 0.1
    v = np.array([[1.0, 0, 0], [-0.2, 0, 0]])
    f = 1.0 / math.sqrt((v**2).sum() / 6)  # rescaling to kT = 1 at the start
    u = 1.2 * f
    dsig = gamma  # ds/dt = gamma / mean diameter = gamma; sigma = s
    t_contact = 1.0 / (u + dsig)
    out = _run([0.5, 0.5], [10, 10, 10], [True] * 3, growth_rate=gamma,
               positions=[[4, 5, 5], [6, 5, 5]], scale=1.0, velocities=v, max_time=t_contact * 1.01, rule=rule)
    assert out["n_collisions"] == 1
    un_after = 2 * dsig + u if rule == "elastic_growing" else max(u, dsig)
    delta = un_after + u
    vf = out["velocities"]
    np.testing.assert_allclose(vf[0], [f * 1.0 - delta / 2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(vf[1], [-f * 0.2 + delta / 2, 0, 0], atol=1e-12)
    assert out["max_contact_error"] < 1e-12


@pytest.mark.parametrize("rule", ["elastic_growing", "legacy"])
def test_wall_collision_of_a_growing_sphere(rule):
    gamma = 0.1
    v = np.array([[-1.0, 0.0, 0.0]])
    f = math.sqrt(3.0)  # kT = 1/3 rescaled to 1
    drho = 0.5 * gamma  # ds/dt = gamma, radius 0.5 s
    t_contact = (2.0 - 0.5) / (f + drho)
    out = _run([0.5], [10, 10, 10], [False, True, True], growth_rate=gamma,
               positions=[[2, 5, 5]], scale=1.0, velocities=v, max_time=t_contact * 1.01, rule=rule)
    expected = 2 * drho + f if rule == "elastic_growing" else f
    assert out["velocities"][0, 0] == pytest.approx(expected, abs=1e-12)


def _carnahan_starling(phi):
    return (1 + phi + phi**2 - phi**3) / (1 - phi) ** 3


@pytest.mark.parametrize("phi", [0.3, 0.45])
def test_equation_of_state_at_fixed_radii(phi):
    start = sp.pack(n=1000, density=phi, seed=1)
    rng = np.random.default_rng(2)
    v = rng.normal(size=(1000, 3))
    v -= v.mean(axis=0)
    out = _run(start.radii, [1.0] * 3, [True] * 3, growth_rate=0.0, positions=start.positions,
               scale=1.0, velocities=v, max_collisions=300_000)
    z = out["history"]["reduced_pressure"][3:]  # skip the first windows (relaxation)
    assert z.mean() == pytest.approx(_carnahan_starling(phi), rel=0.02)


@pytest.mark.slow
def test_slow_growth_follows_equation_of_state():
    # Elastic in the growing frame plus thermostat: slow compression is quasi-static.
    p = sp.pack(n=1000, density=0.5, growth_rate=0.002, seed=3)
    h = p.history
    phi = p.density * (h["scale"] / h["scale"][-1]) ** 3
    sel = (phi > 0.25) & (phi < 0.45)
    np.testing.assert_allclose(h["reduced_pressure"][sel], _carnahan_starling(phi[sel]), rtol=0.05)


SCENARIOS = [
    # dim, container, n, size ratio, density
    (3, sp.PeriodicBox(1.0), 5, 1.0, "max"),
    (3, sp.PeriodicBox(1.0), 30, 1.0, "max"),
    (3, sp.PeriodicBox([1.0, 0.6, 2.0]), 200, 3.0, "max"),
    (3, sp.Box([1.0, 1.0, 1.0], periodic=False), 150, 1.0, "max"),
    (3, sp.Box([1.0, 1.0, 1.0], periodic=[True, False, True]), 200, 5.0, 0.5),
    (3, sp.PeriodicBox(1.0), 400, 10.0, 0.55),
    (2, sp.PeriodicBox(1.0, dim=2), 5, 1.0, "max"),
    (2, sp.PeriodicBox(1.0, dim=2), 300, 1.4, "max"),
    (2, sp.Box([1.0, 3.0], periodic=False), 300, 2.0, "max"),
    (2, sp.Box([1.0, 1.0], periodic=[False, True]), 400, 1.0, 0.8),
]


@pytest.mark.parametrize("dim, container, n, ratio, density", SCENARIOS)
@pytest.mark.parametrize("seed", [1, 2])
def test_stress_scenarios(dim, container, n, ratio, density, seed):
    rng = np.random.default_rng(seed)
    radii = np.exp(rng.uniform(0, math.log(ratio), n)) if ratio > 1 else np.ones(n)
    stop = [sp.stop.Jammed(pressure=1e7), sp.stop.Timeout(60)] if density == "max" else None
    p = sp.pack(radii=radii, density=density, container=container, stop=stop, seed=seed, strict=True)
    assert p.status in ("jammed", "target")
    assert p.max_contact_error < 1e-9
    assert np.all(p.positions >= 0) and np.all(p.positions < np.array(container.lengths))
    assert min_gap_ratio(p.positions, p.radii, container.lengths, container.periodic) > -1e-12
    assert p.shrink_factor > 1 - 1e-12
