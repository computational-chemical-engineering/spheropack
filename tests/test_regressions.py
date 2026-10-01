"""Regression tests for problems found in code review (hangs and edge cases)."""

import numpy as np
import pytest

import spheropack as sp
from spheropack import analysis


def test_largest_sphere_reaching_the_box_size_stops():
    # Used to re-sync forever once the big sphere touched its own periodic image.
    p = sp.pack(radii=[3.0, 1.0], density="max", seed=1, stop=[sp.stop.Jammed(), sp.stop.Timeout(10)])
    assert p.status == "box_limit" and p.success
    assert 2 * p.radii.max() <= 1.0 + 1e-12
    assert p.wall_time < 1


def test_thin_periodic_box_stops():
    p = sp.pack(n=2, density="max", container=sp.PeriodicBox([1.0, 0.3, 1.0]), seed=1,
                stop=[sp.stop.Jammed(), sp.stop.Timeout(10)])
    assert p.status in ("box_limit", "jammed") and p.wall_time < 1


def test_target_with_a_sphere_larger_than_the_box_is_rejected():
    with pytest.raises(ValueError):
        sp.pack(n=1, density=0.6)


def test_zero_growth_rate_is_rejected():
    with pytest.raises(ValueError):
        sp.pack(n=10, density=0.3, growth_rate=0.0)


def test_large_jamming_steps_terminate():
    # A growth step that runs into jamming used to continue forever (Zeno).
    for seed in range(3):
        p = sp.pack(n=200, density="max", seed=seed, stop=[sp.stop.Jammed(step=0.5), sp.stop.Timeout(30)])
        assert p.status == "jammed"
    with pytest.raises(ValueError):
        sp.stop.Jammed(step=0.9)  # cycles without progress: overshoot, then relaxation


def test_force_rattlers_at_moderate_pressure():
    p = sp.pack(n=500, density="max", stop=sp.stop.Pressure(1e3), seed=2)
    rat = analysis.rattlers(p, method="force", force_threshold=0.05)
    assert 0.0 < rat.mean() < 0.5


def test_spheres_held_by_walls_are_not_rattlers():
    box = sp.Box([1.0, 1.0, 1.0], periodic=False)
    p = sp.pack(n=300, density="max", container=box, seed=3)
    rat = analysis.rattlers(p, method="force")
    touching_wall = ((p.positions - p.radii[:, None] < 1e-9) | (p.positions + p.radii[:, None] > 1 - 1e-9)).any(axis=1)
    assert rat[touching_wall].mean() < rat.mean() + 0.05


def test_collision_budget_validation():
    with pytest.raises(ValueError):
        sp.stop.Collisions(total=0)
    with pytest.raises(ValueError):
        sp.stop.Collisions(total=0.5)
    assert sp.stop.Collisions(per_particle=1e-9).budget(10) == 1


def test_neighbor_cutoff_larger_than_box_is_rejected():
    p = sp.pack(n=20, density=0.3, seed=1)
    with pytest.raises(ValueError):
        analysis.neighbor_pairs(p, 1.5)


def test_csv_keeps_touching_spheres_apart(tmp_path):
    p = sp.pack(n=300, density="max", seed=4)
    f = tmp_path / "p.csv"
    p.to_csv(f, header=False)
    data = np.loadtxt(f, delimiter=",")
    from helpers import min_gap_ratio
    assert min_gap_ratio(data[:, :3], data[:, 3], [1.0] * 3, [True] * 3) > -1e-15
