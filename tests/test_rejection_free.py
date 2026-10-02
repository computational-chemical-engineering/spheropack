"""Rejection-free Monte Carlo: exact two-particle distributions, many-body checks."""

import math

import numpy as np
import pytest

import spheropack as sp
from spheropack import rejection_free as rf

POTENTIALS = {
    "lj": rf.LennardJones(cutoff=2.5),
    "wca": rf.WCA(),
    "dpd": rf.DPD(a=25.0),
    "hertz": rf.SoftSpheres(epsilon=10.0, alpha=2.5),
    "hard": rf.HardSpheres(1.0),
}


def test_potential_values():
    lj = POTENTIALS["lj"]
    rm = 2 ** (1 / 6)
    shift = 4 * (2.5**-12 - 2.5**-6)
    np.testing.assert_allclose(lj([rm, 2.5, 3.0]), [-1 - shift, 0, 0], atol=1e-14)
    np.testing.assert_allclose(POTENTIALS["dpd"]([0.0, 0.5, 1.0]), [12.5, 3.125, 0.0])
    np.testing.assert_allclose(POTENTIALS["wca"]([rm]), [0.0], atol=1e-14)
    assert np.isinf(POTENTIALS["hard"]([0.99])[0]) and POTENTIALS["hard"]([1.01])[0] == 0


def _two_particle_chi2(potential, dim, kT, seed, n_samples=20000, interval=3.0, L=5.0, bins=40, **method):
    box = sp.PeriodicBox(L, dim=dim)
    x0 = np.full((2, dim), 1.0)
    x0[1, 0] += 1.5
    s = rf.System(x0, box=box, potential=potential, kT=kT, seed=seed, **method)
    r = []
    for x in s.samples(n_samples, interval):
        d = x[1] - x[0]
        d -= L * np.round(d / L)
        r.append(np.linalg.norm(d))
    r = np.array(r)
    rmax = 0.5 * L
    r = r[r < rmax]
    edges = np.linspace(0.0, rmax, bins + 1)
    obs, _ = np.histogram(r, bins=edges)
    # Exact density in the sphere r < L/2: r^(dim-1) exp(-U/kT), integrated per bin.
    fine = np.linspace(0.0, rmax, 200 * bins + 1)[1:]
    w = fine ** (dim - 1) * np.exp(-potential(fine) / kT)
    w = np.where(np.isfinite(w), w, 0.0)
    exp_bins = np.add.reduceat(w, np.arange(0, len(fine), 200))
    exp_bins *= len(r) / exp_bins.sum()
    keep = exp_bins > 5
    chi2 = ((obs[keep] - exp_bins[keep]) ** 2 / exp_bins[keep]).sum()
    return chi2, keep.sum() - 1


@pytest.mark.parametrize("name", list(POTENTIALS))
@pytest.mark.parametrize("dim", [3, 2])
def test_two_particles_sample_boltzmann(name, dim):
    chi2, dof = _two_particle_chi2(POTENTIALS[name], dim, kT=1.0, seed=1)
    # Successive samples are slightly correlated; allow a generous margin.
    assert chi2 < dof + 6 * math.sqrt(2 * dof), (chi2, dof)


def test_temperature_scales_the_distribution():
    chi2, dof = _two_particle_chi2(POTENTIALS["lj"], 3, kT=0.5, seed=2)
    assert chi2 < dof + 6 * math.sqrt(2 * dof), (chi2, dof)


def test_same_seed_same_trajectory():
    a = rf.System(n=100, density=0.3, potential=POTENTIALS["lj"], seed=5).run(10.0)
    b = rf.System(n=100, density=0.3, potential=POTENTIALS["lj"], seed=5).run(10.0)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert a.n_reflections == b.n_reflections > 0


def test_move_speed_is_conserved():
    s = rf.System(n=200, density=0.5, potential=POTENTIALS["dpd"], seed=3)
    v0 = (s.velocities**2).sum()
    s.run(20.0)
    assert abs((s.velocities**2).sum() - v0) < 1e-9 * v0
    np.testing.assert_allclose(
        s.velocities.sum(axis=0),
        rf.System(n=200, density=0.5, potential=POTENTIALS["dpd"], seed=3).velocities.sum(axis=0),
        atol=1e-9,
    )


def test_hard_spheres_never_overlap():
    p = sp.pack(n=200, density=0.4, seed=1)
    s = rf.System(p.positions, box=p.container, potential=rf.HardSpheres(2 * p.radii[0]), seed=1)
    for _ in s.samples(20, 1.0):
        q = sp.analysis.neighbor_pairs(s.snapshot(), 2 * p.radii[0])
        assert len(q[0]) == 0 or np.linalg.norm(q[2], axis=1).min() >= 2 * p.radii[0] * (1 - 1e-9)


def test_invalid_box():
    with pytest.raises(ValueError):
        rf.System(n=10, box=sp.Box([5.0, 5.0, 5.0], periodic=[True, True, False]), potential=POTENTIALS["lj"])
    with pytest.raises(ValueError):
        rf.System(n=10, box=sp.PeriodicBox(4.0), potential=POTENTIALS["lj"])  # shorter than 2 rc


def _bound_fraction(potential, kT, r_b, dim=3, L=5.0, n_samples=30000, interval=2.0, seed=11, **method):
    """Measured and exact P(r < r_b | r < L/2) for two particles, with a batch-means error."""
    s = rf.System(
        np.array([[1.0] * dim, [2.5] + [1.0] * (dim - 1)]),
        box=sp.PeriodicBox(L, dim=dim),
        potential=potential,
        kT=kT,
        seed=seed,
        **method,
    )
    r = []
    for x in s.samples(n_samples, interval):
        d = x[1] - x[0]
        d -= L * np.round(d / L)
        r.append(np.linalg.norm(d))
    r = np.array(r)
    inside = r[r < 0.5 * L] < r_b
    batches = np.array_split(inside.astype(float), 30)
    measured = inside.mean()
    se = np.std([b.mean() for b in batches], ddof=1) / math.sqrt(len(batches))

    def exact(t):
        fine = np.linspace(0.0, 0.5 * L, 400001)[1:]
        w = fine ** (dim - 1) * np.exp(-potential(fine) / t)
        w = np.where(np.isfinite(w), w, 0.0)
        return w[fine < r_b].sum() / w.sum()

    return measured, se, exact


@pytest.mark.parametrize(
    "potential, kT, L, r_b",
    [
        (rf.LennardJones(cutoff=2.5), 0.5, 5.0, 1.5),
        (rf.DPD(a=5.0), 1.0, 2.0, 0.6),
        (rf.SoftSpheres(2.0, 1.0, 2.5), 0.2, 2.0, 0.6),
        (rf.WCA(), 1.0, 2.3, 1.05),
    ],
    ids=["lj", "dpd", "hertz", "wca"],
)
def test_bound_fraction_is_boltzmann_and_test_is_sensitive(potential, kT, L, r_b):
    measured, se, exact = _bound_fraction(potential, kT, r_b, L=L, n_samples=150_000, interval=1.0)
    assert abs(measured - exact(kT)) < 4 * se, (measured, exact(kT), se)
    # The test can see a 20% error in the temperature.
    assert abs(exact(0.8 * kT) - measured) > 4 * se or abs(exact(1.2 * kT) - measured) > 4 * se, (
        measured,
        exact(0.8 * kT),
        exact(1.2 * kT),
        se,
    )


REFERENCE = __import__("pathlib").Path(__file__).parent / "reference"


@pytest.mark.slow
def test_dpd_liquid_matches_paper():
    ref = np.loadtxt(REFERENCE / "GofR_DPD_110925.dat")
    s = rf.System(n=375, box=sp.PeriodicBox(5.0), potential=rf.DPD(25.0), kT=1.0, seed=2)
    s.run(20.0)
    g = rf.radial_distribution(s, n_samples=400, interval=0.5, r_max=2.5, bins=200)
    np.testing.assert_allclose(g.r, ref[:, 0])
    sel = (g.r > 0.2) & (g.r < 2.0)
    assert np.abs(g.g[sel] - ref[sel, 1]).mean() < 0.012


@pytest.mark.slow
def test_lennard_jones_fluid_matches_paper_metropolis():
    ref = np.loadtxt(REFERENCE / "GofR_LJ_all.dat")
    n, rho, T = 1000, 0.317, 1.085
    start = sp.pack(n=n, radii=0.45, container=sp.PeriodicBox((n / rho) ** (1 / 3)), seed=3)
    s = rf.System(start.positions, box=start.container, potential=rf.LennardJones(cutoff=2.5), kT=T, seed=3)
    s.run(100.0)
    g = rf.radial_distribution(s, n_samples=150, interval=2.0, r_max=5.0, bins=1000)
    metropolis = ref[:, [1, 3, 5]].mean(axis=1)
    coarse = lambda a: a.reshape(100, 10).mean(axis=1)
    assert np.abs(coarse(g.g) - coarse(metropolis)).mean() < 0.006


def test_head_on_lennard_jones_pair_reflects_in_the_core():
    # Exactly head-on: the closest approach rounds to r = 0, where U is inf - inf.
    for seed in range(20):
        s = rf.System(
            np.array([[1.0, 1.0], [2.5, 1.0]]),
            box=sp.PeriodicBox(5.0, dim=2),
            potential=rf.LennardJones(cutoff=2.5),
            kT=1.0,
            seed=seed,
        )
        s.set_velocities(np.array([[1.0, 0.0], [-1.0, 0.0]]))
        dmin = min(abs(x[1, 0] - x[0, 0]) for x in s.samples(150, 0.01))
        assert dmin > 0.8


def test_hard_spheres_starting_in_round_off_contact_separate():
    x = np.array([[1.0, 1.0, 1.0], [2.0 - 1e-15, 1.0, 1.0]])
    s = rf.System(x, box=sp.PeriodicBox(5.0), potential=rf.HardSpheres(1.0), seed=1)
    s.set_velocities(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    for y in s.samples(50, 0.01):
        d = y[1] - y[0]
        d -= 5.0 * np.round(d / 5.0)
        assert np.linalg.norm(d) > 1.0 - 1e-9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        rf.System(np.zeros((0, 3)), box=sp.PeriodicBox(5.0), potential=rf.WCA())
    s = rf.System(n=10, density=0.1, potential=rf.WCA(), seed=1)
    with pytest.raises(ValueError):
        s.run(-1.0)
    with pytest.raises(ValueError):
        s.run(float("nan"))
    with pytest.raises(ValueError):
        rf.radial_distribution(s, n_samples=0, interval=1.0)


EVENT_CHAIN = {
    "irreversible": {"method": "event_chain", "chain_length": 1.0},
    "reversible": {"method": "event_chain", "chain_length": 1.0, "irreversible": False},
}


@pytest.mark.parametrize("variant", list(EVENT_CHAIN))
@pytest.mark.parametrize("name", ["lj", "dpd", "hard"])
@pytest.mark.parametrize("dim", [3, 2])
def test_event_chain_two_particles_sample_boltzmann(variant, name, dim):
    chi2, dof = _two_particle_chi2(POTENTIALS[name], dim, kT=1.0, seed=3, **EVENT_CHAIN[variant])
    assert chi2 < dof + 6 * math.sqrt(2 * dof), (chi2, dof)


@pytest.mark.parametrize("variant", list(EVENT_CHAIN))
def test_event_chain_bound_fraction(variant):
    measured, se, exact = _bound_fraction(
        rf.LennardJones(cutoff=2.5), 0.5, 1.5, L=5.0, n_samples=150_000, interval=1.0, **EVENT_CHAIN[variant]
    )
    assert abs(measured - exact(0.5)) < 4 * se, (measured, exact(0.5), se)
    assert abs(exact(0.4) - measured) > 4 * se or abs(exact(0.6) - measured) > 4 * se


def test_event_chain_has_no_velocities_and_counts_time():
    s = rf.System(n=50, density=0.2, potential=rf.WCA(), seed=1, method="event_chain")
    s.run(3.0)
    assert s.time == pytest.approx(3.0)
    with pytest.raises(AttributeError):
        s.velocities
    with pytest.raises(AttributeError):
        s.redraw_velocities()
    assert np.all((s.positions >= 0) & (s.positions < np.asarray(s.box.lengths)))
    with pytest.raises(ValueError):
        rf.System(n=50, density=0.2, potential=rf.WCA(), method="nonsense")


@pytest.mark.slow
def test_event_chain_lennard_jones_matches_paper():
    ref = np.loadtxt(REFERENCE / "GofR_LJ_all.dat")
    n, rho, T = 1000, 0.317, 1.085
    start = sp.pack(n=n, radii=0.45, container=sp.PeriodicBox((n / rho) ** (1 / 3)), seed=4)
    s = rf.System(
        start.positions, box=start.container, potential=rf.LennardJones(cutoff=2.5), kT=T, seed=4, method="event_chain"
    )
    s.run(100.0)
    g = rf.radial_distribution(s, n_samples=300, interval=1.0, r_max=5.0, bins=1000)
    metropolis = ref[:, [1, 3, 5]].mean(axis=1)
    coarse = lambda a: a.reshape(100, 10).mean(axis=1)
    assert np.abs(coarse(g.g) - coarse(metropolis)).mean() < 0.006
