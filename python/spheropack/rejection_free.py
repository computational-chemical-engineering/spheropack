"""Rejection-free, event-driven Monte Carlo sampling for pair potentials.

The method of E.A.J.F. Peters and G. de With, *Rejection-free Monte Carlo sampling
for general potentials*, Phys. Rev. E 85, 026703 (2012). Particles move along straight
lines with constant velocities. For every pair, the uphill part of the pair potential
along the current path is accumulated, and the pair reflects (an elastic collision of
equal masses) when the accumulated increase reaches :math:`-kT \\ln u` with :math:`u`
uniform. No move is ever rejected. Positions sampled at equidistant times follow the
canonical distribution :math:`\\exp(-U/kT)`; the velocities only set the move
directions and are not physical momenta.

Example
-------
>>> import spheropack as sp
>>> from spheropack import rejection_free as rf
>>> system = rf.System(n=500, density=0.3, potential=rf.LennardJones(cutoff=2.5), kT=1.0, seed=1)
>>> system.run(100.0)
>>> g = rf.radial_distribution(system, n_samples=100, interval=1.0)
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from . import _core, analysis
from .containers import Box, PeriodicBox
from .packing import Packing

__all__ = [
    "DPD",
    "WCA",
    "HardSpheres",
    "LennardJones",
    "PairPotential",
    "SoftSpheres",
    "System",
    "radial_distribution",
]


@dataclass(frozen=True)
class PairPotential:
    """Isotropic pair potential with at most one minimum and zero at the cutoff.

    Use the constructors :class:`LennardJones`, :class:`WCA`, :class:`DPD`,
    :class:`SoftSpheres` and :class:`HardSpheres`.
    """

    kind: Literal["soft", "lennard_jones", "hard"]
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 1.0
    alpha: float = 2.0
    name: str = field(default="", compare=False)

    def __call__(self, r: np.ndarray | float) -> np.ndarray:
        """Potential energy at separation ``r``."""
        r = np.ascontiguousarray(np.atleast_1d(r), dtype=float)
        return _core._potential_values(self.kind, self.epsilon, self.sigma, self.cutoff, self.alpha, r)

    @property
    def display_size(self) -> float:
        """Diameter used to draw the particles (``sigma``)."""
        return self.sigma


def LennardJones(epsilon: float = 1.0, sigma: float = 1.0, cutoff: float = 2.5) -> PairPotential:
    """Truncated and shifted Lennard-Jones potential,
    :math:`4\\epsilon[(\\sigma/r)^{12} - (\\sigma/r)^6] - U_\\mathrm{LJ}(r_c)` for :math:`r < r_c`."""
    return PairPotential("lennard_jones", epsilon, sigma, cutoff, name=f"LennardJones(rc={cutoff:g})")


def WCA(epsilon: float = 1.0, sigma: float = 1.0) -> PairPotential:
    """Weeks-Chandler-Andersen potential: Lennard-Jones truncated and shifted at its minimum."""
    return PairPotential("lennard_jones", epsilon, sigma, 2.0 ** (1.0 / 6.0) * sigma, name="WCA")


def SoftSpheres(epsilon: float = 1.0, sigma: float = 1.0, alpha: float = 2.0) -> PairPotential:
    """Soft repulsion :math:`\\frac{\\epsilon}{\\alpha}(1 - r/\\sigma)^\\alpha` for :math:`r < \\sigma`
    (harmonic for :math:`\\alpha = 2`, Hertzian for :math:`\\alpha = 5/2`)."""
    return PairPotential("soft", epsilon, sigma, sigma, alpha, name=f"SoftSpheres(alpha={alpha:g})")


def DPD(a: float = 25.0, cutoff: float = 1.0) -> PairPotential:
    """Conservative DPD potential :math:`\\frac{a}{2}(1 - r/r_c)^2` for :math:`r < r_c`."""
    return PairPotential("soft", a, cutoff, cutoff, 2.0, name=f"DPD(a={a:g})")


def HardSpheres(diameter: float = 1.0) -> PairPotential:
    """Hard spheres: every approach to contact reflects (event-driven hard-sphere dynamics)."""
    return PairPotential("hard", 1.0, diameter, diameter, name="HardSpheres")


class System:
    """Particles interacting with a pair potential in a periodic box, sampled with the
    rejection-free method.

    Parameters
    ----------
    positions:
        Initial positions, shape ``(n, dim)``. If omitted, ``n`` particles are placed
        uniformly at random (for strongly repulsive potentials start from a packing
        instead, e.g. ``sp.pack(...).positions``, to avoid huge initial energies).
    n, density:
        Number of particles and number density; with ``box`` omitted, a cube (square
        for ``dim=2``) of the matching size is used.
    box:
        Periodic box (:func:`spheropack.PeriodicBox`); edges at least twice the cutoff.
    potential:
        The pair potential.
    kT:
        Temperature.
    seed:
        Random seed. ``None`` draws one.
    dim:
        Dimension when neither positions nor box are given.
    method:
        ``"collisions"`` (default): all particles move simultaneously and pairs reflect
        (the first implementation of the paper). ``"event_chain"``: one particle moves at
        a time along a coordinate axis and passes its motion to the partner at a
        reflection (the straight event-chain variant), usually faster for dense systems.
    chain_length:
        Displacement per chain (``event_chain`` only).
    irreversible:
        ``event_chain`` only: cycle the directions +x, +y, +z (default, faster; global
        balance) instead of random axes and signs (detailed balance).

    Time is measured as the mean displacement per particle in both methods: ``run(t)``
    moves every particle along its path for ``t`` (collisions) or performs chains with
    a total displacement ``n * t`` (event chain).
    """

    def __init__(
        self,
        positions: np.ndarray | None = None,
        *,
        n: int | None = None,
        density: float | None = None,
        box: Box | None = None,
        potential: PairPotential,
        kT: float = 1.0,
        seed: int | None = None,
        dim: int = 3,
        method: Literal["collisions", "event_chain"] = "collisions",
        chain_length: float = 1.0,
        irreversible: bool = True,
    ):
        if seed is None:
            entropy = np.random.SeedSequence().entropy
            assert isinstance(entropy, int)
            seed = entropy % 2**64
        self.seed = int(seed)
        if positions is not None:
            positions = np.ascontiguousarray(positions, dtype=float)
            n, dim = positions.shape
        if n is None:
            raise ValueError("give positions or n")
        if box is None:
            if density is None:
                raise ValueError("give box or density")
            box = PeriodicBox((n / density) ** (1.0 / dim), dim=dim)
        if not all(box.periodic) or box.ball is not None:
            raise ValueError("the box must be fully periodic")
        if positions is None:
            rng = np.random.default_rng(self.seed)
            positions = rng.uniform(0.0, 1.0, (n, box.dim)) * np.asarray(box.lengths)
        if positions.shape != (n, box.dim):
            raise ValueError("positions must have shape (n, dim) matching the box")
        self.box = box
        self.potential = potential
        self.kT = float(kT)
        p = potential
        args = (positions, list(box.lengths), p.kind, p.epsilon, p.sigma, p.cutoff, p.alpha, self.kT, self.seed)
        self._sim: Any  # _RejectionFree2/3 or _EventChain2/3
        if method == "collisions":
            cls = _core._RejectionFree3 if box.dim == 3 else _core._RejectionFree2
            self._sim = cls(*args)
        elif method == "event_chain":
            if not chain_length > 0:
                raise ValueError("chain_length must be positive")
            ec = _core._EventChain3 if box.dim == 3 else _core._EventChain2
            self._sim = ec(*args, irreversible)
        else:
            raise ValueError("method must be 'collisions' or 'event_chain'")
        self.method = method
        self.chain_length = float(chain_length)
        self._n = len(positions)

    @property
    def n(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self.box.dim

    @property
    def density(self) -> float:
        """Number density."""
        return self.n / self.box.volume

    @property
    def positions(self) -> np.ndarray:
        """Current positions (a copy), wrapped into the box."""
        return self._sim.positions()

    @property
    def velocities(self) -> np.ndarray:
        """Current move velocities (a copy); ``collisions`` method only."""
        if self.method != "collisions":
            raise AttributeError("the event-chain method has no velocities")
        return self._sim.velocities()

    @property
    def time(self) -> float:
        """Simulation time since the start: the mean displacement per particle."""
        return self._sim.time if self.method == "collisions" else self._sim.displacement / self._n

    @property
    def n_reflections(self) -> int:
        """Number of reflections (lifts in the event-chain method)."""
        return self._sim.n_reflections if self.method == "collisions" else self._sim.n_lifts

    def run(self, time: float) -> System:
        """Advances the simulation by ``time``. Ctrl-C stops it and raises ``KeyboardInterrupt``."""
        if self.method == "collisions":
            interrupted = self._sim.run(float(time))
        else:
            interrupted = self._sim.run(float(time) * self._n, self.chain_length)
        if interrupted:
            raise KeyboardInterrupt
        return self

    def set_velocities(self, velocities: np.ndarray) -> None:
        """Sets the move velocities, shape ``(n, dim)``; ``collisions`` method only."""
        if self.method != "collisions":
            raise AttributeError("the event-chain method has no velocities")
        self._sim.set_velocities(np.ascontiguousarray(velocities, dtype=float))

    def redraw_velocities(self) -> None:
        """Draws new move velocities (standard normal); ``collisions`` method only."""
        if self.method != "collisions":
            raise AttributeError("the event-chain method has no velocities")
        self._sim.redraw_velocities()

    def potential_energy(self) -> float:
        """Total potential energy of the current configuration."""
        return self._sim.potential_energy()

    def samples(self, n_samples: int, interval: float, redraw_velocities: bool = False) -> Iterator[np.ndarray]:
        """Yields the positions ``n_samples`` times, every ``interval`` of simulation time.

        Equidistant sampling in time is what makes the samples canonical.
        """
        for _ in range(n_samples):
            self.run(interval)
            if redraw_velocities and self.method == "collisions":
                self.redraw_velocities()
            yield self.positions

    def snapshot(self) -> Packing:
        """The current configuration as a :class:`~spheropack.Packing` (spheres of
        diameter ``sigma``), for :mod:`spheropack.analysis` and :mod:`spheropack.io`."""
        x = self.positions
        r = np.full(self.n, 0.5 * self.potential.display_size)
        frac = len(x) * (math.pi / 6 if self.dim == 3 else math.pi / 4) * self.potential.display_size**self.dim
        return Packing(
            x,
            r,
            self.box,
            "sample",
            frac / self.box.volume,
            0.0,
            int(self.n_reflections),
            0,
            self.time,
            0.0,
            0.0,
            1.0,
            self.seed,
            0.0,
        )

    def __repr__(self) -> str:
        return (
            f"System(n={self.n}, dim={self.dim}, density={self.density:.4g}, potential={self.potential.name}, "
            f"kT={self.kT:g}, method={self.method!r}, time={self.time:.4g}, n_reflections={self.n_reflections})"
        )


def radial_distribution(
    system: System, n_samples: int, interval: float, r_max: float | None = None, bins: int = 200
) -> analysis.RadialDistribution:
    """Pair distribution function averaged over ``n_samples`` configurations taken every
    ``interval`` of simulation time (the system advances by ``n_samples * interval``)."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if r_max is None:
        r_max = 0.5 * min(system.box.lengths)
    total = None
    for _ in system.samples(n_samples, interval):
        rdf = analysis.radial_distribution(system.snapshot(), r_max=r_max, bins=bins)
        total = rdf.g if total is None else total + rdf.g
    assert total is not None
    return analysis.RadialDistribution(rdf.r, total / n_samples)
