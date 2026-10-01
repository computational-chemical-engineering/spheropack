"""Stopping criteria for :func:`spheropack.pack`.

A run ends when the target density is reached or when the first of the criteria in
``stop=`` fires. *Arrest* criteria (:class:`Pressure`, :class:`Stall`) detect that the
spheres can no longer grow: the packing is jammed or crystallised. *Limit* criteria
(:class:`Collisions`, :class:`Timeout`) bound the cost of a run.

Examples
--------
>>> import spheropack as sp
>>> p = sp.pack(n=500, density="max", stop=[sp.stop.Pressure(1e9), sp.stop.Timeout(60)])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Iterable

__all__ = ["Collisions", "Criterion", "DEFAULT", "DEFAULT_MAX", "Jammed", "Pressure", "Stall", "Timeout"]


@dataclass(frozen=True)
class Criterion:
    """Base class of all stopping criteria."""

    kind: ClassVar[str] = ""  # "arrest" or "limit"
    name: ClassVar[str] = ""  # status reported when the criterion fires


@dataclass(frozen=True)
class Jammed(Criterion):
    """Compress quasi-statically until the packing is jammed.

    The spheres grow at the requested growth rate until the median per-sphere reduced
    pressure reaches ``start_pressure``. From then on the radii alternate between
    relaxation at fixed size (``relax_windows`` pressure windows of 10 collisions per
    sphere, energy conserved) and growth steps that close the fraction ``step`` of
    the estimated remaining gap to jamming (relative size ``step / Z``). The run
    stops when the median pressure in a relaxation window (fixed radii) exceeds
    ``pressure``.

    The median is insensitive to rattlers and to a few spheres that collide at very
    high rates. Slower protocols (more relaxation, smaller steps) come closer to an
    isostatic contact network at higher cost. For 1000 equal spheres at growth rate
    0.02 the force-bearing contacts reach about 94% of the isostatic number with
    ``relax_windows=2, step=0.5``, 97% with the defaults (1.75 times the run time) and
    98% with ``relax_windows=4, step=0.1`` (3 times). Pressures above about ``1e11``
    bring the gaps between touching spheres close to round-off.

    ``step`` is limited to 0.5: larger steps overshoot, the packing then relaxes back
    to a lower pressure during the next relaxation, and the protocol can cycle without
    progress.

    Requires the ``"elastic_growing"`` collision rule.
    """

    pressure: float = 1e9
    relax_windows: int = 4
    step: float = 0.2
    start_pressure: float = 1e3
    kind: ClassVar[str] = "arrest"
    name: ClassVar[str] = "jammed"

    def __post_init__(self):
        if not self.pressure > 1:
            raise ValueError("Jammed pressure must exceed 1")
        if not (self.relax_windows >= 1 and 0 < self.step <= 0.5 and self.start_pressure > 1):
            raise ValueError("Jammed needs relax_windows >= 1, 0 < step <= 0.5 and start_pressure > 1")


@dataclass(frozen=True)
class Pressure(Criterion):
    """Stop when the reduced pressure ``Z = PV/(N kT)`` exceeds ``value``.

    ``Z`` is the mean over all spheres, measured from the collision virial over
    windows of 10 collisions per sphere while the spheres keep growing. It can spike
    when a few spheres collide at very high rates; :class:`Jammed` uses the median
    and is the better choice to obtain jammed packings. It diverges at jamming as roughly ``Z ~ D / (1 - phi/phi_J)`` for
    quasi-static growth, so ``Pressure(1e6)`` stops at about ``3e-6`` relative
    distance from the jamming density in 3D. A clear contact network (for counting
    contacts) needs about ``1e9``.
    """

    value: float = 1e6
    kind: ClassVar[str] = "arrest"
    name: ClassVar[str] = "pressure"

    def __post_init__(self):
        if not self.value > 1:
            raise ValueError("Pressure value must exceed 1 (the ideal-gas value)")


@dataclass(frozen=True)
class Stall(Criterion):
    """Stop when the radii grew by less than ``tol`` (relative) during the last
    ``window`` collisions per sphere.

    Measures progress directly and does not assume quasi-equilibrium. Together with
    :class:`Jammed` it usually fires first, around ``Z ~ 1e5``, because the protocol
    grows the spheres in small steps; use one or the other.
    """

    tol: float = 1e-6
    window: float = 100.0
    kind: ClassVar[str] = "arrest"
    name: ClassVar[str] = "stall"

    def __post_init__(self):
        if not (0 < self.tol < 1 and self.window > 0):
            raise ValueError("Stall needs 0 < tol < 1 and window > 0")


@dataclass(frozen=True)
class Collisions(Criterion):
    """Stop after ``total`` collisions, or ``per_particle`` collisions per sphere."""

    total: int | None = None
    per_particle: float | None = None
    kind: ClassVar[str] = "limit"
    name: ClassVar[str] = "collisions"

    def __post_init__(self):
        if (self.total is None) == (self.per_particle is None):
            raise ValueError("give exactly one of total and per_particle")
        value = self.total if self.total is not None else self.per_particle
        if not value > 0:
            raise ValueError("the collision budget must be positive")
        if self.total is not None and int(self.total) != self.total:
            raise ValueError("total must be a whole number")

    def budget(self, n: int) -> int:
        return int(self.total) if self.total is not None else max(1, math.ceil(self.per_particle * n))


@dataclass(frozen=True)
class Timeout(Criterion):
    """Stop after ``seconds`` of wall-clock time (not reproducible between machines)."""

    seconds: float
    kind: ClassVar[str] = "limit"
    name: ClassVar[str] = "timeout"

    def __post_init__(self):
        if not self.seconds > 0:
            raise ValueError("Timeout needs seconds > 0")


DEFAULT: tuple[Criterion, ...] = (Pressure(1e6),)
"""Default criteria for a target density: stop early if the packing jams first."""
DEFAULT_MAX: tuple[Criterion, ...] = (Jammed(),)
"""Default criteria for ``density="max"``."""


def _normalise(stop: Criterion | Iterable[Criterion] | None, maximum: bool = False) -> tuple[Criterion, ...]:
    if stop is None:
        return DEFAULT_MAX if maximum else DEFAULT
    if isinstance(stop, Criterion):
        return (stop,)
    stop = tuple(stop)
    for c in stop:
        if not isinstance(c, Criterion):
            raise TypeError(f"not a stopping criterion: {c!r}")
    for kind in (Stall, Jammed):
        if sum(isinstance(c, kind) for c in stop) > 1:
            raise ValueError(f"give at most one {kind.__name__} criterion")
    return stop


def _engine_options(stop: tuple[Criterion, ...], n: int) -> tuple[dict, dict[str, Criterion]]:
    """Reduces criteria to the scalar engine options (strictest value per type) and
    maps every status name to the criterion that produces it."""
    opts = dict(
        max_pressure=math.inf, jammed_pressure=math.inf, jam_start_pressure=1e3, jam_relax_windows=4,
        jam_step=0.2, stall_tol=0.0, stall_window=100.0, max_collisions=0, timeout=math.inf,
    )
    effective: dict[str, Criterion] = {}
    for c in stop:
        if isinstance(c, Pressure) and c.value < opts["max_pressure"]:
            opts["max_pressure"] = c.value
            effective[c.name] = c
        elif isinstance(c, Jammed):
            opts.update(jammed_pressure=c.pressure, jam_start_pressure=c.start_pressure,
                        jam_relax_windows=int(c.relax_windows), jam_step=c.step)
            effective[c.name] = c
        elif isinstance(c, Stall):
            opts["stall_tol"], opts["stall_window"] = c.tol, c.window
            effective[c.name] = c
        elif isinstance(c, Collisions):
            b = c.budget(n)
            if opts["max_collisions"] == 0 or b < opts["max_collisions"]:
                opts["max_collisions"] = b
                effective[c.name] = c
        elif isinstance(c, Timeout) and c.seconds < opts["timeout"]:
            opts["timeout"] = c.seconds
            effective[c.name] = c
    return opts, effective
