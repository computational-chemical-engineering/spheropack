"""High-level packing function."""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Literal, Sequence

import numpy as np

from . import _core
from . import stop as _stop
from .containers import Box, PeriodicBox
from .packing import Packing

_INF = float("inf")
_RULES = ("elastic_growing", "legacy")


class PackingWarning(UserWarning):
    """A packing run stopped before reaching its target."""


class PackingError(RuntimeError):
    """Raised with ``strict=True`` when a run stops before its target.

    The partial result is available as ``.packing``.
    """

    def __init__(self, message: str, packing: Packing):
        super().__init__(message)
        self.packing = packing


def _ball_volume(r: np.ndarray, dim: int) -> np.ndarray:
    return math.pi * r**2 if dim == 2 else 4.0 / 3.0 * math.pi * r**3


def pack(
    n: int | None = None,
    radii: float | Sequence[float] | np.ndarray | None = None,
    container: Box | None = None,
    *,
    density: float | Literal["max"] | None = None,
    growth_rate: float = 0.02,
    stop: _stop.Criterion | Iterable[_stop.Criterion] | None = None,
    seed: int | None = None,
    strict: bool = False,
    collision_rule: Literal["elastic_growing", "legacy"] = "elastic_growing",
) -> Packing:
    r"""Packs spheres (disks in 2D) with the Lubachevsky-Stillinger algorithm.

    The spheres start as points at random positions with Maxwell-Boltzmann
    velocities, move ballistically, collide, and grow at a constant rate until the
    target density is reached or a stopping criterion fires.

    Parameters
    ----------
    n:
        Number of spheres. Needed when ``radii`` is a scalar or omitted.
    radii:
        One radius for all spheres, or one per sphere. With ``density`` given, only
        the ratios matter: all radii are scaled by a common factor. Default: equal
        radii (then ``density`` is required).
    container:
        Where to pack. Default: the periodic unit cube.
    density:
        ``None``: grow until the spheres have the given ``radii``. A number: grow
        until this volume (area) fraction. ``"max"``: no target; grow until an arrest
        criterion in ``stop`` fires (jamming or crystallisation).
    growth_rate:
        Dimensionless growth rate

        .. math:: \Gamma = \frac{1}{v_\mathrm{th}}\frac{\mathrm{d}\langle d\rangle}{\mathrm{d}t},

        where :math:`\langle d\rangle` is the arithmetic mean diameter and
        :math:`v_\mathrm{th} = \sqrt{k_BT/m}` the thermal speed (the standard
        deviation of one velocity component; all spheres have the same mass). While a
        sphere moving at :math:`v_\mathrm{th}` travels one mean diameter, the mean
        diameter grows by the fraction :math:`\Gamma`. Jammed packings of equal
        spheres in 3D reach 0.634 to 0.650 for :math:`\Gamma` from 0.3 down to
        0.001 (0.64 at the default); slower growth gives denser, more ordered
        packings, and below about 0.001 equal spheres start to crystallise. The run
        time grows roughly as :math:`1/\Gamma` for slow growth. See the user guide
        page on the growth rate.
    stop:
        Stopping criteria from :mod:`spheropack.stop`; the first that fires ends the
        run. Default: ``[Jammed()]`` for ``density="max"``, otherwise
        ``[Pressure(1e6)]`` (stop early if the spheres jam before the target).
    seed:
        Random seed. ``None`` draws one; it is stored in the result.
    strict:
        Raise :class:`PackingError` instead of warning when the run fails: the target
        density was not reached, or with ``density="max"`` no arrest criterion fired.
    collision_rule:
        ``"elastic_growing"`` (default): collisions are elastic in the frame of the
        growing surfaces, which gives jammed packings with a well-defined contact
        network. ``"legacy"``: the rule of the original code, which can trap clusters
        of spheres in endless collisions near jamming.

    Returns
    -------
    Packing
        Positions, radii and run statistics; ``status`` says why the run stopped.
    """
    container = PeriodicBox(1.0) if container is None else container
    dim = container.dim
    if not growth_rate > 0:
        raise ValueError("growth_rate must be positive (the spheres start as points)")
    if collision_rule not in _RULES:
        raise ValueError(f"collision_rule must be one of {_RULES}")

    if radii is None:
        if density is None:
            raise ValueError("give density, or radii to grow the spheres to")
        radii = 1.0
    radii = np.asarray(radii, dtype=float)
    if radii.ndim == 0:
        if n is None:
            raise ValueError("give n when radii is a scalar or omitted")
        radii = np.full(int(n), float(radii))
    elif radii.ndim != 1 or (n is not None and len(radii) != n):
        raise ValueError("radii must be a scalar or have one entry per sphere")
    if len(radii) == 0 or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("radii must be positive and finite")
    radii = np.ascontiguousarray(radii)

    if density is None:
        target_scale = 1.0
    elif isinstance(density, str):
        if density != "max":
            raise ValueError("density must be None, a number or 'max'")
        target_scale = _INF
    else:
        if not 0.0 < density < 1.0:
            raise ValueError("density must lie between 0 and 1")
        unit = _ball_volume(radii, dim).sum()
        target_scale = (density * container.volume / unit) ** (1.0 / dim)
    periodic_lengths = [L for L, p in zip(container.lengths, container.periodic) if p]
    if periodic_lengths and target_scale != _INF and 2.0 * radii.max() * target_scale >= min(periodic_lengths):
        raise ValueError("the largest sphere would not fit in the periodic box (diameter >= edge length)")

    criteria = _stop._normalise(stop, maximum=density == "max")
    if not criteria:
        raise ValueError("give at least one stopping criterion")
    if collision_rule == "legacy" and any(isinstance(c, _stop.Jammed) for c in criteria):
        raise ValueError("Jammed needs collision_rule='elastic_growing' (legacy clusters never relax)")
    if density == "max" and not any(c.kind == "arrest" for c in criteria):
        raise ValueError('density="max" needs an arrest criterion (Jammed, Pressure or Stall) in stop')
    opts, effective = _stop._engine_options(criteria, len(radii))

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**64)

    run = _core._ls_pack3 if dim == 3 else _core._ls_pack2
    out = run(
        radii,
        list(container.lengths),
        list(container.periodic),
        target_scale,
        float(growth_rate),
        opts["max_pressure"],
        opts["jammed_pressure"],
        opts["stall_tol"],
        opts["stall_window"],
        opts["max_collisions"],
        _INF,
        opts["timeout"],
        10.0,
        int(seed),
        collision_rule,
        None,
        0.0,
        None,
        jam_start_pressure=opts["jam_start_pressure"],
        jam_relax_windows=opts["jam_relax_windows"],
        jam_step=opts["jam_step"],
    )
    if out["status"] == "interrupted":
        raise KeyboardInterrupt

    stopped_by = effective.get(out["status"])
    if density == "max":
        success = out["status"] == "box_limit" or (stopped_by is not None and stopped_by.kind == "arrest")
    else:
        success = out["status"] == "target"

    result = Packing(
        positions=out["positions"],
        radii=out["radii"],
        container=container,
        status=out["status"],
        density=out["density"],
        reduced_pressure=out["reduced_pressure"],
        median_pressure=out["median_pressure"],
        n_collisions=out["n_collisions"],
        n_events=out["n_events"],
        sim_time=out["sim_time"],
        wall_time=out["wall_time"],
        min_gap=out["min_gap_ratio"],
        shrink_factor=out["shrink_factor"],
        seed=int(seed),
        growth_rate=float(growth_rate),
        collision_rule=collision_rule,
        stop=criteria,
        stopped_by=stopped_by,
        success=success,
        sphere_pressure=out["sphere_pressure"],
        history={**out["history"], "density": _ball_volume(radii, dim).sum() * out["history"]["scale"] ** dim / container.volume},
        max_contact_error=out["max_contact_error"],
        _velocities=out["velocities"],
    )

    if not success:
        message = (
            f"packing stopped with status {result.status!r} at density "
            f"{result.density:.5g} (reduced pressure {result.reduced_pressure:.3g})"
        )
        if strict:
            raise PackingError(message, result)
        warnings.warn(message, PackingWarning, stacklevel=2)
    return result
