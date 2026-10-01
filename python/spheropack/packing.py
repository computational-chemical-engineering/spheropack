"""Result of a packing run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .containers import Box


@dataclass
class Packing:
    """A sphere (disk) packing.

    Attributes
    ----------
    positions:
        Centres, shape ``(n, dim)``, inside the container.
    radii:
        Radii, shape ``(n,)``.
    container:
        The container the spheres are packed in.
    status:
        Why the run stopped: ``"target"`` (density or radii reached), or the name of
        the stopping criterion that fired: ``"pressure"``, ``"stall"``,
        ``"collisions"``, ``"timeout"``; ``"box_limit"`` when the largest sphere grew
        as large as a periodic edge of the box (it would touch its own image);
        ``"no_events"`` if nothing moves.
    success:
        True if the target was reached, or with ``density="max"`` if an arrest
        criterion fired.
    stopped_by:
        The criterion object that ended the run, or None.
    density:
        Volume (area) fraction of the spheres.
    reduced_pressure:
        Last measured reduced pressure ``PV/(NkT)`` (mean over spheres); 0 if the run
        was too short to complete a measurement window.
    median_pressure:
        Median over spheres of the per-sphere reduced pressure in the last window.
    sphere_pressure:
        Per-sphere reduced pressure in the last window, shape ``(n,)``; its mean is
        ``reduced_pressure``. Spheres that carry no force (rattlers) have values near 1.
    history:
        One entry per pressure window: arrays ``time``, ``density``, ``scale`` (the
        factor multiplying the radii passed to the engine), ``reduced_pressure``,
        ``median_pressure``, ``kT``, ``n_collisions`` and ``phase`` (0 growing, 1
        relaxing, 2 growth step of the jamming protocol).
    max_contact_error:
        Largest relative deviation ``|distance / contact distance - 1|`` at any
        collision; a measure of the accumulated round-off (around 1e-13).
    min_gap:
        Smallest ``distance / contact distance - 1`` over all pairs and walls before
        overlaps were removed. Values around -1e-15 are round-off.
    shrink_factor:
        Uniform factor applied to the radii to remove round-off overlaps (1 if none).
    """

    positions: np.ndarray
    radii: np.ndarray
    container: Box
    status: str
    density: float
    reduced_pressure: float
    n_collisions: int
    n_events: int
    sim_time: float
    wall_time: float
    min_gap: float
    shrink_factor: float
    seed: int
    growth_rate: float
    median_pressure: float = 0.0
    sphere_pressure: np.ndarray | None = field(default=None, repr=False)
    history: dict | None = field(default=None, repr=False)
    max_contact_error: float = 0.0
    collision_rule: str = "elastic_growing"
    stop: tuple = ()
    stopped_by: object = None
    success: bool = True
    _velocities: np.ndarray | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"Packing(n={self.n}, dim={self.dim}, density={self.density:.6g}, "
            f"status={self.status!r}, reduced_pressure={self.reduced_pressure:.3g}, "
            f"n_collisions={self.n_collisions}, seed={self.seed})"
        )

    @property
    def n(self) -> int:
        return len(self.radii)

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    @property
    def diameters(self) -> np.ndarray:
        return 2.0 * self.radii

    def periodic_images(self, axes=None) -> "Packing":
        """Copy with extra periodic images of every sphere that cuts a periodic face.

        Useful for visualisation and meshing of the box: every sphere is then
        complete inside the box or continued on the opposite face. ``axes`` limits
        the images to the given axes (default: all periodic axes). Per-sphere data
        such as ``sphere_pressure`` are dropped from the copy.
        """
        x, r = self.positions, self.radii
        lengths = np.asarray(self.container.lengths)
        for k, periodic in enumerate(self.container.periodic):
            if not periodic or (axes is not None and k not in axes):
                continue
            low = x[:, k] < r
            high = x[:, k] + r > lengths[k]
            shift = np.zeros(self.dim)
            shift[k] = lengths[k]
            x = np.concatenate([x, x[low] + shift, x[high] - shift])
            r = np.concatenate([r, r[low], r[high]])
        out = Packing(**{**self.__dict__, "positions": x, "radii": r, "sphere_pressure": None})
        out._velocities = None
        return out

    def to_csv(self, path: str | Path, *, legacy_order: bool = False, header: bool = True) -> None:
        """Writes one sphere per line as ``x,y,z,r`` (``x,y,r`` in 2D).

        ``legacy_order=True`` writes ``z,x,y,r`` without header, the format of the
        legacy ``generate_packed_tube`` and ``generate_periodic_packing`` tools.
        """
        cols = self.positions
        if legacy_order:
            if self.dim != 3:
                raise ValueError("legacy_order needs a 3D packing")
            cols = cols[:, [2, 0, 1]]
            header = False
        data = np.column_stack([cols, self.radii])
        names = "xyz"[: self.dim] if not legacy_order else "zxy"
        np.savetxt(
            path,
            data,
            delimiter=",",
            fmt="%.17g",  # round-trip exact, so touching spheres do not overlap in the file
            header=",".join([*names, "r"]) if header else "",
            comments="",
        )
