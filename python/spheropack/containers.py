"""Containers in which spheres are packed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Box:
    """Rectangular box spanning ``[0, L_k)`` along each axis.

    Each axis is either periodic or bounded by two flat walls. A box with walls along
    one axis and periodic boundaries along the others is a slab.

    Parameters
    ----------
    lengths:
        Edge lengths, one per dimension (2 or 3 values).
    periodic:
        One flag for all axes, or one flag per axis.
    """

    lengths: tuple[float, ...]
    periodic: tuple[bool, ...]

    def __init__(self, lengths: Sequence[float], periodic: bool | Sequence[bool] = True):
        lengths = tuple(float(x) for x in np.atleast_1d(lengths))
        if len(lengths) not in (2, 3):
            raise ValueError("a box needs 2 or 3 edge lengths")
        if any(not np.isfinite(x) or x <= 0 for x in lengths):
            raise ValueError("box edge lengths must be positive and finite")
        if isinstance(periodic, (bool, np.bool_)):
            periodic = (bool(periodic),) * len(lengths)
        else:
            periodic = tuple(bool(p) for p in periodic)
            if len(periodic) != len(lengths):
                raise ValueError("give one periodic flag per axis")
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "periodic", periodic)

    @property
    def dim(self) -> int:
        return len(self.lengths)

    @property
    def volume(self) -> float:
        """Volume (area in 2D)."""
        return float(np.prod(self.lengths))


def PeriodicBox(lengths: float | Sequence[float], dim: int = 3) -> Box:
    """Fully periodic box. A scalar ``lengths`` gives a cube (square for ``dim=2``)."""
    if np.ndim(lengths) == 0:
        lengths = [float(lengths)] * dim
    return Box(lengths, periodic=True)
