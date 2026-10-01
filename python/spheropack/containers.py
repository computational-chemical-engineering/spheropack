"""Containers in which spheres are packed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["Ball", "Box", "Cylinder", "Disk", "PeriodicBox", "SphereContainer"]


@dataclass(frozen=True)
class Ball:
    """Curved wall: the spheres stay inside the ball of ``radius`` around ``center``,
    measured over the axes flagged in ``axes`` only (two axes: a cylinder)."""

    axes: tuple[bool, ...]
    center: tuple[float, ...]
    radius: float


@dataclass(frozen=True)
class Box:
    """Rectangular box spanning ``[0, L_k)`` along each axis.

    Each axis is either periodic or bounded by walls. Without ``ball`` the walls are
    flat (at 0 and ``L_k``); a box with walls along one axis and periodic boundaries
    along the others is a slab. With ``ball`` the axes of the ball are bounded by the
    curved wall instead; use :func:`Cylinder`, :func:`SphereContainer` or
    :func:`Disk` to construct such containers.

    Parameters
    ----------
    lengths:
        Edge lengths, one per dimension (2 or 3 values).
    periodic:
        One flag for all axes, or one flag per axis.
    ball:
        Optional curved wall inside the box.
    """

    lengths: tuple[float, ...]
    periodic: tuple[bool, ...]
    ball: Ball | None = None

    def __init__(self, lengths: Sequence[float], periodic: bool | Sequence[bool] = True, ball: Ball | None = None):
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
        if ball is not None:
            ball = Ball(tuple(bool(a) for a in ball.axes), tuple(float(x) for x in ball.center), float(ball.radius))
            if not (np.isfinite(ball.radius) and ball.radius > 0):
                raise ValueError("ball radius must be positive")
            if len(ball.axes) != len(lengths) or len(ball.center) != len(lengths) or not any(ball.axes):
                raise ValueError("ball axes and centre need one entry per dimension")
            for k, on in enumerate(ball.axes):
                if on and (periodic[k] or ball.center[k] - ball.radius < 0 or ball.center[k] + ball.radius > lengths[k]):
                    raise ValueError("the ball must lie inside the box and its axes cannot be periodic")
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "periodic", periodic)
        object.__setattr__(self, "ball", ball)

    @property
    def dim(self) -> int:
        return len(self.lengths)

    @property
    def volume(self) -> float:
        """Volume (area in 2D) available to the spheres."""
        if self.ball is None:
            return float(np.prod(self.lengths))
        m = sum(self.ball.axes)
        rest = float(np.prod([L for L, on in zip(self.lengths, self.ball.axes) if not on]))
        measure = {1: 2.0 * self.ball.radius, 2: math.pi * self.ball.radius**2, 3: 4.0 / 3.0 * math.pi * self.ball.radius**3}
        return rest * measure[m]


def PeriodicBox(lengths: float | Sequence[float], dim: int = 3) -> Box:
    """Fully periodic box. A scalar ``lengths`` gives a cube (square for ``dim=2``)."""
    if np.ndim(lengths) == 0:
        lengths = [float(lengths)] * dim
    return Box(lengths, periodic=True)


def Cylinder(diameter: float, length: float, axis: int = 2, capped: bool = False) -> Box:
    """Cylindrical tube (3D), periodic along its axis unless ``capped``.

    The tube axis is ``axis`` (default z) at the centre of the bounding box
    ``diameter x diameter x length``. With ``capped=True`` flat walls close the tube at
    both ends; otherwise the packing repeats along the axis, which models a section
    of a long packed tube without end effects.
    """
    if not (diameter > 0 and length > 0):
        raise ValueError("diameter and length must be positive")
    axis = axis % 3
    lengths = [float(diameter)] * 3
    lengths[axis] = float(length)
    axes = tuple(k != axis for k in range(3))
    periodic = [False] * 3
    periodic[axis] = not capped
    center = tuple(0.5 * diameter if on else 0.0 for on in axes)
    return Box(lengths, periodic=periodic, ball=Ball(axes, center, 0.5 * float(diameter)))


def SphereContainer(diameter: float, dim: int = 3) -> Box:
    """Spherical container (a disk for ``dim=2``)."""
    if not diameter > 0:
        raise ValueError("diameter must be positive")
    return Box([float(diameter)] * dim, periodic=False,
               ball=Ball((True,) * dim, (0.5 * float(diameter),) * dim, 0.5 * float(diameter)))


def Disk(diameter: float) -> Box:
    """Circular container in 2D."""
    return SphereContainer(diameter, dim=2)
