"""Matplotlib helpers to look at packings (requires ``matplotlib``)."""

from __future__ import annotations

import numpy as np

from .packing import Packing

__all__ = ["plot_disks", "plot_section"]


def _axes(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(5, 5))
    return ax


def _draw(ax, centres, radii, lengths, color, edgecolor):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Rectangle

    patches = [Circle(c, r) for c, r in zip(centres, radii)]
    ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=edgecolor, linewidth=0.4))
    ax.add_patch(Rectangle((0, 0), lengths[0], lengths[1], fill=False, linewidth=1.0, edgecolor="k"))
    ax.set_xlim(0, lengths[0])
    ax.set_ylim(0, lengths[1])
    ax.set_aspect("equal")
    return ax


def plot_disks(packing: Packing, ax=None, color="tab:blue", edgecolor="k", images: bool = True):
    """Draws a 2D packing. With ``images``, disks cut by periodic edges are completed."""
    if packing.dim != 2:
        raise ValueError("plot_disks needs a 2D packing; use plot_section for 3D")
    p = packing.periodic_images() if images else packing
    return _draw(_axes(ax), p.positions, p.radii, packing.container.lengths, color, edgecolor)


def plot_section(
    packing: Packing, axis: int = 2, position: float | None = None, ax=None, color="tab:blue", edgecolor="k"
):
    """Draws the cross-section of a 3D packing with the plane ``x[axis] = position``.

    Every sphere cut by the plane appears as a disk of radius
    ``sqrt(r**2 - (x[axis] - position)**2)``. Default position: the middle of the box.
    """
    if packing.dim != 3:
        raise ValueError("plot_section needs a 3D packing")
    lengths = np.asarray(packing.container.lengths)
    position = 0.5 * lengths[axis] if position is None else position
    keep = [k for k in range(3) if k != axis]
    p = packing.periodic_images(axes=keep)
    h = p.positions[:, axis] - position
    if packing.container.periodic[axis]:
        h -= lengths[axis] * np.round(h / lengths[axis])
    cut = np.abs(h) < p.radii
    radii = np.sqrt(p.radii[cut] ** 2 - h[cut] ** 2)
    return _draw(_axes(ax), p.positions[cut][:, keep], radii, lengths[keep], color, edgecolor)
