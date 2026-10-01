"""Structure analysis of packings.

All functions take a :class:`~spheropack.Packing` and respect its container:
distances use periodic images along periodic axes. Neighbour searches run in C++
with a cell list, so they scale linearly with the number of spheres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _core
from .packing import Packing

__all__ = [
    "BondOrder",
    "RadialDistribution",
    "bond_order",
    "contact_numbers",
    "contacts",
    "crystalline",
    "density_profile",
    "isostaticity",
    "neighbor_pairs",
    "radial_distribution",
    "radial_profile",
    "rattlers",
]


def neighbor_pairs(packing: Packing, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs closer than ``cutoff`` (centre to centre).

    Returns
    -------
    i, j:
        Indices with ``i < j``, shape ``(m,)``. A pair appears once per periodic image
        within the cutoff.
    r:
        Connecting vectors from ``i`` to the image of ``j``, shape ``(m, dim)``.
    """
    c = packing.container
    periodic_lengths = [L for L, p in zip(c.lengths, c.periodic, strict=True) if p]
    if periodic_lengths and cutoff > min(periodic_lengths):
        raise ValueError("cutoff must not exceed the smallest periodic edge of the box")
    find = _core._neighbor_pairs3 if packing.dim == 3 else _core._neighbor_pairs2
    return find(np.ascontiguousarray(packing.positions), list(c.lengths), list(c.periodic), float(cutoff))


# --------------------------------------------------------------------- contacts


def contacts(packing: Packing, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Pairs in contact: centre distance at most ``(r_i + r_j) * (1 + tol)``.

    In a packing stopped at reduced pressure ``Z`` the gaps between touching spheres
    are of order ``d / Z``, so ``tol`` should be well above ``1 / Z`` and well below
    the gaps between neighbours that do not touch (plot the contact number against
    ``tol`` to find the plateau).

    Returns
    -------
    i, j:
        Indices of the spheres in contact.
    """
    r = packing.radii
    i, j, rv = neighbor_pairs(packing, 2.0 * r.max() * (1.0 + tol))
    dist = np.sqrt((rv**2).sum(axis=1))
    touching = dist <= (r[i] + r[j]) * (1.0 + tol)
    return i[touching], j[touching]


def contact_numbers(packing: Packing, tol: float = 1e-5) -> np.ndarray:
    """Number of contacts of every sphere (see :func:`contacts` for ``tol``).

    Wall contacts are not counted.
    """
    i, j = contacts(packing, tol)
    return np.bincount(i, minlength=packing.n) + np.bincount(j, minlength=packing.n)


def rattlers(packing: Packing, tol: float = 1e-5, method: str = "auto", force_threshold: float = 1e-3) -> np.ndarray:
    """Boolean mask of rattlers: spheres that carry no force in a jammed packing.

    Parameters
    ----------
    method:
        ``"force"``: a sphere is a rattler when its time-averaged collision pressure
        in the last measurement window (:attr:`Packing.sphere_pressure`) is below
        ``force_threshold`` times the median. This is the physical definition: the
        collision impulses of an event-driven simulation are the contact forces.
        ``"contacts"``: spheres with fewer than ``dim + 1`` contacts (see
        :func:`contacts` for ``tol``) are removed together with their contacts,
        repeatedly, until all remaining spheres have at least ``dim + 1``. It needs
        no dynamics but cannot detect spheres whose contacts do not balance.
        ``"auto"``: ``"force"`` when per-sphere pressures are available.
    """
    if method == "auto":
        method = "force" if packing.sphere_pressure is not None and packing.median_pressure > 1 else "contacts"
    if method == "force":
        if packing.sphere_pressure is None or not packing.median_pressure > 1:
            raise ValueError("no per-sphere pressures available; use method='contacts'")
        # Compare the collisional parts: the ideal-gas 1 is common to all spheres.
        return packing.sphere_pressure - 1.0 < force_threshold * (packing.median_pressure - 1.0)
    if method != "contacts":
        raise ValueError("method must be 'auto', 'force' or 'contacts'")
    i, j = contacts(packing, tol)
    alive = np.ones(packing.n, dtype=bool)
    while True:
        keep = alive[i] & alive[j]
        z = np.bincount(i[keep], minlength=packing.n) + np.bincount(j[keep], minlength=packing.n)
        newly_dead = alive & (z < packing.dim + 1)
        if not newly_dead.any():
            return ~alive
        alive &= ~newly_dead


def isostaticity(packing: Packing, tol: float = 1e-7, method: str = "auto") -> float:
    """Ratio of the number of contacts among non-rattlers to the isostatic number.

    A frictionless jammed packing in a periodic box needs ``dim * (n_nr - 1)``
    contacts among its ``n_nr`` non-rattlers to be rigid; the ratio is 1 for an
    isostatic packing and slightly below 1 for packings compressed at a finite rate.
    """
    rat = rattlers(packing, tol, method=method)
    i, j = contacts(packing, tol)
    keep = ~rat[i] & ~rat[j]
    n_nr = int((~rat).sum())
    return float(keep.sum() / (packing.dim * (n_nr - 1)))


# ------------------------------------------------------- radial distribution


@dataclass
class RadialDistribution:
    """Radial distribution function ``g(r)``: bin centres ``r`` and values ``g``."""

    r: np.ndarray
    g: np.ndarray

    def __iter__(self):
        return iter((self.r, self.g))


def radial_distribution(packing: Packing, r_max: float | None = None, bins: int = 200) -> RadialDistribution:
    """Pair distribution function of the sphere centres.

    Normalised by the ideal-gas number of pairs at the mean number density, so that
    ``g -> 1`` at large ``r`` in a periodic box. Along walls no edge correction is
    made, so ``g`` falls below 1 at distances comparable to the box size.

    Parameters
    ----------
    r_max:
        Largest distance. Default: 5 mean diameters, at most half the smallest box
        edge.
    bins:
        Number of bins on ``[0, r_max]``.
    """
    lengths = np.asarray(packing.container.lengths)
    if r_max is None:
        r_max = min(5.0 * packing.diameters.mean(), 0.5 * lengths.min())
    _, _, rv = neighbor_pairs(packing, r_max)
    dist = np.sqrt((rv**2).sum(axis=1))
    counts, edges = np.histogram(dist, bins=bins, range=(0.0, r_max))
    n = packing.n
    rho = n / packing.container.volume
    if packing.dim == 3:
        shell = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    else:
        shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    g = 2.0 * counts / (n * rho * shell)
    return RadialDistribution(0.5 * (edges[1:] + edges[:-1]), g)


# ----------------------------------------------------------------- bond order


@dataclass
class BondOrder:
    """Steinhardt bond-orientational order.

    Attributes
    ----------
    l:
        Order of the spherical harmonics (angular frequency in 2D).
    global_value:
        ``Q_l`` averaged over all bonds (3D), or ``|psi_l|`` averaged over all bonds
        (2D). Close to 0 for disordered packings of many spheres.
    local:
        Per sphere ``q_l`` (3D) or ``|psi_l|`` (2D) over its own bonds; NaN for
        spheres without neighbours.
    n_bonds:
        Number of neighbours of every sphere.
    """

    l: int
    global_value: float
    local: np.ndarray
    n_bonds: np.ndarray


def _legendre_normalised(l: int, x: np.ndarray) -> np.ndarray:
    """Normalised associated Legendre functions for m = 0..l, shape (l+1, len(x)).

    Includes the factor sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) so that
    Y_lm = result[m] * exp(i m phi). Condon-Shortley phase included.
    """
    x = np.asarray(x, dtype=float)
    s = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    out = np.empty((l + 1, x.size))
    # Recursion on fully normalised functions (stable for the small l used here).
    p_mm = np.full(x.size, math.sqrt(1.0 / (4.0 * math.pi)))
    for m in range(l + 1):
        if m > 0:
            p_mm = -math.sqrt((2 * m + 1) / (2 * m)) * s * p_mm
        if m == l:
            out[m] = p_mm
            continue
        p_prev, p_cur = p_mm, math.sqrt(2 * m + 3) * x * p_mm
        for ll in range(m + 2, l + 1):
            a = math.sqrt((4 * ll * ll - 1) / (ll * ll - m * m))
            b = math.sqrt(((ll - 1) ** 2 - m * m) / (4 * (ll - 1) ** 2 - 1))
            p_prev, p_cur = p_cur, a * (x * p_cur - b * p_prev)
        out[m] = p_cur
    return out


def bond_order(packing: Packing, l: int = 6, cutoff: float = 1.4) -> BondOrder:
    """Bond-orientational order parameter of order ``l``.

    Two spheres are bonded when their distance is below ``cutoff * (r_i + r_j)``.
    The default 1.4 lies near the first minimum of ``g(r)`` of dense monodisperse
    packings.

    In 3D this is Steinhardt's ``Q_l``. Reference values for perfect crystals with
    12 nearest neighbours: FCC ``Q_4 = 0.191``, ``Q_6 = 0.575``; HCP ``Q_4 = 0.097``,
    ``Q_6 = 0.485``. In 2D it is the ``l``-fold bond order ``|psi_l|``, 1 for a
    perfect hexagonal crystal with ``l = 6``.
    """
    if l < 1:
        raise ValueError("l must be at least 1")
    r = packing.radii
    i, j, rv = neighbor_pairs(packing, 2.0 * r.max() * cutoff)
    dist = np.sqrt((rv**2).sum(axis=1))
    keep = dist < cutoff * (r[i] + r[j])
    i, j, rv, dist = i[keep], j[keep], rv[keep], dist[keep]
    n = packing.n
    n_bonds = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        if packing.dim == 2:
            theta = np.arctan2(rv[:, 1], rv[:, 0])
            e = np.exp(1j * l * theta)
            e_back = e * (-1) ** l  # bond seen from j points the other way
            per = np.bincount(i, weights=e.real, minlength=n) + np.bincount(j, weights=e_back.real, minlength=n)
            per = per + 1j * (
                np.bincount(i, weights=e.imag, minlength=n) + np.bincount(j, weights=e_back.imag, minlength=n)
            )
            local = np.abs(per) / n_bonds
            global_value = float(np.abs(e.sum()) / len(e)) if len(e) else float("nan")
            return BondOrder(l, global_value, local, n_bonds)

        cos_t = rv[:, 2] / dist
        phi = np.arctan2(rv[:, 1], rv[:, 0])
        plm = _legendre_normalised(l, cos_t)
        sign_back = (-1) ** l  # Y_lm(-r) = (-1)^l Y_lm(r)
        q_local = np.zeros((l + 1, n), dtype=complex)
        q_global = np.zeros(l + 1, dtype=complex)
        for m in range(l + 1):
            y = plm[m] * np.exp(1j * m * phi)
            q_global[m] = y.sum() / len(y) if len(y) else np.nan
            for idx, w in ((i, y), (j, sign_back * y)):
                q_local[m] += np.bincount(idx, weights=w.real, minlength=n) + 1j * np.bincount(
                    idx, weights=w.imag, minlength=n
                )
        q_local /= n_bonds
        weight = np.full(l + 1, 2.0)
        weight[0] = 1.0  # m and -m contribute equally
        norm = 4.0 * math.pi / (2 * l + 1)
        local = np.sqrt(norm * (weight[:, None] * np.abs(q_local) ** 2).sum(axis=0))
        global_value = float(np.sqrt(norm * (weight * np.abs(q_global) ** 2).sum()))
    return BondOrder(l, global_value, local, n_bonds)


# -------------------------------------------------------------- density profile


def _slice_measure(t0: np.ndarray, t1: np.ndarray, R: np.ndarray, dim: int) -> np.ndarray:
    """Volume (3D) or area (2D) of a sphere of radius R between offsets t0 < t1 from its centre."""
    t0 = np.clip(t0, -R, R)
    t1 = np.clip(t1, -R, R)
    if dim == 3:
        F = lambda t: math.pi * (R * R * t - t**3 / 3.0)
    else:
        F = lambda t: t * np.sqrt(np.maximum(R * R - t * t, 0.0)) + R * R * np.arcsin(t / R)
    return np.maximum(F(t1) - F(t0), 0.0)


def density_profile(packing: Packing, axis: int = -1, bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Local volume (area) fraction as a function of position along ``axis``.

    Spheres are sliced exactly, so the profile integrates to the overall density and
    shows the layering next to flat walls. The porosity profile is ``1 - phi``.

    Returns
    -------
    z:
        Bin centres along the axis.
    phi:
        Solid fraction in each bin.
    """
    c = packing.container
    axis = axis % packing.dim
    if c.ball is not None and c.ball.axes[axis]:
        raise ValueError("the cross-section varies along a curved-wall axis; use radial_profile")
    length = c.lengths[axis]
    edges = np.linspace(0.0, length, bins + 1)
    x = packing.positions[:, axis]
    R = packing.radii
    shifts = [-length, 0.0, length] if c.periodic[axis] else [0.0]
    solid = np.zeros(bins)
    for b in range(bins):
        for s in shifts:
            solid[b] += _slice_measure(edges[b] - (x + s), edges[b + 1] - (x + s), R, packing.dim).sum()
    cross_section = c.volume / length
    return 0.5 * (edges[1:] + edges[:-1]), solid / (cross_section * np.diff(edges))


def crystalline(
    packing: Packing, cutoff: float = 1.4, threshold: float | None = None, min_connections: int | None = None
) -> np.ndarray:
    """Boolean mask of spheres in a crystalline environment.

    Uses the criterion of ten Wolde, Ruiz-Montero and Frenkel: every sphere gets a
    normalised vector of local bond order (``q_6m`` over its bonds in 3D, ``psi_6`` in
    2D); a bond is *connected* when the normalised scalar product of the two vectors
    exceeds ``threshold`` (default 0.7 in 3D, 0.8 in 2D); a sphere is crystalline
    when it has at least ``min_connections`` connected bonds (default 7 in 3D, 6 in
    2D). Bonds are defined as in :func:`bond_order`. The 2D defaults are calibrated so
    that slowly grown equal disks score about 90% and a jammed 50:50 mixture with size
    ratio 1.4 about 2%.
    """
    dim = packing.dim
    if threshold is None:
        threshold = 0.7 if dim == 3 else 0.8
    if min_connections is None:
        min_connections = 7 if dim == 3 else 6
    r = packing.radii
    i, j, rv = neighbor_pairs(packing, 2.0 * r.max() * cutoff)
    dist = np.sqrt((rv**2).sum(axis=1))
    keep = dist < cutoff * (r[i] + r[j])
    i, j, rv, dist = i[keep], j[keep], rv[keep], dist[keep]
    n = packing.n
    l = 6

    def accumulate(values_i, values_j):
        out = np.bincount(i, weights=values_i.real, minlength=n) + np.bincount(j, weights=values_j.real, minlength=n)
        return out + 1j * (
            np.bincount(i, weights=values_i.imag, minlength=n) + np.bincount(j, weights=values_j.imag, minlength=n)
        )

    if dim == 2:
        e = np.exp(1j * l * np.arctan2(rv[:, 1], rv[:, 0]))
        q = accumulate(e, e)[None, :]  # l even: the reversed bond has the same phase
        weight = np.ones(1)
    else:
        plm = _legendre_normalised(l, rv[:, 2] / dist)
        phi = np.arctan2(rv[:, 1], rv[:, 0])
        q = np.array([accumulate(plm[m] * np.exp(1j * m * phi), plm[m] * np.exp(1j * m * phi)) for m in range(l + 1)])
        weight = np.full(l + 1, 2.0)
        weight[0] = 1.0  # m and -m contribute equally
    norm = np.sqrt((weight[:, None] * np.abs(q) ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        qn = q / norm
    s = (weight[:, None] * (qn[:, i] * np.conj(qn[:, j])).real).sum(axis=0)
    connected = s > threshold
    count = np.bincount(i[connected], minlength=n) + np.bincount(j[connected], minlength=n)
    return count >= min_connections


def radial_profile(packing: Packing, bins: int = 200, n_theta: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Local solid fraction as a function of the distance from the axis of a cylinder,
    or from the centre of a spherical container or disk.

    The value at radius ``r`` is the fraction of the surface at distance ``r`` (a
    cylinder mantle, a spherical shell or a circle) that lies inside spheres: the
    point-wise radial porosity profile is ``1 - phi``. Spheres are intersected
    exactly with spherical shells and circles; for cylinders the intersection is
    integrated over the angular extent of every sphere with ``n_theta`` points
    (relative error below 1e-3 for the default 32).

    Returns
    -------
    r:
        Radii from 0 to the container radius (bin centres).
    phi:
        Solid fraction at each radius.
    """
    ball = packing.container.ball
    if ball is None:
        raise ValueError("radial_profile needs a Cylinder, SphereContainer or Disk")
    axes = np.flatnonzero(ball.axes)
    q = packing.positions[:, axes] - np.asarray(ball.center)[axes]
    c = np.sqrt((q**2).sum(axis=1))  # distance of the sphere centres from the axis/centre
    a = packing.radii
    R = ball.radius
    r = (np.arange(bins) + 0.5) * R / bins
    phi = np.zeros(bins)
    m, dim = len(axes), packing.dim
    for b, rb in enumerate(r):
        near = np.abs(c - rb) < a
        cn, an = c[near], a[near]
        if m == dim:  # sphere in 3D or disk in 2D: exact
            with np.errstate(invalid="ignore", divide="ignore"):
                cos0 = np.clip((rb * rb + cn * cn - an * an) / (2.0 * rb * cn), -1.0, 1.0)
            cos0 = np.where(cn == 0.0, -1.0, cos0)  # concentric: the whole surface is inside
            if dim == 3:
                phi[b] = (2.0 * math.pi * rb * rb * (1.0 - cos0)).sum() / (4.0 * math.pi * rb * rb)
            else:
                phi[b] = (2.0 * rb * np.arccos(cos0)).sum() / (2.0 * math.pi * rb)
        elif m == 2 and dim == 3:  # cylinder: chord length along the axis, integrated over the angle
            # Integrate every sphere over its own angular support |theta| <= theta0 with the
            # substitution theta = theta0 sin(psi), which also resolves the square-root edges.
            with np.errstate(invalid="ignore", divide="ignore"):
                cos0 = np.clip((rb * rb + cn * cn - an * an) / (2.0 * rb * cn), -1.0, 1.0)
            theta0 = np.where(cn == 0.0, math.pi, np.arccos(cos0))
            psi = (np.arange(n_theta) + 0.5) * math.pi / n_theta - 0.5 * math.pi
            theta = theta0[:, None] * np.sin(psi)[None, :]
            d2 = rb * rb + cn[:, None] ** 2 - 2.0 * rb * cn[:, None] * np.cos(theta)
            chord = 2.0 * np.sqrt(np.maximum(an[:, None] ** 2 - d2, 0.0))
            integral = (chord * (theta0[:, None] * np.cos(psi)[None, :] * math.pi / n_theta)).sum()
            length = packing.container.volume / (math.pi * R * R)
            phi[b] = integral / (2.0 * math.pi * length)
        else:
            raise ValueError("radial_profile supports cylinders (3D), spheres and disks")
    return r, phi
