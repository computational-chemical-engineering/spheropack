import numpy as np


def min_gap_ratio(positions, radii, lengths, periodic):
    """Brute-force min over pairs of distance / (r_i + r_j) - 1, and over walls of
    distance / r_i - 1, with periodic images on periodic axes."""
    x = np.asarray(positions)
    lengths = np.asarray(lengths, dtype=float)
    d = x[:, None, :] - x[None, :, :]
    for k, p in enumerate(periodic):
        if p:
            d[..., k] -= lengths[k] * np.round(d[..., k] / lengths[k])
    dist = np.sqrt((d**2).sum(-1))
    sig = radii[:, None] + radii[None, :]
    iu = np.triu_indices(len(radii), 1)
    gap = (dist[iu] / sig[iu]).min() - 1.0
    for k, p in enumerate(periodic):
        if not p:
            gap = min(gap, (x[:, k] / radii).min() - 1.0, ((lengths[k] - x[:, k]) / radii).min() - 1.0)
    return gap
