"""Sweep of the growth rate: final density, crystallinity and cost of jammed packings.

Writes docs/data/growth_rate_sweep.json, used by the user guide and the notebooks.
Run: python benchmarks/sweep_growth_rate.py [--workers 12]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RATES = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
SEEDS = [1, 2, 3, 4, 5]
N = 1000
SYSTEMS = ["3d_equal", "3d_poly10", "2d_bidisperse"]


def radii_for(system, seed):
    rng = np.random.default_rng(1000 + seed)
    if system == "3d_poly10":
        return rng.lognormal(0.0, 0.1, N)
    if system == "2d_bidisperse":
        return np.where(np.arange(N) < N // 2, 1.0, 1.4)
    return np.ones(N)


def run(task):
    import spheropack as sp
    from spheropack import analysis

    system, rate, seed = task
    dim = 2 if system.startswith("2d") else 3
    p = sp.pack(
        radii=radii_for(system, seed),
        container=sp.PeriodicBox(1.0, dim=dim),
        density="max",
        growth_rate=rate,
        seed=seed,
        stop=[sp.stop.Jammed(), sp.stop.Timeout(3600)],
    )
    return dict(
        system=system,
        growth_rate=rate,
        seed=seed,
        status=p.status,
        density=p.density,
        crystalline=float(analysis.crystalline(p).mean()),
        q6_global=analysis.bond_order(p).global_value,
        isostaticity=analysis.isostaticity(p),
        rattlers=float(analysis.rattlers(p).mean()),
        collisions_per_sphere=p.n_collisions / p.n,
        wall_time=p.wall_time,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    tasks = [(s, r, seed) for r in reversed(RATES) for s in SYSTEMS for seed in SEEDS]  # slowest first
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(run, tasks))
    import spheropack as sp

    out = dict(
        meta=dict(
            version=sp.__version__,
            n=N,
            rates=RATES,
            seeds=SEEDS,
            systems=SYSTEMS,
            stop="Jammed() default",
            total_wall_time=time.time() - t0,
        ),
        runs=runs,
    )
    path = Path(__file__).resolve().parents[1] / "docs" / "data" / "growth_rate_sweep.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path, f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
