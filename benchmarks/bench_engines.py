"""Throughput of the packing and rejection-free engines (single core).

Run: python benchmarks/bench_engines.py
Reference numbers (2026-10-02, one core of an AMD/Intel desktop):
legacy code 2.6e5 collisions/s for the packing cases below.
"""

import math
import time

import spheropack as sp
from spheropack import rejection_free as rf


def packing(n, phi, seed=1):
    box = sp.PeriodicBox((n * math.pi / 6 / phi) ** (1 / 3))
    p = sp.pack(n=n, radii=0.5, container=box, density=phi, growth_rate=0.16, seed=seed)
    return p.n_collisions / p.wall_time


def rejection_free(system, duration):
    t = time.perf_counter()
    system.run(duration)
    return system.n_reflections / (time.perf_counter() - t)


def main():
    print(f"spheropack {sp.__version__}")
    for n in (10_000, 100_000):
        for phi in (0.55, 0.58):
            print(f"packing n={n:>7} to phi={phi}: {packing(n, phi):.3g} collisions/s")
    n, rho, T = 1000, 0.317, 1.085
    start = sp.pack(n=n, radii=0.45, container=sp.PeriodicBox((n / rho) ** (1 / 3)), seed=1)
    lj = rf.System(start.positions, box=start.container, potential=rf.LennardJones(cutoff=2.5), kT=T, seed=1)
    print(f"rejection-free LJ (n=1000, rho=0.317): {rejection_free(lj, 50.0):.3g} reflections/s")
    dpd = rf.System(n=375, box=sp.PeriodicBox(5.0), potential=rf.DPD(25.0), kT=1.0, seed=1)
    print(f"rejection-free DPD (n=375, rho=3): {rejection_free(dpd, 10.0):.3g} reflections/s")


if __name__ == "__main__":
    main()
