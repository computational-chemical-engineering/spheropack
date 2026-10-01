#!/usr/bin/env python3
"""Generate statistical regression baselines from the legacy C++ packing code.

Usage: LEGACY_DIR=/path/to/packing python generate_baselines.py [--only NAME ...]
Needs g++, Boost headers, numpy.
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.environ.get("LEGACY_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "packing"))
MAX_WORKERS = 12
TUBE_BIN = os.path.join(LEGACY_DIR, "build", "generate_packed_tube")


def legacy_commit():
    return subprocess.check_output(
        ["git", "-C", LEGACY_DIR, "rev-parse", "HEAD"], text=True).strip()


def build_driver(build_dir):
    exe = os.path.join(build_dir, "legacy_driver")
    cmd = ["g++", "-O2", "-std=c++17", f"-I{LEGACY_DIR}",
           os.path.join(HERE, "legacy_driver.cpp"), "-o", exe]
    subprocess.run(cmd, check=True)
    return exe, " ".join(cmd)


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def meta(extra):
    m = {"date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "legacy_dir": LEGACY_DIR, "legacy_commit": legacy_commit()}
    m.update(extra)
    return m


def run_periodic(args):
    exe, cmd = args
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    d["command"] = " ".join(cmd)
    return d


def periodic_cases(exe, cases):
    cmds = [(exe, [exe] + [str(x) for x in c]) for c in cases]
    with cf.ThreadPoolExecutor(MAX_WORKERS) as ex:
        return list(ex.map(run_periodic, cmds))


def write(name, obj):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name, flush=True)


def arrest(exe, build_cmd, name, N, growrate, note=""):
    seeds = list(range(1, 21))
    cases = [(N, s, 0.74, growrate, 10000) for s in seeds]
    t0 = time.time()
    runs = periodic_cases(exe, cases)
    write(name, {
        "meta": meta({"description": "Arrest density of legacy LS compression, d=1, periodic cube. "
                      "Runs stop at the collision cap; phi_final is the density when the cap was hit. " + note,
                      "N": N, "seeds": seeds, "target_phi": 0.74, "growth_rate": growrate,
                      "cap_per_particle": 10000, "build_command": build_cmd,
                      "driver_args": "N seed target_phi growrate cap_per_particle",
                      "elapsed_s": time.time() - t0}),
        "runs": runs,
        "summary": {"phi_final": stats([r["phi_final"] for r in runs]),
                    "n_reached": sum(r["reached"] for r in runs)}})


def speed(exe, build_cmd):
    cases = [(10000, s, phi, 0.16, 10000) for phi in (0.55, 0.58) for s in (1, 2, 3)]
    t0 = time.time()
    # run serially-ish in parallel: timing is affected by load, limit workers via pool anyway
    runs = periodic_cases(exe, cases)
    summ = {}
    for phi in (0.55, 0.58):
        rr = [r for r in runs if r["target_phi"] == phi]
        summ[str(phi)] = {"collisions_per_s": stats([r["collisions_per_s"] for r in rr]),
                          "n_collisions": stats([r["n_collisions"] for r in rr]),
                          "wall_time_s": stats([r["wall_time_s"] for r in rr]),
                          "all_reached": all(r["reached"] for r in rr)}
    write("periodic_speed.json", {
        "meta": meta({"description": "Throughput of legacy code, N=10000. Timings taken with up to "
                      f"{MAX_WORKERS} concurrent jobs on the machine.",
                      "N": 10000, "seeds": [1, 2, 3], "target_phis": [0.55, 0.58], "growth_rate": 0.16,
                      "cap_per_particle": 10000, "build_command": build_cmd,
                      "elapsed_s": time.time() - t0}),
        "runs": runs, "summary": summ})


# ---------------------------------------------------------------- tube
TUBE_N = 725
TUBE_d = 3e-3
TUBE_D = 21e-3
TUBE_GR = 106.66666
TUBE_L = TUBE_N * (math.pi / 6) * TUBE_d ** 3 / 0.54 / (0.25 * math.pi * TUBE_D ** 2)
NBINS = 40


def analyse_tube(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    z, x, y, r = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    n = len(r)
    # pair overlaps, periodic in z
    dz = z[:, None] - z[None, :]
    dz -= TUBE_L * np.round(dz / TUBE_L)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dz ** 2 + dx ** 2 + dy ** 2)
    rsum = r[:, None] + r[None, :]
    ov = (rsum - dist) / rsum
    np.fill_diagonal(ov, -np.inf)
    rc = np.hypot(x - TUBE_D / 2, y - TUBE_D / 2)
    pen = rc + r - TUBE_D / 2
    # radial histogram: number density per unit cross-section area relative to mean
    rmax = TUBE_D / 2 - TUBE_d / 2
    edges = np.linspace(0, rmax, NBINS + 1)
    counts, _ = np.histogram(rc, bins=edges)
    area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dens = counts / area
    mean_dens = counts.sum() / (math.pi * rmax ** 2)
    hist = dens / mean_dens
    return {"n_spheres": int(n), "max_overlap_fraction": float(ov.max()),
            "max_wall_penetration": float(pen.max()),
            "mean_radius": float(r.mean()),
            "n_in_histogram": int(counts.sum()),
            "radial_histogram": hist.tolist()}


def run_tube(seed):
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "tube.csv")
        cmd = [TUBE_BIN, f"--num_part={TUBE_N}", f"--part_diam={TUBE_d!r}", f"--tube_diam={TUBE_D!r}",
               f"--tube_length={TUBE_L!r}", f"--growth_rate={TUBE_GR!r}", f"--seed={seed}", f"--file={f}"]
        t0 = time.time()
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        except subprocess.TimeoutExpired:
            return {"seed": seed, "command": " ".join(cmd), "timeout": True}
        wall = time.time() - t0
        res = analyse_tube(f)
    res.update({"seed": seed, "wall_time_s": wall, "command": " ".join(cmd)})
    return res


def tube():
    seeds = list(range(1, 11))
    t0 = time.time()
    with cf.ThreadPoolExecutor(MAX_WORKERS) as ex:
        runs = list(ex.map(run_tube, seeds))
    ok = [r for r in runs if not r.get("timeout")]
    H = np.array([r["radial_histogram"] for r in ok])
    edges = np.linspace(0, TUBE_D / 2 - TUBE_d / 2, NBINS + 1)
    write("tube_D7.json", {
        "meta": meta({"description": "Packing in a tube (D = 7 d) with periodic axis. Histogram: sphere-centre "
                      "distance from axis, number density per unit cross-section area / mean (uniform = 1).",
                      "num_part": TUBE_N, "part_diam": TUBE_d, "tube_diam": TUBE_D, "tube_length": TUBE_L,
                      "growth_rate_cli": TUBE_GR, "seeds": seeds, "timeout_s": 600,
                      "histogram_bins": NBINS, "histogram_edges": edges.tolist(),
                      "overlap_definition": "(r_i + r_j - dist) / (r_i + r_j), periodic along z",
                      "wall_penetration_definition": "r_centre + radius - D/2",
                      "elapsed_s": time.time() - t0}),
        "runs": runs,
        "summary": {"n_spheres": stats([r["n_spheres"] for r in ok]),
                    "max_overlap_fraction": stats([r["max_overlap_fraction"] for r in ok]),
                    "max_wall_penetration": stats([r["max_wall_penetration"] for r in ok]),
                    "wall_time_s": stats([r["wall_time_s"] for r in ok]),
                    "n_timeouts": len(runs) - len(ok),
                    "radial_histogram_mean": H.mean(0).tolist(),
                    "radial_histogram_std": H.std(0, ddof=1).tolist()}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of: n500 n27 slow speed tube")
    a = ap.parse_args()
    want = set(a.only or ["n500", "n27", "slow", "speed", "tube"])
    t0 = time.time()
    with tempfile.TemporaryDirectory() as bd:
        exe, bc = build_driver(bd)
        if "n500" in want:
            arrest(exe, bc, "periodic_arrest_N500.json", 500, 0.16)
        if "n27" in want:
            arrest(exe, bc, "periodic_arrest_N27.json", 27, 0.16)
        if "slow" in want:
            # growth rate scaled so that growth speed / thermal speed matches packing.cpp
            # (L=1, radius from phi=0.62, growrate 0.16): g_eff = 0.16 * r_ref / 0.5.
            for N in (500, 27):
                r_ref = (3 * 0.62 / (4 * math.pi * N)) ** (1 / 3)
                g = round(0.16 * r_ref / 0.5, 4)
                arrest(exe, bc, f"periodic_arrest_N{N}_refrate.json", N, g,
                       note="Growth rate rescaled to match the compression rate of packing.cpp (L=1 units).")
        if "speed" in want:
            speed(exe, bc)
    if "tube" in want:
        tube()
    print(f"total runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
