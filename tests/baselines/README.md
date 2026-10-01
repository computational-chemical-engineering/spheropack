# Legacy regression baselines

Statistical baselines from the legacy C++ event-driven packing code
(`/home/eajfpeters/Code/packing`), to derive test thresholds for the rewrite.

Files
- `legacy_driver.cpp`: periodic 3D monodisperse driver (args: `N seed target_phi growrate cap_per_particle`),
  prints one JSON object. Units: final diameter 1 at `target_phi`. The collision cap is enforced inside
  `findCollision` (exception), because near jamming `step(t)` never returns.
- `generate_baselines.py`: compiles the driver and runs all cases in parallel, writes the JSON files.
- `periodic_arrest_N500.json`, `periodic_arrest_N27.json`: arrest density, growrate 0.16 (d=1 units), 20 seeds.
- `periodic_arrest_N500_refrate.json`, `periodic_arrest_N27_refrate.json`: same with the growrate rescaled to
  the compression rate of `packing.cpp` (box L=1, radius from phi=0.62). Growth speed relative to thermal
  speed is 0.16*r_ref in that code, versus 0.16*0.5 here, hence 0.16 in d=1 units is about 8x faster for N=500.
  These reproduce the usual ~0.64.
- `periodic_speed.json`: collisions/s and collision counts, N=10000, phi 0.55 and 0.58.
  Timings are taken with several jobs running concurrently.
- `tube_D7.json`: tube D=7d packings (overlap, wall penetration, radial density histogram).

Regenerate
    LEGACY_DIR=/path/to/packing /home/eajfpeters/Code/packing/.venv/bin/python generate_baselines.py
(optional `--only n500 n27 slow speed tube`). Needs g++, Boost headers, numpy.

Thresholds for tests are to be derived from the across-seed spreads stored in each `summary`.
