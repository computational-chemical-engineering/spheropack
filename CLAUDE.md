# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

spheropack: Lubachevsky-Stillinger (LS) sphere packing with a header-only C++20 core and a nanobind Python interface. Rewrite of the legacy code in `~/Code/packing` (ancestor `~/Code/event`). Design decisions and phases: `ROADMAP.md`.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install nanobind scikit-build-core numpy pytest
.venv/bin/pip install --no-build-isolation -e .        # rebuild after every C++ change
.venv/bin/python -m pytest                            # fast suite (~1 min)
.venv/bin/python -m pytest -m slow                    # legacy statistics + slow-growth EOS (~9 min)
.venv/bin/python -m pytest tests/test_pack.py::test_csv_roundtrip   # single test
cmake -S . -B build/cpp -G Ninja -DSPHEROPACK_BUILD_TESTS=ON && cmake --build build/cpp && ./build/cpp/tests/cpp/spheropack_tests   # C++ (doctest)
```

The editable install does not rebuild automatically; rerun the pip install line after editing anything in `include/` or `src/`.

## Architecture

- `include/spheropack/ls_packing.hpp`: the engine `LSPacking<D>`. Radii are `a_i * s(t)` with `s = s0 + ds*t`; unit masses, kT = 1 (thermal speed 1). Lazy positions (`x` at particle time `t`), wrapped into the box with integer image counters `img`. One predicted event per particle (cell crossing, wall, pair) in `EventHeap`; a pair event is stale if the partner's `counter` changed. `sync()` advances everyone, resets the time origin, rescales to kT = 1 (only while growing), rebuilds the cell grid and re-predicts everything; it runs every `sync_interval * N` collisions and whenever the radii outgrow the cells. Reduced pressure comes from the collision virial over the windows between syncs and stops the run at `max_pressure`. `finish()` removes round-off overlaps with a uniform shrink and records it.
- Collision rules (`CollisionRule`): `elastic_growing` (default; elastic in the frame of the growing surfaces) and `legacy` (normal relative velocity after contact `max(-u_n, growth speed + margin)`, perfectly inelastic in the growing frame, so clusters can lock into endless collisions near jamming: no contact plateau). Legacy baselines must be compared with `collision_rule="legacy"`.
- Pressure: Z = 1 + W / (2 * integral of KE dt) over windows of `sync_interval * N` collisions; KE is tracked incrementally (`ke_`, `integrate_ke()`). Each sphere gets half the virial of its collisions (`w_`), giving `sphere_pressure` (rattlers ~1) and the median pressure. `history_` stores one `HistoryRecord` per window.
- Jamming protocol (`Phase` grow/relax/step, `jammed_pressure` finite): grow until median Z >= `jam_start_pressure`, then alternate `jam_relax_windows` windows at ds = 0 with growth steps of relative size `jam_step / Z_median`; stop when median Z after relaxation >= `jammed_pressure`. Needs `elastic_growing` (legacy clusters never relax). Finite-rate protocols give ~94-98% of the isostatic contact number; this is physics, not a bug.
- Stop criteria: Python `spheropack.stop` objects (`Jammed`, `Pressure`, `Stall`, `Collisions`, `Timeout`; default `[Jammed()]` for density="max", else `[Pressure(1e6)]`) are reduced in `stop._engine_options` to scalar `PackingOptions` (strictest per type); status names equal the criterion names, plus `target`, `time_limit`, `interrupted`, `no_events`.
- `predict.hpp`: `first_contact(A, B, C)` for `A t^2 + 2 B t + C`, cancellation-free, handles A < 0 (growth) and round-off overlap (C < 0). Any change here needs `tests/cpp/test_predict.cpp` and the no-overlap tests.
- `box.hpp`: `Box<D>` with per-axis periodic flags and an optional `BallWall` (curved wall over a subset of axes: cylinder, sphere, disk). `flat_walls(k)` tells whether axis k has flat walls; `volume()` accounts for the ball. The engine predicts ball contacts with `first_contact` on `(R - rho - drho t)^2 - |q + v t|^2`.
- `cell_grid.hpp`: 3^D stencil yielding (cell, image shift) pairs; on periodic axes with fewer than 3 cells it visits the same cell with different shifts, which keeps tiny boxes correct.
- `rng.hpp`: own xoshiro256++/SplitMix64 and distributions (std distributions are not portable). The integer stream is pinned by tests; never change it.
- `src/bindings.cpp`: `_ls_pack2/_ls_pack3` take everything positionally and return a dict; the GIL is released and Ctrl-C is polled every 65536 loop iterations.
- `python/spheropack/`: `pack()` in `_pack.py` (density None / float / "max", `stop=`), `stop.py`, containers, `Packing` result, `analysis.py` (pairs via C++ `neighbors.hpp`, g(r), contacts, rattlers, Steinhardt Q_l / psi_l, exact density profiles), `plotting.py` (matplotlib sections), `cli.py` (`spheropack pack`, `legacy-periodic`, `legacy-tube`; legacy commands use the legacy rule and stop only on a collision budget).
- Docs: `docs/` (Sphinx + myst-nb notebooks + breathe/Doxygen). Build: `cd docs && ../.venv/bin/sphinx-build -b html . _build/html`. Notebooks in `docs/notebooks/` are executed at build time (cache).

## Conventions

- `growth_rate` = growth speed of the mean diameter / thermal speed. Legacy `generate_*` CLIs at the notebook settings correspond to 0.16; legacy `packing.cpp` to about 0.02.
- Event-driven dynamics is chaotic: compare with legacy statistically (`tests/baselines/*.json`, regenerated by `tests/baselines/generate_baselines.py` from the legacy repo), never trajectory by trajectory.
- No em dashes in docs or comments.
