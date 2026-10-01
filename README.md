# spheropack

Random sphere packings in Python, generated with the event-driven
Lubachevsky-Stillinger algorithm: spheres move ballistically, collide and grow until a
target density is reached or the packing jams or crystallises.

```python
import spheropack as sp

p = sp.pack(n=1000, density=0.6, seed=1)          # target volume fraction
q = sp.pack(n=1000, density="max", seed=1)        # until jammed (isostatic within a few %)
q.positions, q.radii, q.density, q.status
q.to_csv("packing.csv")
```

- Periodic boxes, flat walls (slabs, closed boxes), 2D and 3D, mono- and polydisperse.
- A fast, exact C++20 event-driven core (header-only, no dependencies) with a
  nanobind Python interface. The GIL is released and Ctrl-C works.
- A well-defined, dimensionless growth rate and composable stopping criteria,
  including a quasi-static jamming protocol.
- Structure analysis: pair distribution function, contact numbers, rattlers (from the
  collision forces), isostaticity, Steinhardt bond order, crystallinity, density
  profiles next to walls.
- A command-line tool, including a drop-in replacement of the legacy
  `generate_periodic_packing`.

The package will also make available the rejection-free, event-driven Monte Carlo
method of E.A.J.F. Peters and G. de With, *Rejection-free Monte Carlo sampling for
general potentials*, Phys. Rev. E **85**, 026703 (2012).

> **Status:** alpha, under construction. See [ROADMAP.md](ROADMAP.md).

## Installation

```bash
pip install spheropack
```

Building from source needs a C++20 compiler and CMake 3.18 or newer: `pip install .`

## Documentation

The documentation (user guide, example notebooks, Python and C++ reference) is built
with Sphinx from `docs/`:

```bash
pip install sphinx pydata-sphinx-theme myst-nb breathe sphinx-copybutton sphinxcontrib-bibtex matplotlib
cd docs && sphinx-build -b html . _build/html
```

## License

MIT
