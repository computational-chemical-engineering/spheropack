# spheropack

[![PyPI](https://img.shields.io/pypi/v/spheropack.svg)](https://pypi.org/project/spheropack/)
[![Python versions](https://img.shields.io/pypi/pyversions/spheropack.svg)](https://pypi.org/project/spheropack/)
[![CI](https://github.com/computational-chemical-engineering/spheropack/actions/workflows/test.yml/badge.svg)](https://github.com/computational-chemical-engineering/spheropack/actions/workflows/test.yml)
[![Docs](https://github.com/computational-chemical-engineering/spheropack/actions/workflows/docs.yml/badge.svg)](https://computational-chemical-engineering.github.io/spheropack)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/computational-chemical-engineering/spheropack/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Random sphere packings in Python, generated with the event-driven
Lubachevsky-Stillinger algorithm: spheres move ballistically, collide and grow until a
target density is reached or the packing jams or crystallises.

```python
import spheropack as sp

p = sp.pack(n=1000, density=0.6, seed=1)  # target volume fraction
q = sp.pack(n=1000, density="max", seed=1)  # until jammed (isostatic within a few %)
q.positions, q.radii, q.density, q.status
q.to_csv("packing.csv")
```

- Periodic boxes, flat walls (slabs, closed boxes), cylinders (packed tubes), spherical
  containers and disks; 2D and 3D; mono- and polydisperse.
- A fast, exact C++20 event-driven core (header-only, no dependencies) with a
  nanobind Python interface. The GIL is released and Ctrl-C works.
- A well-defined, dimensionless growth rate and composable stopping criteria,
  including a quasi-static jamming protocol.
- Structure analysis: pair distribution function, contact numbers, rattlers (from the
  collision forces), isostaticity, Steinhardt bond order, crystallinity, density
  profiles next to walls, radial porosity profiles in tubes.
- A command-line tool, including drop-in replacements of the legacy
  `generate_periodic_packing` and `generate_packed_tube`.

The next version will also make available the rejection-free, event-driven Monte Carlo
method of E.A.J.F. Peters and G. de With, *Rejection-free Monte Carlo sampling for
general potentials*, Phys. Rev. E **85**, 026703 (2012).

> **Status:** first release (0.1). The rejection-free Monte Carlo module follows in
> 0.2; see the [roadmap](https://github.com/computational-chemical-engineering/spheropack/blob/main/ROADMAP.md).

## Installation

```bash
pip install spheropack
```

Wheels are available for Linux, macOS and Windows (Python 3.10 and newer). Building
from source needs a C++20 compiler and CMake 3.18 or newer: `pip install .`

Optional extras: `pip install "spheropack[plot]"` for the matplotlib helpers.

## Documentation

https://computational-chemical-engineering.github.io/spheropack: user guide, example
notebooks, Python and C++ reference. To build it locally from `docs/`:

```bash
pip install sphinx pydata-sphinx-theme myst-nb breathe sphinx-copybutton sphinxcontrib-bibtex matplotlib
cd docs && sphinx-build -b html . _build/html
```

## Citing

If you use spheropack, please cite it (see
[CITATION.cff](https://github.com/computational-chemical-engineering/spheropack/blob/main/CITATION.cff)); for the rejection-free
Monte Carlo method, cite Peters and de With (2012).

## Contributing

See [CONTRIBUTING.md](https://github.com/computational-chemical-engineering/spheropack/blob/main/CONTRIBUTING.md).
Bug reports and feature requests are welcome as
[issues](https://github.com/computational-chemical-engineering/spheropack/issues).

## License

MIT
