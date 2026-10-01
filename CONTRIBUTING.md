# Contributing to spheropack

Contributions are welcome: bug reports, documentation, examples, new containers or
analysis functions, and performance work.

## Development setup

```bash
git clone https://github.com/computational-chemical-engineering/spheropack
cd spheropack
python -m venv .venv && source .venv/bin/activate
pip install nanobind scikit-build-core numpy
pip install --no-build-isolation -e ".[dev]"
pre-commit install
```

The editable install does not rebuild the C++ extension automatically: rerun the
`pip install --no-build-isolation -e .` line after changing anything in `include/` or
`src/`.

## Tests

```bash
pytest                    # fast suite, about a minute
pytest -m slow            # statistical comparisons, about ten minutes
cmake -S . -B build/cpp -DSPHEROPACK_BUILD_TESTS=ON && cmake --build build/cpp && ctest --test-dir build/cpp
```

Event-driven dynamics is chaotic: compare results statistically (over seeds), never
sphere by sphere. Changes to the collision rules, the contact prediction or the
stopping criteria need the slow suite and, where possible, a new test with a known
answer (exact collisions, equations of state, conservation laws).

## Style

- Python: `ruff check .` and `ruff format .` (run by pre-commit); numpy-style docstrings;
  `mypy` must pass.
- C++: C++20, header-only core in `include/spheropack/`, Doxygen comments for everything
  public (`cd docs && doxygen Doxyfile` must not warn).
- Documentation: Sphinx in `docs/`; notebooks are committed with their outputs.
- No em dashes in text.

## Pull requests

Describe what changes and why, add tests, update the documentation and
`CHANGELOG.md`. CI runs on Linux, macOS and Windows.
