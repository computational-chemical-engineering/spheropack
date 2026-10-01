# spheropack

Random sphere packings in Python, generated with the event-driven
Lubachevsky-Stillinger algorithm: spheres move ballistically, collide elastically and
grow until a target density is reached or the packing jams or crystallises.

```python
import spheropack as sp

p = sp.pack(n=1000, density=0.6)
p.positions, p.radii
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
guide/algorithm
guide/growth_rate
guide/stopping
cli
```

```{toctree}
:maxdepth: 1
:caption: Examples

notebooks/01_quickstart
notebooks/02_compression_and_jamming
notebooks/03_structure_analysis
notebooks/04_packed_tubes
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/python
api/cpp
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
citing
contributing
references
```
