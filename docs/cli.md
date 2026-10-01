# Command line

Installing the package also installs the `spheropack` command.

```bash
# 1000 spheres of diameter 1 in a periodic box of edge 10, packed until jammed
spheropack pack -n 1000 --box 10 --density max --seed 1 -o packing.csv

# a slab with walls in z, target density 0.55, polydisperse radii from a file
spheropack pack --radii-file radii.txt --box 10,10,20 --walls z --density 0.55 -o slab.csv

# 2D, 500 disks in a channel with walls in y
spheropack pack -n 500 --dim 2 --box 30,20 --walls y --density max -o disks.csv
```

The output is CSV with one sphere per line, `x,y,z,r` (`x,y,r` in 2D), with a header
line; `--format plain` omits the header, `--format legacy` writes `z,x,y,r` without
header. A summary of the run goes to standard error. The exit status is 1 when the
target was not reached. `spheropack pack --help` lists all options, including the
stopping criteria (`--jammed-pressure`, `--max-pressure`, `--max-collisions`,
`--timeout`) and `--growth-rate` (see {doc}`guide/growth_rate`).

## Legacy tool

`spheropack legacy-periodic` replaces the legacy `generate_periodic_packing`: same
options, same output format (`z,x,y,r`), and `--growth_rate` with its legacy meaning
(a rate in 1/time, converted to the dimensionless growth rate as
$\Gamma = g\,d/2$). It uses the legacy collision rule, so existing scripts give
statistically the same packings, and like the legacy tool it does not stop on the
pressure (only after $10^5$ collisions per sphere). `spheropack legacy-tube` does the
same for `generate_packed_tube` (tube axis in the first output column). Both accept
`--collision-rule elastic_growing`, which never gets stuck and is faster, at the price
of a slightly different structure next to walls.

```bash
spheropack legacy-tube --num_part=725 --part_diam=3e-3 --tube_diam=21e-3 \
    --tube_length=0.0599 --growth_rate=106.67 --seed=1 --file=tube.dat
spheropack legacy-periodic --num_part=50 --part_diam=3e-3 --box_size=0.0153 \
    --growth_rate=106.67 --seed=1 --file=periodic.dat --copy_periodic
```
