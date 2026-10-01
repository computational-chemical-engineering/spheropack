"""Command-line interface: ``spheropack pack ...`` and legacy-compatible commands."""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Literal

import numpy as np

from . import stop as _stop
from ._pack import PackingWarning, pack
from ._version import __version__
from .containers import Box, Cylinder, SphereContainer


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def _write(p, path, fmt, images):
    if images:
        p = p.periodic_images()
    if path in (None, "-"):
        path = sys.stdout
    p.to_csv(path, legacy_order=(fmt == "legacy"), header=(fmt == "csv"))


def _container(args):
    dim = args.dim
    if args.container == "cylinder":
        if dim != 3 or args.diameter_container is None or args.length is None:
            raise SystemExit("--container cylinder needs --dim 3, --container-diameter and --length")
        return Cylinder(args.diameter_container, args.length, axis="xyz".index(args.axis), capped=args.capped)
    if args.container == "sphere":
        if args.diameter_container is None:
            raise SystemExit("--container sphere needs --container-diameter")
        return SphereContainer(args.diameter_container, dim=dim)
    if args.box is None:
        raise SystemExit("--container box needs --box")
    lengths = _floats(args.box)
    if len(lengths) == 1:
        lengths *= dim
    walls = set(args.walls or "")
    axes = "xyz"[:dim]
    if not walls <= set(axes):
        raise SystemExit(f"--walls takes axes from {axes!r}")
    return Box(lengths, periodic=[a not in walls for a in axes])


def _cmd_pack(args) -> int:
    container = _container(args)
    if args.radii_file:
        radii = np.loadtxt(args.radii_file, ndmin=1)
    else:
        radii = 0.5 * args.diameter
    density: float | Literal["max"] | None = None
    if args.density is not None:
        density = "max" if args.density == "max" else float(args.density)
    criteria: list[_stop.Criterion] = []
    if args.jammed_pressure:
        criteria.append(_stop.Jammed(pressure=args.jammed_pressure))
    if args.max_pressure:
        criteria.append(_stop.Pressure(args.max_pressure))
    if args.max_collisions:
        criteria.append(_stop.Collisions(per_particle=args.max_collisions))
    if args.timeout:
        criteria.append(_stop.Timeout(args.timeout))
    p = pack(
        args.n,
        radii,
        container,
        density=density,
        growth_rate=args.growth_rate,
        stop=criteria or None,
        seed=args.seed,
        collision_rule=args.collision_rule,
    )
    print(p, file=sys.stderr)
    _write(p, args.output, args.format, args.periodic_images)
    return 0 if p.success else 1


# The legacy tools had no pressure stop: with the legacy rule the mean pressure spikes
# when clusters lock, and such runs usually still reach the target. Only a generous
# collision budget ends a run that is really stuck.
_LEGACY_STOP = [_stop.Collisions(per_particle=1e5)]


def _cmd_legacy_periodic(args) -> int:
    # Same options, semantics and output as the legacy generate_periodic_packing.
    lengths = _floats(args.box_size)
    if len(lengths) == 1:
        lengths *= 3
    gamma = 0.5 * args.growth_rate * args.part_diam  # legacy rate g in 1/time -> Gamma
    print(f"# legacy growth_rate {args.growth_rate} corresponds to growth_rate Gamma = {gamma:.4g}", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PackingWarning)
        p = pack(
            args.num_part,
            0.5 * args.part_diam,
            Box(lengths, periodic=True),
            growth_rate=gamma,
            seed=args.seed,
            collision_rule=args.collision_rule,
            stop=_LEGACY_STOP,
        )
    print(p, file=sys.stderr)
    _write(p, args.file, "legacy", args.copy_periodic)
    return 0 if p.success else 1


def _cmd_legacy_tube(args) -> int:
    # Same options, semantics and output as the legacy generate_packed_tube.
    gamma = 0.5 * args.growth_rate * args.part_diam
    print(f"# legacy growth_rate {args.growth_rate} corresponds to growth_rate Gamma = {gamma:.4g}", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PackingWarning)
        p = pack(
            args.num_part,
            0.5 * args.part_diam,
            Cylinder(args.tube_diam, args.tube_length),
            growth_rate=gamma,
            seed=args.seed,
            collision_rule=args.collision_rule,
            stop=_LEGACY_STOP,
        )
    print(p, file=sys.stderr)
    _write(p, args.file, "legacy", False)
    return 0 if p.success else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="spheropack", description="Random sphere packings (Lubachevsky-Stillinger).")
    ap.add_argument("--version", action="version", version=f"spheropack {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "pack",
        help="pack spheres in a box",
        description="Pack spheres in a box. Writes x,y,z,r (CSV) to --output or stdout; "
        "a summary goes to stderr. Exit status 1 if the target was not reached.",
    )
    p.add_argument("-n", type=int, help="number of spheres (not needed with --radii-file)")
    p.add_argument("--diameter", type=float, default=1.0, help="sphere diameter (default 1)")
    p.add_argument("--radii-file", help="text file with one radius per sphere (polydisperse)")
    p.add_argument(
        "--container",
        choices=("box", "cylinder", "sphere"),
        default="box",
        help="box (default; see --box, --walls), cylinder (see --container-diameter, --length, --axis, "
        "--capped) or sphere (a disk in 2D)",
    )
    p.add_argument("--box", help="edge length L, or Lx,Ly[,Lz] (container box)")
    p.add_argument("--container-diameter", dest="diameter_container", type=float, help="cylinder or sphere diameter")
    p.add_argument("--length", type=float, help="cylinder length")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z", help="cylinder axis (default z)")
    p.add_argument("--capped", action="store_true", help="close the cylinder with flat walls (default: periodic)")
    p.add_argument("--dim", type=int, choices=(2, 3), default=3)
    p.add_argument("--walls", help="axes bounded by flat walls, e.g. 'z' or 'xy' (default: fully periodic)")
    p.add_argument(
        "--density", help="target volume fraction, or 'max' to pack until jammed (default: grow to the given diameters)"
    )
    p.add_argument(
        "--growth-rate",
        type=float,
        default=0.02,
        help="growth speed of the mean diameter / thermal speed (default 0.02)",
    )
    p.add_argument("--seed", type=int)
    p.add_argument("--collision-rule", choices=("elastic_growing", "legacy"), default="elastic_growing")
    p.add_argument("--jammed-pressure", type=float, help="stop: jamming protocol to this median pressure")
    p.add_argument("--max-pressure", type=float, help="stop: reduced pressure threshold")
    p.add_argument("--max-collisions", type=float, help="stop: collisions per sphere")
    p.add_argument("--timeout", type=float, help="stop: wall-clock seconds")
    p.add_argument("-o", "--output", help="output file (default stdout)")
    p.add_argument(
        "--format",
        choices=("csv", "legacy", "plain"),
        default="csv",
        help="csv: x,y,z,r with header; legacy: z,x,y,r without header; plain: x,y,z,r without header",
    )
    p.add_argument("--periodic-images", action="store_true", help="add images of spheres cut by periodic faces")
    p.set_defaults(func=_cmd_pack)

    q = sub.add_parser(
        "legacy-periodic",
        help="drop-in replacement of the legacy generate_periodic_packing",
        description="Options and output (z,x,y,r) of the legacy generate_periodic_packing tool. "
        "--growth_rate has the legacy meaning (1/time) and is converted.",
    )
    q.add_argument("--num_part", type=int, required=True)
    q.add_argument("--part_diam", type=float, required=True)
    q.add_argument("--box_size", required=True)
    q.add_argument("--growth_rate", type=float, default=32e-3)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--file")
    q.add_argument("--copy_periodic", action="store_true")
    q.add_argument(
        "--collision-rule",
        choices=("legacy", "elastic_growing"),
        default="legacy",
        help="legacy (default) reproduces the old code statistically; elastic_growing never gets stuck",
    )
    q.set_defaults(func=_cmd_legacy_periodic)

    t = sub.add_parser(
        "legacy-tube",
        help="drop-in replacement of the legacy generate_packed_tube",
        description="Options and output (z,x,y,r, tube axis first) of the legacy generate_packed_tube "
        "tool. --growth_rate has the legacy meaning (1/time) and is converted.",
    )
    t.add_argument("--num_part", type=int, required=True)
    t.add_argument("--part_diam", type=float, required=True)
    t.add_argument("--tube_diam", type=float, required=True)
    t.add_argument("--tube_length", type=float, required=True)
    t.add_argument("--growth_rate", type=float, default=32e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--file")
    t.add_argument(
        "--collision-rule",
        choices=("legacy", "elastic_growing"),
        default="legacy",
        help="legacy (default) reproduces the old code statistically; elastic_growing never gets stuck "
        "and is faster, with a slightly different wall structure",
    )
    t.set_defaults(func=_cmd_legacy_tube)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pack" and args.n is None and not args.radii_file:
        raise SystemExit("give -n or --radii-file")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
