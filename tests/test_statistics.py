"""Statistical comparison with the legacy code (tests/baselines). Slow: run with -m slow."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import spheropack as sp

BASELINES = Path(__file__).parent / "baselines"


def _arrest(n, growth_rate, seeds):
    # Units and stopping rule of the legacy baseline driver: final diameter 1 at density
    # 0.74, legacy collision rule, stop after 10^4 collisions per sphere.
    box = sp.PeriodicBox((n * np.pi / 6 / 0.74) ** (1 / 3))
    stop = [sp.stop.Collisions(per_particle=1e4), sp.stop.Pressure(1e300)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.PackingWarning)  # a budget stop counts as failure
        return np.array(
            [
                sp.pack(n=n, radii=0.5, container=box, density="max", growth_rate=growth_rate, seed=s,
                        collision_rule="legacy", stop=stop).density
                for s in seeds
            ]
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fname, growth_rate",
    [("periodic_arrest_N500.json", 0.16), ("periodic_arrest_N500_refrate.json", 0.0213)],
)
def test_arrest_density_matches_legacy(fname, growth_rate):
    base = json.loads((BASELINES / fname).read_text())["summary"]["phi_final"]
    phi = _arrest(500, growth_rate, range(101, 121))
    stderr = np.hypot(phi.std(ddof=1), base["std"]) / np.sqrt(len(phi))
    assert abs(phi.mean() - base["mean"]) < 4 * stderr, (phi.mean(), base)
