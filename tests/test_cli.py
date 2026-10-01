import numpy as np

from spheropack.cli import main


def test_pack_to_csv(tmp_path):
    out = tmp_path / "p.csv"
    assert main(["pack", "-n", "200", "--box", "8", "--density", "0.5", "--seed", "1", "-o", str(out)]) == 0
    data = np.loadtxt(out, delimiter=",", skiprows=1)
    assert data.shape == (200, 4)
    assert np.isclose(200 * 4 / 3 * np.pi * (data[:, 3] ** 3).mean() * 1 / 8**3, 0.5)


def test_pack_slab_until_jammed_2d(tmp_path):
    out = tmp_path / "p.csv"
    assert main(["pack", "-n", "150", "--box", "10,20", "--dim", "2", "--walls", "y", "--density", "max",
                 "--jammed-pressure", "1e6", "--seed", "2", "-o", str(out), "--format", "plain"]) == 0
    data = np.loadtxt(out, delimiter=",")
    assert data.shape == (150, 3)
    assert (data[:, 1] - data[:, 2]).min() > -1e-9 and (data[:, 1] + data[:, 2]).max() < 20 + 1e-9


def test_legacy_periodic_matches_legacy_format(tmp_path):
    out = tmp_path / "legacy.dat"
    rc = main(["legacy-periodic", "--num_part=50", "--part_diam=3e-3", "--box_size=0.0153",
               "--growth_rate=106.67", "--seed=3", f"--file={out}"])
    assert rc == 0
    data = np.loadtxt(out, delimiter=",")
    assert data.shape == (50, 4)
    np.testing.assert_allclose(data[:, 3], 1.5e-3)
