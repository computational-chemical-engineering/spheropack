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
    assert (
        main(
            [
                "pack",
                "-n",
                "150",
                "--box",
                "10,20",
                "--dim",
                "2",
                "--walls",
                "y",
                "--density",
                "max",
                "--jammed-pressure",
                "1e6",
                "--seed",
                "2",
                "-o",
                str(out),
                "--format",
                "plain",
            ]
        )
        == 0
    )
    data = np.loadtxt(out, delimiter=",")
    assert data.shape == (150, 3)
    assert (data[:, 1] - data[:, 2]).min() > -1e-9 and (data[:, 1] + data[:, 2]).max() < 20 + 1e-9


def test_legacy_periodic_matches_legacy_format(tmp_path):
    out = tmp_path / "legacy.dat"
    rc = main(
        [
            "legacy-periodic",
            "--num_part=50",
            "--part_diam=3e-3",
            "--box_size=0.0153",
            "--growth_rate=106.67",
            "--seed=3",
            f"--file={out}",
        ]
    )
    assert rc == 0
    data = np.loadtxt(out, delimiter=",")
    assert data.shape == (50, 4)
    np.testing.assert_allclose(data[:, 3], 1.5e-3)


def test_cylinder_from_cli(tmp_path):
    out = tmp_path / "t.csv"
    assert (
        main(
            [
                "pack",
                "-n",
                "200",
                "--container",
                "cylinder",
                "--container-diameter",
                "7",
                "--length",
                "10",
                "--density",
                "0.5",
                "--seed",
                "1",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    data = np.loadtxt(out, delimiter=",", skiprows=1)
    rc = np.hypot(data[:, 0] - 3.5, data[:, 1] - 3.5)
    assert (rc + data[:, 3]).max() <= 3.5 * (1 + 1e-12)


def test_legacy_tube_format(tmp_path):
    out = tmp_path / "tube.dat"
    d, D, n = 3e-3, 21e-3, 100
    L = n * (np.pi / 6) * d**3 / 0.5 / (0.25 * np.pi * D**2)
    assert (
        main(
            [
                "legacy-tube",
                f"--num_part={n}",
                f"--part_diam={d}",
                f"--tube_diam={D}",
                f"--tube_length={L}",
                "--growth_rate=106.67",
                "--seed=1",
                f"--file={out}",
            ]
        )
        == 0
    )
    data = np.loadtxt(out, delimiter=",")
    assert data.shape == (n, 4)
    assert data[:, 0].max() <= L  # first column: tube axis
    rc = np.hypot(data[:, 1] - D / 2, data[:, 2] - D / 2)
    assert (rc + data[:, 3]).max() <= D / 2 * (1 + 1e-12)
