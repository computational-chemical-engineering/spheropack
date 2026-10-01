import pytest

import spheropack as sp

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
from spheropack.plotting import plot_disks, plot_section  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_disks_periodic_and_disk():
    p = sp.pack(n=100, density=0.6, container=sp.PeriodicBox(1.0, dim=2), seed=1)
    ax = plot_disks(p)
    assert len(ax.collections[0].get_paths()) >= p.n  # images added at the edges
    q = sp.pack(n=50, density=0.5, container=sp.Disk(1.0), seed=1)
    ax = plot_disks(q)
    assert len(ax.patches) == 1  # circular outline


@pytest.mark.parametrize("container", [sp.PeriodicBox(1.0), sp.Cylinder(1.0, 1.0), sp.SphereContainer(1.0)])
@pytest.mark.parametrize("axis", [0, 2])
def test_plot_section(container, axis):
    p = sp.pack(n=150, density=0.45, container=container, seed=2)
    ax = plot_section(p, axis=axis)
    assert len(ax.collections[0].get_paths()) > 0
    assert len(ax.patches) == 1  # box outline or wall circle


def test_plot_rejects_wrong_dimension():
    p3 = sp.pack(n=20, density=0.3, seed=1)
    p2 = sp.pack(n=20, density=0.3, container=sp.PeriodicBox(1.0, dim=2), seed=1)
    with pytest.raises(ValueError):
        plot_disks(p3)
    with pytest.raises(ValueError):
        plot_section(p2)
