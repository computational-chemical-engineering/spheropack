import numpy as np

from spheropack import _core


def test_raw_stream_is_fixed():
    # Reference values pin the integer stream: they must never change between
    # releases or platforms, because seeded packings depend on them.
    assert _core._rng_raw(0) == REFERENCE[0]
    assert _core._rng_raw(0, skip=1) == REFERENCE[1]
    assert _core._rng_raw(12345) == REFERENCE[2]


def test_same_seed_same_stream():
    a = _core._rng_sample(42, 1000, "uniform")
    b = _core._rng_sample(42, 1000, "uniform")
    c = _core._rng_sample(43, 1000, "uniform")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_uniform_moments():
    x = _core._rng_sample(1, 200_000, "uniform")
    assert x.min() >= 0.0 and x.max() < 1.0
    assert abs(x.mean() - 0.5) < 5 * np.sqrt(1 / 12 / x.size)
    assert abs(x.var() - 1 / 12) < 1e-3


def test_normal_moments():
    x = _core._rng_sample(2, 200_000, "normal")
    assert abs(x.mean()) < 5 / np.sqrt(x.size)
    assert abs(x.var() - 1.0) < 0.01
    assert abs(np.mean(x**4) - 3.0) < 0.05


def test_exponential_moments():
    x = _core._rng_sample(3, 200_000, "exponential")
    assert x.min() > 0.0
    assert abs(x.mean() - 1.0) < 5 / np.sqrt(x.size)
    assert abs(x.var() - 1.0) < 0.02


# First outputs of xoshiro256++ seeded through SplitMix64, cross-checked against an
# independent pure-Python implementation (SplitMix64 from 0 starts 0xe220a8397b1dcdaf).
REFERENCE = [5987356902031041503, 7051070477665621255, 10201931350592234856]
