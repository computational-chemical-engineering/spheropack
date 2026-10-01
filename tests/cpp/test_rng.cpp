#include <doctest/doctest.h>

#include "spheropack/rng.hpp"

using spheropack::Rng;

TEST_CASE("splitmix64 matches the published reference") {
  std::uint64_t state = 0;
  CHECK(spheropack::splitmix64(state) == 0xe220a8397b1dcdafULL);
}

TEST_CASE("xoshiro256++ stream is pinned") {
  Rng rng(0);
  CHECK(rng() == 5987356902031041503ULL);
  CHECK(rng() == 7051070477665621255ULL);
  CHECK(Rng(12345)() == 10201931350592234856ULL);
}

TEST_CASE("reseed restarts the stream and clears the cached normal") {
  Rng a(7);
  const double first = a.normal();
  a.reseed(7);
  CHECK(a.normal() == first);
}

TEST_CASE("uniform variates lie in their documented intervals") {
  Rng rng(1);
  for (int i = 0; i < 100000; ++i) {
    const double u = rng.uniform();
    const double v = rng.uniform_open0();
    CHECK_UNARY(u >= 0.0 && u < 1.0);
    CHECK_UNARY(v > 0.0 && v <= 1.0);
  }
}
