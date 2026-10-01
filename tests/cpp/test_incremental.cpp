#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "spheropack/ls_packing.hpp"
#include "spheropack/rejection_free.hpp"

using namespace spheropack;

// After a cell crossing only the new cell layer is scanned. Forcing a full scan must give
// the same events. Positions are stored at different moments in the two modes, so the
// trajectories differ by round-off, which the chaotic dynamics amplifies by about a
// decade per 600 events; a missed or spurious event would show as an O(1) jump instead.
// Hence the comparison over a short horizon.

namespace {
template <class Sim>
double max_difference(Sim& a, Sim& b, double L) {
  const auto xa = a.positions(), xb = b.positions();
  double m = 0.0;
  for (std::size_t i = 0; i < xa.size(); ++i)
    for (int k = 0; k < 3; ++k) {
      double d = std::abs(xa[i][k] - xb[i][k]);
      m = std::max(m, std::min(d, L - d));  // a wrap is not a difference
    }
  return m;
}
}  // namespace

TEST_CASE("packing: incremental crossing scans match full scans") {
  const int n = 400;
  const double L = std::cbrt(n * 3.141592653589793 / 6 / 0.55);
  Box<3> box;
  box.L = {L, L, L};
  box.periodic = {true, true, true};
  PackingOptions opt;
  opt.growth_rate = 0.05;
  opt.seed = 3;
  opt.rule = CollisionRule::elastic_growing;
  opt.max_time = 4.0;
  LSPacking<3> a(box, std::vector<double>(n, 0.5), opt), b(box, std::vector<double>(n, 0.5), opt);
  b.set_full_rescan(true);
  a.run();
  b.run();
  CHECK(a.stats().n_collisions == b.stats().n_collisions);
  CHECK(a.stats().n_collisions > 50);
  CHECK(max_difference(a, b, L) < 1e-8);
}

TEST_CASE("rejection-free: incremental crossing scans match full scans") {
  for (double L : {5.2, 9.0}) {  // 4 cells per axis (fewer than 2m + 1: wraps, full scans) and 7
    Box<3> box;
    box.L = {L, L, L};
    box.periodic = {true, true, true};
    Rng rng(5);
    std::vector<Vec<3>> x(static_cast<std::size_t>(0.3 * L * L * L));
    for (auto& p : x)
      for (int k = 0; k < 3; ++k) p[k] = L * rng.uniform();
    PairPotential pot;
    pot.kind = PairPotential::Kind::lennard_jones;
    pot.cutoff = 2.5;
    RejectionFreeMC<3> a(box, x, pot, 1.0, 9), b(box, x, pot, 1.0, 9);
    b.set_full_rescan(true);
    a.run(2.0);
    b.run(2.0);
    CHECK(a.stats().n_reflections == b.stats().n_reflections);
    CHECK(a.stats().n_reflections > 200);
    CHECK(max_difference(a, b, L) < 1e-8);
  }
}
