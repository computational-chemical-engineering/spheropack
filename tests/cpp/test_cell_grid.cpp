#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "spheropack/cell_grid.hpp"
#include "spheropack/rng.hpp"

using spheropack::Box;
using spheropack::CellGrid;
using spheropack::Vec;

namespace {

using Key = std::tuple<std::uint32_t, int, int, int>;  // particle j and image shift

// All (j, shift) whose image lies within `reach` of particle i, by brute force.
std::set<Key> brute_force(const std::vector<Vec<3>>& x, const Box<3>& box, std::uint32_t i, double reach) {
  std::set<Key> out;
  for (std::uint32_t j = 0; j < x.size(); ++j)
    for (int a = -3; a <= 3; ++a)
      for (int b = -3; b <= 3; ++b)
        for (int c = -3; c <= 3; ++c) {
          const std::array<int, 3> s{a, b, c};
          bool ok = true;
          double d2 = 0.0;
          for (int k = 0; k < 3; ++k) {
            if (!box.periodic[k] && s[k] != 0) ok = false;
            const double d = x[j][k] + s[k] * box.L[k] - x[i][k];
            d2 += d * d;
          }
          if (ok && d2 < reach * reach && !(j == i && a == 0 && b == 0 && c == 0)) out.insert({j, a, b, c});
        }
  return out;
}

void check_grid(const Box<3>& box, double reach, int stencil, std::uint64_t seed) {
  spheropack::Rng rng(seed);
  std::vector<Vec<3>> x(60);
  for (auto& p : x)
    for (int k = 0; k < 3; ++k) p[k] = box.L[k] * rng.uniform();
  CellGrid<3> grid;
  grid.build(box, reach / stencil, 1000, stencil);
  grid.clear_particles(x.size());
  for (std::uint32_t i = 0; i < x.size(); ++i) grid.insert(i, grid.flat(grid.coords_of(x[i])));
  for (std::uint32_t i = 0; i < x.size(); ++i) {
    std::multiset<Key> visited;
    grid.for_each_neighbor(grid.cell_of(i), [&](std::uint32_t j, const std::array<int, 3>& s) {
      if (!(j == i && s[0] == 0 && s[1] == 0 && s[2] == 0)) visited.insert({j, s[0], s[1], s[2]});
    });
    // every image within reach is visited, and no image twice
    for (const auto& key : brute_force(x, box, i, reach)) CHECK(visited.count(key) == 1);
    std::set<Key> unique(visited.begin(), visited.end());
    CHECK(unique.size() == visited.size());
  }
}

}  // namespace

TEST_CASE("stencils find every image within reach exactly once") {
  for (int stencil : {1, 2}) {
    Box<3> periodic;
    periodic.L = {4.0, 5.0, 6.0};
    periodic.periodic = {true, true, true};
    check_grid(periodic, 1.0, stencil, 1);   // many cells
    check_grid(periodic, 2.0, stencil, 2);   // the box is only twice the reach
    Box<3> mixed;
    mixed.L = {3.0, 3.0, 5.0};
    mixed.periodic = {true, false, true};
    check_grid(mixed, 1.2, stencil, 3);
  }
}

TEST_CASE("layer iteration covers the cells that become adjacent after a crossing") {
  for (int stencil : {1, 2}) {
    Box<3> box;
    box.L = {6.0, 6.0, 6.0};
    box.periodic = {true, true, true};
    CellGrid<3> grid;
    grid.build(box, 1.0 / stencil, 10000, stencil);
    std::vector<Vec<3>> x;
    spheropack::Rng rng(7);
    for (int n = 0; n < 200; ++n) x.push_back({6.0 * rng.uniform(), 6.0 * rng.uniform(), 6.0 * rng.uniform()});
    grid.clear_particles(x.size());
    for (std::uint32_t i = 0; i < x.size(); ++i) grid.insert(i, grid.flat(grid.coords_of(x[i])));
    for (int axis = 0; axis < 3; ++axis)
      for (int dir : {-1, 1}) {
        const auto from = grid.coords_of({3.01, 3.01, 3.01});
        auto to = from;
        to[axis] += dir;
        auto collect = [&](auto&& visit) {
          std::set<Key> out;
          visit([&](std::uint32_t j, const std::array<int, 3>& s) { out.insert({j, s[0], s[1], s[2]}); });
          return out;
        };
        const auto before = collect([&](auto f) { grid.for_each_neighbor(grid.flat(from), f); });
        const auto after = collect([&](auto f) { grid.for_each_neighbor(grid.flat(to), f); });
        const auto layer = collect([&](auto f) { grid.for_each_neighbor_layer(grid.flat(to), axis, dir, f); });
        for (const auto& key : after) CHECK((before.count(key) == 1 || layer.count(key) == 1));
      }
  }
}
