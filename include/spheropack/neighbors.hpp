/// @file neighbors.hpp
/// @brief Enumeration of particle pairs within a cutoff distance, including periodic images.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "box.hpp"
#include "cell_grid.hpp"
#include "vec.hpp"

namespace spheropack {

/// @brief Calls `f(i, j, r)` once for every pair `i < j` and every periodic image of
/// `j` with `|r| < cutoff`, where `r = x_j + shift * L - x_i`.
///
/// Positions must lie inside the box. On periodic axes the cutoff may exceed half the
/// box length but not the box length itself; every image within the cutoff is then
/// reported separately. Images of a particle with itself are not reported.
///
/// @tparam D dimension (2 or 3)
/// @param x positions, inside the box
/// @param box container
/// @param cutoff largest distance of interest
/// @param f callable `f(std::uint32_t i, std::uint32_t j, const Vec<D>& r)`
template <int D, class F>
void for_each_pair(const std::vector<Vec<D>>& x, const Box<D>& box, double cutoff, F&& f) {
  CellGrid<D> grid;
  grid.build(box, cutoff, 2 * x.size() + 27);
  grid.clear_particles(x.size());
  for (std::uint32_t i = 0; i < x.size(); ++i) grid.insert(i, grid.flat(grid.coords_of(x[i])));
  const double cutoff2 = cutoff * cutoff;
  for (std::uint32_t i = 0; i < x.size(); ++i) {
    grid.for_each_neighbor(grid.cell_of(i), [&](std::uint32_t j, const std::array<int, D>& shift) {
      if (j <= i) return;
      Vec<D> r;
      for (int k = 0; k < D; ++k) r[k] = x[j][k] + shift[k] * box.L[k] - x[i][k];
      if (norm2<D>(r) < cutoff2) f(i, j, r);
    });
  }
}

}  // namespace spheropack
