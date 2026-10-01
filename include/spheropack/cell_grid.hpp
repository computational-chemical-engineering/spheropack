/// @file cell_grid.hpp
/// @brief Cell list on a box: doubly linked lists of particles per cell, and a 3^D stencil
/// that yields every (cell, periodic image shift) pair a particle may touch.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "box.hpp"

namespace spheropack {

/// @brief Cell list with O(1) insertion, removal and move of particles.
///
/// Particles are stored in one doubly linked list per cell. Particle indices and cell
/// indices are 32-bit; `none` marks the end of a list and a particle in no cell.
/// @tparam D dimension (2 or 3)
template <int D>
class CellGrid {
 public:
  /// Integer cell coordinates, one per axis.
  using Index = std::array<int, D>;
  /// Sentinel for "no particle" / "no cell".
  static constexpr std::int32_t none = -1;

  /// @brief Choose the cell layout for a box and empty all cells.
  ///
  /// Cells are at least `min_width` wide along every axis; the total number of cells
  /// is capped near `max_cells` by widening cells (never by narrowing them). Call
  /// `clear_particles` afterwards, before inserting.
  /// @param box container (copied; its periodicity is used by `for_each_neighbor`)
  /// @param min_width lower bound on the cell width along every axis, typically the
  ///        largest interaction distance; a value <= 0 requests the finest grid allowed
  /// @param max_cells approximate upper bound on the total number of cells
  void build(const Box<D>& box, double min_width, std::size_t max_cells) {
    box_ = box;
    for (int k = 0; k < D; ++k) {
      const double n = min_width > 0.0 ? std::floor(box.L[k] / min_width) : 1e9;
      n_[k] = static_cast<int>(std::clamp(n, 1.0, 1e6));
    }
    auto total = [&] {
      double t = 1.0;
      for (int k = 0; k < D; ++k) t *= n_[k];
      return t;
    };
    while (total() > static_cast<double>(std::max<std::size_t>(max_cells, 1))) {
      int kmax = 0;
      for (int k = 1; k < D; ++k)
        if (n_[k] > n_[kmax]) kmax = k;
      if (n_[kmax] == 1) break;
      n_[kmax] = std::max(1, n_[kmax] * 3 / 4);
    }
    std::size_t cells = 1;
    for (int k = 0; k < D; ++k) {
      w_[k] = box.L[k] / n_[k];
      cells *= static_cast<std::size_t>(n_[k]);
    }
    head_.assign(cells, none);
  }

  /// @brief Remove all particles and size the per-particle arrays.
  /// @param n_particles number of particles that will be inserted
  void clear_particles(std::size_t n_particles) {
    std::fill(head_.begin(), head_.end(), none);
    next_.assign(n_particles, none);
    prev_.assign(n_particles, none);
    cell_.assign(n_particles, none);
  }

  /// @return smallest cell width over all axes
  double min_width() const noexcept {
    double w = w_[0];
    for (int k = 1; k < D; ++k) w = std::min(w, w_[k]);
    return w;
  }

  /// @return number of cells along axis `k`
  int n(int k) const noexcept { return n_[k]; }
  /// @return cell width along axis `k`
  double width(int k) const noexcept { return w_[k]; }
  /// @return total number of cells
  std::size_t num_cells() const noexcept { return head_.size(); }

  /// @brief Cell coordinates containing a position.
  /// @param x position; coordinates outside the box are clamped to the edge cells
  /// @return integer cell coordinates
  Index coords_of(const Vec<D>& x) const noexcept {
    Index c;
    for (int k = 0; k < D; ++k)
      c[k] = std::clamp(static_cast<int>(std::floor(x[k] / w_[k])), 0, n_[k] - 1);
    return c;
  }

  /// @brief Flatten cell coordinates (axis 0 varies fastest).
  /// @param c cell coordinates, within range
  /// @return cell index
  std::int32_t flat(const Index& c) const noexcept {
    std::int32_t f = 0;
    for (int k = D - 1; k >= 0; --k) f = f * n_[k] + c[k];
    return f;
  }

  /// @brief Inverse of `flat`.
  /// @param f cell index
  /// @return cell coordinates
  Index coords(std::int32_t f) const noexcept {
    Index c;
    for (int k = 0; k < D; ++k) {
      c[k] = f % n_[k];
      f /= n_[k];
    }
    return c;
  }

  /// @param i particle index
  /// @return cell holding particle `i`, or `none` if it is not inserted
  std::int32_t cell_of(std::uint32_t i) const noexcept { return cell_[i]; }

  /// @brief Put particle `i` at the head of the list of `cell`.
  /// @param i particle index; must not currently be in a cell
  /// @param cell destination cell index
  void insert(std::uint32_t i, std::int32_t cell) noexcept {
    cell_[i] = cell;
    prev_[i] = none;
    next_[i] = head_[cell];
    if (head_[cell] != none) prev_[head_[cell]] = static_cast<std::int32_t>(i);
    head_[cell] = static_cast<std::int32_t>(i);
  }

  /// @brief Take particle `i` out of its cell; afterwards `cell_of(i) == none`.
  /// @param i particle index; must currently be in a cell
  void remove(std::uint32_t i) noexcept {
    const std::int32_t c = cell_[i];
    if (prev_[i] != none) next_[prev_[i]] = next_[i];
    else head_[c] = next_[i];
    if (next_[i] != none) prev_[next_[i]] = prev_[i];
    cell_[i] = none;
  }

  /// @brief Move particle `i` to another cell (`remove` followed by `insert`).
  /// @param i particle index; must currently be in a cell
  /// @param cell destination cell index
  void move(std::uint32_t i, std::int32_t cell) noexcept {
    remove(i);
    insert(i, cell);
  }

  /// @brief Like for_each_neighbor(), but only the layer of the stencil at offset `dir`
  /// (+1 or -1) along `axis`: the cells that become adjacent when a particle moves into
  /// `cell` across that face. Requires at least three cells along `axis`.
  /// @param cell centre cell of the stencil
  /// @param axis axis of the move
  /// @param dir direction of the move along `axis` (+1 or -1)
  /// @param f callable `f(std::uint32_t j, const std::array<int, D>& shift)`
  template <class F>
  void for_each_neighbor_layer(std::int32_t cell, int axis, int dir, F&& f) const {
    const Index c = coords(cell);
    std::array<int, D> o;
    o.fill(-1);
    o[axis] = dir;
    for (;;) {
      Index nc;
      std::array<int, D> shift{};
      bool valid = true;
      for (int k = 0; k < D && valid; ++k) {
        int ck = c[k] + o[k];
        if (ck < 0) {
          if (box_.periodic[k]) {
            ck += n_[k];
            shift[k] = -1;
          } else valid = false;
        } else if (ck >= n_[k]) {
          if (box_.periodic[k]) {
            ck -= n_[k];
            shift[k] = 1;
          } else valid = false;
        }
        nc[k] = ck;
      }
      if (valid)
        for (std::int32_t j = head_[flat(nc)]; j != none; j = next_[j]) f(static_cast<std::uint32_t>(j), shift);
      int k = 0;
      while (k < D && (k == axis || ++o[k] > 1)) {
        if (k != axis) o[k] = -1;
        ++k;
      }
      if (k == D) break;
    }
  }

  /// @brief Visit every particle in the \f$3^D\f$ stencil of cells around `cell`.
  ///
  /// Calls `f(j, shift)` for each particle `j` found, where the image of `j` to use is
  /// \f$x_j + \mathrm{shift}\cdot L\f$ (componentwise, shift entries in {-1, 0, 1}).
  /// On periodic axes with fewer than three cells the stencil visits the same cell
  /// with different shifts, which is what makes small periodic boxes correct. On
  /// non-periodic axes stencil cells outside the grid are skipped. Includes the
  /// particle itself with shift 0.
  /// @param cell centre cell of the stencil
  /// @param f callable `f(std::uint32_t j, const std::array<int, D>& shift)`; the grid
  ///        must not be modified from inside `f`
  template <class F>
  void for_each_neighbor(std::int32_t cell, F&& f) const {
    const Index c = coords(cell);
    std::array<int, D> o;
    o.fill(-1);
    for (;;) {
      Index nc;
      std::array<int, D> shift{};
      bool valid = true;
      for (int k = 0; k < D && valid; ++k) {
        int ck = c[k] + o[k];
        if (ck < 0) {
          if (box_.periodic[k]) {
            ck += n_[k];
            shift[k] = -1;
          } else valid = false;
        } else if (ck >= n_[k]) {
          if (box_.periodic[k]) {
            ck -= n_[k];
            shift[k] = 1;
          } else valid = false;
        }
        nc[k] = ck;
      }
      if (valid)
        for (std::int32_t j = head_[flat(nc)]; j != none; j = next_[j])
          f(static_cast<std::uint32_t>(j), shift);
      int k = 0;
      while (k < D && ++o[k] > 1) o[k++] = -1;
      if (k == D) break;
    }
  }

 private:
  Box<D> box_{};                   ///< container, for periodicity
  std::array<int, D> n_{};         ///< cells per axis
  Vec<D> w_{};                     ///< cell width per axis
  /// Linked lists: `head_[cell]` first particle, `next_`/`prev_` neighbours in the list,
  /// `cell_[i]` the cell of particle i; all `none` when absent.
  std::vector<std::int32_t> head_, next_, prev_, cell_;
};

}  // namespace spheropack
