/// @file box.hpp
/// @brief Rectangular container, periodic or bounded by flat walls along each axis.
#pragma once

#include <array>
#include <numbers>
#include <stdexcept>

#include "vec.hpp"

namespace spheropack {

/// @brief Axis-aligned box \f$[0, L_0) \times \dots \times [0, L_{D-1})\f$ with a boundary type per axis.
/// @tparam D dimension (2 or 3)
template <int D>
struct Box {
  Vec<D> L{};                    ///< edge lengths; the box spans [0, L_k) along axis k
  std::array<bool, D> periodic{};  ///< per axis; false: flat walls at 0 and L_k

  /// @brief Check the geometry.
  /// @throws std::invalid_argument if any edge length is not positive (or is NaN)
  void validate() const {
    for (int k = 0; k < D; ++k)
      if (!(L[k] > 0.0)) throw std::invalid_argument("box edge lengths must be positive");
  }

  /// @return volume (D=3) or area (D=2), \f$\prod_k L_k\f$
  double volume() const noexcept {
    double v = 1.0;
    for (int k = 0; k < D; ++k) v *= L[k];
    return v;
  }
};

/// @brief Volume (D=3) or area (D=2) of a ball of radius `r`.
/// @tparam D dimension (2 or 3)
/// @param r radius
/// @return \f$\pi r^2\f$ for D=2, \f$\tfrac{4}{3}\pi r^3\f$ for D=3
template <int D>
constexpr double ball_volume(double r) noexcept {
  if constexpr (D == 2) return std::numbers::pi * r * r;
  else return 4.0 / 3.0 * std::numbers::pi * r * r * r;
}

}  // namespace spheropack
