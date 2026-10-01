/// @file box.hpp
/// @brief Containers: a box that is periodic or walled per axis, optionally with a
/// curved (ball) wall over a subset of the axes (cylinder, spherical container, disk).
#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "vec.hpp"

namespace spheropack {

/// @brief Measure of a ball of radius `r` in `m` dimensions (m = 1, 2, 3).
/// @return \f$2r\f$, \f$\pi r^2\f$ or \f$\tfrac{4}{3}\pi r^3\f$
inline double ball_measure(int m, double r) noexcept {
  switch (m) {
    case 1: return 2.0 * r;
    case 2: return std::numbers::pi * r * r;
    default: return 4.0 / 3.0 * std::numbers::pi * r * r * r;
  }
}

/// @brief Volume (D=3) or area (D=2) of a ball of radius `r`.
/// @tparam D dimension (2 or 3)
/// @param r radius
/// @return \f$\pi r^2\f$ for D=2, \f$\tfrac{4}{3}\pi r^3\f$ for D=3
template <int D>
constexpr double ball_volume(double r) noexcept {
  if constexpr (D == 2) return std::numbers::pi * r * r;
  else return 4.0 / 3.0 * std::numbers::pi * r * r * r;
}

/// @brief Curved wall: the region \f$\sum_{k \in A} (x_k - c_k)^2 \le R^2\f$ over a set
/// of axes \f$A\f$. Over two axes of a 3D box it is a cylinder, over all axes a
/// spherical container (3D) or a disk (2D).
/// @tparam D dimension (2 or 3)
template <int D>
struct BallWall {
  bool active = false;         ///< false: no curved wall
  std::array<bool, D> axes{};  ///< axes the ball constrains
  Vec<D> center{};             ///< centre (only the components on `axes` are used)
  double radius = 0.0;         ///< radius \f$R\f$
};

/// @brief Axis-aligned box \f$[0, L_0) \times \dots \times [0, L_{D-1})\f$ with a boundary
/// type per axis, and an optional curved wall inside it.
///
/// Along a periodic axis the box repeats. Along a non-periodic axis that the ball does
/// not constrain, flat walls bound the box at 0 and \f$L_k\f$. Along the axes of the
/// ball, the ball wall bounds the spheres; it must lie inside the box.
/// @tparam D dimension (2 or 3)
template <int D>
struct Box {
  Vec<D> L{};                      ///< edge lengths; the box spans [0, L_k) along axis k
  std::array<bool, D> periodic{};  ///< per axis; false: walls (flat, or the ball)
  BallWall<D> ball{};              ///< optional curved wall

  /// @return true if axis `k` is bounded by flat walls at 0 and \f$L_k\f$
  bool flat_walls(int k) const noexcept { return !periodic[k] && !(ball.active && ball.axes[k]); }

  /// @brief Check the geometry.
  /// @throws std::invalid_argument if an edge length is not positive (or NaN), or if
  /// the ball wall is periodic along one of its axes or does not fit in the box
  void validate() const {
    for (int k = 0; k < D; ++k)
      if (!(L[k] > 0.0)) throw std::invalid_argument("box edge lengths must be positive");
    if (!ball.active) return;
    if (!(ball.radius > 0.0)) throw std::invalid_argument("ball wall radius must be positive");
    int m = 0;
    for (int k = 0; k < D; ++k) {
      if (!ball.axes[k]) continue;
      ++m;
      if (periodic[k]) throw std::invalid_argument("a ball wall axis cannot be periodic");
      if (ball.center[k] - ball.radius < 0.0 || ball.center[k] + ball.radius > L[k] * (1.0 + 1e-12))
        throw std::invalid_argument("the ball wall must lie inside the box");
    }
    if (m == 0) throw std::invalid_argument("a ball wall needs at least one axis");
  }

  /// @return volume (D=3) or area (D=2) available to the spheres: the box, or the ball
  /// over its axes times the box edges along the other axes
  double volume() const noexcept {
    double v = 1.0;
    int m = 0;
    for (int k = 0; k < D; ++k) {
      if (ball.active && ball.axes[k]) ++m;
      else v *= L[k];
    }
    return m > 0 ? v * ball_measure(m, ball.radius) : v;
  }

  /// @brief Squared distance from the ball centre, over the ball axes only.
  double ball_distance2(const Vec<D>& x) const noexcept {
    double q2 = 0.0;
    for (int k = 0; k < D; ++k)
      if (ball.axes[k]) q2 += (x[k] - ball.center[k]) * (x[k] - ball.center[k]);
    return q2;
  }
};

}  // namespace spheropack
