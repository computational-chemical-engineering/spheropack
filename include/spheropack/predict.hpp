/// @file predict.hpp
/// @brief Contact time prediction for a gap that is quadratic in time.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace spheropack {

/// Event time meaning "no contact" (positive infinity).
inline constexpr double never = std::numeric_limits<double>::infinity();

/// @brief Earliest \f$\tau \ge 0\f$ at which the quadratic
/// \f$f(\tau) = A\tau^2 + 2B\tau + C\f$ reaches zero while decreasing.
///
/// Here \f$f\f$ is the squared distance of a pair minus its squared contact
/// distance, so the root is the first contact. \f$C \le 0\f$ means contact (or
/// round-off overlap) now. The roots use the cancellation-free form
/// \f$q = -(B + \mathrm{sign}(B)\sqrt{B^2 - AC})\f$, with roots \f$q/A\f$ and
/// \f$C/q\f$. With growing particles \f$A < 0\f$ is legal.
///
/// @param A coefficient of \f$\tau^2\f$
/// @param B half the coefficient of \f$\tau\f$ (\f$B<0\f$: pair closing now)
/// @param C value at \f$\tau = 0\f$
/// @return contact time, 0 if already in contact, or `never` if the pair does not meet
inline double first_contact(double A, double B, double C) noexcept {
  if (B < 0.0) {  // closing now
    const double disc = B * B - A * C;
    // disc < 0: with C > 0 (then A > 0) the pair passes by; with C < 0 it overlaps by
    // round-off and is closing, so it collides now.
    if (disc < 0.0) return C < 0.0 ? 0.0 : never;
    return std::max(0.0, C / (-B + std::sqrt(disc)));
  }
  if (A < 0.0) {  // opening now, but contact distance outgrows the separation
    const double disc = B * B - A * C;
    // disc < 0 needs C < 0: a round-off overlap that never opens; collide when it
    // starts closing again so the collision rule can separate the pair.
    if (disc < 0.0) return -B / A;
    return (B + std::sqrt(disc)) / (-A);
  }
  return never;
}

/// @brief Earliest \f$\tau \ge 0\f$ at which \f$\mathrm{gap} + \mathrm{rate}\,\tau\f$
/// reaches zero (flat wall, gap linear in time).
/// @param gap current distance to the wall
/// @param rate rate of change of the gap
/// @return contact time (0 if the gap is already non-positive and closing), or `never`
///         if the gap does not decrease
inline double first_contact_linear(double gap, double rate) noexcept {
  if (rate >= 0.0) return never;
  return std::max(0.0, gap / -rate);
}

}  // namespace spheropack
