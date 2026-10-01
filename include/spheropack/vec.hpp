/// @file vec.hpp
/// @brief Small fixed-size vector helpers.
#pragma once

#include <array>
#include <cstddef>

namespace spheropack {

/// @brief Fixed-size vector of `D` doubles.
/// @tparam D dimension (2 or 3)
template <int D>
using Vec = std::array<double, D>;

/// @brief Euclidean inner product.
/// @tparam D dimension
/// @return \f$\sum_k a_k b_k\f$
template <int D>
inline double dot(const Vec<D>& a, const Vec<D>& b) noexcept {
  double s = 0.0;
  for (int k = 0; k < D; ++k) s += a[k] * b[k];
  return s;
}

/// @brief Squared Euclidean norm.
/// @tparam D dimension
/// @return \f$\sum_k a_k^2\f$
template <int D>
inline double norm2(const Vec<D>& a) noexcept {
  return dot<D>(a, a);
}

}  // namespace spheropack
