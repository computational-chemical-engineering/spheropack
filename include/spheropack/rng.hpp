/// @file rng.hpp
/// @brief Portable random numbers: xoshiro256++ seeded through SplitMix64.
///
/// The standard library distributions are not portable between implementations, so
/// the uniform, normal and exponential variates are defined here. For a given seed
/// the integer stream is identical on every platform; the floating point variates
/// differ at most in the last bit where libm (log, cos, sin) differs.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace spheropack {

/// @brief One step of SplitMix64.
/// @param state generator state, advanced by the golden-ratio increment
/// @return a well-mixed 64-bit value
constexpr std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/// @brief Stateless 64-bit mix of a key, used for counter-based random numbers.
/// @param key input value (for example a counter or an index)
/// @return one SplitMix64 output for the state `key`
constexpr std::uint64_t mix64(std::uint64_t key) noexcept {
  std::uint64_t state = key;
  return splitmix64(state);
}

/// @brief Uniform double in \f$[0, 1)\f$ from the 53 high bits of a 64-bit integer.
/// @param bits raw 64-bit random value
/// @return `(bits >> 11) * 2^-53`, exactly representable
constexpr double to_unit_interval(std::uint64_t bits) noexcept {
  return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

/// @brief xoshiro256++ pseudo-random generator (Blackman and Vigna, 2019).
///
/// Satisfies the interface of a C++ uniform random bit generator (`result_type`,
/// `min`, `max`, `operator()`), and provides portable uniform, exponential and
/// normal variates. The state is 256 bits, seeded through SplitMix64.
class Rng {
 public:
  /// Type of the raw integer output.
  using result_type = std::uint64_t;

  /// @brief Construct and seed the generator.
  /// @param seed any 64-bit seed; the four state words are drawn from SplitMix64
  explicit Rng(std::uint64_t seed = 0) noexcept { reseed(seed); }

  /// @brief Reset the stream to the one defined by `seed` and drop any cached normal variate.
  /// @param seed any 64-bit seed
  void reseed(std::uint64_t seed) noexcept {
    std::uint64_t sm = seed;
    for (auto& s : s_) s = splitmix64(sm);
    has_cached_normal_ = false;
  }

  /// Smallest raw output (0).
  static constexpr result_type min() noexcept { return 0; }
  /// Largest raw output (all bits set).
  static constexpr result_type max() noexcept { return ~result_type{0}; }

  /// @brief Next raw 64-bit value of the xoshiro256++ stream.
  /// @return uniformly distributed 64-bit integer
  result_type operator()() noexcept {
    const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  /// @return uniform variate in \f$[0, 1)\f$
  double uniform() noexcept { return to_unit_interval((*this)()); }

  /// @return uniform variate in \f$(0, 1]\f$, safe as argument of `log`
  double uniform_open0() noexcept { return 1.0 - uniform(); }

  /// @return standard exponential variate (rate 1), \f$-\ln u\f$ with \f$u \in (0,1]\f$
  double exponential() noexcept { return -std::log(uniform_open0()); }

  /// @brief Standard normal variate by the Box-Muller transform.
  ///
  /// Each transform yields two independent values; the second is cached and returned
  /// by the next call.
  /// @return normal variate with zero mean and unit variance
  double normal() noexcept {
    if (has_cached_normal_) {
      has_cached_normal_ = false;
      return cached_normal_;
    }
    const double r = std::sqrt(-2.0 * std::log(uniform_open0()));
    const double phi = 2.0 * std::numbers::pi * uniform();
    cached_normal_ = r * std::sin(phi);
    has_cached_normal_ = true;
    return r * std::cos(phi);
  }

  /// @return copy of the four 64-bit state words (the cached normal is not included)
  std::array<std::uint64_t, 4> state() const noexcept { return s_; }

 private:
  /// Rotate `x` left by `k` bits, 0 < k < 64.
  static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  std::array<std::uint64_t, 4> s_{};  ///< xoshiro256++ state
  double cached_normal_ = 0.0;        ///< second Box-Muller value, valid if `has_cached_normal_`
  bool has_cached_normal_ = false;    ///< true if `cached_normal_` has not been consumed
};

}  // namespace spheropack
