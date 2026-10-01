/// @file event_queue.hpp
/// @brief Indexed binary min-heap holding one key (event time) per particle.
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace spheropack {

/// @brief Indexed binary min-heap with one key per particle.
///
/// Particle `i` always owns slot `i`; its key can be changed in O(log n). The top is
/// the particle with the smallest key (the next event). Keys equal to infinity mean
/// "no event".
class EventHeap {
 public:
  /// @brief Resize to `n` particles and set every key to infinity.
  /// @param n number of particles
  void reset(std::uint32_t n) {
    key_.assign(n, std::numeric_limits<double>::infinity());
    heap_.resize(n);
    pos_.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) heap_[i] = pos_[i] = i;
  }

  /// @brief Set all keys at once (O(n) heapify).
  /// @param keys one key per particle; must have the size given to `reset`
  void assign(const std::vector<double>& keys) {
    key_ = keys;
    for (std::uint32_t i = 0; i < key_.size(); ++i) heap_[i] = pos_[i] = i;
    for (std::size_t p = heap_.size() / 2; p-- > 0;) sift_down(p);
  }

  /// @brief Change the key of particle `i` and restore the heap order, O(log n).
  /// @param i particle index
  /// @param t new key (event time)
  void update(std::uint32_t i, double t) {
    const double old = key_[i];
    key_[i] = t;
    if (t < old) sift_up(pos_[i]);
    else sift_down(pos_[i]);
  }

  /// @return particle with the smallest key; undefined if `empty()`
  std::uint32_t top() const noexcept { return heap_.front(); }
  /// @return smallest key; undefined if `empty()`
  double top_key() const noexcept { return key_[heap_.front()]; }
  /// @param i particle index
  /// @return current key of particle `i`
  double key(std::uint32_t i) const noexcept { return key_[i]; }
  /// @return true if the heap holds no particles
  bool empty() const noexcept { return heap_.empty(); }

 private:
  /// Compare the keys at heap positions `a` and `b`.
  bool less(std::size_t a, std::size_t b) const noexcept { return key_[heap_[a]] < key_[heap_[b]]; }

  /// Swap heap positions `a` and `b` and update the position map.
  void swap_nodes(std::size_t a, std::size_t b) noexcept {
    std::swap(heap_[a], heap_[b]);
    pos_[heap_[a]] = static_cast<std::uint32_t>(a);
    pos_[heap_[b]] = static_cast<std::uint32_t>(b);
  }

  /// Move the node at heap position `p` towards the root until the order holds.
  void sift_up(std::size_t p) noexcept {
    while (p > 0) {
      const std::size_t parent = (p - 1) / 2;
      if (!less(p, parent)) break;
      swap_nodes(p, parent);
      p = parent;
    }
  }

  /// Move the node at heap position `p` towards the leaves until the order holds.
  void sift_down(std::size_t p) noexcept {
    const std::size_t n = heap_.size();
    for (;;) {
      std::size_t best = p;
      const std::size_t l = 2 * p + 1, r = l + 1;
      if (l < n && less(l, best)) best = l;
      if (r < n && less(r, best)) best = r;
      if (best == p) return;
      swap_nodes(p, best);
      p = best;
    }
  }

  std::vector<double> key_;          ///< key per particle
  std::vector<std::uint32_t> heap_;  // heap position -> particle
  std::vector<std::uint32_t> pos_;   // particle -> heap position
};

}  // namespace spheropack
