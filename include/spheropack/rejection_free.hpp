/// @file rejection_free.hpp
/// @brief Rejection-free, event-driven Monte Carlo sampling for pair potentials.
///
/// Implements the method of E.A.J.F. Peters and G. de With, Phys. Rev. E 85, 026703
/// (2012). All particles move along straight lines with constant "velocities" (move
/// directions, not physical momenta). For every pair potential separately, the uphill
/// part of the potential along the current straight path is accumulated; the pair
/// reflects (an elastic collision of equal masses along the line of centres) when the
/// accumulated increase reaches \f$-kT \ln u\f$ with \f$u\f$ uniform on (0, 1]. The
/// positions sampled at equidistant times follow the canonical distribution
/// \f$\exp(-U/kT)\f$.
///
/// Each particle keeps one predicted event (cell crossing or reflection) in an indexed
/// heap, as in the packing engine. Because a pair can be predicted by either particle,
/// and again after cell crossings, the random number of a pair is a counter-based hash
/// of (seed, i, j, c_i, c_j), with c the number of velocity changes of a particle, and
/// the uphill integral starts at the beginning of the pair's current straight segment.
/// A prediction therefore depends only on the state, and duplicate predictions agree.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "box.hpp"
#include "cell_grid.hpp"
#include "event_queue.hpp"
#include "neighbors.hpp"
#include "predict.hpp"
#include "rng.hpp"
#include "vec.hpp"

namespace spheropack {

/// @brief Isotropic pair potential with at most one minimum, zero at and beyond the cutoff.
///
/// U decreases on \f$(0, r_m]\f$ and increases on \f$[r_m, r_c]\f$ to \f$U(r_c) = 0\f$.
/// Purely repulsive potentials have \f$r_m = r_c\f$. Hard spheres have an infinite
/// step at \f$\sigma\f$.
struct PairPotential {
  /// Potential families.
  enum class Kind {
    soft,           ///< \f$U = \tfrac{\epsilon}{\alpha}(1 - r/\sigma)^\alpha\f$ for \f$r < \sigma\f$ (DPD: \f$\alpha = 2\f$)
    lennard_jones,  ///< \f$4\epsilon[(\sigma/r)^{12} - (\sigma/r)^6] - U(r_c)\f$ for \f$r < r_c\f$
    hard,           ///< infinite for \f$r < \sigma\f$, zero otherwise
  };
  Kind kind = Kind::soft;  ///< potential family
  double epsilon = 1.0;  ///< energy scale
  double sigma = 1.0;    ///< length scale (soft and hard: range / diameter)
  double cutoff = 1.0;   ///< \f$r_c\f$ (soft and hard: equal to sigma)
  double alpha = 2.0;    ///< exponent of the soft potential

  /// @brief Check parameters and derive the shift and the minimum.
  void prepare() {
    if (!(epsilon > 0.0) || !(sigma > 0.0)) throw std::invalid_argument("epsilon and sigma must be positive");
    if (kind == Kind::lennard_jones) {
      if (!(cutoff > 0.0)) throw std::invalid_argument("cutoff must be positive");
      shift_ = lj_raw(cutoff);
      r_min_ = std::min(std::pow(2.0, 1.0 / 6.0) * sigma, cutoff);
    } else {
      if (kind == Kind::soft && !(alpha > 0.0)) throw std::invalid_argument("alpha must be positive");
      cutoff = sigma;
      shift_ = 0.0;
      r_min_ = sigma;
    }
    u_min_ = value(r_min_);
  }

  /// @return radius of the minimum \f$r_m\f$
  double r_min() const noexcept { return r_min_; }
  /// @return \f$U(r_m)\f$, the lowest value (0 for purely repulsive potentials)
  double u_min() const noexcept { return u_min_; }

  /// @return \f$U(r)\f$
  double value(double r) const noexcept {
    if (r >= cutoff) return 0.0;
    switch (kind) {
      case Kind::soft: {
        const double x = 1.0 - r / sigma;
        return epsilon / alpha * (alpha == 2.0 ? x * x : std::pow(x, alpha));
      }
      case Kind::lennard_jones: return lj_raw(r) - shift_;
      case Kind::hard: return never;
    }
    return 0.0;
  }

  /// @brief Radius on the repulsive branch \f$r \le r_m\f$ where \f$U(r) = u\f$, for \f$u \ge U(r_m)\f$.
  double inner_radius(double u) const noexcept {
    switch (kind) {
      case Kind::soft: return sigma * (1.0 - std::pow(alpha * u / epsilon, 1.0 / alpha));
      case Kind::lennard_jones: {
        const double x = 0.5 + std::sqrt(std::max(0.0, 0.25 + (u + shift_) / (4.0 * epsilon)));  // (sigma/r)^6
        return sigma * std::pow(x, -1.0 / 6.0);
      }
      case Kind::hard: return sigma;
    }
    return sigma;
  }

  /// @brief Radius on the attractive branch \f$r_m \le r \le r_c\f$ where \f$U(r) = u\f$, for \f$U(r_m) \le u \le 0\f$.
  double outer_radius(double u) const noexcept {
    if (kind != Kind::lennard_jones) return cutoff;
    const double x = 0.5 - std::sqrt(std::max(0.0, 0.25 + (u + shift_) / (4.0 * epsilon)));
    return x > 0.0 ? sigma * std::pow(x, -1.0 / 6.0) : cutoff;
  }

 private:
  double lj_raw(double r) const noexcept {
    if (!(r > 0.0)) return never;
    const double s2 = (sigma * sigma) / (r * r);
    const double s6 = s2 * s2 * s2;
    return 4.0 * epsilon * (s6 * s6 - s6);
  }
  double shift_ = 0.0;
  double r_min_ = 1.0;
  double u_min_ = 0.0;
};

/// @brief Time \f$\tau \ge 0\f$ along a straight relative path at which a pair reflects.
///
/// The pair separation is \f$\mathbf r(\tau) = \mathbf r_0 + \mathbf u\,\tau\f$. Along the
/// inward leg (up to the closest approach) the potential rises inside \f$r_m\f$, along
/// the outward leg it rises between \f$r_m\f$ and \f$r_c\f$. The reflection happens where
/// the accumulated rise equals `du`.
/// @param r2 \f$|\mathbf r_0|^2\f$
/// @param b \f$\mathbf r_0 \cdot \mathbf u\f$
/// @param u2 \f$|\mathbf u|^2\f$
/// @param du drawn energy, \f$-kT \ln u\f$
/// @param pot the pair potential
/// @return \f$\tau\f$, or `never` if the pair does not reflect before it separates beyond the cutoff
inline double reflection_time(double r2, double b, double u2, double du, const PairPotential& pot) noexcept {
  if (!(u2 > 0.0)) return never;
  const double rc = pot.cutoff, rm = pot.r_min();
  const double r0 = std::sqrt(r2);
  // Closest approach along the path (only ahead if moving inward).
  const double r_close2 = std::max(0.0, r2 - b * b / u2);
  const double r_close = std::sqrt(r_close2);
  auto time_at = [&](double r, bool outward) {  // roots of |r0 + u t|^2 = r^2
    const double disc = std::max(0.0, b * b - u2 * (r2 - r * r));
    return outward ? (-b + std::sqrt(disc)) / u2 : (-b - std::sqrt(disc)) / u2;
  };
  double r_start_out = r0;  // where the outward leg starts
  if (b < 0.0) {
    r_start_out = r_close;
    if (r_close < rm) {  // inward leg climbs the repulsive branch
      const double u_from = pot.value(std::min(r0, rm));
      const double rise = pot.value(r_close) - u_from;
      // !(du >= rise) also catches rise = NaN (inf - inf: a head-on approach to r = 0, or
      // an overlapping hard-sphere pair), which must reflect rather than pass through.
      if (!(du >= rise)) return std::max(0.0, time_at(pot.inner_radius(u_from + du), false));
      du -= rise;
    }
  }
  // Outward leg climbs the attractive branch from max(start, r_m) to the cutoff.
  if (rm >= rc || r_start_out >= rc) return never;
  const double u_from = pot.value(std::max(r_start_out, rm));
  if (du >= -u_from) return never;  // U(r_c) = 0 is reached without reflection
  return std::max(0.0, time_at(pot.outer_radius(u_from + du), true));
}

/// Run statistics of a rejection-free simulation.
struct RejectionFreeStats {
  double time = 0.0;               ///< simulation time ("contour length" of the moves)
  std::uint64_t n_reflections = 0;  ///< number of pair reflections
  std::uint64_t n_events = 0;       ///< processed events (reflections, cell crossings, stale predictions)
  double wall_time = 0.0;           ///< wall-clock seconds spent in run()
  bool interrupted = false;         ///< the last run() was stopped by the interrupt callback
};

/// @brief Rejection-free event-driven Monte Carlo in a periodic box.
/// @tparam D dimension (2 or 3)
template <int D>
class RejectionFreeMC {
 public:
  /// @brief Set up particles at the given positions with Gaussian move velocities.
  /// @param box periodic box (all axes periodic)
  /// @param positions initial positions inside the box
  /// @param potential pair potential (prepared here)
  /// @param kT temperature in energy units
  /// @param seed random seed (velocities and reflection energies)
  RejectionFreeMC(const Box<D>& box, const std::vector<Vec<D>>& positions, PairPotential potential, double kT,
                  std::uint64_t seed)
      : box_(box), pot_(potential), kT_(kT), seed_(mix64(seed ^ 0x5bd1e995ULL)), rng_(seed) {
    box_.validate();
    for (int k = 0; k < D; ++k)
      if (!box_.periodic[k]) throw std::invalid_argument("rejection-free MC needs a fully periodic box");
    if (box_.ball.active) throw std::invalid_argument("rejection-free MC needs a box without curved wall");
    if (!(kT > 0.0)) throw std::invalid_argument("kT must be positive");
    pot_.prepare();
    for (int k = 0; k < D; ++k)
      if (box_.L[k] < 2.0 * pot_.cutoff) throw std::invalid_argument("box edges must be at least twice the cutoff");
    if (positions.empty()) throw std::invalid_argument("at least one particle is required");
    p_.resize(positions.size());
    for (std::size_t i = 0; i < p_.size(); ++i) {
      p_[i].x = positions[i];
      for (auto& v : p_[i].v) v = rng_.normal();
    }
    ev_.resize(p_.size());
    wrap_positions();
  }

  /// @brief Replace the move velocities (for example to redraw them).
  void set_velocities(const std::vector<Vec<D>>& v) {
    if (v.size() != p_.size()) throw std::invalid_argument("velocities: wrong number of particles");
    sync();
    for (std::size_t i = 0; i < p_.size(); ++i) {
      p_[i].v = v[i];
      ++p_[i].counter;
      p_[i].t_seg = 0.0;
    }
    ready_ = false;
  }

  /// @brief Redraw all move velocities from a standard normal distribution.
  void redraw_velocities() {
    std::vector<Vec<D>> v(p_.size());
    for (auto& vi : v)
      for (auto& c : vi) c = rng_.normal();
    set_velocities(v);
  }

  /// @brief Advance the simulation by `duration` (time = contour length along the moves).
  /// @param duration simulation time to advance
  /// @param interrupted polled regularly; returning true stops the run early
  /// @return the accumulated statistics
  const RejectionFreeStats& run(double duration, const std::function<bool()>& interrupted = {}) {
    if (!(duration >= 0.0) || !std::isfinite(duration))
      throw std::invalid_argument("duration must be finite and non-negative");
    const auto t_start = std::chrono::steady_clock::now();
    if (!ready_) prepare_events();
    double t_end = now_ + duration;
    const std::uint64_t sync_every = std::max<std::uint64_t>(1, 20 * p_.size());
    for (std::uint64_t it = 1;; ++it) {
      if ((it & 0xffff) == 0 && interrupted && interrupted()) {
        stats_.interrupted = true;
        t_end = now_;
        break;
      }
      const std::uint32_t i = heap_.top();
      const double te = heap_.top_key();
      if (te >= t_end) break;
      now_ = te;
      ++stats_.n_events;
      handle(i);
      if (since_sync_ >= sync_every) {  // new time origin keeps the times small
        const double remaining = t_end - now_;
        sync();
        prepare_events();
        t_end = remaining;
      }
    }
    now_ = t_end;
    sync();
    stats_.wall_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    return stats_;
  }

  /// @return simulation time since the start
  double time() const noexcept { return stats_.time + now_; }

  /// @return positions at the current time, wrapped into the box
  std::vector<Vec<D>> positions() {
    sync();
    std::vector<Vec<D>> x(p_.size());
    for (std::size_t i = 0; i < p_.size(); ++i) x[i] = p_[i].x;
    return x;
  }

  /// @return current move velocities
  std::vector<Vec<D>> velocities() const {
    std::vector<Vec<D>> v(p_.size());
    for (std::size_t i = 0; i < p_.size(); ++i) v[i] = p_[i].v;
    return v;
  }

  /// @return total potential energy at the current time
  double potential_energy() {
    const auto x = positions();
    double u = 0.0;
    for_each_pair<D>(x, box_, pot_.cutoff, [&](std::uint32_t, std::uint32_t, const Vec<D>& r) {
      u += pot_.value(std::sqrt(norm2<D>(r)));
    });
    return u;
  }

  /// @return the accumulated run statistics
  const RejectionFreeStats& stats() const noexcept { return stats_; }
  /// @return the number of particles
  std::size_t size() const noexcept { return p_.size(); }

 private:
  struct Particle {
    Vec<D> x{};                         ///< position at time t
    Vec<D> v{};                         ///< move velocity
    std::array<std::int32_t, D> img{};  ///< periodic image counters
    double t = 0.0;                     ///< time of the stored position
    double t_seg = 0.0;                 ///< time of the last velocity change
    std::uint64_t counter = 0;          ///< number of velocity changes
  };
  enum class EventType : std::uint8_t { none, cross, pair };
  struct Event {
    EventType type = EventType::none;
    std::int8_t axis = 0, dir = 0;
    std::uint32_t partner = 0;
    std::uint64_t partner_counter = 0;
    std::array<std::int32_t, D> image{};
  };

  void advance(Particle& q) const noexcept {
    const double dt = now_ - q.t;
    for (int k = 0; k < D; ++k) q.x[k] += q.v[k] * dt;
    q.t = now_;
  }

  void wrap_positions() noexcept {
    for (auto& q : p_)
      for (int k = 0; k < D; ++k) {
        const double shift = std::floor(q.x[k] / box_.L[k]);
        if (shift != 0.0) {
          q.x[k] -= shift * box_.L[k];
          q.img[k] += static_cast<std::int32_t>(shift);
        }
        if (q.x[k] >= box_.L[k]) {
          q.x[k] -= box_.L[k];
          ++q.img[k];
        }
      }
  }

  // Brings all particles to the current time and makes it the new time origin.
  void sync() {
    for (auto& q : p_) advance(q);
    stats_.time += now_;
    for (auto& q : p_) {
      q.t = 0.0;
      q.t_seg -= now_;
    }
    now_ = 0.0;
    since_sync_ = 0;
    wrap_positions();
    ready_ = false;
  }

  void prepare_events() {
    grid_.build(box_, pot_.cutoff, 2 * p_.size() + 27);
    grid_.clear_particles(p_.size());
    for (std::uint32_t i = 0; i < p_.size(); ++i) grid_.insert(i, grid_.flat(grid_.coords_of(p_[i].x)));
    std::vector<double> keys(p_.size());
    for (std::uint32_t i = 0; i < p_.size(); ++i) keys[i] = predict(i);
    heap_.reset(static_cast<std::uint32_t>(p_.size()));
    heap_.assign(keys);
    ready_ = true;
  }

  // Uniform in (0, 1] for the current straight segment of the pair term (i, j, image).
  // `image` is the periodic image of j seen from i (constant in time); each image is a
  // separate potential term and gets its own random number.
  double pair_uniform(std::uint32_t i, std::uint32_t j, const std::array<std::int32_t, D>& image) const noexcept {
    const bool swap = j < i;
    const std::uint32_t a = swap ? j : i, b = swap ? i : j;
    std::uint64_t h = mix64(seed_ ^ a);
    h = mix64(h ^ (static_cast<std::uint64_t>(b) << 1));
    h = mix64(h ^ p_[a].counter);
    h = mix64(h ^ (p_[b].counter * 0x9e3779b97f4a7c15ULL));
    for (int k = 0; k < D; ++k)  // image as seen from the lower index
      h = mix64(h ^ static_cast<std::uint64_t>(static_cast<std::int64_t>(swap ? -image[k] : image[k]) + 1024));
    return 1.0 - to_unit_interval(h);
  }

  // Whether the straight segment has any uphill part (a cheap necessary condition).
  bool may_reflect(double r2, double b, double u2) const noexcept {
    if (!(u2 > 0.0)) return false;
    const double rc2 = pot_.cutoff * pot_.cutoff, rm2 = pot_.r_min() * pot_.r_min();
    if (b >= 0.0) return pot_.r_min() < pot_.cutoff && r2 < rc2;  // outward: attractive branch only
    const double close2 = r2 - b * b / u2;
    if (close2 < rm2) return true;                     // reaches the repulsive branch
    return pot_.r_min() < pot_.cutoff && close2 < rc2;  // outward leg after closest approach
  }

  double predict(std::uint32_t i) {
    const Particle& pi = p_[i];
    Event best;
    double t_best = never;
    // Cell crossing (positions advanced to now for the cell bookkeeping).
    const auto c = grid_.coords(grid_.cell_of(i));
    for (int k = 0; k < D; ++k) {
      const double v = pi.v[k];
      const double xk = pi.x[k] + v * (now_ - pi.t);
      double t = never;
      std::int8_t dir = 0;
      if (v > 0.0) {
        t = now_ + std::max(0.0, ((c[k] + 1) * grid_.width(k) - xk) / v);
        dir = 1;
      } else if (v < 0.0) {
        t = now_ + std::max(0.0, (c[k] * grid_.width(k) - xk) / v);
        dir = -1;
      }
      if (t < t_best) {
        t_best = t;
        best.type = EventType::cross;
        best.axis = static_cast<std::int8_t>(k);
        best.dir = dir;
      }
    }
    grid_.for_each_neighbor(grid_.cell_of(i), [&](std::uint32_t j, const std::array<int, D>& shift) {
      if (j == i) return;
      const Particle& pj = p_[j];
      const double t0 = std::max(pi.t_seg, pj.t_seg);  // start of the pair's straight segment
      Vec<D> r, u;
      for (int k = 0; k < D; ++k) {
        r[k] = pj.x[k] + pj.v[k] * (t0 - pj.t) + shift[k] * box_.L[k] - pi.x[k] - pi.v[k] * (t0 - pi.t);
        u[k] = pj.v[k] - pi.v[k];
      }
      std::array<std::int32_t, D> image;
      for (int k = 0; k < D; ++k) image[k] = shift[k] - pj.img[k] + pi.img[k];
      const double r2 = norm2<D>(r), b = dot<D>(r, u), u2 = norm2<D>(u);
      if (!may_reflect(r2, b, u2)) return;  // no uphill part on this segment: skip the random number
      const double du = -kT_ * std::log(pair_uniform(i, j, image));
      const double tau = reflection_time(r2, b, u2, du, pot_);
      if (tau == never) return;
      const double t = std::max(now_, t0 + tau);
      if (t < t_best) {
        t_best = t;
        best.type = EventType::pair;
        best.partner = j;
        best.partner_counter = pj.counter;
        best.image = image;
      }
    });
    ev_[i] = best;
    return t_best;
  }

  void handle(std::uint32_t i) {
    const Event e = ev_[i];
    Particle& pi = p_[i];
    if (e.type == EventType::cross) {
      advance(pi);
      auto c = grid_.coords(grid_.cell_of(i));
      const int k = e.axis;
      c[k] += e.dir;
      if (c[k] < 0) {
        c[k] = grid_.n(k) - 1;
        pi.x[k] += box_.L[k];
        --pi.img[k];
      } else if (c[k] >= grid_.n(k)) {
        c[k] = 0;
        pi.x[k] -= box_.L[k];
        ++pi.img[k];
      }
      grid_.move(i, grid_.flat(c));
      heap_.update(i, predict(i));
      return;
    }
    if (e.type != EventType::pair) {
      heap_.update(i, never);
      return;
    }
    const std::uint32_t j = e.partner;
    Particle& pj = p_[j];
    if (pj.counter != e.partner_counter) {  // partner changed direction: stale
      heap_.update(i, predict(i));
      return;
    }
    advance(pi);
    advance(pj);
    Vec<D> r;
    for (int k = 0; k < D; ++k)
      r[k] = pj.x[k] - pi.x[k] + (e.image[k] + pj.img[k] - pi.img[k]) * box_.L[k];
    const double dist = std::sqrt(norm2<D>(r));
    double un = 0.0;
    for (int k = 0; k < D; ++k) un += (pj.v[k] - pi.v[k]) * r[k] / dist;
    // Elastic collision of equal masses: exchange the normal velocity components.
    for (int k = 0; k < D; ++k) {
      const double dv = un * r[k] / dist;
      pi.v[k] += dv;
      pj.v[k] -= dv;
    }
    ++pi.counter;
    ++pj.counter;
    pi.t_seg = pj.t_seg = now_;
    ++stats_.n_reflections;
    ++since_sync_;
    heap_.update(i, predict(i));
    heap_.update(j, predict(j));
  }

  Box<D> box_;
  PairPotential pot_;
  double kT_;
  std::uint64_t seed_;
  Rng rng_;
  std::vector<Particle> p_;
  std::vector<Event> ev_;
  EventHeap heap_;
  CellGrid<D> grid_;
  RejectionFreeStats stats_;
  double now_ = 0.0;
  std::uint64_t since_sync_ = 0;
  bool ready_ = false;
};

}  // namespace spheropack
