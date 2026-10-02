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

/// @brief Whether a straight relative path has any uphill part for `pot` (a cheap
/// necessary condition for reflection_time() to be finite).
/// @param r2 \f$|\mathbf r_0|^2\f$
/// @param b \f$\mathbf r_0 \cdot \mathbf u\f$
/// @param u2 \f$|\mathbf u|^2\f$
/// @param pot the pair potential
inline bool may_reflect(double r2, double b, double u2, const PairPotential& pot) noexcept {
  if (!(u2 > 0.0)) return false;
  const double rc2 = pot.cutoff * pot.cutoff, rm2 = pot.r_min() * pot.r_min();
  const bool attractive = pot.r_min() < pot.cutoff;
  if (b >= 0.0) return attractive && r2 < rc2;  // outward: attractive branch only
  const double close2 = r2 - b * b / u2;
  if (close2 < rm2) return true;         // reaches the repulsive branch
  return attractive && close2 < rc2;     // outward leg after the closest approach
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
    pair_ev_.resize(p_.size());
    pair_t_.assign(p_.size(), never);
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
    stats_.interrupted = false;
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

  /// @brief Testing aid: re-predict with a full neighbour scan after every cell crossing
  /// instead of scanning only the new cell layer. The trajectory must be the same.
  void set_full_rescan(bool on) noexcept { full_rescan_ = on; }

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
    // Cells of half the cutoff with a 5^D stencil scan 15.6 r_c^3 instead of 27 r_c^3 per
    // full prediction in 3D. The cap on the number of cells only matters for very dilute
    // systems.
    grid_.build(box_, 0.5 * pot_.cutoff, 8 * p_.size() + 125, 2);
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
    // Two SplitMix64 finalisations of the key combined with distinct odd multipliers.
    std::uint64_t img = 0;
    for (int k = 0; k < D; ++k)  // image as seen from the lower index, 21 bits per axis
      img = (img << 21) ^ (static_cast<std::uint64_t>((swap ? -image[k] : image[k]) + (1 << 20)) & 0x1FFFFFULL);
    std::uint64_t h = mix64(seed_ ^ (static_cast<std::uint64_t>(a) * 0x9e3779b97f4a7c15ULL) ^
                            (static_cast<std::uint64_t>(b) * 0xc2b2ae3d27d4eb4fULL));
    h = mix64(h ^ (p_[a].counter * 0x165667b19e3779f9ULL) ^ (p_[b].counter * 0xd6e8feb86659fd93ULL) ^
              (img * 0xff51afd7ed558ccdULL));
    return 1.0 - to_unit_interval(h);
  }

  double predict(std::uint32_t i) {
    Event best;
    double t_best = never;
    grid_.for_each_neighbor(grid_.cell_of(i), [&](std::uint32_t j, const std::array<int, D>& shift) {
      consider_pair(i, j, shift, best, t_best);
    });
    pair_ev_[i] = best;
    pair_t_[i] = t_best;
    return combine_with_crossing(i);
  }

  // After particle i crossed a cell face its straight path is unchanged: the stored pair
  // event stays valid unless the partner changed direction, and only the cell layer
  // that became adjacent needs scanning.
  double predict_after_crossing(std::uint32_t i, int axis, int dir) {
    const Event& old = pair_ev_[i];
    const bool stale = old.type == EventType::pair && p_[old.partner].counter != old.partner_counter;
    if (full_rescan_ || stale || grid_.n(axis) < 2 * grid_.stencil() + 1) return predict(i);
    Event best = old;
    double t_best = pair_t_[i];
    grid_.for_each_neighbor_layer(grid_.cell_of(i), axis, dir, [&](std::uint32_t j, const std::array<int, D>& shift) {
      consider_pair(i, j, shift, best, t_best);
    });
    pair_ev_[i] = best;
    pair_t_[i] = t_best;
    return combine_with_crossing(i);
  }

  void consider_pair(std::uint32_t i, std::uint32_t j, const std::array<int, D>& shift, Event& best,
                     double& t_best) const noexcept {
    if (j == i) return;
    const Particle& pi = p_[i];
    const Particle& pj = p_[j];
    const double t0 = std::max(pi.t_seg, pj.t_seg);  // start of the pair's straight segment
    Vec<D> r, u;
    for (int k = 0; k < D; ++k) {
      r[k] = pj.x[k] + pj.v[k] * (t0 - pj.t) + shift[k] * box_.L[k] - pi.x[k] - pi.v[k] * (t0 - pi.t);
      u[k] = pj.v[k] - pi.v[k];
    }
    const double r2 = norm2<D>(r), b = dot<D>(r, u), u2 = norm2<D>(u);
    if (!may_reflect(r2, b, u2, pot_)) return;  // no uphill part on this segment: skip the random number
    std::array<std::int32_t, D> image;
    for (int k = 0; k < D; ++k) image[k] = shift[k] - pj.img[k] + pi.img[k];
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
  }

  // Next cell crossing of particle i, combined with its stored pair event into ev_[i].
  double combine_with_crossing(std::uint32_t i) {
    const Particle& pi = p_[i];
    const auto c = grid_.coords(grid_.cell_of(i));
    Event cross;
    double t_cross = never;
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
      if (t < t_cross) {
        t_cross = t;
        cross.type = EventType::cross;
        cross.axis = static_cast<std::int8_t>(k);
        cross.dir = dir;
      }
    }
    if (t_cross < pair_t_[i]) {
      ev_[i] = cross;
      return t_cross;
    }
    ev_[i] = pair_ev_[i];
    return pair_t_[i];
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
      heap_.update(i, predict_after_crossing(i, k, e.dir));
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
  std::vector<Event> ev_;          ///< next event of each particle
  std::vector<Event> pair_ev_;     ///< best pair event of each particle from its last scan
  std::vector<double> pair_t_;     ///< time of pair_ev_
  EventHeap heap_;
  CellGrid<D> grid_;
  RejectionFreeStats stats_;
  double now_ = 0.0;
  std::uint64_t since_sync_ = 0;
  bool ready_ = false;
  bool full_rescan_ = false;
};

/// Run statistics of an event-chain simulation.
struct EventChainStats {
  double displacement = 0.0;  ///< total displacement of all chains
  std::uint64_t n_lifts = 0;  ///< number of lifts (the motion passed to another particle)
  std::uint64_t n_chains = 0;  ///< number of chains
  double wall_time = 0.0;      ///< wall-clock seconds spent in run()
  bool interrupted = false;    ///< the last run() was stopped by the interrupt callback
};

/// @brief Straight event-chain variant of the rejection-free method (Peters and de With
/// 2012, section "straight event-chain collision"; Bernard, Krauth and Wilson 2009 for
/// hard spheres).
///
/// One particle moves at a time, along a coordinate axis. For every pair with the moving
/// particle the uphill part of the pair potential along the path is accumulated; at a
/// reflection the moving particle stops and the partner continues with the same
/// displacement direction (a lift). A chain ends after a total displacement
/// `chain_length`; the next chain starts at a random particle. In the irreversible mode
/// (the default) the direction cycles through +x, +y, +z; this breaks detailed balance
/// but satisfies global balance, so the canonical distribution is sampled. In the
/// reversible mode each chain gets a random axis and sign.
/// @tparam D dimension (2 or 3)
template <int D>
class EventChainMC {
 public:
  /// @brief Set up particles at the given positions.
  /// @param box periodic box (all axes periodic), edges at least twice the cutoff
  /// @param positions initial positions
  /// @param potential pair potential (prepared here)
  /// @param kT temperature in energy units
  /// @param seed random seed
  /// @param irreversible cycle the directions +x, +y, +z (true) or draw random axes and signs
  EventChainMC(const Box<D>& box, const std::vector<Vec<D>>& positions, PairPotential potential, double kT,
               std::uint64_t seed, bool irreversible = true)
      : box_(box), pot_(potential), kT_(kT), rng_(seed), irreversible_(irreversible) {
    box_.validate();
    for (int k = 0; k < D; ++k)
      if (!box_.periodic[k]) throw std::invalid_argument("event-chain MC needs a fully periodic box");
    if (box_.ball.active) throw std::invalid_argument("event-chain MC needs a box without curved wall");
    if (!(kT > 0.0)) throw std::invalid_argument("kT must be positive");
    if (positions.empty()) throw std::invalid_argument("at least one particle is required");
    pot_.prepare();
    for (int k = 0; k < D; ++k)
      if (box_.L[k] < 2.0 * pot_.cutoff) throw std::invalid_argument("box edges must be at least twice the cutoff");
    x_ = positions;
    for (auto& xi : x_) wrap(xi);
    grid_.build(box_, 0.5 * pot_.cutoff, 8 * x_.size() + 125, 2);
    grid_.clear_particles(x_.size());
    for (std::uint32_t i = 0; i < x_.size(); ++i) grid_.insert(i, grid_.flat(grid_.coords_of(x_[i])));
  }

  /// @brief Run chains until the total displacement reaches `displacement`.
  /// @param displacement total displacement of all chains together
  /// @param chain_length displacement per chain
  /// @param interrupted polled regularly; returning true stops the run early
  /// @return the accumulated statistics
  const EventChainStats& run(double displacement, double chain_length, const std::function<bool()>& interrupted = {}) {
    if (!(displacement >= 0.0) || !std::isfinite(displacement) || !(chain_length > 0.0))
      throw std::invalid_argument("displacement must be non-negative and chain_length positive");
    stats_.interrupted = false;
    const auto t_start = std::chrono::steady_clock::now();
    double left = displacement;
    std::uint64_t polled_at = stats_.n_lifts + stats_.n_chains;
    while (left > 0.0) {
      // Poll the interrupt about every 65536 lifts or chains, whichever comes first.
      if (stats_.n_lifts + stats_.n_chains - polled_at > 0xffff) {
        polled_at = stats_.n_lifts + stats_.n_chains;
        if (interrupted && interrupted()) {
          stats_.interrupted = true;
          break;
        }
      }
      const double ell = std::min(chain_length, left);
      chain(ell);
      left -= ell;
      stats_.displacement += ell;
      ++stats_.n_chains;
    }
    stats_.wall_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    return stats_;
  }

  /// @return current positions, wrapped into the box
  std::vector<Vec<D>> positions() const {
    std::vector<Vec<D>> x = x_;
    for (auto& xi : x) wrap(xi);  // a particle can sit exactly on the upper face after a crossing
    return x;
  }

  /// @return total potential energy
  double potential_energy() const {
    double u = 0.0;
    for_each_pair<D>(x_, box_, pot_.cutoff, [&](std::uint32_t, std::uint32_t, const Vec<D>& r) {
      u += pot_.value(std::sqrt(norm2<D>(r)));
    });
    return u;
  }

  /// @return the accumulated statistics
  const EventChainStats& stats() const noexcept { return stats_; }

 private:
  void wrap(Vec<D>& x) const noexcept {
    for (int k = 0; k < D; ++k) {
      x[k] -= std::floor(x[k] / box_.L[k]) * box_.L[k];
      if (x[k] >= box_.L[k]) x[k] -= box_.L[k];
    }
  }

  // One chain of total displacement `ell`.
  void chain(double ell) {
    int axis;
    int sign = 1;
    if (irreversible_) {
      axis = next_axis_;
      next_axis_ = (next_axis_ + 1) % D;
    } else {
      axis = static_cast<int>(rng_.uniform() * D);
      sign = rng_.uniform() < 0.5 ? -1 : 1;
    }
    auto i = static_cast<std::uint32_t>(rng_.uniform() * static_cast<double>(x_.size()));
    while (ell > 0.0) {
      // Distance to the next cell face along the move, which bounds the step so that the
      // neighbour scan stays valid.
      const auto c = grid_.coords(grid_.cell_of(i));
      const double face = sign > 0 ? (c[axis] + 1) * grid_.width(axis) : c[axis] * grid_.width(axis);
      const double s_face = std::max(0.0, sign * (face - x_[i][axis]));
      double s_best = std::min(ell, s_face);
      std::uint32_t lift = std::numeric_limits<std::uint32_t>::max();
      grid_.for_each_neighbor(grid_.cell_of(i), [&](std::uint32_t j, const std::array<int, D>& shift) {
        if (j == i) return;
        Vec<D> r;
        for (int k = 0; k < D; ++k) r[k] = x_[j][k] + shift[k] * box_.L[k] - x_[i][k];
        const double r2 = norm2<D>(r);
        const double b = -sign * r[axis];  // relative velocity of j seen from i: -e
        if (!may_reflect(r2, b, 1.0, pot_)) return;
        const double tau = reflection_time(r2, b, 1.0, kT_ * rng_.exponential(), pot_);
        if (tau < s_best) {
          s_best = tau;
          lift = j;
        }
      });
      x_[i][axis] += sign * s_best;
      ell -= s_best;
      if (lift != std::numeric_limits<std::uint32_t>::max()) {
        i = lift;
        ++stats_.n_lifts;
        continue;
      }
      if (s_best == s_face && ell > 0.0) {  // into the next cell, wrapping at the box face
        auto nc = c;
        nc[axis] += sign;
        if (nc[axis] < 0) {
          nc[axis] = grid_.n(axis) - 1;
          x_[i][axis] += box_.L[axis];
        } else if (nc[axis] >= grid_.n(axis)) {
          nc[axis] = 0;
          x_[i][axis] -= box_.L[axis];
        }
        grid_.move(i, grid_.flat(nc));
      } else {
        wrap(x_[i]);
        const auto cell = grid_.flat(grid_.coords_of(x_[i]));
        if (cell != grid_.cell_of(i)) grid_.move(i, cell);
      }
    }
  }

  Box<D> box_;
  PairPotential pot_;
  double kT_;
  Rng rng_;
  bool irreversible_;
  int next_axis_ = 0;
  std::vector<Vec<D>> x_;
  CellGrid<D> grid_;
  EventChainStats stats_;
};

}  // namespace spheropack
