/// @file ls_packing.hpp
/// @brief Event-driven Lubachevsky-Stillinger packing of growing hard spheres (D = 2, 3).
///
/// Particle i has radius \f$a_i s(t)\f$, with \f$s(t) = s_0 + \dot s\, t\f$ between
/// synchronisations. All particles have unit mass and the temperature is kT = 1, so the
/// thermal speed is one length unit per time unit. Positions are updated lazily: particle i
/// stores its position at its own time \f$t_i\f$. Each particle has one predicted event
/// (cell crossing, wall contact or pair contact) in an indexed min-heap. A pair event is
/// valid only if the partner's collision counter is unchanged since the prediction.
///
/// Typical use: construct an LSPacking, optionally call set_positions() and
/// set_velocities(), call run(), then read the result with stats(), positions() and radii().
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "box.hpp"
#include "cell_grid.hpp"
#include "event_queue.hpp"
#include "predict.hpp"
#include "rng.hpp"
#include "vec.hpp"

namespace spheropack {

/// @brief Reason why LSPacking::run() returned.
enum class Status {
  running,      ///< run() has not finished (initial value)
  target,       ///< radii reached a_i * target_scale
  pressure,     ///< reduced pressure exceeded max_pressure (jammed or crystallised)
  jammed,       ///< jamming protocol reached jammed_pressure (median over spheres)
  stall,        ///< radii grew by less than stall_tol over stall_window collisions per particle
  collisions,   ///< max_collisions reached
  time_limit,   ///< max_time (simulation time) reached
  timeout,      ///< wall-clock timeout reached
  interrupted,  ///< the interrupt callback returned true (e.g. Ctrl-C)
  no_events,    ///< no future events (all particles at rest)
  box_limit,    ///< the largest sphere touches its own periodic image
};

/// @brief Collision rule for growing spheres.
///
/// Here \f$u_n\f$ is the normal relative velocity before contact (negative when
/// approaching) and the growth speed is the rate at which the contact distance grows.
enum class CollisionRule {
  /// Normal relative velocity after contact: max(-u_n, growth speed + margin). Elastic
  /// when the approach is faster than the growth, perfectly inelastic in the frame of
  /// the growing surfaces otherwise (the legacy rule).
  legacy,
  /// Elastic in the frame of the growing surfaces: u_n' = 2 * growth speed - u_n
  /// (Donev, Torquato and Stillinger). Heats the system; the thermostat removes it.
  elastic_growing,
};

/// @brief Name of a status as used in the Python interface.
/// @param s status value
/// @return the enumerator name as a string, e.g. `"time_limit"`; `"unknown"` for invalid values
inline const char* to_string(Status s) noexcept {
  switch (s) {
    case Status::running: return "running";
    case Status::target: return "target";
    case Status::pressure: return "pressure";
    case Status::jammed: return "jammed";
    case Status::stall: return "stall";
    case Status::collisions: return "collisions";
    case Status::time_limit: return "time_limit";
    case Status::timeout: return "timeout";
    case Status::interrupted: return "interrupted";
    case Status::no_events: return "no_events";
    case Status::box_limit: return "box_limit";
  }
  return "unknown";
}

/// @brief Parameters of a run. Every stopping criterion is off when set to its default.
///
/// Lengths are in the units of the box; `never` is positive infinity (see predict.hpp).
struct PackingOptions {
  /// Final radii are a_i * target_scale; infinity: no target.
  double target_scale = 1.0;
  /// Growth speed of the mean diameter in units of the thermal speed sqrt(kT/m).
  double growth_rate = 0.02;
  /// Reduced pressure PV/(NkT) at which growth stops (mean over a measurement window).
  double max_pressure = never;
  /// @name Jamming protocol
  /// Off when jammed_pressure is infinite: grow until the median reduced pressure reaches
  /// jam_start_pressure, then alternate relaxation at fixed radii (jam_relax_windows
  /// pressure windows) with growth steps of relative size jam_step / Z_median, until
  /// Z_median after relaxation reaches jammed_pressure.
  /// @{
  double jammed_pressure = never;    ///< median reduced pressure at which the packing counts as jammed
  double jam_start_pressure = 1e3;   ///< median reduced pressure at which the protocol starts
  int jam_relax_windows = 4;         ///< pressure windows per relaxation at fixed radii
  double jam_step = 0.2;             ///< fraction of the remaining relative growth closed per step
  /// @}
  double stall_tol = 0.0;              ///< 0: off. Stop when s grew by less than this (relative)
  double stall_window = 100.0;         ///<   over this many collisions per particle
  std::uint64_t max_collisions = 0;    ///< total collisions at which to stop; 0: unlimited
  CollisionRule rule = CollisionRule::legacy;  ///< collision rule
  double max_time = never;             ///< simulation time
  double timeout = never;              ///< wall-clock seconds
  double sync_interval = 10.0;         ///< collisions per particle between synchronisations
  /// After a collision the pair separates at least this much faster than the contact
  /// distance grows (thermal speed units); keeps the gap opening in floating point.
  double separation_margin = 1e-9;
  std::uint64_t seed = 0;              ///< seed of the random number generator
};

/// @brief One record per pressure window.
struct HistoryRecord {
  double time;              ///< simulation time at the end of the window
  double scale;             ///< s at the end of the window
  double reduced_pressure;  ///< mean over spheres
  double median_pressure;   ///< median over spheres
  double kT;                ///< time-averaged temperature over the window
  std::uint64_t n_collisions;  ///< collisions since the start of the run
  int phase;                ///< 0 growing, 1 relaxing (fixed radii), 2 growth step
};

/// @brief Summary of a finished run, returned by LSPacking::run().
struct PackingStats {
  Status status = Status::running;  ///< why the run ended
  double scale = 0.0;             ///< final s: radii are a_i * scale
  double density = 0.0;           ///< volume (area) fraction
  double reduced_pressure = 0.0;  ///< last measured PV/(NkT) (mean over spheres); 0 if never measured
  double median_pressure = 0.0;   ///< median over spheres of the per-sphere reduced pressure
  std::uint64_t n_collisions = 0;  ///< number of collisions that changed velocities
  std::uint64_t n_events = 0;      ///< number of processed events, including cell crossings
  double sim_time = 0.0;           ///< simulation time
  double wall_time = 0.0;          ///< wall-clock seconds
  double min_gap_ratio = 0.0;  ///< min over contacts of distance / contact distance - 1, before shrink
  double shrink_factor = 1.0;  ///< uniform radius factor applied to remove round-off overlaps
  double max_contact_error = 0.0;  ///< max over collisions of |distance / contact distance - 1|
};

/// @brief Event-driven Lubachevsky-Stillinger packing of polydisperse growing hard spheres.
///
/// The spheres move ballistically with unit mass at temperature kT = 1 and collide
/// elastically while their radii \f$a_i s(t)\f$ grow at a constant rate. The temperature
/// is restored to 1 at every synchronisation while the spheres grow. Pressure is
/// measured in windows of `sync_interval` collisions per particle as the reduced pressure
/// \f$Z = PV/(NkT) = 1 + \sum \mathbf r\cdot\mathbf J / (2 \int KE\, dt)\f$, from the virial of
/// the collision impulses \f$\mathbf J\f$ and the time integral of the kinetic energy.
/// The container may have periodic axes, flat walls and a spherical wall (see Box).
///
/// @tparam D dimension (2 or 3)
template <int D>
class LSPacking {
 public:
  /// @brief Creates a packing with random positions at zero radius and random velocities.
  ///
  /// The growth rate is set so that the mean diameter grows at `opt.growth_rate` thermal
  /// speeds. The generator is seeded with `opt.seed`.
  ///
  /// @param box container; validated here
  /// @param radii relative radii a_i; the radii at scale s are a_i * s
  /// @param opt run parameters
  /// @throws std::invalid_argument if the box is invalid, `radii` is empty or has too many
  /// entries, a radius is not positive and finite, `growth_rate < 0` or `target_scale <= 0`
  LSPacking(const Box<D>& box, std::vector<double> radii, const PackingOptions& opt)
      : box_(box), opt_(opt), rng_(opt.seed) {
    box_.validate();
    const std::size_t n = radii.size();
    if (n == 0) throw std::invalid_argument("at least one particle is required");
    if (n >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      throw std::invalid_argument("too many particles");
    p_.resize(n);
    double sum_a = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!(radii[i] > 0.0) || !std::isfinite(radii[i]))
        throw std::invalid_argument("radii must be positive and finite");
      p_[i].a = radii[i];
      a_max_ = std::max(a_max_, radii[i]);
      sum_a += radii[i];
      unit_volume_ += ball_volume<D>(radii[i]);
    }
    if (!(opt_.growth_rate >= 0.0)) throw std::invalid_argument("growth_rate must be >= 0");
    if (!(opt_.target_scale > 0.0)) throw std::invalid_argument("target_scale must be > 0");
    ds_user_ = ds_ = opt_.growth_rate / (2.0 * sum_a / static_cast<double>(n));
    w_.assign(n, 0.0);
    ev_.resize(n);
    randomize_positions();
    randomize_velocities();
  }

  /// @brief Places the particles uniformly at random (inside the ball wall, if any) and
  /// sets the scale to s = 0, so that the radii are zero.
  void randomize_positions() {
    for (auto& q : p_) {
      do {  // rejection sampling inside the ball wall, if any
        for (int k = 0; k < D; ++k) q.x[k] = box_.L[k] * rng_.uniform();
      } while (box_.ball.active && box_.ball_distance2(q.x) >= box_.ball.radius * box_.ball.radius);
    }
    s0_ = 0.0;
  }

  /// @brief Draws Maxwellian velocities with zero total momentum, scaled to kT = 1 exactly.
  void randomize_velocities() {
    for (auto& q : p_)
      for (int k = 0; k < D; ++k) q.v[k] = rng_.normal();
    if (p_.size() > 1) {
      Vec<D> mean{};
      for (const auto& q : p_)
        for (int k = 0; k < D; ++k) mean[k] += q.v[k];
      for (auto& q : p_)
        for (int k = 0; k < D; ++k) q.v[k] -= mean[k] / static_cast<double>(p_.size());
    }
    rescale_temperature();
  }

  /// @brief Starts from given positions at scale s (radii a_i * s); they must not overlap.
  ///
  /// Resets the periodic image counters. The positions are not checked for overlaps.
  ///
  /// @param x positions, inside the box
  /// @param scale initial scale s, at least 0
  /// @throws std::invalid_argument if the number of positions differs from the number of
  /// particles or `scale < 0`
  void set_positions(const std::vector<Vec<D>>& x, double scale) {
    if (x.size() != p_.size()) throw std::invalid_argument("positions: wrong number of particles");
    if (!(scale >= 0.0)) throw std::invalid_argument("initial scale must be >= 0");
    for (std::size_t i = 0; i < p_.size(); ++i) {
      p_[i].x = x[i];
      p_[i].img.fill(0);
    }
    s0_ = scale;
  }

  /// @brief Sets the velocities, replacing the random ones.
  ///
  /// The temperature is rescaled to kT = 1 at the first synchronisation of a growing run.
  ///
  /// @param v velocities, one per particle
  /// @throws std::invalid_argument if the number of velocities differs from the number of particles
  void set_velocities(const std::vector<Vec<D>>& v) {
    if (v.size() != p_.size()) throw std::invalid_argument("velocities: wrong number of particles");
    for (std::size_t i = 0; i < p_.size(); ++i) p_[i].v = v[i];
  }

  /// @brief Runs until a stopping criterion is met.
  ///
  /// The criteria are the target scale, the pressure limits, the jamming protocol, stall
  /// detection, the collision, simulation time and wall-clock limits, the interrupt
  /// callback, the absence of events and the box limit; see Status. On return the radii
  /// are shrunk uniformly if round-off left overlaps (PackingStats::shrink_factor).
  ///
  /// @param interrupted optional callback polled regularly; the run stops with
  /// Status::interrupted when it returns true
  /// @return the statistics of the run, valid until the next call of run()
  /// @throws std::invalid_argument if the growth rate is zero and the initial scale is zero
  /// @throws std::runtime_error if the box is smaller than the largest particle diameter
  const PackingStats& run(const std::function<bool()>& interrupted = {}) {
    const auto clock_start = std::chrono::steady_clock::now();
    auto wall_seconds = [&] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_start).count();
    };
    stats_ = PackingStats{};
    history_.clear();
    phase_ = Phase::grow;
    ds_ = ds_user_;
    s_step_ = never;
    window_ke_time_ = window_virial_ = window_time_ = 0.0;
    std::fill(w_.begin(), w_.end(), 0.0);
    sphere_pressure_.clear();
    if (ds_ == 0.0 && s0_ == 0.0)
      throw std::invalid_argument("zero growth rate needs initial positions with nonzero radii");
    // A sphere touches its own periodic image when its diameter reaches L_k.
    s_self_ = never;
    for (int k = 0; k < D; ++k)
      if (box_.periodic[k] || box_.flat_walls(k))  // touches its own image, or both walls
        s_self_ = std::min(s_self_, box_.L[k] / (2.0 * a_max_));
    if (box_.ball.active) s_self_ = std::min(s_self_, box_.ball.radius / a_max_);  // fills the ball
    if (opt_.target_scale != never && s0_ >= opt_.target_scale) {
      // Already at (or beyond) the target: only the final checks remain.
      sync();
      finish(Status::target, wall_seconds());
      return stats_;
    }
    sync();
    const std::uint64_t n = p_.size();
    const auto stall_window = static_cast<std::uint64_t>(opt_.stall_window * n);
    stall_history_.clear();
    const auto sync_every = static_cast<std::uint64_t>(std::max(1.0, opt_.sync_interval * n));
    for (std::uint64_t iteration = 1;; ++iteration) {
      if ((iteration & 0xffff) == 0) {
        if (wall_seconds() > opt_.timeout) {
          finish(Status::timeout, wall_seconds());
          return stats_;
        }
        if (interrupted && interrupted()) {
          finish(Status::interrupted, wall_seconds());
          return stats_;
        }
      }
      const double t_global =
          std::min({time_end(), time_self(), time_grid(), time_step(), opt_.max_time - elapsed_});
      const std::uint32_t i = heap_.top();
      const double te = heap_.top_key();
      if (te >= t_global) {
        if (t_global == never) {
          finish(Status::no_events, wall_seconds());
          return stats_;
        }
        now_ = t_global;
        if (t_global == time_end()) {
          finish(Status::target, wall_seconds());
          return stats_;
        }
        if (t_global == time_self()) {
          finish(Status::box_limit, wall_seconds());
          return stats_;
        }
        if (t_global == opt_.max_time - elapsed_) {
          finish(Status::time_limit, wall_seconds());
          return stats_;
        }
        if (t_global == time_step()) {  // growth step of the jamming protocol done
          sync();
          s0_ = s_step_;  // exact, without round-off from s0 + ds * t
          start_relaxation();
          continue;
        }
        sync();  // cell grid no longer valid for the grown radii
        continue;
      }
      now_ = te;
      ++stats_.n_events;
      handle(i);
      if (opt_.max_collisions != 0 && stats_.n_collisions >= opt_.max_collisions) {
        finish(Status::collisions, wall_seconds());
        return stats_;
      }

      if (since_sync_ >= sync_every) {
        sync();
        if (window_ke_time_ > 0.0) {
          close_window();
          if (stats_.reduced_pressure > opt_.max_pressure) {
            finish(Status::pressure, wall_seconds());
            return stats_;
          }
          if (opt_.jammed_pressure != never) {
            const double zmed = stats_.median_pressure;
            if (phase_ == Phase::grow && zmed >= std::min(opt_.jam_start_pressure, opt_.jammed_pressure)) {
              start_relaxation();
            } else if (phase_ == Phase::step &&
                       zmed >= std::min(opt_.jammed_pressure, 4.0 * z_step_start_ / (1.0 - opt_.jam_step))) {
              // The step should raise Z by about 1 / (1 - jam_step); far more means it
              // ran into jamming before reaching s_step_.
              start_relaxation();
            } else if (phase_ == Phase::relax && zmed >= opt_.jammed_pressure) {
              // Measured at fixed radii: jammed. (Waiting for the last relaxation window
              // can cycle forever: near jamming small rearrangements lower the pressure
              // again after each step.)
              finish(Status::jammed, wall_seconds());
              return stats_;
            } else if (phase_ == Phase::relax && --relax_left_ <= 0) {
              // Close the remaining gap to jamming, about s / Z_median, by a fraction.
              phase_ = Phase::step;
              z_step_start_ = zmed;
              s_step_ = s0_ * (1.0 + opt_.jam_step / zmed);
              ds_ = ds_user_;
              sync();
            }
          }
        }
        if (opt_.stall_tol > 0.0) {
          // Compare with the oldest sync point at least stall_window collisions back.
          stall_history_.push_back({stats_.n_collisions, s0_});
          while (stall_history_.size() > 1 &&
                 stats_.n_collisions - stall_history_[1].first >= stall_window)
            stall_history_.pop_front();
          const auto& [n_then, s_then] = stall_history_.front();
          if (stats_.n_collisions - n_then >= stall_window && s0_ - s_then < opt_.stall_tol * s0_) {
            finish(Status::stall, wall_seconds());
            return stats_;
          }
        }
      }
    }
  }

  /// @return the number of particles
  std::size_t size() const noexcept { return p_.size(); }
  /// @return one record per pressure window of the last run
  const std::vector<HistoryRecord>& history() const noexcept { return history_; }
  /// @brief Per-sphere reduced pressure of the last completed window: 1 + N w_i / (2 int KE dt).
  /// Without walls its mean over spheres is the reduced pressure; wall contacts add to
  /// the spheres that touch a wall. Close to 1 for rattlers.
  /// @return one value per sphere; empty before the first window has closed
  const std::vector<double>& sphere_pressure() const noexcept { return sphere_pressure_; }
  /// @return the statistics of the last run (status Status::running before the first run)
  const PackingStats& stats() const noexcept { return stats_; }
  /// @return the container
  const Box<D>& box() const noexcept { return box_; }

  /// @brief Positions wrapped into the box; valid after run().
  /// @return one position per particle
  std::vector<Vec<D>> positions() const {
    std::vector<Vec<D>> x(p_.size());
    for (std::size_t i = 0; i < p_.size(); ++i) x[i] = p_[i].x;
    return x;
  }
  /// @return one velocity per particle
  std::vector<Vec<D>> velocities() const {
    std::vector<Vec<D>> v(p_.size());
    for (std::size_t i = 0; i < p_.size(); ++i) v[i] = p_[i].v;
    return v;
  }
  /// @brief Final radii a_i * stats().scale; valid after run().
  /// @return one radius per particle
  std::vector<double> radii() const {
    std::vector<double> r(p_.size());
    for (std::size_t i = 0; i < p_.size(); ++i) r[i] = p_[i].a * stats_.scale;
    return r;
  }

 private:
  /// State of one particle, advanced lazily to its own time `t`.
  struct Particle {
    Vec<D> x{};                     ///< position at time t (wrapped on periodic axes)
    Vec<D> v{};                     ///< velocity
    std::array<std::int32_t, D> img{};  ///< periodic image counters: unwrapped = x + img * L
    double t = 0.0;                 ///< time (since the last synchronisation) of x
    double a = 0.0;                 ///< relative radius
    std::uint64_t counter = 0;      ///< number of velocity changes
  };

  /// Kind of the predicted event of a particle: grid cell crossing, flat wall contact,
  /// ball wall contact or pair contact.
  enum class EventType : std::uint8_t { none, cross, wall, ball, pair };

  /// Predicted next event of a particle.
  struct Event {
    EventType type = EventType::none;
    std::int8_t axis = 0;        ///< cross and wall: axis
    std::int8_t dir = 0;         ///< cross and wall: -1 lower, +1 upper
    std::uint32_t partner = 0;   ///< pair: the other particle
    std::uint64_t partner_counter = 0;  ///< pair: partner's counter at prediction time
    /// pair: r = X_partner - X_self + image * L with unwrapped positions X
    std::array<std::int32_t, D> image{};
  };

  /// Scale s at time t after the last synchronisation.
  double scale_at(double t) const noexcept { return s0_ + ds_ * t; }

  /// Time (since the last synchronisation) at which the target scale is reached, or `never`.
  double time_end() const noexcept {
    if (opt_.target_scale == never || ds_ == 0.0) return never;
    return (opt_.target_scale - s0_) / ds_;
  }

  /// Time at which the radii reach the limit that the cell grid allows, or `never`.
  double time_grid() const noexcept {
    if (ds_ == 0.0 || s_grid_limit_ == never) return never;
    return (s_grid_limit_ - s0_) / ds_;
  }

  /// Moves a particle ballistically to the current time `now_`.
  void advance(Particle& q) const noexcept {
    const double dt = now_ - q.t;
    for (int k = 0; k < D; ++k) q.x[k] += q.v[k] * dt;
    q.t = now_;
  }

  /// Total kinetic energy of all particles.
  double kinetic_energy() const noexcept {
    double e = 0.0;
    for (const auto& q : p_) e += 0.5 * norm2<D>(q.v);
    return e;
  }

  /// Scales all velocities so that kT = 1.
  void rescale_temperature() {
    const double kT = 2.0 * kinetic_energy() / (D * static_cast<double>(p_.size()));
    if (kT > 0.0) {
      const double f = 1.0 / std::sqrt(kT);
      for (auto& q : p_)
        for (int k = 0; k < D; ++k) q.v[k] *= f;
    }
  }

  /// Accumulates the time integral of the kinetic energy up to now (it only changes at
  /// collisions and synchronisations).
  void integrate_ke() noexcept {
    window_ke_time_ += ke_ * (now_ - ke_since_);
    ke_since_ = now_;
  }

  /// Brings every particle to the current time, resets the time origin, restores
  /// kT = 1 while growing, rebuilds the cell grid for the current radii and re-predicts all events.
  void sync() {
    for (auto& q : p_) advance(q);
    integrate_ke();
    elapsed_ += now_;
    window_time_ += now_;
    s0_ = scale_at(now_);
    now_ = 0.0;
    ke_since_ = 0.0;
    for (auto& q : p_) q.t = 0.0;
    since_sync_ = 0;
    if (ds_ > 0.0) rescale_temperature();  // growth changes the energy; fixed radii conserve it
    ke_ = kinetic_energy();
    wrap_positions();
    build_grid();
    std::vector<double> keys(p_.size());
    for (std::uint32_t i = 0; i < p_.size(); ++i) keys[i] = predict(i);
    heap_.reset(static_cast<std::uint32_t>(p_.size()));
    heap_.assign(keys);
  }

  /// Wraps positions into the box on periodic axes and updates the image counters.
  void wrap_positions() noexcept {
    for (auto& q : p_)
      for (int k = 0; k < D; ++k) {
        if (!box_.periodic[k]) continue;
        const double shift = std::floor(q.x[k] / box_.L[k]);
        if (shift != 0.0) {
          q.x[k] -= shift * box_.L[k];
          q.img[k] += static_cast<std::int32_t>(shift);
        }
        if (q.x[k] >= box_.L[k]) {  // e.g. -1e-17 + L rounds to L
          q.x[k] -= box_.L[k];
          ++q.img[k];
        }
      }
  }

  /// Rebuilds the cell grid with cells strictly wider than the current largest diameter,
  /// so that the radii can grow before the next rebuild.
  /// @throws std::runtime_error if the box is smaller than the largest particle diameter
  void build_grid() {
    const double sigma_max = 2.0 * a_max_ * s0_;
    grid_.build(box_, sigma_max * (1.0 + 1e-6), 2 * p_.size() + 27);
    if (grid_.min_width() < sigma_max)
      throw std::runtime_error("the box is smaller than the largest particle diameter");
    s_grid_limit_ = grid_.min_width() / (2.0 * a_max_);
    grid_.clear_particles(p_.size());
    for (std::uint32_t i = 0; i < p_.size(); ++i) grid_.insert(i, grid_.flat(grid_.coords_of(p_[i].x)));
  }

  /// Predicts the next event of particle i (advanced to now) and stores it in ev_[i].
  /// @return the event time, in the time since the last synchronisation
  double predict(std::uint32_t i) {
    Particle& pi = p_[i];
    advance(pi);
    Event best;
    double tau_best = never;
    const double s = scale_at(now_);

    const auto c = grid_.coords(grid_.cell_of(i));
    for (int k = 0; k < D; ++k) {
      const double v = pi.v[k];
      double tau = never;
      std::int8_t dir = 0;
      if (v > 0.0 && (box_.periodic[k] || c[k] < grid_.n(k) - 1)) {
        tau = std::max(0.0, ((c[k] + 1) * grid_.width(k) - pi.x[k]) / v);
        dir = 1;
      } else if (v < 0.0 && (box_.periodic[k] || c[k] > 0)) {
        tau = std::max(0.0, (c[k] * grid_.width(k) - pi.x[k]) / v);
        dir = -1;
      }
      if (tau < tau_best) {
        tau_best = tau;
        best.type = EventType::cross;
        best.axis = static_cast<std::int8_t>(k);
        best.dir = dir;
      }
      if (box_.flat_walls(k)) {
        const double rho = pi.a * s, drho = pi.a * ds_;
        const double t_lo = first_contact_linear(pi.x[k] - rho, v - drho);
        const double t_hi = first_contact_linear(box_.L[k] - pi.x[k] - rho, -v - drho);
        if (t_lo < tau_best) {
          tau_best = t_lo;
          best.type = EventType::wall;
          best.axis = static_cast<std::int8_t>(k);
          best.dir = -1;
        }
        if (t_hi < tau_best) {
          tau_best = t_hi;
          best.type = EventType::wall;
          best.axis = static_cast<std::int8_t>(k);
          best.dir = 1;
        }
      }
    }

    if (box_.ball.active) {
      // f(tau) = (R - rho - drho tau)^2 - |q + v tau|^2 over the ball axes; f > 0 inside.
      const double rho = pi.a * s, drho = pi.a * ds_;
      const double gap = box_.ball.radius - rho;
      double qv = 0.0, vv = 0.0, qq = 0.0;
      for (int k = 0; k < D; ++k) {
        if (!box_.ball.axes[k]) continue;
        const double q = pi.x[k] - box_.ball.center[k];
        qv += q * pi.v[k];
        vv += pi.v[k] * pi.v[k];
        qq += q * q;
      }
      const double tau = first_contact(drho * drho - vv, -gap * drho - qv, gap * gap - qq);
      if (tau < tau_best) {
        tau_best = tau;
        best.type = EventType::ball;
      }
    }

    grid_.for_each_neighbor(grid_.cell_of(i), [&](std::uint32_t j, const std::array<int, D>& shift) {
      if (j == i) return;  // self images would need a box smaller than a diameter
      const Particle& pj = p_[j];
      const double dtj = now_ - pj.t;
      Vec<D> r, u;
      for (int k = 0; k < D; ++k) {
        r[k] = pj.x[k] + pj.v[k] * dtj + shift[k] * box_.L[k] - pi.x[k];
        u[k] = pj.v[k] - pi.v[k];
      }
      const double sig = (pi.a + pj.a) * s, dsig = (pi.a + pj.a) * ds_;
      const double tau = first_contact(norm2<D>(u) - dsig * dsig, dot<D>(r, u) - sig * dsig,
                                       norm2<D>(r) - sig * sig);
      if (tau < tau_best) {
        tau_best = tau;
        best.type = EventType::pair;
        best.partner = j;
        best.partner_counter = pj.counter;
        for (int k = 0; k < D; ++k) best.image[k] = shift[k] - pj.img[k] + pi.img[k];
      }
    });

    ev_[i] = best;
    return now_ + tau_best;
  }

  /// Processes the predicted event of particle i at the current time: moves it across a
  /// cell, collides it with a wall or a partner, and re-predicts the affected particles.
  void handle(std::uint32_t i) {
    const Event e = ev_[i];
    Particle& pi = p_[i];
    switch (e.type) {
      case EventType::none:
        heap_.update(i, never);
        return;
      case EventType::cross: {
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
      case EventType::wall: {
        advance(pi);
        const int k = e.axis;
        const double drho = pi.a * ds_;
        const double away = e.dir < 0 ? pi.v[k] : -pi.v[k];  // velocity away from the wall
        const double target = separation_speed(away, drho);
        if (target > away) {
          integrate_ke();
          ke_ += 0.5 * (target * target - away * away);
          pi.v[k] = e.dir < 0 ? target : -target;
          // Load on the sphere: impulse times the distance from centre to contact.
          w_[i] += (target - away) * pi.a * scale_at(now_);
        }
        ++pi.counter;
        count_collision();
        heap_.update(i, predict(i));
        return;
      }
      case EventType::ball: {
        advance(pi);
        Vec<D> n{};  // outward normal of the ball wall at the contact
        double qn = 0.0;
        for (int k = 0; k < D; ++k)
          if (box_.ball.axes[k]) {
            n[k] = pi.x[k] - box_.ball.center[k];
            qn += n[k] * n[k];
          }
        qn = std::sqrt(qn);
        if (qn > 0.0) {
          for (int k = 0; k < D; ++k) n[k] /= qn;
          const double rho = pi.a * scale_at(now_), drho = pi.a * ds_;
          stats_.max_contact_error =
              std::max(stats_.max_contact_error, std::abs((box_.ball.radius - qn) / rho - 1.0));
          const double away = -dot<D>(pi.v, n);  // velocity away from the wall
          // On a concave wall the legacy rule (separation at the growth speed) traps a
          // sphere grazing along the wall in ever more frequent collisions; the legacy
          // tube code separated 1% faster than the growth (fctr = 1.01), as here.
          const double target = opt_.rule == CollisionRule::legacy
                                    ? std::max(-away, 1.01 * drho + opt_.separation_margin)
                                    : separation_speed(away, drho);
          if (target > away) {
            integrate_ke();
            ke_ -= 0.5 * norm2<D>(pi.v);
            for (int k = 0; k < D; ++k) pi.v[k] -= (target - away) * n[k];
            ke_ += 0.5 * norm2<D>(pi.v);
            w_[i] += (target - away) * rho;
          }
        }
        ++pi.counter;
        count_collision();
        heap_.update(i, predict(i));
        return;
      }
      case EventType::pair: {
        const std::uint32_t j = e.partner;
        Particle& pj = p_[j];
        if (pj.counter != e.partner_counter) {  // partner changed course: stale event
          heap_.update(i, predict(i));
          return;
        }
        advance(pi);
        advance(pj);
        Vec<D> r, u;
        for (int k = 0; k < D; ++k) {
          r[k] = pj.x[k] - pi.x[k] + (e.image[k] + pj.img[k] - pi.img[k]) * box_.L[k];
          u[k] = pj.v[k] - pi.v[k];
        }
        const double dist = std::sqrt(norm2<D>(r));
        stats_.max_contact_error = std::max(stats_.max_contact_error,
                                            std::abs(dist / ((pi.a + pj.a) * scale_at(now_)) - 1.0));
        const double un = dot<D>(r, u) / dist;
        const double dsig = (pi.a + pj.a) * ds_;
        const double delta = separation_speed(un, dsig) - un;
        if (delta > 0.0) {
          integrate_ke();
          ke_ -= 0.5 * (norm2<D>(pi.v) + norm2<D>(pj.v));
          for (int k = 0; k < D; ++k) {
            const double dv = 0.5 * delta * r[k] / dist;
            pi.v[k] -= dv;
            pj.v[k] += dv;
          }
          ke_ += 0.5 * (norm2<D>(pi.v) + norm2<D>(pj.v));
          window_virial_ += 0.5 * delta * dist;
          w_[i] += 0.25 * delta * dist;  // each sphere takes half of the pair virial
          w_[j] += 0.25 * delta * dist;
          ++pi.counter;
          ++pj.counter;
          count_collision();
        }
        heap_.update(i, predict(i));
        heap_.update(j, predict(j));
        return;
      }
    }
  }

  /// Stage of the jamming protocol: growth, relaxation at fixed radii, or a growth step.
  enum class Phase { grow, relax, step };

  /// Time at which the largest sphere touches its own periodic image (or fills the ball),
  /// or `never`.
  double time_self() const noexcept {
    if (s_self_ == never || ds_ == 0.0) return never;
    return std::max(0.0, (s_self_ - s0_) / ds_);
  }

  /// Time at which the current growth step of the jamming protocol ends, or `never`.
  double time_step() const noexcept {
    if (phase_ != Phase::step || ds_ == 0.0) return never;
    return (s_step_ - s0_) / ds_;
  }

  /// Stops growth and starts a fresh pressure measurement at fixed radii. Must be
  /// called right after sync() (now_ == 0).
  void start_relaxation() {
    phase_ = Phase::relax;
    relax_left_ = std::max(1, opt_.jam_relax_windows);
    s_step_ = never;
    ds_ = 0.0;
    window_ke_time_ = window_virial_ = window_time_ = 0.0;
    std::fill(w_.begin(), w_.end(), 0.0);
    sync();  // re-predict all events without growth
  }

  /// Evaluates the pressures of the window that just ended and starts a new one.
  void close_window() {
    const double n = static_cast<double>(p_.size());
    stats_.reduced_pressure = 1.0 + window_virial_ / (2.0 * window_ke_time_);
    std::vector<double> z(w_);
    const auto mid = z.begin() + static_cast<std::ptrdiff_t>(z.size() / 2);
    std::nth_element(z.begin(), mid, z.end());
    stats_.median_pressure = 1.0 + n * (*mid) / (2.0 * window_ke_time_);
    sphere_pressure_.resize(w_.size());
    for (std::size_t i = 0; i < w_.size(); ++i) sphere_pressure_[i] = 1.0 + n * w_[i] / (2.0 * window_ke_time_);
    const double kT = 2.0 * window_ke_time_ / (D * n * window_time_);
    history_.push_back({elapsed_, s0_, stats_.reduced_pressure, stats_.median_pressure, kT,
                        stats_.n_collisions, static_cast<int>(phase_)});
    window_ke_time_ = window_virial_ = window_time_ = 0.0;
    std::fill(w_.begin(), w_.end(), 0.0);
  }

  /// Normal separation speed after a contact with approach velocity `un` (negative when
  /// approaching) and contact-distance growth speed `dsig`.
  double separation_speed(double un, double dsig) const noexcept {
    if (opt_.rule == CollisionRule::elastic_growing)
      return std::max(2.0 * dsig - un, dsig + opt_.separation_margin);
    return std::max(-un, dsig + opt_.separation_margin);
  }

  /// Counts a collision in the statistics and towards the next synchronisation.
  void count_collision() noexcept {
    ++stats_.n_collisions;
    ++since_sync_;
  }

  /// Ends the run: brings all particles to the current time, fixes the final scale, shrinks
  /// the radii uniformly to remove round-off overlaps and fills in the statistics.
  void finish(Status status, double wall_time) {
    for (auto& q : p_) advance(q);
    elapsed_ += now_;
    s0_ = status == Status::target      ? opt_.target_scale
          : status == Status::box_limit ? std::min(s_self_, scale_at(now_))
                                        : scale_at(now_);
    now_ = 0.0;
    for (auto& q : p_) q.t = 0.0;
    wrap_positions();

    const double r_min = min_contact_scale();
    stats_.status = status;
    stats_.min_gap_ratio = r_min / s0_ - 1.0;
    stats_.shrink_factor = std::min(1.0, r_min / s0_);
    stats_.scale = s0_ * stats_.shrink_factor;
    stats_.density = unit_volume_ * std::pow(stats_.scale, D) / box_.volume();
    stats_.sim_time = elapsed_;
    stats_.wall_time = wall_time;
  }

  /// Smallest scale s at which some pair or particle-wall contact occurs for the
  /// current positions (pairs: distance / (a_i + a_j); walls: distance / a_i).
  double min_contact_scale() {
    double r_min = never;
    CellGrid<D> grid;
    grid.build(box_, 2.0 * a_max_ * s0_, 2 * p_.size() + 27);
    grid.clear_particles(p_.size());
    for (std::uint32_t i = 0; i < p_.size(); ++i) grid.insert(i, grid.flat(grid.coords_of(p_[i].x)));
    for (std::uint32_t i = 0; i < p_.size(); ++i) {
      const Particle& pi = p_[i];
      for (int k = 0; k < D; ++k)
        if (box_.flat_walls(k))
          r_min = std::min({r_min, pi.x[k] / pi.a, (box_.L[k] - pi.x[k]) / pi.a});
      if (box_.ball.active)
        r_min = std::min(r_min, (box_.ball.radius - std::sqrt(box_.ball_distance2(pi.x))) / pi.a);
      grid.for_each_neighbor(grid.cell_of(i), [&](std::uint32_t j, const std::array<int, D>& shift) {
        if (j <= i) return;
        Vec<D> r;
        for (int k = 0; k < D; ++k) r[k] = p_[j].x[k] + shift[k] * box_.L[k] - pi.x[k];
        r_min = std::min(r_min, std::sqrt(norm2<D>(r)) / (pi.a + p_[j].a));
      });
    }
    return r_min;
  }

  Box<D> box_;                ///< container
  PackingOptions opt_;        ///< run parameters
  Rng rng_;                   ///< random number generator
  std::vector<Particle> p_;   ///< particles
  std::vector<Event> ev_;     ///< predicted event of each particle
  EventHeap heap_;            ///< event times, one per particle
  CellGrid<D> grid_;          ///< cell grid for neighbour search
  PackingStats stats_;        ///< statistics of the run

  double a_max_ = 0.0;        ///< largest relative radius
  double unit_volume_ = 0.0;  ///< total particle volume at s = 1
  double ds_ = 0.0;           ///< ds/dt
  double s0_ = 0.0;           ///< scale at now_ = 0
  double now_ = 0.0;          ///< time since the last synchronisation
  double elapsed_ = 0.0;      ///< simulation time up to the last synchronisation
  double s_grid_limit_ = never;  ///< largest scale for which the cell grid is valid
  std::uint64_t since_sync_ = 0;  ///< collisions since the last synchronisation
  double ds_user_ = 0.0;         ///< ds/dt requested by the user
  Phase phase_ = Phase::grow;    ///< current stage of the jamming protocol
  double s_step_ = never;        ///< end of the current growth step (jamming protocol)
  double s_self_ = never;        ///< scale at which the largest sphere touches its own image
  double z_step_start_ = 0.0;    ///< median pressure when the current growth step started
  int relax_left_ = 0;           ///< relaxation windows left (jamming protocol)
  std::vector<double> w_;        ///< per-sphere virial in the current window
  std::vector<double> sphere_pressure_;  ///< per-sphere reduced pressure of the last window
  double window_time_ = 0.0;     ///< duration of the current pressure window
  std::vector<HistoryRecord> history_;   ///< one record per pressure window
  double ke_ = 0.0;              ///< current kinetic energy
  double ke_since_ = 0.0;        ///< time up to which ke_ has been integrated
  double window_ke_time_ = 0.0;  ///< integral of the kinetic energy over the pressure window
  double window_virial_ = 0.0;   ///< sum of r . J over pair collisions in the window
  std::deque<std::pair<std::uint64_t, double>> stall_history_;  ///< (collisions, s) at syncs
};

}  // namespace spheropack
