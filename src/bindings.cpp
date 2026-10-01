#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "spheropack/ls_packing.hpp"
#include "spheropack/neighbors.hpp"
#include "spheropack/rng.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using InArray1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using InArray2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, double>;

// Moves a vector into a new numpy array of the given shape (no copy).
template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& data, std::initializer_list<std::size_t> shape) {
  auto* heap = new std::vector<T>(std::move(data));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(heap->data(), shape, owner);
}

template <int D>
OutArray points_to_numpy(const std::vector<spheropack::Vec<D>>& x) {
  std::vector<double> flat(x.size() * D);
  for (std::size_t i = 0; i < x.size(); ++i)
    for (int k = 0; k < D; ++k) flat[i * D + k] = x[i][k];
  return to_numpy(std::move(flat), {x.size(), static_cast<std::size_t>(D)});
}

template <int D>
std::vector<spheropack::Vec<D>> numpy_to_points(const InArray2& a, std::size_t n, const char* name) {
  if (a.shape(0) != n || a.shape(1) != static_cast<std::size_t>(D))
    throw nb::value_error((std::string(name) + " must have shape (n, dim)").c_str());
  std::vector<spheropack::Vec<D>> x(n);
  const double* p = a.data();
  for (std::size_t i = 0; i < n; ++i)
    for (int k = 0; k < D; ++k) x[i][k] = p[i * D + k];
  return x;
}

template <int D>
nb::dict ls_pack(InArray1 radii, std::vector<double> lengths, std::vector<bool> periodic,
                 double target_scale, double growth_rate, double max_pressure,
                 double jammed_pressure, double stall_tol, double stall_window, std::uint64_t max_collisions,
                 double max_time, double timeout, double sync_interval, std::uint64_t seed,
                 const std::string& rule, std::optional<InArray2> positions,
                 double initial_scale, std::optional<InArray2> velocities,
                 double jam_start_pressure, int jam_relax_windows, double jam_step) {
  if (lengths.size() != D || periodic.size() != D)
    throw nb::value_error("box lengths and periodic flags need one entry per dimension");
  spheropack::Box<D> box;
  for (int k = 0; k < D; ++k) {
    box.L[k] = lengths[k];
    box.periodic[k] = periodic[k];
  }
  spheropack::PackingOptions opt;
  opt.target_scale = target_scale;
  opt.growth_rate = growth_rate;
  opt.max_pressure = max_pressure;
  opt.jammed_pressure = jammed_pressure;
  opt.jam_start_pressure = jam_start_pressure;
  opt.jam_relax_windows = jam_relax_windows;
  opt.jam_step = jam_step;
  opt.stall_tol = stall_tol;
  opt.stall_window = stall_window;
  opt.max_collisions = max_collisions;
  if (rule == "legacy") opt.rule = spheropack::CollisionRule::legacy;
  else if (rule == "elastic_growing") opt.rule = spheropack::CollisionRule::elastic_growing;
  else throw nb::value_error("rule must be 'legacy' or 'elastic_growing'");
  opt.max_time = max_time;
  opt.timeout = timeout;
  opt.sync_interval = sync_interval;
  opt.seed = seed;

  const std::size_t n = radii.shape(0);
  std::vector<double> a(radii.data(), radii.data() + n);
  spheropack::LSPacking<D> sim(box, std::move(a), opt);
  if (positions) sim.set_positions(numpy_to_points<D>(*positions, n, "positions"), initial_scale);
  if (velocities) sim.set_velocities(numpy_to_points<D>(*velocities, n, "velocities"));

  {
    nb::gil_scoped_release release;
    sim.run([] {
      nb::gil_scoped_acquire acquire;
      if (PyErr_CheckSignals() != 0) {
        PyErr_Clear();
        return true;
      }
      return false;
    });
  }

  const auto& s = sim.stats();
  nb::dict out;
  out["positions"] = points_to_numpy<D>(sim.positions());
  out["velocities"] = points_to_numpy<D>(sim.velocities());
  out["radii"] = to_numpy(sim.radii(), {n});
  out["status"] = std::string(spheropack::to_string(s.status));
  out["scale"] = s.scale;
  out["density"] = s.density;
  out["reduced_pressure"] = s.reduced_pressure;
  out["median_pressure"] = s.median_pressure;
  out["sphere_pressure"] = to_numpy(std::vector<double>(sim.sphere_pressure()), {sim.sphere_pressure().size()});
  {
    const auto& h = sim.history();
    const std::size_t m = h.size();
    std::vector<double> t(m), sc(m), z(m), zm(m), kt(m), nc(m), ph(m);
    for (std::size_t k = 0; k < m; ++k) {
      t[k] = h[k].time;
      sc[k] = h[k].scale;
      z[k] = h[k].reduced_pressure;
      zm[k] = h[k].median_pressure;
      kt[k] = h[k].kT;
      nc[k] = static_cast<double>(h[k].n_collisions);
      ph[k] = h[k].phase;
    }
    nb::dict hist;
    hist["time"] = to_numpy(std::move(t), {m});
    hist["scale"] = to_numpy(std::move(sc), {m});
    hist["reduced_pressure"] = to_numpy(std::move(z), {m});
    hist["median_pressure"] = to_numpy(std::move(zm), {m});
    hist["kT"] = to_numpy(std::move(kt), {m});
    hist["n_collisions"] = to_numpy(std::move(nc), {m});
    hist["phase"] = to_numpy(std::move(ph), {m});
    out["history"] = hist;
  }
  out["n_collisions"] = s.n_collisions;
  out["n_events"] = s.n_events;
  out["sim_time"] = s.sim_time;
  out["wall_time"] = s.wall_time;
  out["min_gap_ratio"] = s.min_gap_ratio;
  out["shrink_factor"] = s.shrink_factor;
  out["max_contact_error"] = s.max_contact_error;
  return out;
}

template <int D>
nb::tuple neighbor_pairs(InArray2 positions, std::vector<double> lengths, std::vector<bool> periodic,
                         double cutoff) {
  if (lengths.size() != D || periodic.size() != D)
    throw nb::value_error("box lengths and periodic flags need one entry per dimension");
  if (!(cutoff > 0.0)) throw nb::value_error("cutoff must be positive");
  spheropack::Box<D> box;
  for (int k = 0; k < D; ++k) {
    box.L[k] = lengths[k];
    box.periodic[k] = periodic[k];
  }
  box.validate();
  const auto x = numpy_to_points<D>(positions, positions.shape(0), "positions");
  std::vector<std::int64_t> i_out, j_out;
  std::vector<double> r_out;
  {
    nb::gil_scoped_release release;
    spheropack::for_each_pair<D>(x, box, cutoff, [&](std::uint32_t i, std::uint32_t j, const spheropack::Vec<D>& r) {
      i_out.push_back(i);
      j_out.push_back(j);
      r_out.insert(r_out.end(), r.begin(), r.end());
    });
  }
  const std::size_t m = i_out.size();
  return nb::make_tuple(to_numpy(std::move(i_out), {m}), to_numpy(std::move(j_out), {m}),
                        to_numpy(std::move(r_out), {m, static_cast<std::size_t>(D)}));
}

// Draws `n` variates of the given kind from a fresh generator; used to test the
// generator from Python and to check reproducibility across platforms.
OutArray rng_sample(std::uint64_t seed, std::size_t n, const std::string& kind) {
  spheropack::Rng rng(seed);
  double (spheropack::Rng::*draw)() = nullptr;
  if (kind == "uniform") draw = &spheropack::Rng::uniform;
  else if (kind == "normal") draw = &spheropack::Rng::normal;
  else if (kind == "exponential") draw = &spheropack::Rng::exponential;
  else throw nb::value_error("kind must be 'uniform', 'normal' or 'exponential'");
  std::vector<double> data(n);
  for (auto& x : data) x = (rng.*draw)();
  return to_numpy(std::move(data), {n});
}

std::uint64_t rng_raw(std::uint64_t seed, std::size_t skip) {
  spheropack::Rng rng(seed);
  for (std::size_t i = 0; i < skip; ++i) rng();
  return rng();
}

template <int D>
void def_ls_pack(nb::module_& m, const char* name) {
  m.def(name, &ls_pack<D>, "radii"_a, "lengths"_a, "periodic"_a, "target_scale"_a,
        "growth_rate"_a, "max_pressure"_a, "jammed_pressure"_a, "stall_tol"_a, "stall_window"_a, "max_collisions"_a,
        "max_time"_a, "timeout"_a, "sync_interval"_a, "seed"_a, "rule"_a, "positions"_a.none(), "initial_scale"_a,
        "velocities"_a.none(), "jam_start_pressure"_a = 1e3, "jam_relax_windows"_a = 4,
        "jam_step"_a = 0.2);
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "spheropack C++ core";
  def_ls_pack<2>(m, "_ls_pack2");
  def_ls_pack<3>(m, "_ls_pack3");
  m.def("_neighbor_pairs2", &neighbor_pairs<2>, "positions"_a, "lengths"_a, "periodic"_a, "cutoff"_a);
  m.def("_neighbor_pairs3", &neighbor_pairs<3>, "positions"_a, "lengths"_a, "periodic"_a, "cutoff"_a);
  m.def("_rng_sample", &rng_sample, "seed"_a, "n"_a, "kind"_a);
  m.def("_rng_raw", &rng_raw, "seed"_a, "skip"_a = 0);
}
