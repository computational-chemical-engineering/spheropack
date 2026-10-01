// Periodic 3D monodisperse event-driven Lubachevsky-Stillinger driver (legacy code).
// Usage: legacy_driver N seed target_phi growrate collision_cap_per_particle
// Prints one JSON object on stdout. Final diameter d=1 at target_phi.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "packing.hpp"

using boost::ranlux3;
using boost::uniform_01;
using boost::normal_distribution;
using boost::variate_generator;
using ed::Particle;
using ed::PackingHS;

// Time can stall near jamming, so step(t) may never return. The cap is therefore
// enforced from inside the simulation via the virtual findCollision hook.
struct CapReached {};

template <unsigned dim>
class CappedHS : public PackingHS<dim>
{
public:
  using PackingHS<dim>::PackingHS;
  size_t maxColl = 0;
  virtual bool findCollision(size_t i, const ed::Nbr<dim> &nbr)
  {
    if (this->numCollisions() > maxColl) throw CapReached();
    return PackingHS<dim>::findCollision(i, nbr);
  }
};

int main(int argc, char *argv[])
{
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s N seed target_phi growrate cap_per_particle\n", argv[0]);
    return 1;
  }
  const unsigned dim(3);
  const double pi(3.1415926535897932384626433832795);
  const size_t N = std::strtoull(argv[1], nullptr, 10);
  const unsigned seed = (unsigned)std::strtoul(argv[2], nullptr, 10);
  const double phi = std::strtod(argv[3], nullptr);
  const double growrate = std::strtod(argv[4], nullptr);
  const double cap = std::strtod(argv[5], nullptr);

  const double radius(0.5);
  const double Lc = std::pow(N * pi / 6.0 / phi, 1.0 / 3.0);
  double L[3] = {Lc, Lc, Lc};
  const double tEnd(1.0 / growrate);

  ranlux3 ran;
  uniform_01<double> unif;
  normal_distribution<double> gaussDistr(0.0, 1.0);
  variate_generator<boost::ranlux3 &, boost::normal_distribution<double> > normalVar(ran, gaussDistr);
  ran.seed(seed);

  std::vector<Particle<dim> > particle(N);
  std::vector<double> radii(N);
  for (size_t i(0); i < N; ++i) {
    radii[i] = radius;
    for (unsigned j(0); j < dim; ++j) {
      particle[i].xyz[j] = L[j] * unif(ran);
      particle[i].v[j] = normalVar();
    }
  }

  CappedHS<dim> sim(particle, radii, growrate, L, 2.0 * radius, 20);
  sim.maxColl = (size_t)(cap * (double)N);
  sim.setup();
  sim.getNbrList().putInBox();
  sim.settEnd(tEnd);

  const double dt(2e-3 * tEnd);
  double t(0.0);
  bool cont(true), capped(false);
  auto t0 = std::chrono::steady_clock::now();
  try {
    while (cont) {
      cont = sim.step(t);
      t += dt;
    }
  } catch (const CapReached &) {
    capped = true;
  }
  auto t1 = std::chrono::steady_clock::now();
  double wall = std::chrono::duration<double>(t1 - t0).count();

  double tsim = sim.getTime();
  double r = std::min(tsim, tEnd) / tEnd;
  double phiFinal = phi * r * r * r;
  size_t nc = sim.numCollisions();
  std::printf("{\"N\": %zu, \"seed\": %u, \"target_phi\": %.10g, \"growth_rate\": %.10g, "
              "\"cap_per_particle\": %.10g, \"reached\": %s, \"phi_final\": %.10g, "
              "\"t_sim\": %.10g, \"n_collisions\": %zu, \"wall_time_s\": %.6f, "
              "\"collisions_per_s\": %.6g}\n",
              N, seed, phi, growrate, cap, (!capped && !cont) ? "true" : "false",
              phiFinal, tsim, nc, wall, wall > 0 ? nc / wall : 0.0);
  return 0;
}
