#include <doctest/doctest.h>

#include <cmath>

#include "spheropack/predict.hpp"

using spheropack::first_contact;
using spheropack::first_contact_linear;
using spheropack::never;

namespace {
// Gap function of two spheres on a line: distance r0 + u t, contact distance s0 + ds t.
struct Pair1D {
  double r0, u, s0, ds;
  double A() const { return u * u - ds * ds; }
  double B() const { return r0 * u - s0 * ds; }
  double C() const { return r0 * r0 - s0 * s0; }
};
}  // namespace

TEST_CASE("approaching fixed spheres touch when the gap closes") {
  const Pair1D p{3.0, -1.0, 1.0, 0.0};
  CHECK(first_contact(p.A(), p.B(), p.C()) == doctest::Approx(2.0));
}

TEST_CASE("receding or passing spheres never touch") {
  const Pair1D p{3.0, 1.0, 1.0, 0.0};
  CHECK(first_contact(p.A(), p.B(), p.C()) == never);
  // Passing by in 2D: offset 2 > contact distance 1, A > 0, B < 0, disc < 0.
  CHECK(first_contact(1.0, -3.0, 9.0 + 4.0 - 1.0) == never);
}

TEST_CASE("growth alone brings resting spheres into contact") {
  const Pair1D p{3.0, 0.0, 1.0, 0.5};  // contact distance reaches 3 at t = 4
  CHECK(first_contact(p.A(), p.B(), p.C()) == doctest::Approx(4.0));
}

TEST_CASE("growth faster than separation catches up with receding spheres") {
  const Pair1D p{2.0, 0.25, 1.0, 0.5};  // 2 + 0.25 t = 1 + 0.5 t at t = 4
  CHECK(first_contact(p.A(), p.B(), p.C()) == doctest::Approx(4.0));
}

TEST_CASE("contact now: collide if closing, otherwise wait") {
  CHECK(first_contact(1.0, -1.0, 0.0) == 0.0);
  CHECK(first_contact(1.0, -1.0, -1e-14) == 0.0);  // round-off overlap, closing
  CHECK(first_contact(1.0, 1.0, 0.0) == never);    // opening, no growth
  // In contact and separating faster than the contact distance grows: never again.
  const Pair1D p{1.0, 0.501, 1.0, 0.5};
  CHECK(first_contact(p.A(), p.B(), p.C()) == never);
}

TEST_CASE("round-off overlap that is closing collides now even when disc < 0") {
  // A < 0, B < 0, C < 0, B^2 - A C < 0: overlapping and closing.
  CHECK(first_contact(-1.0, -0.5, -1.0) == 0.0);
}

TEST_CASE("round-off overlap that never opens collides when it starts closing") {
  // A < 0, B >= 0, C < 0 with disc < 0.
  const double t = first_contact(-1.0, 1e-9, -1e-12);
  CHECK(t == doctest::Approx(1e-9));
}

TEST_CASE("no catastrophic cancellation for near-grazing growth") {
  // A is tiny: relative speed equals the growth speed up to 1e-12.
  const Pair1D p{2.0, -0.5 - 1e-12, 1.0, 0.5};  // gap closes at 1 + 1e-12 per time
  const double t = first_contact(p.A(), p.B(), p.C());
  CHECK(t == doctest::Approx(1.0 / (1.0 + 1e-12)).epsilon(1e-12));
}

TEST_CASE("flat wall contact") {
  CHECK(first_contact_linear(1.0, -2.0) == doctest::Approx(0.5));
  CHECK(first_contact_linear(1.0, 0.0) == never);
  CHECK(first_contact_linear(-1e-15, -1.0) == 0.0);
}
