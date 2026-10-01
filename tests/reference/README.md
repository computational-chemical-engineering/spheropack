# Reference data from Peters and de With, Phys. Rev. E 85, 026703 (2012)

- `GofR_LJ_all.dat`: radial distribution function of the truncated and shifted
  Lennard-Jones fluid, N = 1000 (999 for the rejection-free run), density 0.317,
  T = 1.085, cutoff 2.5 sigma; bins of width 0.005. Column pairs (r, g): three long
  Metropolis runs (columns 1-6), the rejection-free method (7-8) and the straight
  event-chain variant (9-10). The header lines give the run details.
- `GofR_DPD_110925.dat`: radial distribution function of a DPD liquid, N = 375 in a
  periodic cube of edge 5 (density 3), a = 25 kT, cutoff 1; bins of width 0.0125.

These are the data behind the figures of the paper; the tests and the documentation
notebook compare spheropack's implementation with them.
