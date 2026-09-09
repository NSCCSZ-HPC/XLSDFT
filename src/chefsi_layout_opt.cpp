#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "chefsi_layout.h"
#include "sme.hpp"

namespace chefsi_layout {

namespace {

#if !defined(__ARM_FEATURE_SME)
constexpr int B_TILE = 16;
#endif

#if defined(__ARM_FEATURE_SME)

__arm_locally_streaming __arm_new("za") void untile_16_sme_worker(
    const size_t M, const size_t K, const size_t K_OUT,
    double* __restrict__ out, const double* __restrict__ in,
    const size_t worker, const size_t nworkers) {
  constexpr size_t B = 16;

  const size_t band_tiles = M / B;
  const size_t grid_blks = (K + B - 1) / B;
  const size_t ntasks = band_tiles * grid_blks;

  const svbool_t pg_all = svptrue_b64();

  for (size_t task = worker; task < ntasks; task += nworkers) {
    const size_t bt = task / grid_blks;
    const size_t gb = task % grid_blks;

    const size_t b0 = bt * B;
    const size_t g0 = gb * B;

    const double* src = in + bt * K_OUT * B + g0 * B;

#pragma clang loop unroll(full)
    for (uint32_t s = 0; s < 8; ++s) {
      const double* r0 = src + s * B;
      const double* r1 = src + (s + 8) * B;

      svld1_hor_za64(0, s, pg_all, r0);
      svld1_hor_za64(1, s, pg_all, r0 + 8);
      svld1_hor_za64(4, s, pg_all, r1);
      svld1_hor_za64(5, s, pg_all, r1 + 8);
    }

    const size_t rem = K - g0;
    const size_t n0 = std::min<size_t>(rem, 8);
    const size_t n1 = rem > 8 ? std::min<size_t>(rem - 8, 8) : 0;

    const svbool_t pg0 = svwhilelt_b64(uint64_t{0}, static_cast<uint64_t>(n0));
    const svbool_t pg1 = svwhilelt_b64(uint64_t{0}, static_cast<uint64_t>(n1));

#pragma clang loop unroll(full)
    for (uint32_t s = 0; s < 8; ++s) {
      double* dst0 = out + (b0 + s) * K + g0;
      double* dst1 = out + (b0 + s + 8) * K + g0;

      // bands b0+s
      svst1_ver_za64(0, s, pg0, dst0);
      svst1_ver_za64(4, s, pg1, dst0 + 8);

      // bands b0+s+8
      svst1_ver_za64(1, s, pg0, dst1);
      svst1_ver_za64(5, s, pg1, dst1 + 8);
    }
  }
}

#else

void untile_16_opt_scalar(const size_t M, const size_t K,
                          double* __restrict__ out,
                          const double* __restrict__ in) {
  const size_t K_OUT = k_out(K);
  const size_t band_tiles = M / B_TILE;
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t bt = 0; bt < band_tiles; ++bt) {
    for (size_t g = 0; g < K; ++g) {
      const double* src = in + bt * K_OUT * B_TILE + g * B_TILE;
      for (size_t bl = 0; bl < B_TILE; ++bl) {
        const size_t b = bt * B_TILE + bl;
        out[g + b * K] = src[bl];
      }
    }
  }
}

#endif

}  // namespace

void untile_16_opt(const size_t M, const size_t K, double* __restrict__ out,
                   const double* __restrict__ in) {
#if defined(__ARM_FEATURE_SME)
  const size_t K_OUT = k_out(K);
#pragma omp parallel
  {
    const size_t worker = omp_get_thread_num();
    const size_t nworkers = omp_get_num_threads();
    untile_16_sme_worker(M, K, K_OUT, out, in, worker, nworkers);
  }
#else
  untile_16_opt_scalar(M, K, out, in);
#endif
}

void untile_16_production(const size_t M, const size_t K,
                          double* __restrict__ out,
                          const double* __restrict__ in) {
  if (std::getenv("CHEFSI_UNTILE_REF") != nullptr) {
    untile_16_ref(M, K, out, in);
  } else {
    untile_16_opt(M, K, out, in);
  }
}

}  // namespace chefsi_layout
