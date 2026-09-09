#include "sr_gemm_opt.h"

#include <omp.h>

#include <algorithm>

#include "sme.hpp"

namespace sr_gemm_opt {

/*
A (K x M) x B (M x M) = C (K x M)
Input layouts:
- A: (M x K) packed in tile_16 grid-tiled
- B: (M x M)
- C: (K x M)
Constraints:
- NT = 38
- (K / NT / 16) % BM == 0
- (M / 32) % BN == 0
- M % BK == 0
- BK % 16 == 0
- Full sizes:
  - K = 200704 padded to 201248
  - M = 640
- Best performance:
  - BM = 343
  - BN = 5
  - BK = 160
  - No PRFDIST
*/

#ifndef NT
#define NT 38
#endif

#ifndef BM
#define BM 343
#endif
#ifndef BN
#define BN 5
#endif
#ifndef BK
#define BK 160
#endif

inline void svld1_f64_2x4_tile16(
    const double* __restrict__ lo,
    const double* __restrict__ hi) __arm_streaming __arm_out("za") {
  const svbool_t pg = svptrue_b64();

#pragma clang loop unroll(full)
  for (uint32_t s = 0; s < 8; ++s) {
    const double* lo0 = lo + s * 16;
    const double* hi0 = hi + s * 16;
    svld1_hor_za64(0, s, pg, lo0 + 0);
    svld1_hor_za64(1, s, pg, lo0 + 8);
    svld1_hor_za64(2, s, pg, hi0 + 0);
    svld1_hor_za64(3, s, pg, hi0 + 8);
    const double* lo1 = lo + (s + 8) * 16;
    const double* hi1 = hi + (s + 8) * 16;
    svld1_hor_za64(4, s, pg, lo1 + 0);
    svld1_hor_za64(5, s, pg, lo1 + 8);
    svld1_hor_za64(6, s, pg, hi1 + 0);
    svld1_hor_za64(7, s, pg, hi1 + 8);
  }
}

inline void svst1_f64_2x4_tile16(
    double* __restrict__ lo,
    double* __restrict__ hi) __arm_streaming __arm_preserves("za") {
  const svbool_t pg = svptrue_b64();

#pragma clang loop unroll(full)
  for (uint32_t s = 0; s < 8; ++s) {
    double* lo0 = lo + s * 16;
    double* hi0 = hi + s * 16;
    svst1_hor_za64(0, s, pg, lo0 + 0);
    svst1_hor_za64(1, s, pg, lo0 + 8);
    svst1_hor_za64(2, s, pg, hi0 + 0);
    svst1_hor_za64(3, s, pg, hi0 + 8);
    double* lo1 = lo + (s + 8) * 16;
    double* hi1 = hi + (s + 8) * 16;
    svst1_hor_za64(4, s, pg, lo1 + 0);
    svst1_hor_za64(5, s, pg, lo1 + 8);
    svst1_hor_za64(6, s, pg, hi1 + 0);
    svst1_hor_za64(7, s, pg, hi1 + 8);
  }
}

__arm_locally_streaming __arm_new("za") void gemm_sr(
    const size_t K_local, const size_t K_global, const size_t grid_base,
    const size_t M, uint8_t* __restrict__ c, uint8_t* __restrict__ a,
    uint8_t* __restrict__ b) {
  double* c64 = reinterpret_cast<double*>(c);

  for (size_t _i2_ = 0; _i2_ < (M + BK - 1) / BK; ++_i2_) {
    for (size_t _i1_ = 0; _i1_ < ((M / 32) / BN); ++_i1_) {
      for (size_t _i0_ = 0; _i0_ < ((K_local / 16) + BM - 1) / BM; ++_i0_) {
        const size_t cur_BM = std::min((size_t)BM, (K_local / 16) - _i0_ * BM);
        for (size_t _i3_ = 0; _i3_ < cur_BM; ++_i3_) {
          const size_t grid = grid_base + 16 * (_i0_ * BM + _i3_);
          for (size_t _i4_ = 0; _i4_ < BN; ++_i4_) {
            const size_t col32 = _i1_ * BN + _i4_;
            const size_t band16_lo = 2 * col32;
            const size_t band16_hi = band16_lo + 1;
            double* c_lo = c64 + band16_lo * K_global * 16 + grid * 16;
            double* c_hi = c64 + band16_hi * K_global * 16 + grid * 16;
            if (_i2_ > 0) {
              svld1_f64_2x4_tile16(c_lo, c_hi);
            } else {
              svzero_za();
            }
            const size_t max_K = std::min((size_t)BK, M - _i2_ * BK);
            const uint8_t* a_base = a + 128 * (BK * _i2_) + 128 * (M * _i3_) +
                                    128 * (BM * M * _i0_);
            const uint8_t* b_base =
                b + 256 * _i4_ + 256 * (BN * _i1_) + 8 * (BK * M * _i2_);
#pragma clang loop unroll_count(4)
            for (size_t _i7_ = 0; _i7_ < max_K; ++_i7_) {
              fmopa_f64_2x4(reinterpret_cast<const double*>(a_base),
                            reinterpret_cast<const double*>(b_base));
              a_base += 128;
              b_base += 8 * M;
            }
            svst1_f64_2x4_tile16(c_lo, c_hi);
          }
        }
      }
    }
  }
}

}  // namespace sr_gemm_opt

void sr_gemm(const int M, const int K, double* c, double* a, double* b) {
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    const int SK = K / nt;
    const int SK_start = tid * SK;
    sr_gemm_opt::gemm_sr(SK, K, SK_start, M, reinterpret_cast<uint8_t*>(c),
                         reinterpret_cast<uint8_t*>(a + SK_start * M),
                         reinterpret_cast<uint8_t*>(b));
  }
}
