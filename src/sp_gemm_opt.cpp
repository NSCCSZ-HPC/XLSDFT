#include "sp_gemm_opt.h"

#include <omp.h>

#include <algorithm>

#include "inst_sve.hpp"
#include "sme.hpp"

namespace sp_gemm_opt {

/*
A (M x K) x B (K x M) = C (M x M)
Input layouts:
- A: (K x M) packed in tile_16t band-tiled grid-tiled for threads
- B: (K x M) packed in tile_32t band-tiled grid-tiled for threads
- C: (M x M)
Constraints:
- NT = 32
- (M / 16) % BM == 0
- (M / 16) % BN == 0
- (K / NT) % BK == 0
- K % NT == 0
- Full sizes:
  - K = 200704 padded to 200716
  - M = 640
- Best performance:
  - BM = 40
  - BN = 5
  - BK = 343
  - PRFDIST = 24
*/

#ifndef NT
#define NT 38
#endif

#ifndef BM
#define BM 40
#endif
#ifndef BN
#define BN 5
#endif
#ifndef BK
#define BK 343
#endif

#ifndef PRFDIST
#define PRFDIST 24
#endif

// Cluster-only: -DCHEFSI_SWPF4 in Makefile.config.LX2 (not armie64/clang++-22).
#ifdef CHEFSI_SWPF4
#define SWPF4
#endif

__arm_locally_streaming __arm_new("za") void gemm_spt(const size_t M,
                                                      const size_t K,
                                                      uint8_t* __restrict__ c,
                                                      uint8_t* __restrict__ a,
                                                      uint8_t* __restrict__ b) {
#ifdef SWPF4
  const svuint32_t idx = svindex_u32(0, 16);
#endif
  for (size_t _i0_ = 0; _i0_ < (K + BK - 1) / BK; _i0_ += 1) {
    for (size_t _i2_ = 0; _i2_ < ((M / 32) / BN); _i2_ += 1) {
      for (size_t _i1_ = 0; _i1_ < ((M / 16) / BM); _i1_ += 1) {
        for (size_t _i3_ = 0; _i3_ < BM; _i3_ += 1) {
          for (size_t _i4_ = 0; _i4_ < BN; _i4_ += 1) {
            if (_i0_ > 0) {
              uint8_t const* _in0_ptr =
                  c +
                  ((((256 * _i4_) + (256 * (BN * _i2_))) + (128 * (M * _i3_))) +
                   (128 * ((BM * M) * _i1_)));
              svld1_f64_2x4(reinterpret_cast<const double*>(_in0_ptr), M);
            } else {
              svzero_za();
            }
            const size_t max_K = std::min((size_t)BK, K - _i0_ * BK);
            uint8_t const* a_base =
                a + (((128 * (BK * _i0_)) + (128 * (K * _i3_))) +
                     (128 * ((BM * K) * _i1_)));
            uint8_t const* b_base =
                b + (((256 * (BK * _i0_)) + (256 * (K * _i4_))) +
                     (256 * ((BN * K) * _i2_)));
#pragma clang loop unroll_count(4)
            for (size_t _i7_ = 0; _i7_ < max_K; _i7_ += 1) {
              {
#ifdef SWPF4
                if ((_i7_ & 7) == 0 && _i7_ + PRFDIST + 7 < max_K) {
                  uint8_t const* prfa_ptr = a_base + PRFDIST * 128;
                  uint8_t const* prfb_ptr = b_base + PRFDIST * 256;
                  svprfw_gather_index(svptrue_b32(), prfa_ptr, idx,
                                      SV_PLDL1STRM);
                  svprfw_gather_index(svptrue_b32(), prfb_ptr, idx,
                                      SV_PLDL1STRM);
                } else if ((_i7_ & 7) == 4 && _i7_ + PRFDIST + 7 < max_K) {
                  uint8_t const* prfb_ptr = b_base + PRFDIST * 256;
                  svprfw_gather_index(svptrue_b32(), prfb_ptr, idx,
                                      SV_PLDL1STRM);
                }
#endif
                fmopa_f64_2x4(reinterpret_cast<const double*>(a_base),
                              reinterpret_cast<const double*>(b_base));
                a_base += 128;
                b_base += 256;
              }
            }
            {
              uint8_t* _out_ptr =
                  c +
                  ((((256 * _i4_) + (256 * (BN * _i2_))) + (128 * (M * _i3_))) +
                   (128 * ((BM * M) * _i1_)));
              svst1_f64_2x4(reinterpret_cast<double*>(_out_ptr), M);
            }
          }
        }
      }
    }
  }
}

inline void reduce_streamk(const size_t M, uint8_t* __restrict__ d,
                           uint8_t* __restrict__ a) {
#pragma omp parallel for collapse(2)
  for (size_t _i0_ = 0; _i0_ < M; _i0_ += 1) {
    for (size_t _i1_ = 0; _i1_ < M; _i1_ += 8) {
      uint8_t* _out_ptr = d + ((8 * _i1_) + (8 * (M * _i0_)));
      uint8_t const* _in0_ptr =
          a + (((8 * _i1_) + (8 * (M * _i0_))) + (8 * ((M * M) * 0)));
      vmov_f64(reinterpret_cast<double*>(_out_ptr),
               reinterpret_cast<const double*>(_in0_ptr));
      for (size_t _i2_ = 1; _i2_ < NT; _i2_ += 1) {
        uint8_t const* _in_ptr =
            a + (((8 * _i1_) + (8 * (M * _i0_))) + (8 * ((M * M) * _i2_)));
        vadd_f64(reinterpret_cast<double*>(_out_ptr),
                 reinterpret_cast<double*>(_out_ptr),
                 reinterpret_cast<const double*>(_in_ptr));
      }
    }
  }
}

}  // namespace sp_gemm_opt

void sp_gemm(const int M, const int K, double* c, double* c_nt, double* a,
             double* b) {
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int SK = (K / omp_get_num_threads());
    const int SK_start = tid * SK;
    const int SK_end = (tid + 1) * SK;
    sp_gemm_opt::gemm_spt(M, SK_end - SK_start, (uint8_t*)(c_nt + tid * M * M),
                          (uint8_t*)(a + SK_start * M),
                          (uint8_t*)(b + SK_start * M));
  }
  sp_gemm_opt::reduce_streamk(M, (uint8_t*)c, (uint8_t*)c_nt);
}
