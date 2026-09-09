#include "sp_syrk_opt.h"

#include <omp.h>

#include <algorithm>

#include "inst_sve.hpp"
#include "sme.hpp"

namespace sp_syrk_opt {

/*
A (M x K) = C (M x M)
Input layouts:
- A: (K x M) packed in tile_16t band-tiled grid-tiled for threads
- C: (M x M)
Constraints:
- NT = 32
- (M / 16) % BM == 0
- (M / 32) % BN == 0
- (K / NT) % BK == 0
- K % NT == 0
- Full sizes:
  - K = 200704 padded to 200716
  - M = 640
- Best performance:
  - BM = 40
  - BN = 5
  - BK = 256
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

#define SYRK_PRF_HINT SV_PLDL2KEEP

// Cluster-only: -DCHEFSI_SWPF4 in Makefile.config.LX2 (not armie64/clang++-22).
#ifdef CHEFSI_SWPF4
#define SWPF4
#endif

inline void fmopa_f64_2x4_split(
    const double* a, const double* blo,
    const double* bhi) __arm_streaming __arm_inout("za") {
  svfloat64_t a0 = svld1_vnum(svptrue_b64(), a, 0);
  svfloat64_t a1 = svld1_vnum(svptrue_b64(), a, 1);
  svfloat64_t b0 = svld1_vnum(svptrue_b64(), blo, 0);
  svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), a0, b0);
  svmopa_za64_f64_m(4, svptrue_b64(), svptrue_b64(), a1, b0);

  svfloat64_t b1 = svld1_vnum(svptrue_b64(), blo, 1);
  svmopa_za64_f64_m(1, svptrue_b64(), svptrue_b64(), a0, b1);
  svmopa_za64_f64_m(5, svptrue_b64(), svptrue_b64(), a1, b1);

  svfloat64_t b2 = svld1_vnum(svptrue_b64(), bhi, 0);
  svmopa_za64_f64_m(2, svptrue_b64(), svptrue_b64(), a0, b2);
  svmopa_za64_f64_m(6, svptrue_b64(), svptrue_b64(), a1, b2);

  svfloat64_t b3 = svld1_vnum(svptrue_b64(), bhi, 1);
  svmopa_za64_f64_m(3, svptrue_b64(), svptrue_b64(), a0, b3);
  svmopa_za64_f64_m(7, svptrue_b64(), svptrue_b64(), a1, b3);
}

__arm_locally_streaming __arm_new("za") void syrk_spt(const size_t M,
                                                      const size_t K,
                                                      uint8_t* __restrict__ c,
                                                      uint8_t* __restrict__ a) {
  const size_t num_k_blocks = (K + BK - 1) / BK;

#ifdef SWPF4
  alignas(64) const uint32_t prfb_array[] = {0,
                                             16,
                                             32,
                                             48,
                                             64,
                                             80,
                                             96,
                                             112,
                                             (uint32_t)(32 * K + 0),
                                             (uint32_t)(32 * K + 16),
                                             (uint32_t)(32 * K + 32),
                                             (uint32_t)(32 * K + 48),
                                             (uint32_t)(32 * K + 64),
                                             (uint32_t)(32 * K + 80),
                                             (uint32_t)(32 * K + 96),
                                             (uint32_t)(32 * K + 112)};
  const svuint32_t prfb_ind = svld1_u32(svptrue_b32(), prfb_array);
  const svuint32_t prfa_ind = svindex_u32(0, 16);
#endif

  for (size_t kb = 0; kb < num_k_blocks; ++kb) {
    const size_t k0 = kb * BK;
    const size_t max_K = std::min((size_t)BK, K - k0);
    const bool final_k_block = (kb + 1 == num_k_blocks);

    for (size_t bn_outer = 0; bn_outer < ((M / 32) / BN); ++bn_outer) {
      for (size_t bm_outer = 0; bm_outer < ((M / 16) / BM); ++bm_outer) {
        for (size_t bm = 0; bm < BM; ++bm) {
          const size_t row16 = bm_outer * BM + bm;

          for (size_t bn = 0; bn < BN; ++bn) {
            const size_t col32 = bn_outer * BN + bn;
            if (row16 > 2 * col32 + 1) {
              continue;
            }
            uint8_t* c_ptr = c + 128 * M * row16 + 256 * col32;
            if (kb > 0) {
              svld1_f64_2x4(reinterpret_cast<const double*>(c_ptr), M);
            } else {
              svzero_za();
            }

#pragma clang loop unroll_count(4)
            for (size_t kk = 0; kk < max_K; ++kk) {
              const size_t k = k0 + kk;
              const uint8_t* in0_ptr = a + 128 * k + 128 * K * row16;
              const uint8_t* in1_ptr = a + 128 * k + 256 * K * col32;
              const uint8_t* in2_ptr = in1_ptr + 128 * K;
#ifdef SWPF4
              if (((kk & 7) == 0) && kk + PRFDIST + 7 < max_K) {
                const uint8_t* prfa_ptr = in0_ptr + PRFDIST * 128;
                const uint8_t* prfb_ptr = in1_ptr + PRFDIST * 128;
                svprfw_gather_index(svptrue_b32(), prfa_ptr, prfa_ind,
                                    SYRK_PRF_HINT);
                svprfw_gather_index(svptrue_b32(), prfb_ptr, prfb_ind,
                                    SYRK_PRF_HINT);
              } else if (((kk & 7) == 4) && kk + PRFDIST + 3 < max_K) {
                const uint8_t* prfb_ptr = in1_ptr + PRFDIST * 128;
                svprfw_gather_index(svptrue_b32(), prfb_ptr, prfb_ind,
                                    SYRK_PRF_HINT);
              }
#endif
              fmopa_f64_2x4_split(reinterpret_cast<const double*>(in0_ptr),
                                  reinterpret_cast<const double*>(in1_ptr),
                                  reinterpret_cast<const double*>(in2_ptr));
            }
            svst1_f64_2x4(reinterpret_cast<double*>(c_ptr), M);
            if (final_k_block && row16 < 2 * col32) {
              uint8_t* mirror_ptr = c + 256 * M * col32 + 128 * row16;
              svst1_f64_4x2_ver(reinterpret_cast<double*>(mirror_ptr), M);
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

}  // namespace sp_syrk_opt

void sp_syrk(const int M, const int K, double* c, double* c_nt, double* a) {
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int SK = (K / omp_get_num_threads());
    const int SK_start = tid * SK;
    const int SK_end = (tid + 1) * SK;
    sp_syrk_opt::syrk_spt(M, SK_end - SK_start, (uint8_t*)(c_nt + tid * M * M),
                          (uint8_t*)(a + SK_start * M));
  }
  sp_syrk_opt::reduce_streamk(M, (uint8_t*)c, (uint8_t*)c_nt);
}
