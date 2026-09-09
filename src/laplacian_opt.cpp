#include "laplacian_opt.h"

#include <arm_sve.h>
#include <omp.h>

#include <algorithm>

#include "chefsi_layout.h"

namespace laplacian_opt {

#ifndef MC
#define MC 64
#endif
#ifndef NC
#define NC 4
#endif
#ifndef KC
#define KC 8
#endif

#ifndef RADIUS
#define RADIUS 6
#endif

constexpr int B_TILE = 16;

inline const double* ptr_16(const double* base, const size_t g, const size_t bt,
                            const size_t k_out) {
  return base + bt * k_out * B_TILE + g * B_TILE;
}

template <int OutTile>
static inline double* out_ptr(double* base, const size_t g, const size_t bt,
                              const size_t k_out, const size_t nb,
                              const size_t klocal) {
  if constexpr (OutTile == 16) {
    return base + bt * k_out * B_TILE + g * B_TILE;
  } else if constexpr (OutTile == -16) {
    const size_t tid = g % NT;
    const size_t ploc = g / NT;
    return base + tid * (nb / 16) * klocal * 16 + bt * klocal * 16 + ploc * 16;
  } else {
    const size_t tid = g % NT;
    const size_t ploc = g / NT;
    const size_t bo = bt / 2;
    const size_t half = bt & 1;
    return base + tid * (nb / 32) * klocal * 32 + bo * klocal * 32 + ploc * 32 +
           half * 16;
  }
}

inline void axis_generic_16x2(const double* s_base, const size_t coord,
                              const size_t extent, const size_t stride,
                              const svbool_t ptrue, const double* coeff,
                              svfloat64_t& res0, svfloat64_t& res1) {
#pragma clang loop unroll(full)
  for (size_t r = 1; r <= RADIUS; ++r) {
    const ptrdiff_t off = static_cast<ptrdiff_t>(r * stride) * B_TILE;
    const svfloat64_t c = svdup_f64(coeff[r]);
    if (coord >= r) {
      const double* n = s_base - off;
      res0 = svmla_x(ptrue, res0, svld1(ptrue, n), c);
      res1 = svmla_x(ptrue, res1, svld1(ptrue, n + 8), c);
    }
    if (coord + r < extent) {
      const double* n = s_base + off;
      res0 = svmla_x(ptrue, res0, svld1(ptrue, n), c);
      res1 = svmla_x(ptrue, res1, svld1(ptrue, n + 8), c);
    }
  }
}

}  // namespace laplacian_opt

namespace {

template <int OutTile, bool FusePsiTile16t>
void laplacian_4d_impl(const size_t ni, const size_t nj, const size_t nk,
                       const size_t nb, double* __restrict__ h,
                       const double* __restrict__ psi,
                       const double* __restrict__ vloc,
                       const double* __restrict__ cx,
                       const double* __restrict__ cy,
                       const double* __restrict__ cz, const double coef0,
                       double* __restrict__ psi_tile16t_out) {
  static_assert(OutTile == 16 || OutTile == -16 || OutTile == -32);
  static_assert(!FusePsiTile16t || OutTile == -32);

  const size_t grids = ni * nj * nk;
  const size_t stride_j = ni;
  const size_t stride_k = ni * nj;
  const size_t band_tiles = nb / laplacian_opt::B_TILE;
  const size_t k_out = chefsi_layout::k_out(grids);
  const size_t k_in = chefsi_layout::k_in(grids);
  const size_t klocal = k_in / NT;
  const svbool_t ptrue = svptrue_b64();

#pragma omp parallel
  {
#pragma omp for collapse(3) schedule(static)
    for (size_t bt = 0; bt < band_tiles; ++bt) {
      for (size_t kk = 0; kk < nk; kk += KC) {
        for (size_t jj = 0; jj < nj; jj += NC) {
          const size_t k_end = std::min(kk + KC, nk);
          const size_t j_end = std::min(jj + NC, nj);

          for (size_t ii = 0; ii < ni; ii += MC) {
            const size_t i_end = std::min(ii + MC, ni);

            for (size_t k = kk; k < k_end; ++k) {
              for (size_t j = jj; j < j_end; ++j) {
                for (size_t i = ii; i < i_end; ++i) {
                  const size_t g = i + j * stride_j + k * stride_k;
                  const svfloat64_t sve_A_plus_coef0 =
                      svdup_f64(vloc[g] + coef0);
                  double* d_base = laplacian_opt::out_ptr<OutTile>(
                      h, g, bt, k_out, nb, klocal);
                  const double* s_base =
                      laplacian_opt::ptr_16(psi, g, bt, k_out);

                  const svfloat64_t psi0 = svld1(ptrue, s_base);
                  const svfloat64_t psi1 = svld1(ptrue, s_base + 8);
                  svfloat64_t res0 = svmul_x(ptrue, psi0, sve_A_plus_coef0);
                  svfloat64_t res1 = svmul_x(ptrue, psi1, sve_A_plus_coef0);

                  laplacian_opt::axis_generic_16x2(s_base, i, ni, 1, ptrue, cx,
                                                   res0, res1);
                  laplacian_opt::axis_generic_16x2(s_base, j, nj, stride_j,
                                                   ptrue, cy, res0, res1);
                  laplacian_opt::axis_generic_16x2(s_base, k, nk, stride_k,
                                                   ptrue, cz, res0, res1);

                  svstnt1(ptrue, d_base, res0);
                  svstnt1(ptrue, d_base + 8, res1);

                  if constexpr (FusePsiTile16t) {
                    double* psi_t = laplacian_opt::out_ptr<-16>(
                        psi_tile16t_out, g, bt, k_out, nb, klocal);
                    svstnt1(ptrue, psi_t, psi0);
                    svstnt1(ptrue, psi_t + 8, psi1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

template <int OutTile>
void laplacian_4d(const size_t ni, const size_t nj, const size_t nk,
                  const size_t nb, double* __restrict__ h,
                  const double* __restrict__ psi,
                  const double* __restrict__ vloc,
                  const double* __restrict__ cx, const double* __restrict__ cy,
                  const double* __restrict__ cz, const double coef0,
                  double* __restrict__ psi_tile16t_out) {
  if constexpr (OutTile == -32) {
    if (psi_tile16t_out != nullptr) {
      laplacian_4d_impl<-32, true>(ni, nj, nk, nb, h, psi, vloc, cx, cy, cz,
                                   coef0, psi_tile16t_out);
    } else {
      laplacian_4d_impl<-32, false>(ni, nj, nk, nb, h, psi, vloc, cx, cy, cz,
                                    coef0, psi_tile16t_out);
    }
  } else {
    laplacian_4d_impl<OutTile, false>(ni, nj, nk, nb, h, psi, vloc, cx, cy, cz,
                                      coef0, psi_tile16t_out);
  }
}

template void laplacian_4d<16>(
    const size_t ni, const size_t nj, const size_t nk, const size_t nb,
    double* __restrict__ h, const double* __restrict__ psi,
    const double* __restrict__ vloc, const double* __restrict__ cx,
    const double* __restrict__ cy, const double* __restrict__ cz,
    const double coef0, double* __restrict__ psi_tile16t_out);

template void laplacian_4d<-16>(
    const size_t ni, const size_t nj, const size_t nk, const size_t nb,
    double* __restrict__ h, const double* __restrict__ psi,
    const double* __restrict__ vloc, const double* __restrict__ cx,
    const double* __restrict__ cy, const double* __restrict__ cz,
    const double coef0, double* __restrict__ psi_tile16t_out);

template void laplacian_4d<-32>(
    const size_t ni, const size_t nj, const size_t nk, const size_t nb,
    double* __restrict__ h, const double* __restrict__ psi,
    const double* __restrict__ vloc, const double* __restrict__ cx,
    const double* __restrict__ cy, const double* __restrict__ cz,
    const double coef0, double* __restrict__ psi_tile16t_out);
