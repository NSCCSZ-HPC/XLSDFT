#include <arm_sve.h>
#include <omp.h>

namespace axpy_opt {

#ifndef NT
#define NT 38
#endif

constexpr int B_TILE = 16;

inline size_t k_out(const size_t K) {
  return K + ((NT * 16 - K % (NT * 16)) % (NT * 16));
}

inline size_t KLOCAL(const size_t K) { return k_out(K) / NT; }

template <int OutTile>
static inline double* out_ptr(double* base, const size_t g, const size_t bt,
                              const size_t grids, const size_t nb,
                              const size_t KLOCAL) {
  if constexpr (OutTile == 16) {
    return base + bt * k_out(grids) * B_TILE + g * B_TILE;
  } else {
    const size_t tid = g % NT;
    const size_t ploc = g / NT;
    return base + tid * (nb / 16) * KLOCAL * 16 + bt * KLOCAL * 16 + ploc * 16;
  }
}

inline const double* ptr_tile_16(const double* base, const size_t g,
                                 const size_t bt, const size_t K) {
  return base + bt * k_out(K) * B_TILE + g * B_TILE;
}

inline double* ptr_tile_16(double* base, const size_t g, const size_t bt,
                           const size_t K) {
  return const_cast<double*>(
      ptr_tile_16(static_cast<const double*>(base), g, bt, K));
}

}  // namespace axpy_opt

void cheb_rec_scale(double* __restrict__ packed, const double a, const size_t M,
                    const size_t K) {
  const size_t n = M * K;
  const svbool_t ptrue = svptrue_b64();
  const svfloat64_t va = svdup_f64(a);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i += 8) {
    svstnt1(ptrue, packed + i, svmul_x(ptrue, svld1(ptrue, packed + i), va));
  }
}

template <int OutTile>
void cheb_rec_axpy(double* __restrict__ y_packed, const double a,
                   const double* __restrict__ z_packed, const double b,
                   const size_t M, const size_t K) {
  const svbool_t ptrue = svptrue_b64();
  const svfloat64_t va = svdup_f64(a);
  const svfloat64_t vb = svdup_f64(b);

  static_assert(OutTile == 16 || OutTile == -16);

  if constexpr (OutTile == 16) {
    const size_t grids = K;
    const size_t band_tiles = M / axpy_opt::B_TILE;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t bt = 0; bt < band_tiles; ++bt) {
      for (size_t g = 0; g < grids; ++g) {
        double* y_row =
            axpy_opt::out_ptr<OutTile>(y_packed, g, bt, grids, M, 0);
        const double* z_row = axpy_opt::ptr_tile_16(z_packed, g, bt, K);

        const svfloat64_t z0 = svld1(ptrue, z_row);
        const svfloat64_t z1 = svld1(ptrue, z_row + 8);
        const svfloat64_t y0 = svld1(ptrue, y_row);
        const svfloat64_t y1 = svld1(ptrue, y_row + 8);

        svstnt1(ptrue, y_row, svmls_x(ptrue, svmul_x(ptrue, y0, va), z0, vb));
        svstnt1(ptrue, y_row + 8,
                svmls_x(ptrue, svmul_x(ptrue, y1, va), z1, vb));
      }
    }
  } else {
    const size_t grids = K;
    const size_t band_tiles = M / axpy_opt::B_TILE;
    const size_t kl = axpy_opt::KLOCAL(K);

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t bt = 0; bt < band_tiles; ++bt) {
      for (size_t g = 0; g < grids; ++g) {
        double* y_row =
            axpy_opt::out_ptr<OutTile>(y_packed, g, bt, grids, M, kl);
        const double* z_row = axpy_opt::ptr_tile_16(z_packed, g, bt, K);

        const svfloat64_t z0 = svld1(ptrue, z_row);
        const svfloat64_t z1 = svld1(ptrue, z_row + 8);
        const svfloat64_t y0 = svld1(ptrue, y_row);
        const svfloat64_t y1 = svld1(ptrue, y_row + 8);

        svstnt1(ptrue, y_row, svmls_x(ptrue, svmul_x(ptrue, y0, va), z0, vb));
        svstnt1(ptrue, y_row + 8,
                svmls_x(ptrue, svmul_x(ptrue, y1, va), z1, vb));
      }
    }
  }
}

template void cheb_rec_axpy<16>(double* __restrict__ y_packed, const double a,
                                const double* __restrict__ z_packed,
                                const double b, const size_t M, const size_t K);

template void cheb_rec_axpy<-16>(double* __restrict__ y_packed, const double a,
                                 const double* __restrict__ z_packed,
                                 const double b, const size_t M, const size_t K);
