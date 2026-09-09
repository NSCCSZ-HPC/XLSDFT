#ifndef LVTX_BLAS_DETAIL_KML_DSYGVD_OPT_KERNELS_HPP_
#define LVTX_BLAS_DETAIL_KML_DSYGVD_OPT_KERNELS_HPP_

// SVE implementations in detail_opt/; scalar ref fallback when SVE unavailable.
#include "../detail/lvtx_kml_dsygvd_detail.hpp"
#include "../lvtx_blas_common.hpp"

#if defined(__ARM_FEATURE_SVE)
#include "lvtx_kml_dsygvd_detail_opt.hpp"
#else

namespace lvtx_blas_detail {
namespace kml_dsygvd_opt {

namespace ref = kml_dsygvd;

inline int upper_cholesky(const int n, double* const b, const int ldb,
                          const int threads) noexcept {
    return ref::upper_cholesky(n, b, ldb);
}

inline bool form_standard_problem(const int n, const double* const a,
                                  const int lda, const double* const b,
                                  const int ldb, double* const standard,
                                  const int threads) noexcept {
    return ref::form_standard_problem(n, a, lda, b, ldb, standard, threads);
}

inline bool reduce_to_tridiagonal(const int n, double* const standard,
                                  double* const diagonal,
                                  double* const off_diagonal, double* const tau,
                                  double* const temporary,
                                  double* const panel_v, double* const panel_w,
                                  lvtx_dsygvd_tridiag_kernel_times_f64*
                                      const kernel_times,
                                  const int threads) noexcept {
    (void)panel_v;
    (void)panel_w;
    (void)kernel_times;
    return ref::reduce_to_tridiagonal(n, standard, diagonal, off_diagonal, tau,
                                    temporary, threads);
}

inline void form_householder_product(const int n, const double* const reflectors,
                                     const double* const tau, double* const vectors,
                                     const int ldv, const int threads) noexcept {
    ref::form_householder_product(n, reflectors, tau, vectors, ldv, threads);
}

inline int tridiagonal_ql(const int n, double* const diagonal,
                          double* const off_diagonal, double* const vectors,
                          const int ldv) noexcept {
    return ref::tridiagonal_ql(n, diagonal, off_diagonal, vectors, ldv);
}

inline int tridiagonal_ql_640_parallel(double* const diagonal,
                                       double* const off_diagonal,
                                       double* const vectors, const int ldv,
                                       double* const workspace,
                                       const int threads,
                                       const int tile) noexcept {
    return ref::tridiagonal_ql_640_parallel(diagonal, off_diagonal, vectors, ldv,
                                            workspace, threads, tile);
}

inline bool generalized_backsolve(const int n, double* const vectors,
                                  const int ldv, const double* const factor,
                                  const int ldf, double* const upper_packed,
                                  const int threads) noexcept {
    return ref::generalized_backsolve(n, vectors, ldv, factor, ldf, threads);
}

}  // namespace kml_dsygvd_opt
}  // namespace lvtx_blas_detail

#endif  // __ARM_FEATURE_SVE

#endif  // LVTX_BLAS_DETAIL_KML_DSYGVD_OPT_KERNELS_HPP_
