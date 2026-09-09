#ifndef LVTX_DSYGVD_UPPER_F64_HPP_
#define LVTX_DSYGVD_UPPER_F64_HPP_

#include "detail/lvtx_kml_dsygvd_upper_impl.hpp"
#include "detail_opt/lvtx_kml_dsygvd_upper_impl.hpp"
#include "lvtx_dsygvd_internal_config.hpp"

#include <cstddef>

// Caller-workspace itype=1, jobz='V', uplo='U' generalized
// symmetric-definite eigensolver for n in [0,640]. The upper triangles of a
// and b are authoritative. On success eigenvalues are ascending, a contains
// generalized eigenvectors, and upper(b) is its Cholesky factor.
extern "C" inline int lvtx_dsygvd_upper_f64(
    const int n, double* const a, const int lda, double* const b, const int ldb,
    double* const w, double* const work, const std::size_t work_doubles,
    const int nthreads) noexcept {
    using namespace lvtx_blas_detail;
    if (xlsdft_dsygvd_internal::use_opt_path) {
        return kml_dsygvd_opt::dsygvd_upper_impl(
            n, a, lda, b, ldb, w, work, work_doubles, nthreads, nullptr,
            kml_dsygvd::kProductionTile);
    }
    return kml_dsygvd::dsygvd_upper_impl(n, a, lda, b, ldb, w, work,
                                         work_doubles, nthreads, nullptr,
                                         kml_dsygvd::kProductionTile);
}

// SME-optimized path; used by CheFSI opt diagonalization only.
extern "C" inline int lvtx_dsygvd_upper_f64_opt(
    const int n, double* const a, const int lda, double* const b, const int ldb,
    double* const w, double* const work, const std::size_t work_doubles,
    const int nthreads) noexcept {
    using namespace lvtx_blas_detail;
    return kml_dsygvd_opt::dsygvd_upper_impl(
        n, a, lda, b, ldb, w, work, work_doubles, nthreads, nullptr,
        kml_dsygvd::kProductionTile);
}

#endif  // LVTX_DSYGVD_UPPER_F64_HPP_
