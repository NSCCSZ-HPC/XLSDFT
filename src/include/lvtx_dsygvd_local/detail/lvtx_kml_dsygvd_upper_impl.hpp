#ifndef LVTX_BLAS_DETAIL_KML_DSYGVD_UPPER_IMPL_HPP_
#define LVTX_BLAS_DETAIL_KML_DSYGVD_UPPER_IMPL_HPP_

#include "lvtx_kml_dsygvd_detail.hpp"
#include "../lvtx_dsygvd_work_doubles.hpp"

#include <cstddef>
#include <omp.h>

namespace lvtx_blas_detail {
namespace kml_dsygvd {

inline int dsygvd_upper_impl(
    const int n, double* const a, const int lda, double* const b, const int ldb,
    double* const w, double* const work, const std::size_t work_doubles,
    const int nthreads,
    lvtx_dsygvd_phase_times_f64* const phase_times,
    const int production_tile) noexcept {
    if (n < 0 || n > kMaxOrder || nthreads < 0 || nthreads > kMaxThreads) {
        return kStatusDim;
    }
    if (!valid_production_tile(production_tile)) {
        return kStatusDim;
    }
    if (phase_times != nullptr && !aligned_64(phase_times)) {
        return kStatusAlign;
    }
    if (n == 0) {
        if (phase_times != nullptr) {
            *phase_times = {};
        }
        return kStatusOk;
    }
    if (a == nullptr || b == nullptr || w == nullptr || work == nullptr) {
        return kStatusNull;
    }
    if (lda < n || ldb < n) {
        return kStatusLd;
    }
    if (!aligned_64(a) || !aligned_64(b) || !aligned_64(w) ||
        !aligned_64(work)) {
        return kStatusAlign;
    }

    const std::size_t required = lvtx_dsygvd_work_doubles(n);
    if (work_doubles < required) {
        return kStatusWorkspace;
    }
    AddressRange a_range{};
    AddressRange b_range{};
    AddressRange w_range{};
    AddressRange work_range{};
    AddressRange phase_range{};
    if (!make_range(a, static_cast<std::size_t>(lda) *
                           static_cast<std::size_t>(n),
                    &a_range) ||
        !make_range(b, static_cast<std::size_t>(ldb) *
                           static_cast<std::size_t>(n),
                    &b_range) ||
        !make_range(w, static_cast<std::size_t>(n), &w_range) ||
        !make_range(work, work_doubles, &work_range) ||
        (phase_times != nullptr &&
         !make_range(reinterpret_cast<const double*>(phase_times), 10U,
                     &phase_range))) {
        return kStatusDim;
    }
    if (overlaps(a_range, b_range) || overlaps(a_range, w_range) ||
        overlaps(a_range, work_range) || overlaps(b_range, w_range) ||
        overlaps(b_range, work_range) || overlaps(w_range, work_range)) {
        return kStatusOverlap;
    }
    if (phase_times != nullptr &&
        (overlaps(phase_range, a_range) || overlaps(phase_range, b_range) ||
         overlaps(phase_range, w_range) ||
         overlaps(phase_range, work_range))) {
        return kStatusOverlap;
    }
    if (omp_in_parallel() != 0) {
        return kStatusNestedOmp;
    }
    if (!finite_upper_triangles(n, a, lda, b, ldb)) {
        return kStatusNonfinite;
    }

    double total_start = 0.0;
    if (phase_times != nullptr) {
        *phase_times = {};
        total_start = omp_get_wtime();
    }
    const auto finish = [phase_times, total_start](const int status) noexcept {
        if (phase_times != nullptr) {
            phase_times->total_seconds = omp_get_wtime() - total_start;
        }
        return status;
    };

    const int threads = nthreads == 0 ? kDefaultThreads : nthreads;
    const int tile = production_tile;
    double* const standard = work;
    const std::size_t matrix_doubles =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    double* const diagonal = standard + matrix_doubles;
    double* const off_diagonal = diagonal + n;
    double* const tau = off_diagonal + n;
    double* const temporary = tau + n;

    double phase_start =
        phase_times != nullptr ? omp_get_wtime() : 0.0;
    const int cholesky_status = upper_cholesky(n, b, ldb);
    if (phase_times != nullptr) {
        phase_times->cholesky_seconds = omp_get_wtime() - phase_start;
    }
    if (cholesky_status != kStatusOk) {
        return finish(cholesky_status);
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    if (!form_standard_problem(n, a, lda, b, ldb, standard, threads)) {
        return finish(kStatusNonfinite);
    }
    if (phase_times != nullptr) {
        phase_times->standard_transform_seconds =
            omp_get_wtime() - phase_start;
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    if (!reduce_to_tridiagonal(n, standard, diagonal, off_diagonal, tau,
                               temporary, threads)) {
        return finish(kStatusNonfinite);
    }
    if (phase_times != nullptr) {
        phase_times->tridiagonal_reduction_seconds =
            omp_get_wtime() - phase_start;
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    form_householder_product(n, standard, tau, a, lda, threads);
    if (phase_times != nullptr) {
        phase_times->householder_product_seconds =
            omp_get_wtime() - phase_start;
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    const int eigen_status =
        n == kMaxOrder
            ? tridiagonal_ql_640_parallel(diagonal, off_diagonal, a, lda,
                                          standard, threads, tile)
            : tridiagonal_ql(n, diagonal, off_diagonal, a, lda);
    if (phase_times != nullptr) {
        phase_times->tridiagonal_eigensolve_seconds =
            omp_get_wtime() - phase_start;
    }
    if (eigen_status != kStatusOk) {
        return finish(eigen_status);
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    if (!generalized_backsolve(n, a, lda, b, ldb, threads)) {
        return finish(kStatusNonfinite);
    }
    if (phase_times != nullptr) {
        phase_times->backtransform_seconds = omp_get_wtime() - phase_start;
    }
    for (int index = 0; index < n; ++index) {
        w[index] = diagonal[index];
    }
    return finish(kStatusOk);
}

}  // namespace kml_dsygvd
}  // namespace lvtx_blas_detail

#endif  // LVTX_BLAS_DETAIL_KML_DSYGVD_UPPER_IMPL_HPP_
