#ifndef LVTX_BLAS_DETAIL_KML_DSYGVD_OPT_UPPER_IMPL_HPP_
#define LVTX_BLAS_DETAIL_KML_DSYGVD_OPT_UPPER_IMPL_HPP_

#include "lvtx_kml_dsygvd_kernels.hpp"
#include "lvtx_dsygvd_opt_diag.hpp"
#include "../detail/lvtx_kml_dsygvd_detail.hpp"
#include "../lvtx_dsygvd_work_doubles.hpp"

#include <cstddef>
#include <omp.h>

namespace lvtx_blas_detail {
namespace kml_dsygvd_opt {

namespace ref = kml_dsygvd;

inline int dsygvd_upper_impl(
    const int n, double* const a, const int lda, double* const b, const int ldb,
    double* const w, double* const work, const std::size_t work_doubles,
    const int nthreads,
    lvtx_dsygvd_phase_times_f64* const phase_times,
    const int production_tile) noexcept {
    if (n < 0 || n > ref::kMaxOrder || nthreads < 0 || nthreads > ref::kMaxThreads) {
        return ref::kStatusDim;
    }
    if (!ref::valid_production_tile(production_tile)) {
        return ref::kStatusDim;
    }
    if (phase_times != nullptr && !ref::aligned_64(phase_times)) {
        return ref::kStatusAlign;
    }
    if (n == 0) {
        if (phase_times != nullptr) {
            *phase_times = {};
        }
        return ref::kStatusOk;
    }
    if (a == nullptr || b == nullptr || w == nullptr || work == nullptr) {
        return ref::kStatusNull;
    }
    if (lda < n || ldb < n) {
        return ref::kStatusLd;
    }
    if (!ref::aligned_64(a) || !ref::aligned_64(b) || !ref::aligned_64(w) ||
        !ref::aligned_64(work)) {
        return ref::kStatusAlign;
    }

    const std::size_t required = lvtx_dsygvd_work_doubles(n);
    if (work_doubles < required) {
        return ref::kStatusWorkspace;
    }
    ref::AddressRange a_range{};
    ref::AddressRange b_range{};
    ref::AddressRange w_range{};
    ref::AddressRange work_range{};
    ref::AddressRange phase_range{};
    if (!ref::make_range(a, static_cast<std::size_t>(lda) *
                                   static_cast<std::size_t>(n),
                         &a_range) ||
        !ref::make_range(b, static_cast<std::size_t>(ldb) *
                                   static_cast<std::size_t>(n),
                         &b_range) ||
        !ref::make_range(w, static_cast<std::size_t>(n), &w_range) ||
        !ref::make_range(work, work_doubles, &work_range) ||
        (phase_times != nullptr &&
         !ref::make_range(reinterpret_cast<const double*>(phase_times), 10U,
                          &phase_range))) {
        return ref::kStatusDim;
    }
    if (ref::overlaps(a_range, b_range) || ref::overlaps(a_range, w_range) ||
        ref::overlaps(a_range, work_range) || ref::overlaps(b_range, w_range) ||
        ref::overlaps(b_range, work_range) || ref::overlaps(w_range, work_range)) {
        return ref::kStatusOverlap;
    }
    if (phase_times != nullptr &&
        (ref::overlaps(phase_range, a_range) || ref::overlaps(phase_range, b_range) ||
         ref::overlaps(phase_range, w_range) ||
         ref::overlaps(phase_range, work_range))) {
        return ref::kStatusOverlap;
    }
    if (omp_in_parallel() != 0) {
        return ref::kStatusNestedOmp;
    }
    if (!ref::finite_upper_triangles(n, a, lda, b, ldb)) {
        diag::report_fail("input", "nonfinite_upper_triangle");
        return ref::kStatusNonfinite;
    }
#if defined(__ARM_FEATURE_SVE)
    {
        const int sve_status = validate_sve_environment();
        if (sve_status != ref::kStatusOk) {
            diag::report_fail("input", "sve_environment", sve_status);
            return sve_status;
        }
    }
#endif

    diag::reset_fail_report();

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

    const int threads = nthreads == 0 ? ref::kDefaultThreads : nthreads;
    const int tile = production_tile;
    double* const standard = work;
    const std::size_t matrix_doubles =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    double* const diagonal = standard + matrix_doubles;
    double* const off_diagonal = diagonal + n;
    double* const tau = off_diagonal + n;
    double* const temporary = tau + n;
    double* const panel_v =
        temporary + static_cast<std::size_t>(n);
    double* const panel_w =
        panel_v + static_cast<std::size_t>(n) *
                      static_cast<std::size_t>(kHouseholderPanelNb);
    lvtx_dsygvd_tridiag_kernel_times_f64 tridiag_kernels{};

    double phase_start =
        phase_times != nullptr ? omp_get_wtime() : 0.0;
    diag::report_progress("cholesky_start");
    const int cholesky_status = upper_cholesky(n, b, ldb, threads);
    diag::report_progress("cholesky_done");
    if (phase_times != nullptr) {
        phase_times->cholesky_seconds = omp_get_wtime() - phase_start;
    }
    if (cholesky_status != ref::kStatusOk) {
        return finish(cholesky_status);
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    diag::report_progress("standard_transform_start");
    if (!form_standard_problem(n, a, lda, b, ldb, standard, threads)) {
        diag::report_fail("standard_transform", "failed");
        return finish(ref::kStatusNonfinite);
    }
    diag::report_progress("standard_transform_done");
    if (phase_times != nullptr) {
        phase_times->standard_transform_seconds =
            omp_get_wtime() - phase_start;
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    diag::report_progress("tridiagonal_reduction_start");
    if (!reduce_to_tridiagonal(n, standard, diagonal, off_diagonal, tau,
                               temporary, panel_v, panel_w, &tridiag_kernels,
                               threads)) {
        diag::report_fail("tridiagonal_reduction", "failed");
        return finish(ref::kStatusNonfinite);
    }
    if (phase_times != nullptr) {
        phase_times->tridiagonal_matvec_seconds =
            tridiag_kernels.matvec_seconds;
        phase_times->tridiagonal_panel_correction_seconds =
            tridiag_kernels.panel_correction_seconds;
        phase_times->tridiagonal_trailing_rank2_seconds =
            tridiag_kernels.trailing_rank2_seconds;
    }
    if (phase_times != nullptr) {
        phase_times->tridiagonal_reduction_seconds =
            omp_get_wtime() - phase_start;
    }
    diag::report_progress("tridiagonal_reduction_core_done");
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    form_householder_product(n, standard, tau, a, lda, threads);
    if (phase_times != nullptr) {
        phase_times->householder_product_seconds =
            omp_get_wtime() - phase_start;
    }
    diag::report_progress("householder_product_done");
    diag::report_progress("tridiagonal_reduction_done");
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    diag::report_progress("tridiagonal_eigensolve_start");
    const int eigen_status =
        n == ref::kMaxOrder
            ? tridiagonal_ql_640_parallel(diagonal, off_diagonal, a, lda,
                                          standard, threads, tile)
            : tridiagonal_ql(n, diagonal, off_diagonal, a, lda);
    diag::report_progress("tridiagonal_eigensolve_done");
    if (phase_times != nullptr) {
        phase_times->tridiagonal_eigensolve_seconds =
            omp_get_wtime() - phase_start;
    }
    if (eigen_status != ref::kStatusOk) {
        diag::report_fail("tridiagonal_eigensolve", "status", eigen_status);
        return finish(eigen_status);
    }
    phase_start = phase_times != nullptr ? omp_get_wtime() : 0.0;
    diag::report_progress("backtransform_start");
    if (!generalized_backsolve(n, a, lda, b, ldb, standard, threads)) {
        diag::report_fail("backtransform", "failed");
        return finish(ref::kStatusNonfinite);
    }
    diag::report_progress("backtransform_done");
    if (phase_times != nullptr) {
        phase_times->backtransform_seconds = omp_get_wtime() - phase_start;
    }
    for (int index = 0; index < n; ++index) {
        w[index] = diagonal[index];
    }
    return finish(ref::kStatusOk);
}

}  // namespace kml_dsygvd_opt
}  // namespace lvtx_blas_detail

#endif  // LVTX_BLAS_DETAIL_KML_DSYGVD_OPT_UPPER_IMPL_HPP_
