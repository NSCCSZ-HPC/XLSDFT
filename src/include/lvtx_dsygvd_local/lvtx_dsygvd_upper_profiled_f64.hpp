#ifndef LVTX_DSYGVD_UPPER_PROFILED_F64_HPP_
#define LVTX_DSYGVD_UPPER_PROFILED_F64_HPP_

#include "detail/lvtx_kml_dsygvd_upper_impl.hpp"
#include "detail_opt/lvtx_kml_dsygvd_upper_impl.hpp"
#include "lvtx_dsygvd_internal_config.hpp"

#include <cstddef>

// The same operation and validation as lvtx_dsygvd_upper_f64, with measured
// internal phase durations returned through a distinct 64-byte-aligned timing
// record. The ordinary production entry point performs no timing calls.
extern "C" inline int lvtx_dsygvd_upper_profiled_f64(
    const int n, double* const a, const int lda, double* const b, const int ldb,
    double* const w, double* const work, const std::size_t work_doubles,
    const int nthreads,
    lvtx_dsygvd_phase_times_f64* const phase_times) noexcept {
    using namespace lvtx_blas_detail;
    if (n < 0 || n > kml_dsygvd::kMaxOrder || nthreads < 0 ||
        nthreads > kml_dsygvd::kMaxThreads) {
        return kml_dsygvd::kStatusDim;
    }
    if (phase_times == nullptr) {
        return kml_dsygvd::kStatusNull;
    }
    if (xlsdft_dsygvd_internal::use_opt_path) {
        return kml_dsygvd_opt::dsygvd_upper_impl(
            n, a, lda, b, ldb, w, work, work_doubles, nthreads, phase_times,
            kml_dsygvd::kProductionTile);
    }
    return kml_dsygvd::dsygvd_upper_impl(n, a, lda, b, ldb, w, work,
                                         work_doubles, nthreads, phase_times,
                                         kml_dsygvd::kProductionTile);
}

#endif  // LVTX_DSYGVD_UPPER_PROFILED_F64_HPP_
