#ifndef LVTX_DSYGVD_OPT_DIAG_HPP_
#define LVTX_DSYGVD_OPT_DIAG_HPP_

#ifndef LVTX_DSYGVD_OPT_DIAG
#define LVTX_DSYGVD_OPT_DIAG 0
#endif

#include <cmath>
#include <cstdio>

#include <omp.h>

namespace lvtx_blas_detail {
namespace kml_dsygvd_opt {
namespace diag {

#if LVTX_DSYGVD_OPT_DIAG

inline bool& fail_reported() noexcept {
    static bool reported = false;
    return reported;
}

inline void reset_fail_report() noexcept { fail_reported() = false; }

inline void report_fail(const char* phase, const char* site, const int i0 = -1,
                        const int i1 = -1, const double value = 0.0,
                        const bool has_value = false) noexcept {
    std::fprintf(stderr, "LVTX_DSYGVD_OPT_FAIL phase=%s site=%s", phase, site);
    if (i0 >= 0) {
        std::fprintf(stderr, " i0=%d", i0);
    }
    if (i1 >= 0) {
        std::fprintf(stderr, " i1=%d", i1);
    }
    if (has_value) {
        std::fprintf(stderr, " value=%a", value);
    }
    std::fprintf(stderr, "\n");
}

inline void report_fail_once(const char* phase, const char* site,
                             const int i0 = -1, const int i1 = -1,
                             const double value = 0.0,
                             const bool has_value = false) noexcept {
#pragma omp critical(lvtx_dsygvd_opt_fail_report)
    {
        if (!fail_reported()) {
            fail_reported() = true;
            report_fail(phase, site, i0, i1, value, has_value);
        }
    }
}

inline void report_cholesky_trailing(const int column,
                                     const int trailing_column,
                                     const double input, const double factor,
                                     const double dot,
                                     const double result) noexcept {
    std::fprintf(stderr,
                 "chol trailing col=%d tc=%d input=%a factor=%a dot=%a "
                 "result=%a\n",
                 column, trailing_column, input, factor, dot, result);
}

inline void report_opt_input(const int iter, const double b00, const double b01,
                             const double b11) noexcept {
    std::fprintf(stderr,
                 "OPT INPUT iter=%d b00=%a b01=%a b11=%a finite=%d,%d,%d\n",
                 iter, b00, b01, b11, std::isfinite(b00), std::isfinite(b01),
                 std::isfinite(b11));
    std::fflush(stderr);
}

inline void report_progress(const char* phase, const int step = -1) noexcept {
    std::fprintf(stderr, "LVTX_DSYGVD_OPT_PROGRESS phase=%s", phase);
    if (step >= 0) {
        std::fprintf(stderr, " step=%d", step);
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

inline void report_tridiag_step(const int step, const int active,
                                const bool ok) noexcept {
    std::fprintf(stderr, "TRIDIAG step=%d active=%d ok=%d\n", step, active,
                 ok ? 1 : 0);
    std::fflush(stderr);
}

#else

inline void reset_fail_report() noexcept {}

inline void report_fail(const char* /*phase*/, const char* /*site*/,
                        const int /*i0*/ = -1, const int /*i1*/ = -1,
                        const double /*value*/ = 0.0,
                        const bool /*has_value*/ = false) noexcept {}

inline void report_fail_once(const char* /*phase*/, const char* /*site*/,
                             const int /*i0*/ = -1, const int /*i1*/ = -1,
                             const double /*value*/ = 0.0,
                             const bool /*has_value*/ = false) noexcept {}

inline void report_cholesky_trailing(const int /*column*/,
                                     const int /*trailing_column*/,
                                     const double /*input*/,
                                     const double /*factor*/,
                                     const double /*dot*/,
                                     const double /*result*/) noexcept {}

inline void report_opt_input(const int /*iter*/, const double /*b00*/,
                             const double /*b01*/,
                             const double /*b11*/) noexcept {}

inline void report_progress(const char* /*phase*/,
                            const int /*step*/ = -1) noexcept {}

inline void report_tridiag_step(const int /*step*/, const int /*active*/,
                                const bool /*ok*/) noexcept {}

#endif  // LVTX_DSYGVD_OPT_DIAG

}  // namespace diag
}  // namespace kml_dsygvd_opt
}  // namespace lvtx_blas_detail

#endif  // LVTX_DSYGVD_OPT_DIAG_HPP_
