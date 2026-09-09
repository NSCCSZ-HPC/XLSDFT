#ifndef LVTX_BLAS_DETAIL_KML_DSYGVD_DETAIL_OPT_HPP_
#define LVTX_BLAS_DETAIL_KML_DSYGVD_DETAIL_OPT_HPP_

#if !defined(__ARM_FEATURE_SVE)
#error "lvtx_kml_dsygvd_detail_opt.hpp requires SVE (-march=...+sve)"
#endif

#include "../detail/lvtx_kml_dsygvd_detail.hpp"
#include "../lvtx_blas_common.hpp"
#include "lvtx_dsygvd_opt_diag.hpp"
#include "lvtx_sve_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <omp.h>

#include <arm_sve.h>

namespace lvtx_blas_detail {
namespace kml_dsygvd_opt {

namespace ref = kml_dsygvd;

inline constexpr int kHouseholderPanelNb = 16;

inline int validate_sve_environment() noexcept {
    if (!sve::fp64_lanes_supported()) {
        return ref::kStatusUnsupportedVectorLength;
    }
    return ref::kStatusOk;
}

inline int upper_cholesky(const int n, double* const b, const int ldb,
                          const int threads) noexcept {
    if (threads <= 1) {
        for (int column = 0; column < n; ++column) {
            const double* const pivot_column =
                b + static_cast<std::size_t>(column) *
                        static_cast<std::size_t>(ldb);
            double diagonal = pivot_column[column];
            diagonal -= sve::contiguous_squared_norm_f64(pivot_column, column);
            if (!std::isfinite(diagonal)) {
                diag::report_fail("cholesky", "diagonal_nonfinite", column, -1,
                                  diagonal, true);
                return ref::kStatusNonfinite;
            }
            if (!(diagonal > 0.0)) {
                diag::report_fail("cholesky", "not_spd_minor", column);
                return n + column + 1;
            }
            const double factor_diagonal = std::sqrt(diagonal);
            if (!(factor_diagonal > 0.0) || !std::isfinite(factor_diagonal)) {
                diag::report_fail("cholesky", "factor_nonfinite", column, -1,
                                  factor_diagonal, true);
                return ref::kStatusNonfinite;
            }
            ref::matrix(b, ldb, column, column) = factor_diagonal;

            for (int trailing_column = column + 1; trailing_column < n;
                 ++trailing_column) {
                const double* const trailing_ptr =
                    b + static_cast<std::size_t>(trailing_column) *
                            static_cast<std::size_t>(ldb);
                const double input = trailing_ptr[column];
                const double dot =
                    sve::contiguous_dot_f64(pivot_column, trailing_ptr, column);
                double value = input;
                value -= dot;
                value /= factor_diagonal;
                if (!std::isfinite(value)) {
                    diag::report_cholesky_trailing(column, trailing_column,
                                                   input, factor_diagonal, dot,
                                                   value);
                    diag::report_fail("cholesky", "trailing_nonfinite", column,
                                      trailing_column, value, true);
                    return ref::kStatusNonfinite;
                }
                ref::matrix(b, ldb, column, trailing_column) = value;
            }
        }
        return ref::kStatusOk;
    }

    int status = ref::kStatusOk;
#pragma omp parallel num_threads(threads) default(none) shared(b, ldb, n, status)
    {
        for (int column = 0; column < n; ++column) {
#pragma omp single
            {
                if (status == ref::kStatusOk) {
                    const double* const pivot_column =
                        b + static_cast<std::size_t>(column) *
                                static_cast<std::size_t>(ldb);
                    double diagonal = pivot_column[column];
                    diagonal -=
                        sve::contiguous_squared_norm_f64(pivot_column, column);
                    if (!std::isfinite(diagonal)) {
                        status = ref::kStatusNonfinite;
                        diag::report_fail("cholesky", "diagonal_nonfinite",
                                          column, -1, diagonal, true);
                    } else if (!(diagonal > 0.0)) {
                        status = n + column + 1;
                        diag::report_fail("cholesky", "not_spd_minor", column);
                    } else {
                        const double factor = std::sqrt(diagonal);
                        if (!(factor > 0.0) || !std::isfinite(factor)) {
                            status = ref::kStatusNonfinite;
                            diag::report_fail("cholesky", "factor_nonfinite",
                                              column, -1, factor, true);
                        } else {
                            ref::matrix(b, ldb, column, column) = factor;
                        }
                    }
                }
            }

#pragma omp for schedule(static)
            for (int trailing_column = column + 1; trailing_column < n;
                 ++trailing_column) {
                if (status != ref::kStatusOk) {
                    continue;
                }
                const double* const pivot_column =
                    b + static_cast<std::size_t>(column) *
                            static_cast<std::size_t>(ldb);
                const double* const trailing_ptr =
                    b + static_cast<std::size_t>(trailing_column) *
                            static_cast<std::size_t>(ldb);
                const double factor =
                    ref::matrix(b, ldb, column, column);
                const double input = trailing_ptr[column];
                const double dot =
                    sve::contiguous_dot_f64(pivot_column, trailing_ptr, column);
                double value = input;
                value -= dot;
                value /= factor;
                if (!std::isfinite(value)) {
#pragma omp critical(dsygvd_cholesky_status)
                    {
                        if (status == ref::kStatusOk) {
                            status = ref::kStatusNonfinite;
                            diag::report_cholesky_trailing(
                                column, trailing_column, input, factor, dot,
                                value);
                            diag::report_fail_once(
                                "cholesky", "trailing_nonfinite", column,
                                trailing_column, value, true);
                        }
                    }
                    continue;
                }
                ref::matrix(b, ldb, column, trailing_column) = value;
            }
        }
    }
    return status;
}

inline bool form_standard_problem(const int n, const double* const a,
                                  const int lda, const double* const upper,
                                  const int ldu, double* const standard,
                                  const int threads) noexcept {
#pragma omp parallel for schedule(static) num_threads(threads) \
    if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        double* const dst_column =
            standard + static_cast<std::size_t>(column) *
                           static_cast<std::size_t>(n);
        for (int row = 0; row < n; ++row) {
            dst_column[row] =
                row <= column ? ref::matrix(a, lda, row, column)
                              : ref::matrix(a, lda, column, row);
        }
    }

#pragma omp parallel for schedule(static) num_threads(threads) \
    if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        double* const dst_column =
            standard + static_cast<std::size_t>(column) *
                           static_cast<std::size_t>(n);
        for (int row = 0; row < n; ++row) {
            double value = dst_column[row];
            const double* const upper_row =
                upper + static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(ldu);
            value -= sve::contiguous_dot_f64(upper_row, dst_column, row);
            value /= upper_row[row];
            dst_column[row] = value;
        }
    }

#pragma omp parallel for schedule(static) num_threads(threads) \
    if (threads > 1 && n >= 64)
    for (int row = 0; row < n; row += sve::kFp64Lanes) {
        const int block = std::min(sve::kFp64Lanes, n - row);
        const svbool_t pg = sve::pg_all();
        for (int column = 0; column < n; ++column) {
            const double pivot = ref::matrix(upper, ldu, column, column);
            if (block < sve::kFp64Lanes) {
                for (int r = row; r < n; ++r) {
                    double value = ref::matrix(standard, n, r, column);
                    for (int inner = 0; inner < column; ++inner) {
                        value -= ref::matrix(standard, n, r, inner) *
                                 ref::matrix(upper, ldu, inner, column);
                    }
                    ref::matrix(standard, n, r, column) = value / pivot;
                }
                continue;
            }
            svfloat64_t value_vec =
                svld1_f64(pg, standard + static_cast<std::size_t>(row) +
                                         static_cast<std::size_t>(column) * n);
            for (int inner = 0; inner < column; ++inner) {
                const svfloat64_t prev_vec = svld1_f64(
                    pg, standard + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(inner) * n);
                value_vec = svmls_n_f64_x(
                    pg, value_vec, prev_vec,
                    ref::matrix(upper, ldu, inner, column));
            }
            value_vec = svdiv_n_f64_x(pg, value_vec, pivot);
            svst1_f64(pg, standard + static_cast<std::size_t>(row) +
                                   static_cast<std::size_t>(column) * n,
                      value_vec);
        }
    }

    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : finite) \
    if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row <= column; ++row) {
            const double value =
                0.5 * ref::matrix(standard, n, row, column) +
                0.5 * ref::matrix(standard, n, column, row);
            finite = finite && std::isfinite(value);
            if (!std::isfinite(value)) {
                diag::report_fail_once("standard_transform", "symmetrize",
                                       row, column, value, true);
            }
            ref::matrix(standard, n, row, column) = value;
            ref::matrix(standard, n, column, row) = value;
        }
    }
    return finite;
}

inline bool active_matvec_sve(const int first, const int active, const int n,
                              const double* const standard,
                              const double* const vector,
                              double* const temporary,
                              const double reflector_scale,
                              const int threads) noexcept {
    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) \
    reduction(&& : finite) if (threads > 1 && active >= 128)
    for (int row = 0; row < active; row += sve::kFp64Lanes) {
        const int block = std::min(sve::kFp64Lanes, active - row);
        if (block < sve::kFp64Lanes) {
            for (int r = row; r < active; ++r) {
                double product = 0.0;
                for (int column = 0; column < active; ++column) {
                    product += ref::matrix(standard, n, first + r, first + column) *
                               vector[column];
                }
                const double value = reflector_scale * product;
                temporary[r] = value;
                finite = finite && std::isfinite(value);
            }
            continue;
        }
        const svbool_t pg = sve::pg_all();
        const svfloat64_t scale_vec = svdup_f64(reflector_scale);
        svfloat64_t sum_vec = svdup_f64(0.0);
        for (int column = 0; column < active; ++column) {
            const double vector_entry = vector[column];
            const double* const matrix_ptr =
                standard + static_cast<std::size_t>(first + row) +
                static_cast<std::size_t>(first + column) *
                    static_cast<std::size_t>(n);
            const svfloat64_t matrix_vec = svld1_f64(pg, matrix_ptr);
            sum_vec = svmla_f64_m(pg, sum_vec, matrix_vec,
                                  svdup_f64(vector_entry));
        }
        sum_vec = svmul_f64_m(pg, sum_vec, scale_vec);
        svst1_f64(pg, temporary + row, sum_vec);
        for (int lane = 0; lane < sve::kFp64Lanes; ++lane) {
            finite = finite && std::isfinite(temporary[row + lane]);
        }
    }
    return finite;
}

inline bool panel_matvec_sve(const int first, const int active, const int n,
                             const int step, const int local_t,
                             const double* const standard,
                             const double* const vector,
                             double* const temporary,
                             const double* const panel_v,
                             const double* const panel_w,
                             const double reflector_scale,
                             const int threads) noexcept {
    (void)step;
    double dot_w[kHouseholderPanelNb]{};
    double dot_v[kHouseholderPanelNb]{};
    for (int j = 0; j < local_t; ++j) {
        double sum_w = 0.0;
        double sum_v = 0.0;
        for (int r = 0; r < active; ++r) {
            const int global_row = first + r;
            sum_w += panel_w[global_row + static_cast<std::size_t>(j) * n] *
                     vector[r];
            sum_v += panel_v[global_row + static_cast<std::size_t>(j) * n] *
                     vector[r];
        }
        dot_w[j] = sum_w;
        dot_v[j] = sum_v;
    }

    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) \
    reduction(&& : finite) if (threads > 1 && active >= 128)
    for (int row = 0; row < active; row += sve::kFp64Lanes) {
        const int block = std::min(sve::kFp64Lanes, active - row);
        if (block < sve::kFp64Lanes) {
            for (int r = row; r < active; ++r) {
                const int global_row = first + r;
                double product = 0.0;
                for (int column = 0; column < active; ++column) {
                    product += ref::matrix(standard, n, first + r, first + column) *
                               vector[column];
                }
                for (int j = 0; j < local_t; ++j) {
                    product -=
                        panel_v[global_row + static_cast<std::size_t>(j) * n] *
                            dot_w[j] +
                        panel_w[global_row + static_cast<std::size_t>(j) * n] *
                            dot_v[j];
                }
                const double value = reflector_scale * product;
                temporary[r] = value;
                finite = finite && std::isfinite(value);
            }
            continue;
        }
        const svbool_t pg = sve::pg_all();
        const svfloat64_t scale_vec = svdup_f64(reflector_scale);
        svfloat64_t sum_vec = svdup_f64(0.0);
        for (int column = 0; column < active; ++column) {
            const double vector_entry = vector[column];
            const double* const matrix_ptr =
                standard + static_cast<std::size_t>(first + row) +
                static_cast<std::size_t>(first + column) *
                    static_cast<std::size_t>(n);
            const svfloat64_t matrix_vec = svld1_f64(pg, matrix_ptr);
            sum_vec = svmla_f64_m(pg, sum_vec, matrix_vec,
                                  svdup_f64(vector_entry));
        }
        for (int j = 0; j < local_t; ++j) {
            const svfloat64_t dot_w_vec = svdup_f64(dot_w[j]);
            const svfloat64_t dot_v_vec = svdup_f64(dot_v[j]);
            const double* const v_col =
                panel_v + static_cast<std::size_t>(first + row) +
                static_cast<std::size_t>(j) * n;
            const double* const w_col =
                panel_w + static_cast<std::size_t>(first + row) +
                static_cast<std::size_t>(j) * n;
            const svfloat64_t v_vec = svld1_f64(pg, v_col);
            const svfloat64_t w_vec = svld1_f64(pg, w_col);
            sum_vec = svmls_f64_m(pg, sum_vec, v_vec, dot_w_vec);
            sum_vec = svmls_f64_m(pg, sum_vec, w_vec, dot_v_vec);
        }
        sum_vec = svmul_f64_m(pg, sum_vec, scale_vec);
        svst1_f64(pg, temporary + row, sum_vec);
        for (int lane = 0; lane < sve::kFp64Lanes; ++lane) {
            finite = finite && std::isfinite(temporary[row + lane]);
        }
    }
    return finite;
}

inline bool apply_trailing_panel_rank2k_sve(
    const int n, double* const standard, const int trailing_first,
    const int panel_cols, const double* const panel_v,
    const double* const panel_w, const int threads) noexcept {
    const int trailing_active = n - trailing_first;
    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) \
    reduction(&& : finite) if (threads > 1 && trailing_active >= 128)
    for (int column = trailing_first; column < n; ++column) {
        int row = column;
        while (row < n && ((row - column) & (sve::kFp64Lanes - 1)) != 0) {
            double delta = 0.0;
            for (int k = 0; k < panel_cols; ++k) {
                delta += panel_v[row + static_cast<std::size_t>(k) * n] *
                             panel_w[column + static_cast<std::size_t>(k) * n] +
                         panel_w[row + static_cast<std::size_t>(k) * n] *
                             panel_v[column + static_cast<std::size_t>(k) * n];
            }
            double value = ref::matrix(standard, n, row, column) - delta;
            finite = finite && std::isfinite(value);
            ref::matrix(standard, n, row, column) = value;
            ref::matrix(standard, n, column, row) = value;
            ++row;
        }
        for (; row + sve::kFp64Lanes <= n; row += sve::kFp64Lanes) {
            const svbool_t pg = sve::pg_all();
            svfloat64_t values = svld1_f64(
                pg, standard + static_cast<std::size_t>(row) +
                        static_cast<std::size_t>(column) * n);
            for (int k = 0; k < panel_cols; ++k) {
                const svfloat64_t v_rows = svld1_f64(
                    pg, panel_v + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(k) * n);
                const svfloat64_t w_rows = svld1_f64(
                    pg, panel_w + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(k) * n);
                const double w_col = panel_w[column + static_cast<std::size_t>(k) * n];
                const double v_col = panel_v[column + static_cast<std::size_t>(k) * n];
                values = svmls_f64_m(pg, values, v_rows, svdup_f64(w_col));
                values = svmls_f64_m(pg, values, w_rows, svdup_f64(v_col));
            }
            svst1_f64(pg, standard + static_cast<std::size_t>(row) +
                                   static_cast<std::size_t>(column) * n,
                      values);
            double mirrored[sve::kFp64Lanes];
            svst1_f64(pg, mirrored, values);
            for (int lane = 0; lane < sve::kFp64Lanes; ++lane) {
                finite = finite && std::isfinite(mirrored[lane]);
                ref::matrix(standard, n, column, row + lane) = mirrored[lane];
            }
        }
        for (; row < n; ++row) {
            double delta = 0.0;
            for (int k = 0; k < panel_cols; ++k) {
                delta += panel_v[row + static_cast<std::size_t>(k) * n] *
                             panel_w[column + static_cast<std::size_t>(k) * n] +
                         panel_w[row + static_cast<std::size_t>(k) * n] *
                             panel_v[column + static_cast<std::size_t>(k) * n];
            }
            double value = ref::matrix(standard, n, row, column) - delta;
            finite = finite && std::isfinite(value);
            ref::matrix(standard, n, row, column) = value;
            ref::matrix(standard, n, column, row) = value;
        }
    }
    return finite;
}

inline void reconstruct_panel_column_sve(
    const int first, const int active, const int n, const int step,
    const int local_t, const double* const standard,
    const double* const panel_v, const double* const panel_w,
    double* const vector) noexcept {
    const double* const column =
        standard + static_cast<std::size_t>(first) +
        static_cast<std::size_t>(step) * static_cast<std::size_t>(n);
    if (local_t == 0) {
        sve::copy_f64(vector, column, active);
        return;
    }

    const svbool_t pg = sve::pg_all();
    int row = 0;
    for (; row + sve::kFp64Lanes <= active; row += sve::kFp64Lanes) {
        svfloat64_t values = svld1_f64(pg, column + row);
        for (int j = 0; j < local_t; ++j) {
            const std::size_t panel_stride =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
            const double w_step = panel_w[step + panel_stride];
            const double v_step = panel_v[step + panel_stride];
            const svfloat64_t v_rows =
                svld1_f64(pg, panel_v + static_cast<std::size_t>(first + row) +
                                   panel_stride);
            const svfloat64_t w_rows =
                svld1_f64(pg, panel_w + static_cast<std::size_t>(first + row) +
                                   panel_stride);
            values = svmls_n_f64_m(pg, values, v_rows, w_step);
            values = svmls_n_f64_m(pg, values, w_rows, v_step);
        }
        svst1_f64(pg, vector + row, values);
    }
    for (; row < active; ++row) {
        const int global_row = first + row;
        double value = column[row];
        for (int j = 0; j < local_t; ++j) {
            const std::size_t panel_stride =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
            value -= panel_v[global_row + panel_stride] *
                         panel_w[step + panel_stride] +
                     panel_w[global_row + panel_stride] *
                         panel_v[step + panel_stride];
        }
        vector[row] = value;
    }
}

inline void apply_householder_w_correction_sve(
    const int active, const double* const vector, double* const temporary,
    double* const wcol_active, const double reflector_scale) noexcept {
    const double dot = sve::contiguous_dot_f64(vector, temporary, active);
    const double correction = -0.5 * reflector_scale * dot;
    const svbool_t pg = sve::pg_all();
    const svfloat64_t correction_vec = svdup_f64(correction);

    int row = 0;
    for (; row + sve::kFp64Lanes <= active; row += sve::kFp64Lanes) {
        svfloat64_t temp_vec = svld1_f64(pg, temporary + row);
        const svfloat64_t vector_vec = svld1_f64(pg, vector + row);
        temp_vec = svmla_f64_m(pg, temp_vec, vector_vec, correction_vec);
        svst1_f64(pg, temporary + row, temp_vec);
        svst1_f64(pg, wcol_active + row, temp_vec);
    }
    for (; row < active; ++row) {
        const double corrected = temporary[row] + correction * vector[row];
        temporary[row] = corrected;
        wcol_active[row] = corrected;
    }
}

inline bool reduce_to_tridiagonal(
    const int n, double* const standard, double* const diagonal,
    double* const off_diagonal, double* const tau, double* const temporary,
    double* const panel_v, double* const panel_w,
    lvtx_dsygvd_tridiag_kernel_times_f64* const kernel_times,
    const int threads) noexcept {
    if (kernel_times != nullptr) {
        *kernel_times = {};
    }

    for (int panel_start = 0; panel_start + 2 < n;
         panel_start += kHouseholderPanelNb) {
        const int panel_cols =
            std::min(kHouseholderPanelNb, n - 2 - panel_start);

        for (int local_col = 0; local_col < panel_cols; ++local_col) {
            for (int row = 0; row < n; ++row) {
                panel_v[row + static_cast<std::size_t>(local_col) * n] = 0.0;
                panel_w[row + static_cast<std::size_t>(local_col) * n] = 0.0;
            }
        }

        for (int local_t = 0; local_t < panel_cols; ++local_t) {
            const int step = panel_start + local_t;
            const int first = step + 1;
            const int active = n - first;
            double* const vcol =
                panel_v + static_cast<std::size_t>(local_t) * n;
            double* const wcol =
                panel_w + static_cast<std::size_t>(local_t) * n;
            double* const vector = vcol + first;

            double correction_start = 0.0;
            if (kernel_times != nullptr) {
                correction_start = omp_get_wtime();
            }

            double diag = ref::matrix(standard, n, step, step);
            for (int j = 0; j < local_t; ++j) {
                diag -= 2.0 * panel_v[step + static_cast<std::size_t>(j) * n] *
                                panel_w[step + static_cast<std::size_t>(j) * n];
            }
            ref::matrix(standard, n, step, step) = diag;

            reconstruct_panel_column_sve(first, active, n, step, local_t,
                                         standard, panel_v, panel_w, vector);

            const double alpha = vector[0];
            const double tail_norm = ref::stable_norm(vector + 1, active - 1);

            if (tail_norm == 0.0) {
                tau[step] = 0.0;
                off_diagonal[step] = alpha;
                for (int r = 0; r < active; ++r) {
                    vector[r] = 0.0;
                    (wcol + first)[r] = 0.0;
                }
                if (kernel_times != nullptr) {
                    kernel_times->panel_correction_seconds +=
                        omp_get_wtime() - correction_start;
                }
                continue;
            }

            const double beta =
                -std::copysign(std::hypot(alpha, tail_norm), alpha);
            const long double denominator =
                static_cast<long double>(alpha) - static_cast<long double>(beta);
            const long double reflector_scale_wide =
                (static_cast<long double>(beta) -
                 static_cast<long double>(alpha)) /
                static_cast<long double>(beta);
            const double reflector_scale =
                static_cast<double>(reflector_scale_wide);
            if (!std::isfinite(beta) || !std::isfinite(denominator) ||
                !std::isfinite(reflector_scale) || denominator == 0.0L) {
                diag::report_fail("tridiagonal_reduction", "setup_reflector",
                                  step);
                return false;
            }

            vector[0] = 1.0;
            const long double inverse_denominator = 1.0L / denominator;
            sve::scale_f64(vector + 1, active - 1,
                           static_cast<double>(inverse_denominator));
            tau[step] = reflector_scale;
            off_diagonal[step] = beta;

            if ((step % 64) == 0) {
                diag::report_tridiag_step(step, active, true);
            }
            if (step == 0) {
                diag::report_progress("step0_setup_done");
            }

            if (kernel_times != nullptr) {
                kernel_times->panel_correction_seconds +=
                    omp_get_wtime() - correction_start;
            }

            double matvec_start = 0.0;
            if (kernel_times != nullptr) {
                matvec_start = omp_get_wtime();
            }

            const bool matvec_finite =
                local_t == 0
                    ? active_matvec_sve(first, active, n, standard, vector,
                                        temporary, reflector_scale, threads)
                    : panel_matvec_sve(first, active, n, step, local_t, standard,
                                       vector, temporary, panel_v, panel_w,
                                       reflector_scale, threads);
            if (!matvec_finite) {
                diag::report_fail("tridiagonal_reduction", "matvec", step);
                return false;
            }
            if (step == 0) {
                diag::report_progress("step0_matvec_done");
            }

            if (kernel_times != nullptr) {
                kernel_times->matvec_seconds += omp_get_wtime() - matvec_start;
                correction_start = omp_get_wtime();
            }

            apply_householder_w_correction_sve(active, vector, temporary,
                                               wcol + first, reflector_scale);
            if (step == 0) {
                diag::report_progress("step0_correction_done");
            }

            if (kernel_times != nullptr) {
                kernel_times->panel_correction_seconds +=
                    omp_get_wtime() - correction_start;
            }
        }

        const int trailing_first = panel_start + panel_cols;
        if (trailing_first < n) {
            double rank2_start = 0.0;
            if (kernel_times != nullptr) {
                rank2_start = omp_get_wtime();
            }
            if (!apply_trailing_panel_rank2k_sve(n, standard, trailing_first,
                                                 panel_cols, panel_v, panel_w,
                                                 threads)) {
                diag::report_fail("tridiagonal_reduction", "rank2",
                                  panel_start);
                return false;
            }
            if (kernel_times != nullptr) {
                kernel_times->trailing_rank2_seconds +=
                    omp_get_wtime() - rank2_start;
            }
        }
        for (int local_t = 0; local_t < panel_cols; ++local_t) {
            const int step = panel_start + local_t;
            if (tau[step] == 0.0) {
                continue;
            }
            const int first = step + 1;
            const int active = n - first;
            const double beta = off_diagonal[step];
            for (int r = 0; r < active; ++r) {
                const int global_row = first + r;
                ref::matrix(standard, n, global_row, step) =
                    panel_v[global_row + static_cast<std::size_t>(local_t) * n];
            }
            ref::matrix(standard, n, first, step) = beta;
            ref::matrix(standard, n, step, first) = beta;
        }
        if (panel_start == 0) {
            diag::report_progress("step0_rank2_done");
        }
    }

    for (int index = 0; index < n; ++index) {
        diagonal[index] = ref::matrix(standard, n, index, index);
        if (index + 1 < n && index + 2 >= n) {
            off_diagonal[index] = ref::matrix(standard, n, index + 1, index);
        }
    }
    if (n > 0) {
        off_diagonal[n - 1] = 0.0;
        tau[n - 1] = 0.0;
    }
    if (n > 1) {
        tau[n - 2] = 0.0;
    }
    return true;
}

inline void form_householder_product(const int n, const double* const reflectors,
                                     const double* const tau, double* const q,
                                     const int ldq, const int threads) noexcept {
    const auto apply_reflector_rows = [&](const int first,
                                          const double reflector_scale,
                                          const int step) noexcept {
        for (int row = 0; row < n; row += sve::kFp64Lanes) {
            const int block = std::min(sve::kFp64Lanes, n - row);
            if (block < sve::kFp64Lanes) {
                for (int r = row; r < n; ++r) {
                    double dot = ref::matrix(q, ldq, r, first);
                    for (int column = first + 1; column < n; ++column) {
                        dot += ref::matrix(q, ldq, r, column) *
                               ref::matrix(reflectors, n, column, step);
                    }
                    const double multiplier = reflector_scale * dot;
                    ref::matrix(q, ldq, r, first) -= multiplier;
                    for (int column = first + 1; column < n; ++column) {
                        ref::matrix(q, ldq, r, column) -=
                            multiplier *
                            ref::matrix(reflectors, n, column, step);
                    }
                }
                continue;
            }
            const svbool_t pg = sve::pg_all();
            const svfloat64_t scale_vec = svdup_f64(reflector_scale);
            svfloat64_t dot_vec = svld1_f64(
                pg, q + static_cast<std::size_t>(row) +
                        static_cast<std::size_t>(first) * ldq);
            for (int column = first + 1; column < n; ++column) {
                const svfloat64_t q_vec = svld1_f64(
                    pg, q + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(column) * ldq);
                const double reflector_entry =
                    ref::matrix(reflectors, n, column, step);
                dot_vec = svmla_f64_m(pg, dot_vec, q_vec,
                                      svdup_f64(reflector_entry));
            }
            const svfloat64_t multiplier_vec =
                svmul_f64_m(pg, scale_vec, dot_vec);
            svfloat64_t first_vec = svld1_f64(
                pg, q + static_cast<std::size_t>(row) +
                        static_cast<std::size_t>(first) * ldq);
            first_vec = svsub_f64_m(pg, first_vec, multiplier_vec);
            svst1_f64(pg, q + static_cast<std::size_t>(row) +
                               static_cast<std::size_t>(first) * ldq,
                      first_vec);
            for (int column = first + 1; column < n; ++column) {
                svfloat64_t q_vec = svld1_f64(
                    pg, q + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(column) * ldq);
                const double reflector_entry =
                    ref::matrix(reflectors, n, column, step);
                q_vec = svmls_f64_m(pg, q_vec, multiplier_vec,
                                    svdup_f64(reflector_entry));
                svst1_f64(pg, q + static_cast<std::size_t>(row) +
                                   static_cast<std::size_t>(column) * ldq,
                          q_vec);
            }
        }
    };

    if (threads <= 1) {
        for (int column = 0; column < n; ++column) {
            for (int row = 0; row < n; ++row) {
                ref::matrix(q, ldq, row, column) = row == column ? 1.0 : 0.0;
            }
        }
        for (int step = 0; step + 2 < n; ++step) {
            const double reflector_scale = tau[step];
            if (reflector_scale == 0.0) {
                continue;
            }
            apply_reflector_rows(step + 1, reflector_scale, step);
        }
        return;
    }

#pragma omp parallel num_threads(threads) default(none) \
    shared(n, reflectors, tau, q, ldq)
    {
#pragma omp for schedule(static)
        for (int column = 0; column < n; ++column) {
            for (int row = 0; row < n; ++row) {
                ref::matrix(q, ldq, row, column) = row == column ? 1.0 : 0.0;
            }
        }

        for (int step = 0; step + 2 < n; ++step) {
            const double reflector_scale = tau[step];
            const int first = step + 1;
#pragma omp for schedule(static)
            for (int row = 0; row < n; row += sve::kFp64Lanes) {
                if (reflector_scale == 0.0) {
                    continue;
                }
                const int block = std::min(sve::kFp64Lanes, n - row);
                if (block < sve::kFp64Lanes) {
                    for (int r = row; r < n; ++r) {
                        double dot = ref::matrix(q, ldq, r, first);
                        for (int column = first + 1; column < n; ++column) {
                            dot += ref::matrix(q, ldq, r, column) *
                                   ref::matrix(reflectors, n, column, step);
                        }
                        const double multiplier = reflector_scale * dot;
                        ref::matrix(q, ldq, r, first) -= multiplier;
                        for (int column = first + 1; column < n; ++column) {
                            ref::matrix(q, ldq, r, column) -=
                                multiplier *
                                ref::matrix(reflectors, n, column, step);
                        }
                    }
                    continue;
                }
                const svbool_t pg = sve::pg_all();
                const svfloat64_t scale_vec = svdup_f64(reflector_scale);
                svfloat64_t dot_vec = svld1_f64(
                    pg, q + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(first) * ldq);
                for (int column = first + 1; column < n; ++column) {
                    const svfloat64_t q_vec = svld1_f64(
                        pg, q + static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * ldq);
                    const double reflector_entry =
                        ref::matrix(reflectors, n, column, step);
                    dot_vec = svmla_f64_m(pg, dot_vec, q_vec,
                                          svdup_f64(reflector_entry));
                }
                const svfloat64_t multiplier_vec =
                    svmul_f64_m(pg, scale_vec, dot_vec);
                svfloat64_t first_vec = svld1_f64(
                    pg, q + static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(first) * ldq);
                first_vec = svsub_f64_m(pg, first_vec, multiplier_vec);
                svst1_f64(pg, q + static_cast<std::size_t>(row) +
                                   static_cast<std::size_t>(first) * ldq,
                          first_vec);
                for (int column = first + 1; column < n; ++column) {
                    svfloat64_t q_vec = svld1_f64(
                        pg, q + static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * ldq);
                    const double reflector_entry =
                        ref::matrix(reflectors, n, column, step);
                    q_vec = svmls_f64_m(pg, q_vec, multiplier_vec,
                                        svdup_f64(reflector_entry));
                    svst1_f64(pg,
                              q + static_cast<std::size_t>(row) +
                                  static_cast<std::size_t>(column) * ldq,
                              q_vec);
                }
            }
        }
    }
}

inline void apply_givens_rows_sve(double* const vectors, const int ldv,
                                  const int index, const int /*n*/,
                                  const double sine, const double cosine,
                                  const int tile_first,
                                  const int tile_last) noexcept {
    const svbool_t pg = sve::pg_all();
    const svfloat64_t sine_vec = svdup_f64(sine);
    const svfloat64_t cosine_vec = svdup_f64(cosine);
    const std::size_t lower_base =
        static_cast<std::size_t>(index) * static_cast<std::size_t>(ldv);
    const std::size_t upper_base =
        static_cast<std::size_t>(index + 1) * static_cast<std::size_t>(ldv);
    int row = tile_first;
    for (; row + sve::kFp64Lanes <= tile_last; row += sve::kFp64Lanes) {
        svfloat64_t lower = svld1_f64(pg, vectors + lower_base + row);
        svfloat64_t upper = svld1_f64(pg, vectors + upper_base + row);
        const svfloat64_t new_upper =
            svmla_f64_m(pg, svmul_f64_m(pg, sine_vec, lower), upper, cosine_vec);
        const svfloat64_t new_lower =
            svsub_f64_m(pg, svmul_f64_m(pg, cosine_vec, lower),
                        svmul_f64_m(pg, sine_vec, upper));
        svst1_f64(pg, vectors + upper_base + row, new_upper);
        svst1_f64(pg, vectors + lower_base + row, new_lower);
    }
    for (; row < tile_last; ++row) {
        const double lower_value = ref::matrix(vectors, ldv, row, index);
        const double upper_value = ref::matrix(vectors, ldv, row, index + 1);
        ref::matrix(vectors, ldv, row, index + 1) =
            sine * lower_value + cosine * upper_value;
        ref::matrix(vectors, ldv, row, index) =
            cosine * lower_value - sine * upper_value;
    }
}

__attribute__((noinline)) inline int tridiagonal_ql_rows(
    const int n, double* const diagonal, double* const off_diagonal,
    double* const vectors, const int ldv, const int row_worker,
    const int row_workers, const int row_tile) noexcept {
    if (n <= 1) {
        return ref::kStatusOk;
    }

    const double epsilon = std::numeric_limits<double>::epsilon();
    const double safe_minimum = std::numeric_limits<double>::min();
    const double safe_maximum = std::sqrt(std::numeric_limits<double>::max()) / 3.0;
    const double safe_small = std::sqrt(safe_minimum) / (epsilon * epsilon);
    off_diagonal[n - 1] = 0.0;

    int total_iterations = 0;
    const int iteration_limit = 30 * n;
    int block_first = 0;
    while (block_first < n) {
        int block_last = block_first;
        while (block_last + 1 < n && off_diagonal[block_last] != 0.0) {
            ++block_last;
        }

        double block_norm = 0.0;
        for (int index = block_first; index <= block_last; ++index) {
            block_norm = std::max(block_norm, std::abs(diagonal[index]));
            if (index < block_last) {
                block_norm =
                    std::max(block_norm, std::abs(off_diagonal[index]));
            }
        }
        int scale_exponent = 0;
        if (block_norm > safe_maximum) {
            scale_exponent =
                std::ilogb(safe_maximum) - std::ilogb(block_norm);
        } else if (block_norm != 0.0 && block_norm < safe_small) {
            scale_exponent =
                std::ilogb(safe_small) - std::ilogb(block_norm);
        }
        const double block_scale = std::scalbn(1.0, scale_exponent);
        if (scale_exponent != 0) {
            for (int index = block_first; index <= block_last; ++index) {
                diagonal[index] *= block_scale;
                if (index < block_last) {
                    off_diagonal[index] *= block_scale;
                }
            }
        }

        int failure_status = ref::kStatusOk;
        for (int left = block_first; left <= block_last; ++left) {
            while (true) {
                int right = left;
                for (; right < block_last; ++right) {
                    const double local_scale =
                        std::abs(diagonal[right]) +
                        std::abs(diagonal[right + 1]);
                    if (std::abs(off_diagonal[right]) <=
                        epsilon * local_scale + safe_minimum) {
                        off_diagonal[right] = 0.0;
                        break;
                    }
                }
                if (right == left) {
                    break;
                }
                if (++total_iterations > iteration_limit) {
                    failure_status = ref::count_unconverged(n, off_diagonal);
                    break;
                }

                const long double shift_ratio =
                    (static_cast<long double>(diagonal[left + 1]) -
                     static_cast<long double>(diagonal[left])) /
                    (2.0L * static_cast<long double>(off_diagonal[left]));
                const long double shift_root =
                    std::hypot(shift_ratio, 1.0L);
                const long double denominator =
                    shift_ratio + std::copysign(shift_root, shift_ratio);
                const long double shift =
                    denominator == 0.0L
                        ? 0.0L
                        : static_cast<long double>(off_diagonal[left]) /
                              denominator;
                double bulge = static_cast<double>(
                    static_cast<long double>(diagonal[right]) -
                    static_cast<long double>(diagonal[left]) + shift);
                double sine = 1.0;
                double cosine = 1.0;
                double correction = 0.0;

                for (int index = right - 1; index >= left; --index) {
                    const double sine_edge = sine * off_diagonal[index];
                    const double cosine_edge = cosine * off_diagonal[index];
                    double radius = 0.0;
                    if (std::abs(sine_edge) >= std::abs(bulge)) {
                        if (sine_edge == 0.0) {
                            cosine = 1.0;
                            sine = 0.0;
                            off_diagonal[index + 1] = 0.0;
                        } else {
                            cosine = bulge / sine_edge;
                            radius = std::hypot(cosine, 1.0);
                            off_diagonal[index + 1] = sine_edge * radius;
                            sine = 1.0 / radius;
                            cosine *= sine;
                        }
                    } else {
                        sine = sine_edge / bulge;
                        radius = std::hypot(sine, 1.0);
                        off_diagonal[index + 1] = bulge * radius;
                        cosine = 1.0 / radius;
                        sine *= cosine;
                    }

                    bulge = diagonal[index + 1] - correction;
                    radius =
                        (diagonal[index] - bulge) * sine +
                        2.0 * cosine * cosine_edge;
                    correction = sine * radius;
                    diagonal[index + 1] = bulge + correction;
                    bulge = cosine * radius - cosine_edge;

                    for (int tile_first = row_worker * row_tile;
                         tile_first < n;
                         tile_first += row_workers * row_tile) {
                        const int tile_last =
                            std::min(tile_first + row_tile, n);
                        apply_givens_rows_sve(vectors, ldv, index, n, sine,
                                              cosine, tile_first, tile_last);
                    }
                }
                diagonal[left] -= correction;
                off_diagonal[left] = bulge;
                off_diagonal[right] = 0.0;
            }
            if (failure_status != ref::kStatusOk) {
                break;
            }
        }

        if (scale_exponent != 0) {
            for (int index = block_first; index <= block_last; ++index) {
                diagonal[index] /= block_scale;
                if (index < block_last) {
                    off_diagonal[index] /= block_scale;
                }
            }
        }
        if (failure_status != ref::kStatusOk) {
            return failure_status;
        }
        block_first = block_last + 1;
    }

    for (int column = 0; column + 1 < n; ++column) {
        int smallest = column;
        for (int candidate = column + 1; candidate < n; ++candidate) {
            if (diagonal[candidate] < diagonal[smallest]) {
                smallest = candidate;
            }
        }
        if (smallest != column) {
            std::swap(diagonal[column], diagonal[smallest]);
            for (int tile_first = row_worker * row_tile;
                 tile_first < n;
                 tile_first += row_workers * row_tile) {
                const int tile_last = std::min(tile_first + row_tile, n);
                int row = tile_first;
                for (; row + sve::kFp64Lanes <= tile_last;
                     row += sve::kFp64Lanes) {
                    const svbool_t pg = sve::pg_all();
                    svfloat64_t left = svld1_f64(
                        pg, vectors + static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * ldv);
                    svfloat64_t right = svld1_f64(
                        pg, vectors + static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(smallest) * ldv);
                    svst1_f64(pg, vectors + static_cast<std::size_t>(row) +
                                           static_cast<std::size_t>(column) * ldv,
                              right);
                    svst1_f64(pg, vectors + static_cast<std::size_t>(row) +
                                           static_cast<std::size_t>(smallest) * ldv,
                              left);
                }
                for (; row < tile_last; ++row) {
                    std::swap(ref::matrix(vectors, ldv, row, column),
                              ref::matrix(vectors, ldv, row, smallest));
                }
            }
        }
    }
    return ref::kStatusOk;
}

inline int tridiagonal_ql(const int n, double* const diagonal,
                          double* const off_diagonal, double* const vectors,
                          const int ldv) noexcept {
    return tridiagonal_ql_rows(n, diagonal, off_diagonal, vectors, ldv, 0, 1,
                               std::max(n, 1));
}

inline int tridiagonal_ql_640_parallel(
    double* const diagonal, double* const off_diagonal,
    double* const vectors, const int ldv, double* const private_states,
    const int threads, const int tile) noexcept {
    constexpr int n = ref::kMaxOrder;
    constexpr int state_doubles = 2 * n;
    int statuses[ref::kMaxThreads]{};
    for (int index = 0; index < ref::kMaxThreads; ++index) {
        statuses[index] = std::numeric_limits<int>::min();
    }
    int workers = 1;
    const int requested_workers =
        std::min(threads, (n + tile - 1) / tile);
    for (int worker = 1; worker < requested_workers; ++worker) {
        double* const state =
            private_states + static_cast<std::size_t>(worker - 1) *
                                 state_doubles;
        std::memcpy(state, diagonal,
                    static_cast<std::size_t>(n) * sizeof(double));
        std::memcpy(state + n, off_diagonal,
                    static_cast<std::size_t>(n) * sizeof(double));
    }

#pragma omp parallel num_threads(threads) shared(workers, statuses)
    {
        const int worker = omp_get_thread_num();
        const int active_workers =
            std::min(omp_get_num_threads(), (n + tile - 1) / tile);
        if (worker == 0) {
            workers = active_workers;
        }
        double* worker_diagonal = diagonal;
        double* worker_off_diagonal = off_diagonal;
        if (worker > 0 && worker < active_workers) {
            double* const state =
                private_states + static_cast<std::size_t>(worker - 1) *
                                     state_doubles;
            worker_diagonal = state;
            worker_off_diagonal = state + n;
        }

        if (worker < active_workers) {
            statuses[worker] = tridiagonal_ql_rows(
                n, worker_diagonal, worker_off_diagonal, vectors, ldv,
                worker, active_workers, tile);
        }
    }

    const int status = statuses[0];
    for (int worker = 1; worker < workers; ++worker) {
        if (statuses[worker] != status) {
            std::fprintf(stderr,
                         "LVTX_DSYGVD_REPLICA_MISMATCH kind=status worker=%d"
                         " reference=%d actual=%d\n",
                         worker, status, statuses[worker]);
            return ref::kStatusReplicaMismatch;
        }
    }
    if (status != ref::kStatusOk) {
        return status;
    }
    for (int worker = 1; worker < workers; ++worker) {
        const double* const state =
            private_states + static_cast<std::size_t>(worker - 1) *
                                 state_doubles;
        for (int index = 0; index < n; ++index) {
            if (std::memcmp(diagonal + index, state + index,
                            sizeof(double)) != 0) {
                std::fprintf(
                    stderr,
                    "LVTX_DSYGVD_REPLICA_MISMATCH kind=diagonal worker=%d"
                    " index=%d reference=%a actual=%a\n",
                    worker, index, diagonal[index], state[index]);
                return ref::kStatusReplicaMismatch;
            }
            if (std::memcmp(off_diagonal + index, state + n + index,
                            sizeof(double)) != 0) {
                std::fprintf(
                    stderr,
                    "LVTX_DSYGVD_REPLICA_MISMATCH kind=off_diagonal"
                    " worker=%d index=%d reference=%a actual=%a\n",
                    worker, index, off_diagonal[index], state[n + index]);
                return ref::kStatusReplicaMismatch;
            }
        }
    }
    return ref::kStatusOk;
}

inline bool generalized_backsolve(const int n, double* const vectors,
                                  const int ldv, const double* const upper,
                                  const int ldu, double* const upper_packed,
                                  const int threads) noexcept {
    const std::size_t row_stride = static_cast<std::size_t>(n);
#pragma omp parallel for schedule(static) num_threads(threads) \
    if (threads > 1 && n >= 64)
    for (int row = 0; row < n; ++row) {
        const std::size_t packed_base =
            static_cast<std::size_t>(row) * row_stride;
        for (int col = row; col < n; ++col) {
            upper_packed[packed_base + static_cast<std::size_t>(col)] =
                ref::matrix(upper, ldu, row, col);
        }
    }

    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : finite) \
    if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        double* const vector_column =
            vectors + static_cast<std::size_t>(column) *
                          static_cast<std::size_t>(ldv);

        double scale = 0.0;
        double sum_squares = 1.0;
        for (int row = 0; row < n; ++row) {
            const double magnitude = std::abs(vector_column[row]);
            if (magnitude == 0.0) {
                continue;
            }
            if (scale < magnitude) {
                const double ratio = scale / magnitude;
                sum_squares = 1.0 + sum_squares * ratio * ratio;
                scale = magnitude;
            } else {
                const double ratio = magnitude / scale;
                sum_squares += ratio * ratio;
            }
        }
        const double vector_norm =
            scale == 0.0 ? 0.0 : scale * std::sqrt(sum_squares);
        if (!(vector_norm > 0.0) || !std::isfinite(vector_norm)) {
            diag::report_fail_once("backtransform", "vector_norm", column, -1,
                                   vector_norm, true);
            finite = false;
            continue;
        }
        sve::divide_f64(vector_column, n, vector_norm);

        for (int row = n - 1; row >= 0; --row) {
            double value = vector_column[row];
            const double* const packed_row =
                upper_packed +
                static_cast<std::size_t>(row) * row_stride;
            value -= sve::contiguous_dot_f64_range(
                packed_row, vector_column, row + 1, n);
            value /= packed_row[row];
            if (!std::isfinite(value)) {
                diag::report_fail_once("backtransform", "backsolve", column,
                                       row, value, true);
            }
            finite = finite && std::isfinite(value);
            vector_column[row] = value;
        }

        int pivot = 0;
        double pivot_magnitude = std::abs(vector_column[0]);
        for (int row = 1; row < n; ++row) {
            const double magnitude = std::abs(vector_column[row]);
            if (magnitude > pivot_magnitude) {
                pivot = row;
                pivot_magnitude = magnitude;
            }
        }
        if (std::signbit(vector_column[pivot])) {
            sve::negate_f64(vector_column, n);
        }
    }
    return finite;
}

}  // namespace kml_dsygvd_opt
}  // namespace lvtx_blas_detail

#endif  // LVTX_BLAS_DETAIL_KML_DSYGVD_DETAIL_OPT_HPP_
