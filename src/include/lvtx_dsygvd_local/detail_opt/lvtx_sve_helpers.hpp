#ifndef LVTX_DSYGVD_DETAIL_OPT_SVE_HELPERS_HPP_
#define LVTX_DSYGVD_DETAIL_OPT_SVE_HELPERS_HPP_

#include <arm_sve.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lvtx_blas_detail {
namespace kml_dsygvd_opt {
namespace sve {

inline constexpr int kFp64Lanes = 8;

inline svbool_t pg_all() noexcept { return svptrue_b64(); }

inline bool fp64_lanes_supported() noexcept {
    return svcntd() == static_cast<uint64_t>(kFp64Lanes);
}

inline double horizontal_sum(svfloat64_t value) noexcept {
    return svaddv_f64(pg_all(), value);
}

inline double contiguous_dot_f64(const double* a, const double* b,
                               int count) noexcept {
    if (count <= 0) {
        return 0.0;
    }
    const svbool_t pg = pg_all();
    svfloat64_t acc0 = svdup_f64(0.0);
    svfloat64_t acc1 = svdup_f64(0.0);
    int index = 0;
    for (; index + 16 <= count; index += 16) {
        acc0 = svmla_f64_m(pg, acc0, svld1_f64(pg, a + index),
                           svld1_f64(pg, b + index));
        acc1 = svmla_f64_m(pg, acc1, svld1_f64(pg, a + index + kFp64Lanes),
                           svld1_f64(pg, b + index + kFp64Lanes));
    }
    for (; index + kFp64Lanes <= count; index += kFp64Lanes) {
        acc0 = svmla_f64_m(pg, acc0, svld1_f64(pg, a + index),
                           svld1_f64(pg, b + index));
    }
    double sum = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; index < count; ++index) {
        sum += a[index] * b[index];
    }
    return sum;
}

inline double contiguous_dot_f64_range(const double* a, const double* b,
                                       int begin, int end) noexcept {
    int index = begin;
    double sum = 0.0;
    while (index < end && (index & (kFp64Lanes - 1)) != 0) {
        sum += a[index] * b[index];
        ++index;
    }
    sum += contiguous_dot_f64(a + index, b + index, end - index);
    return sum;
}

inline double contiguous_squared_norm_f64(const double* values,
                                          int count) noexcept {
    if (count <= 0) {
        return 0.0;
    }
    const svbool_t pg = pg_all();
    svfloat64_t acc0 = svdup_f64(0.0);
    svfloat64_t acc1 = svdup_f64(0.0);
    int index = 0;
    for (; index + 16 <= count; index += 16) {
        const svfloat64_t v0 = svld1_f64(pg, values + index);
        const svfloat64_t v1 = svld1_f64(pg, values + index + kFp64Lanes);
        acc0 = svmla_f64_m(pg, acc0, v0, v0);
        acc1 = svmla_f64_m(pg, acc1, v1, v1);
    }
    for (; index + kFp64Lanes <= count; index += kFp64Lanes) {
        const svfloat64_t v = svld1_f64(pg, values + index);
        acc0 = svmla_f64_m(pg, acc0, v, v);
    }
    double sum = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; index < count; ++index) {
        sum += values[index] * values[index];
    }
    return sum;
}

inline void scale_f64(double* values, int count, double scale) noexcept {
    if (count <= 0) {
        return;
    }
    const svbool_t pg = pg_all();
    const svfloat64_t scale_vec = svdup_f64(scale);
    int index = 0;
    for (; index + kFp64Lanes <= count; index += kFp64Lanes) {
        svfloat64_t value = svld1_f64(pg, values + index);
        value = svmul_f64_m(pg, value, scale_vec);
        svst1_f64(pg, values + index, value);
    }
    for (; index < count; ++index) {
        values[index] *= scale;
    }
}

inline void divide_f64(double* values, int count, double divisor) noexcept {
    scale_f64(values, count, 1.0 / divisor);
}

inline void negate_f64(double* values, int count) noexcept {
    scale_f64(values, count, -1.0);
}

inline void axpy_f64(double* y, const double* x, int count,
                     double alpha) noexcept {
    if (count <= 0) {
        return;
    }
    const svbool_t pg = pg_all();
    const svfloat64_t alpha_vec = svdup_f64(alpha);
    int index = 0;
    for (; index + kFp64Lanes <= count; index += kFp64Lanes) {
        svfloat64_t y_vec = svld1_f64(pg, y + index);
        const svfloat64_t x_vec = svld1_f64(pg, x + index);
        y_vec = svmla_f64_m(pg, y_vec, x_vec, alpha_vec);
        svst1_f64(pg, y + index, y_vec);
    }
    for (; index < count; ++index) {
        y[index] += alpha * x[index];
    }
}

inline void copy_f64(double* dst, const double* src, int count) noexcept {
    if (count <= 0) {
        return;
    }
    const svbool_t pg = pg_all();
    int index = 0;
    for (; index + kFp64Lanes <= count; index += kFp64Lanes) {
        svst1_f64(pg, dst + index, svld1_f64(pg, src + index));
    }
    for (; index < count; ++index) {
        dst[index] = src[index];
    }
}

}  // namespace sve
}  // namespace kml_dsygvd_opt
}  // namespace lvtx_blas_detail

#endif  // LVTX_DSYGVD_DETAIL_OPT_SVE_HELPERS_HPP_
