#ifndef LVTX_BLAS_DETAIL_KML_DSYGVD_HPP_
#define LVTX_BLAS_DETAIL_KML_DSYGVD_HPP_

// Source-local copy of the DSYGVD implementation pinned at LVTX commit
// 3a068939753c68d412dc0b7154c21ec3f41cf8da.  XLSDFT keeps the shared LVTX
// tree immutable.  The generic implementation remains byte-for-byte
// equivalent to that pin; only the production n=640 tridiagonal-eigenvector
// path below changes its parallel organization and row traversal.

#include "../lvtx_blas_common.hpp"
#include "../lvtx_dsygvd_internal_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <omp.h>

namespace lvtx_blas_detail {
namespace kml_dsygvd {

inline constexpr int kMaxOrder = 640;
inline constexpr int kMaxThreads = LVTX_DSYGVD_MAX_THREADS;
inline constexpr int kDefaultThreads = LVTX_DSYGVD_DEFAULT_THREADS;
inline constexpr int kProductionTile =
    xlsdft_dsygvd_internal::production_tile;

inline constexpr int kStatusOk = LVTX_SUCCESS;
inline constexpr int kStatusNull = LVTX_ERR_NULL;
inline constexpr int kStatusDim = LVTX_ERR_DIM;
inline constexpr int kStatusLd = LVTX_ERR_LD;
inline constexpr int kStatusAlign = LVTX_ERR_ALIGN;
inline constexpr int kStatusOverlap = LVTX_ERR_OVERLAP;
inline constexpr int kStatusWorkspace = LVTX_ERR_WORKSPACE;
inline constexpr int kStatusNestedOmp = LVTX_ERR_NESTED_OMP;
inline constexpr int kStatusNonfinite = LVTX_ERR_NONFINITE;
inline constexpr int kStatusUnsupportedVectorLength = LVTX_ERR_UNSUPPORTED_VL;
inline constexpr int kStatusReplicaMismatch =
    xlsdft_dsygvd_internal::replica_mismatch_status;

inline constexpr bool valid_production_tile(const int tile) noexcept {
    return tile == 32 || tile == 64 || tile == 96 || tile == 128;
}

static_assert(valid_production_tile(kProductionTile),
              "invalid fixed DSYGVD production tile");
static_assert(xlsdft_dsygvd_internal::production_threads > 0 &&
                  xlsdft_dsygvd_internal::production_threads <= kMaxThreads,
              "invalid fixed DSYGVD production team");

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

inline bool aligned_64(const void* const pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & std::uintptr_t{63}) == 0;
}

inline bool make_range(const double* const pointer, const std::size_t count,
                AddressRange* const range) noexcept {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    constexpr std::uintptr_t kAddressMax = std::numeric_limits<std::uintptr_t>::max();
    if (count > static_cast<std::size_t>((kAddressMax - begin) / sizeof(double))) {
        return false;
    }
    range->begin = begin;
    range->end = begin + static_cast<std::uintptr_t>(count * sizeof(double));
    return true;
}

inline bool overlaps(const AddressRange& left, const AddressRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

inline double& matrix(double* const values, const int leading_dimension,
               const int row, const int column) noexcept {
    return values[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(column) *
                      static_cast<std::size_t>(leading_dimension)];
}

inline double matrix(const double* const values, const int leading_dimension,
              const int row, const int column) noexcept {
    return values[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(column) *
                      static_cast<std::size_t>(leading_dimension)];
}

inline double stable_norm(const double* const values, const int count) noexcept {
    double scale = 0.0;
    double sum_squares = 1.0;
    for (int index = 0; index < count; ++index) {
        const double absolute_value = std::abs(values[index]);
        if (absolute_value == 0.0) {
            continue;
        }
        if (scale < absolute_value) {
            const double ratio = scale / absolute_value;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = absolute_value;
        } else {
            const double ratio = absolute_value / scale;
            sum_squares += ratio * ratio;
        }
    }
    return scale == 0.0 ? 0.0 : scale * std::sqrt(sum_squares);
}

inline bool finite_upper_triangles(const int n, const double* const a, const int lda,
                            const double* const b, const int ldb) noexcept {
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row <= column; ++row) {
            if (!std::isfinite(matrix(a, lda, row, column)) ||
                !std::isfinite(matrix(b, ldb, row, column))) {
                return false;
            }
        }
    }
    return true;
}

inline int upper_cholesky(const int n, double* const b, const int ldb) noexcept {
    for (int column = 0; column < n; ++column) {
        double diagonal = matrix(b, ldb, column, column);
        for (int row = 0; row < column; ++row) {
            const double value = matrix(b, ldb, row, column);
            diagonal -= value * value;
        }

        if (!std::isfinite(diagonal)) {
            return kStatusNonfinite;
        }
        if (!(diagonal > 0.0)) {
            return n + column + 1;
        }
        const double factor_diagonal = std::sqrt(diagonal);
        if (!(factor_diagonal > 0.0) || !std::isfinite(factor_diagonal)) {
            return kStatusNonfinite;
        }
        matrix(b, ldb, column, column) = factor_diagonal;

        for (int trailing_column = column + 1; trailing_column < n;
             ++trailing_column) {
            double value = matrix(b, ldb, column, trailing_column);
            for (int row = 0; row < column; ++row) {
                value -= matrix(b, ldb, row, column) *
                         matrix(b, ldb, row, trailing_column);
            }
            value /= factor_diagonal;
            if (!std::isfinite(value)) {
                return kStatusNonfinite;
            }
            matrix(b, ldb, column, trailing_column) = value;
        }
    }
    return kStatusOk;
}

inline bool form_standard_problem(const int n, const double* const a, const int lda,
                           const double* const upper, const int ldu,
                           double* const standard, const int threads) noexcept {
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            matrix(standard, n, row, column) =
                row <= column ? matrix(a, lda, row, column)
                              : matrix(a, lda, column, row);
        }
    }

    // Left solve U^T X=A.  Columns are independent.
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            double value = matrix(standard, n, row, column);
            for (int inner = 0; inner < row; ++inner) {
                value -= matrix(upper, ldu, inner, row) *
                         matrix(standard, n, inner, column);
            }
            value /= matrix(upper, ldu, row, row);
            matrix(standard, n, row, column) = value;
        }
    }

    // Right solve C U=X.  Rows are independent; columns of each row must be
    // visited in increasing order because U is upper triangular.
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && n >= 64)
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            double value = matrix(standard, n, row, column);
            for (int inner = 0; inner < column; ++inner) {
                value -= matrix(standard, n, row, inner) *
                         matrix(upper, ldu, inner, column);
            }
            value /= matrix(upper, ldu, column, column);
            matrix(standard, n, row, column) = value;
        }
    }

    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : finite) if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row <= column; ++row) {
            const double value =
                0.5 * matrix(standard, n, row, column) +
                0.5 * matrix(standard, n, column, row);
            finite = finite && std::isfinite(value);
            matrix(standard, n, row, column) = value;
            matrix(standard, n, column, row) = value;
        }
    }
    return finite;
}

inline bool reduce_to_tridiagonal(const int n, double* const standard,
                           double* const diagonal, double* const off_diagonal,
                           double* const tau, double* const temporary,
                           const int threads) noexcept {
    for (int step = 0; step + 2 < n; ++step) {
        const int first = step + 1;
        const int active = n - first;
        double* const vector = &matrix(standard, n, first, step);
        const double alpha = vector[0];
        const double tail_norm = stable_norm(vector + 1, active - 1);

        if (tail_norm == 0.0) {
            tau[step] = 0.0;
            off_diagonal[step] = alpha;
            continue;
        }

        const double beta = -std::copysign(std::hypot(alpha, tail_norm), alpha);
        const long double denominator =
            static_cast<long double>(alpha) - static_cast<long double>(beta);
        const long double reflector_scale_wide =
            (static_cast<long double>(beta) - static_cast<long double>(alpha)) /
            static_cast<long double>(beta);
        const double reflector_scale =
            static_cast<double>(reflector_scale_wide);
        if (!std::isfinite(beta) || !std::isfinite(denominator) ||
            !std::isfinite(reflector_scale) || denominator == 0.0L) {
            return false;
        }

        vector[0] = 1.0;
        for (int row = 1; row < active; ++row) {
            vector[row] = static_cast<double>(
                static_cast<long double>(vector[row]) / denominator);
        }
        tau[step] = reflector_scale;
        off_diagonal[step] = beta;

        bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : finite) if (threads > 1 && active >= 128)
        for (int row = 0; row < active; ++row) {
            double product = 0.0;
            for (int column = 0; column < active; ++column) {
                product += matrix(standard, n, first + row, first + column) *
                           vector[column];
            }
            const double value = reflector_scale * product;
            temporary[row] = value;
            finite = finite && std::isfinite(value);
        }
        if (!finite) {
            return false;
        }

        long double dot = 0.0L;
        for (int row = 0; row < active; ++row) {
            dot += static_cast<long double>(vector[row]) *
                   static_cast<long double>(temporary[row]);
        }
        const long double correction =
            -0.5L * static_cast<long double>(reflector_scale) * dot;
        for (int row = 0; row < active; ++row) {
            const long double corrected =
                static_cast<long double>(temporary[row]) +
                correction * static_cast<long double>(vector[row]);
            temporary[row] = static_cast<double>(corrected);
            if (!std::isfinite(temporary[row])) {
                return false;
            }
        }

        finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : finite) if (threads > 1 && active >= 128)
        for (int column = 0; column < active; ++column) {
            for (int row = column; row < active; ++row) {
                const double value =
                    matrix(standard, n, first + row, first + column) -
                    vector[row] * temporary[column] -
                    temporary[row] * vector[column];
                finite = finite && std::isfinite(value);
                matrix(standard, n, first + row, first + column) = value;
                matrix(standard, n, first + column, first + row) = value;
            }
        }
        if (!finite) {
            return false;
        }

        // Preserve the reflector tail below the first subdiagonal.  The first
        // element is implicit and the visible tridiagonal entry is beta.
        vector[0] = beta;
        matrix(standard, n, step, first) = beta;
    }

    for (int index = 0; index < n; ++index) {
        diagonal[index] = matrix(standard, n, index, index);
        if (index + 1 < n && index + 2 >= n) {
            off_diagonal[index] = matrix(standard, n, index + 1, index);
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
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            matrix(q, ldq, row, column) = row == column ? 1.0 : 0.0;
        }
    }

    // If T=Q^T C Q, the reduction produces Q=H_0 H_1 ... H_{n-3}.
    for (int step = 0; step + 2 < n; ++step) {
        const double reflector_scale = tau[step];
        if (reflector_scale == 0.0) {
            continue;
        }
        const int first = step + 1;
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && n >= 64)
        for (int row = 0; row < n; ++row) {
            double dot = matrix(q, ldq, row, first);
            for (int column = first + 1; column < n; ++column) {
                dot += matrix(q, ldq, row, column) *
                       matrix(reflectors, n, column, step);
            }
            const double multiplier = reflector_scale * dot;
            matrix(q, ldq, row, first) -= multiplier;
            for (int column = first + 1; column < n; ++column) {
                matrix(q, ldq, row, column) -=
                    multiplier * matrix(reflectors, n, column, step);
            }
        }
    }
}

inline int count_unconverged(const int n, const double* const off_diagonal) noexcept {
    int count = 0;
    for (int index = 0; index + 1 < n; ++index) {
        if (off_diagonal[index] != 0.0) {
            ++count;
        }
    }
    return count == 0 ? 1 : count;
}

__attribute__((noinline)) inline int tridiagonal_ql_rows(
    const int n, double* const diagonal, double* const off_diagonal,
    double* const vectors, const int ldv, const int row_worker,
    const int row_workers, const int row_tile) noexcept {
    if (n <= 1) {
        return kStatusOk;
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
        // Scale each exactly unreduced block independently.  A single global
        // scale can erase a small block when another split block is enormous.
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

        int failure_status = kStatusOk;
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
                    failure_status = count_unconverged(n, off_diagonal);
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
                        for (int row = tile_first; row < tile_last; ++row) {
                            const double upper =
                                matrix(vectors, ldv, row, index + 1);
                            const double lower =
                                matrix(vectors, ldv, row, index);
                            matrix(vectors, ldv, row, index + 1) =
                                sine * lower + cosine * upper;
                            matrix(vectors, ldv, row, index) =
                                cosine * lower - sine * upper;
                        }
                    }
                }
                diagonal[left] -= correction;
                off_diagonal[left] = bulge;
                off_diagonal[right] = 0.0;
            }
            if (failure_status != kStatusOk) {
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
        if (failure_status != kStatusOk) {
            return failure_status;
        }
        block_first = block_last + 1;
    }

    // DSTEQR does not promise a deterministic basis inside a degenerate
    // eigenspace.  Ordering is fixed here; signs are fixed after backsolve.
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
                for (int row = tile_first; row < tile_last; ++row) {
                    std::swap(matrix(vectors, ldv, row, column),
                              matrix(vectors, ldv, row, smallest));
                }
            }
        }
    }
    return kStatusOk;
}

inline int tridiagonal_ql(const int n, double* const diagonal,
                          double* const off_diagonal,
                          double* const vectors, const int ldv) noexcept {
    return tridiagonal_ql_rows(n, diagonal, off_diagonal, vectors, ldv, 0, 1,
                               std::max(n, 1));
}

inline int tridiagonal_ql_640_parallel(
    double* const diagonal, double* const off_diagonal,
    double* const vectors, const int ldv, double* const private_states,
    const int threads, const int tile) noexcept {
    constexpr int n = kMaxOrder;
    constexpr int state_doubles = 2 * n;
    int statuses[kMaxThreads]{};
    for (int index = 0; index < kMaxThreads; ++index) {
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

    // The QL scalar recurrence is tiny compared with applying its rotations
    // to the 640 eigenvector rows.  Each worker therefore runs an identical
    // private d/e recurrence while updating only its own cyclic row tiles.
    // There is one team formation and no synchronization in the hot rotation
    // sequence.  Worker zero owns the authoritative output d/e arrays.
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
            return kStatusReplicaMismatch;
        }
    }
    if (status != kStatusOk) {
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
                return kStatusReplicaMismatch;
            }
            if (std::memcmp(off_diagonal + index, state + n + index,
                            sizeof(double)) != 0) {
                std::fprintf(
                    stderr,
                    "LVTX_DSYGVD_REPLICA_MISMATCH kind=off_diagonal"
                    " worker=%d index=%d reference=%a actual=%a\n",
                    worker, index, off_diagonal[index], state[n + index]);
                return kStatusReplicaMismatch;
            }
        }
    }
    return kStatusOk;
}

inline bool generalized_backsolve(const int n, double* const vectors, const int ldv,
                           const double* const upper, const int ldu,
                           const int threads) noexcept {
    bool finite = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : finite) if (threads > 1 && n >= 64)
    for (int column = 0; column < n; ++column) {
        // Normalize the orthogonal standard-problem eigenvector before the
        // triangular solve.  Since U*x=y, this is also the B norm of x, up to
        // the backward error of the solve, and avoids rebuilding B.
        double scale = 0.0;
        double sum_squares = 1.0;
        for (int row = 0; row < n; ++row) {
            const double magnitude =
                std::abs(matrix(vectors, ldv, row, column));
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
            finite = false;
            continue;
        }
        for (int row = 0; row < n; ++row) {
            matrix(vectors, ldv, row, column) /= vector_norm;
        }

        for (int row = n - 1; row >= 0; --row) {
            double value = matrix(vectors, ldv, row, column);
            for (int inner = row + 1; inner < n; ++inner) {
                value -= matrix(upper, ldu, row, inner) *
                         matrix(vectors, ldv, inner, column);
            }
            value /= matrix(upper, ldu, row, row);
            finite = finite && std::isfinite(value);
            matrix(vectors, ldv, row, column) = value;
        }

        int pivot = 0;
        double pivot_magnitude = std::abs(matrix(vectors, ldv, 0, column));
        for (int row = 1; row < n; ++row) {
            const double magnitude =
                std::abs(matrix(vectors, ldv, row, column));
            if (magnitude > pivot_magnitude) {
                pivot = row;
                pivot_magnitude = magnitude;
            }
        }
        if (std::signbit(matrix(vectors, ldv, pivot, column))) {
            for (int row = 0; row < n; ++row) {
                matrix(vectors, ldv, row, column) =
                    -matrix(vectors, ldv, row, column);
            }
        }
    }
    return finite;
}

}  // namespace kml_dsygvd
}  // namespace lvtx_blas_detail

#endif  // LVTX_BLAS_DETAIL_KML_DSYGVD_HPP_
