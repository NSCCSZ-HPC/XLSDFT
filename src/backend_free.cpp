#include "xlsdft_backend.h"

#include "lvtx_backend.h"
#include "lvtx_dsygvd_local/lvtx_dsygvd_internal_config.hpp"

#if !defined(XLSDFT_BACKEND_FREE) || defined(XLSDFT_BACKEND_KBLAS)
#error "backend_free.cpp requires the exclusive free-backend compile flag"
#endif

#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include <cstdio>
#include <cstdlib>

#include <omp.h>
#include <unistd.h>

namespace Xlsdft_backend {
namespace {

struct Address_range {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool make_range(const double* const pointer, const std::size_t doubles,
                Address_range* const range) noexcept {
    if (pointer == nullptr || range == nullptr ||
        doubles > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    const std::size_t bytes = doubles * sizeof(double);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    range->begin = begin;
    range->end = begin + bytes;
    return true;
}

bool overlaps(const Address_range& left, const Address_range& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <std::size_t Count>
bool pairwise_disjoint(const std::array<Address_range, Count>& ranges) noexcept {
    for (std::size_t left = 0; left < Count; ++left) {
        for (std::size_t right = left + 1U; right < Count; ++right) {
            if (overlaps(ranges[left], ranges[right])) return false;
        }
    }
    return true;
}

Lvtx_backend::Operation to_lvtx_operation(const Operation operation) noexcept {
    switch (operation) {
        case Operation::nloc_forward:
            return Lvtx_backend::Operation::nloc_forward;
        case Operation::nloc_back:
            return Lvtx_backend::Operation::nloc_back;
        case Operation::pulay_gram:
            return Lvtx_backend::Operation::pulay_gram;
        case Operation::pulay_tmv:
            return Lvtx_backend::Operation::pulay_tmv;
        case Operation::pulay_update:
            return Lvtx_backend::Operation::pulay_update;
        case Operation::pulay_gelsd:
            return Lvtx_backend::Operation::pulay_gelsd;
        case Operation::dsterf:
            return Lvtx_backend::Operation::dsterf;
        case Operation::dsygvd_upper:
            return Lvtx_backend::Operation::dsygvd_upper;
        case Operation::dsygvd_upper_profiled:
            return Lvtx_backend::Operation::dsygvd_upper_profiled;
        case Operation::dgemm_poj:
            return Lvtx_backend::Operation::dgemm_poj;
        case Operation::dsyrk_upper:
            return Lvtx_backend::Operation::dsyrk_upper;
        case Operation::dgemm_rotation:
            return Lvtx_backend::Operation::dgemm_rot_compute;
    }
    return Lvtx_backend::Operation::dgemm_rot_compute;
}

}  // namespace

static_assert(alignment_bytes == Lvtx_backend::alignment_bytes);
static_assert(nloc_max_bands == Lvtx_backend::nloc_max_bands);
static_assert(dsterf_max_n == Lvtx_backend::dsterf_max_n);
static_assert(dsygvd_max_n == Lvtx_backend::dsygvd_max_n);
static_assert(pulay_default_threads == Lvtx_backend::pulay_default_threads);
static_assert(pulay_max_threads == Lvtx_backend::pulay_max_threads);
static_assert(dsygvd_default_threads == Lvtx_backend::dsygvd_default_threads);
static_assert(dsygvd_max_threads == Lvtx_backend::dsygvd_max_threads);
static_assert(dsygvd_production_threads ==
              xlsdft_dsygvd_internal::production_threads);
static_assert(fixed_streaming_fp64_lanes ==
              Lvtx_backend::fixed_streaming_fp64_lanes);
static_assert(fixed_threads == Lvtx_backend::fixed_threads);
static_assert(fixed_configuration_error ==
              Lvtx_backend::fixed_configuration_error);
static_assert(poj_m == Lvtx_backend::poj_m);
static_assert(poj_n == Lvtx_backend::poj_n);
static_assert(poj_k == Lvtx_backend::poj_k);
static_assert(poj_lda == Lvtx_backend::poj_lda);
static_assert(poj_ldb == Lvtx_backend::poj_ldb);
static_assert(poj_ldc == Lvtx_backend::poj_ldc);
static_assert(dsyrk_n == Lvtx_backend::dsyrk_n);
static_assert(dsyrk_k == Lvtx_backend::dsyrk_k);
static_assert(dsyrk_lda == Lvtx_backend::dsyrk_lda);
static_assert(dsyrk_ldc == Lvtx_backend::dsyrk_ldc);
static_assert(rot_m == Lvtx_backend::rot_m);
static_assert(rot_n == Lvtx_backend::rot_n);
static_assert(rot_k == Lvtx_backend::rot_k);
static_assert(rot_lda == Lvtx_backend::rot_lda);
static_assert(rot_ldb == Lvtx_backend::rot_ldb);
static_assert(rot_ldc == Lvtx_backend::rot_ldc);
static_assert(sizeof(Dsygvd_phase_times) ==
              sizeof(Lvtx_backend::Dsygvd_phase_times));
static_assert(alignof(Dsygvd_phase_times) ==
              alignof(Lvtx_backend::Dsygvd_phase_times));

const char* backend_name() noexcept {
    return "kBLAS-free";
}

void initialize() noexcept {
    omp_set_dynamic(0);
    omp_set_num_threads(fixed_threads);
}

bool requires_streaming_fp64_lanes() noexcept {
    return true;
}

bool is_aligned(const void* const pointer) noexcept {
    return Lvtx_backend::is_aligned(pointer);
}

int streaming_fp64_lanes() noexcept {
    return Lvtx_backend::streaming_fp64_lanes();
}

int fixed_team_size_probe() noexcept {
    return Lvtx_backend::fixed_team_size_probe();
}

std::size_t pulay_gelsd_workspace_doubles(const int history) noexcept {
    return Lvtx_backend::pulay_gelsd_workspace_doubles(history);
}

std::size_t dsygvd_workspace_doubles(const int n) noexcept {
    return Lvtx_backend::dsygvd_workspace_doubles(n);
}

int dsygvd_profile_tile() noexcept {
    return xlsdft_dsygvd_internal::production_tile;
}

std::size_t projection_workspace_doubles() noexcept {
    return Lvtx_backend::poj_dsyrk_workspace_doubles;
}

std::size_t rotation_workspace_doubles() noexcept {
    return Lvtx_backend::rot_packed_a_doubles +
           Lvtx_backend::rot_packed_b_doubles;
}

int nloc_forward(const int nr, const int nchi, const int bands,
                 const double dv, const double* const chi, const int ldchi,
                 const double* const x, const int ldx,
                 double* const coefficients,
                 const int ldcoefficients) noexcept {
    return Lvtx_backend::nloc_forward(nr, nchi, bands, dv, chi, ldchi, x,
                                      ldx, coefficients, ldcoefficients);
}

int nloc_back(const int nr, const int nchi, const int bands,
              const double* const chi, const int ldchi,
              const double* const coefficients, const int ldcoefficients,
              double* const y, const int ldy) noexcept {
    return Lvtx_backend::nloc_back(nr, nchi, bands, chi, ldchi, coefficients,
                                   ldcoefficients, y, ldy);
}

int pulay_gram(const int ngrid, const int history, const double* const f,
               const int ldf, double* const gram, const int ldgram,
               const int nthreads) noexcept {
    return Lvtx_backend::pulay_gram(ngrid, history, f, ldf, gram, ldgram,
                                    nthreads);
}

int pulay_tmv(const int ngrid, const int history, const double* const f,
              const int ldf, const double* const vector, double* const g,
              const int nthreads) noexcept {
    return Lvtx_backend::pulay_tmv(ngrid, history, f, ldf, vector, g,
                                   nthreads);
}

int pulay_update(const int ngrid, const int history, const double* const f,
                 const int ldf, const double* const g, double* const y,
                 const int nthreads) noexcept {
    return Lvtx_backend::pulay_update(ngrid, history, f, ldf, g, y,
                                      nthreads);
}

int pulay_gelsd(const int history, double* const a, const int lda,
                double* const rhs, double* const singular_values,
                const double rcond, int* const rank, double* const workspace,
                const std::size_t workspace_doubles) noexcept {
    return Lvtx_backend::pulay_gelsd(history, a, lda, rhs, singular_values,
                                     rcond, rank, workspace,
                                     workspace_doubles);
}

int dsterf(const int n, double* const diagonal,
           double* const off_diagonal) noexcept {
    return Lvtx_backend::dsterf(n, diagonal, off_diagonal);
}

int dsygvd_upper(const int n, double* const a, const int lda,
                 double* const b, const int ldb, double* const eigenvalues,
                 double* const workspace,
                 const std::size_t workspace_doubles,
                 const int nthreads) noexcept {
    return Lvtx_backend::dsygvd_upper(n, a, lda, b, ldb, eigenvalues,
                                      workspace, workspace_doubles, nthreads);
}

int dsygvd_upper_opt(const int n, double* const a, const int lda,
                     double* const b, const int ldb,
                     double* const eigenvalues, double* const workspace,
                     const std::size_t workspace_doubles,
                     const int nthreads) noexcept {
    return Lvtx_backend::dsygvd_upper_opt(n, a, lda, b, ldb, eigenvalues,
                                          workspace, workspace_doubles,
                                          nthreads);
}

int dsygvd_upper_profiled(
    const int n, double* const a, const int lda, double* const b,
    const int ldb, double* const eigenvalues, double* const workspace,
    const std::size_t workspace_doubles, Dsygvd_phase_times* const phase_times,
    const int nthreads) noexcept {
    if (n < 0 || n > dsygvd_max_n || nthreads < 0 ||
        nthreads > dsygvd_max_threads) {
        return -2;
    }
    if (phase_times == nullptr) return -1;
    if (!is_aligned(phase_times)) return -4;
    if (n > 0) {
        if (a == nullptr || b == nullptr || eigenvalues == nullptr ||
            workspace == nullptr) {
            return -1;
        }
        if (lda < n || ldb < n) return -3;
        if (!is_aligned(a) || !is_aligned(b) || !is_aligned(eigenvalues) ||
            !is_aligned(workspace)) {
            return -4;
        }
        if (workspace_doubles < dsygvd_workspace_doubles(n)) return -6;
        Address_range a_range{};
        Address_range b_range{};
        Address_range eigenvalue_range{};
        Address_range workspace_range{};
        Address_range phase_range{};
        if (!make_range(a,
                        static_cast<std::size_t>(lda) *
                            static_cast<std::size_t>(n),
                        &a_range) ||
            !make_range(b,
                        static_cast<std::size_t>(ldb) *
                            static_cast<std::size_t>(n),
                        &b_range) ||
            !make_range(eigenvalues, static_cast<std::size_t>(n),
                        &eigenvalue_range) ||
            !make_range(workspace, workspace_doubles, &workspace_range) ||
            !make_range(reinterpret_cast<const double*>(phase_times),
                        sizeof(Dsygvd_phase_times) / sizeof(double),
                        &phase_range)) {
            return -2;
        }
        if (overlaps(phase_range, a_range) ||
            overlaps(phase_range, b_range) ||
            overlaps(phase_range, eigenvalue_range) ||
            overlaps(phase_range, workspace_range)) {
            return -5;
        }
    }
    alignas(Lvtx_backend::alignment_bytes)
        Lvtx_backend::Dsygvd_phase_times lvtx_times{};
    const int status = Lvtx_backend::dsygvd_upper_profiled(
        n, a, lda, b, ldb, eigenvalues, workspace, workspace_doubles,
        &lvtx_times, nthreads);
    if (status >= 0) {
        phase_times->cholesky_seconds = lvtx_times.cholesky_seconds;
        phase_times->standard_transform_seconds =
            lvtx_times.standard_transform_seconds;
        phase_times->tridiagonal_reduction_seconds =
            lvtx_times.tridiagonal_reduction_seconds;
        phase_times->householder_product_seconds =
            lvtx_times.householder_product_seconds;
        phase_times->tridiagonal_eigensolve_seconds =
            lvtx_times.tridiagonal_eigensolve_seconds;
        phase_times->backtransform_seconds = lvtx_times.backtransform_seconds;
        phase_times->total_seconds = lvtx_times.total_seconds;
        phase_times->tridiagonal_matvec_seconds =
            lvtx_times.tridiagonal_matvec_seconds;
        phase_times->tridiagonal_panel_correction_seconds =
            lvtx_times.tridiagonal_panel_correction_seconds;
        phase_times->tridiagonal_trailing_rank2_seconds =
            lvtx_times.tridiagonal_trailing_rank2_seconds;
    }
    return status;
}

int dgemm_poj(const double* const a, const double* const b, double* const c,
              double* const workspace,
              const std::size_t workspace_doubles) noexcept {
    return Lvtx_backend::dgemm_poj(a, b, c, workspace, workspace_doubles);
}

int dgemm_poj(const double* const a, const double* const b, double* const c,
              double* const workspace) noexcept {
    return Lvtx_backend::dgemm_poj(a, b, c, workspace);
}

int dsyrk_upper(const double* const a, double* const c,
                double* const workspace,
                const std::size_t workspace_doubles) noexcept {
    return Lvtx_backend::dsyrk_upper(a, c, workspace, workspace_doubles);
}

int rotate_raw(const double* const a, const double* const b, double* const c,
               double* const workspace,
               const std::size_t workspace_doubles) noexcept {
    if (a == nullptr || b == nullptr || c == nullptr) return -1;
    if (workspace == nullptr ||
        workspace_doubles < rotation_workspace_doubles())
        return -2;
    if (!is_aligned(a) || !is_aligned(b) || !is_aligned(c) ||
        !is_aligned(workspace))
        return rot_invalid_alignment;
    std::array<Address_range, 4> ranges{};
    if (!make_range(a, rot_a_doubles, &ranges[0]) ||
        !make_range(b, rot_b_doubles, &ranges[1]) ||
        !make_range(c, rot_c_doubles, &ranges[2]) ||
        !make_range(workspace, workspace_doubles, &ranges[3]))
        return rot_invalid_range;
    if (!pairwise_disjoint(ranges)) return rot_overlap;

    double* const packed_a = workspace;
    double* const packed_b =
        workspace + Lvtx_backend::rot_packed_a_doubles;
    int status = Lvtx_backend::dgemm_rot_pack_raw(a, b, packed_a, packed_b);
    if (status != 0) {
        return status;
    }
    status = Lvtx_backend::dgemm_rot_compute_packed(packed_a, packed_b, c);
    return status;
}

const char* operation_name(const Operation operation) noexcept {
    if (operation == Operation::dgemm_rotation) {
        return "dgemm_rotation";
    }
    return Lvtx_backend::operation_name(to_lvtx_operation(operation));
}

const char* status_diagnostic(const Operation operation,
                              const int status) noexcept {
    if (status == fixed_configuration_error) {
        return "fixed XLSDFT shape, communicator, or mode contract violated";
    }
    if (operation == Operation::dgemm_rotation && status == -2) {
        return "invalid or undersized rotation workspace";
    }
    return Lvtx_backend::status_diagnostic(to_lvtx_operation(operation),
                                            status);
}

void require_success(const Operation operation, const int status,
                     const MPI_Comm comm, const Failure_context& context) {
    if (status == 0) {
        return;
    }

    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }

    (void)comm;
    int rank = -1;
    const bool mpi_usable = mpi_initialized != 0 && mpi_finalized == 0;
    if (mpi_usable) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    char hostname[256] = "unknown";
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        std::snprintf(hostname, sizeof(hostname), "unknown");
    } else {
        hostname[sizeof(hostname) - 1U] = '\0';
    }

    std::fprintf(
        stderr,
        "LVTX backend failure: backend=%s operation=%s status=%d "
        "diagnostic=%s host=%s rank=%d omp_in_parallel=%d omp_thread=%d "
        "omp_team=%d omp_max_threads=%d label=%s m=%lld n=%lld k=%lld "
        "lda=%lld ldb=%lld ldc=%lld a=%p/a64=%d b=%p/a64=%d "
        "c=%p/a64=%d workspace=%p/a64=%d workspace_doubles=%zu\n",
        backend_name(), operation_name(operation), status,
        status_diagnostic(operation, status), hostname, rank,
        omp_in_parallel(), omp_get_thread_num(), omp_get_num_threads(),
        omp_get_max_threads(), context.label != nullptr ? context.label : "-",
        static_cast<long long>(context.m), static_cast<long long>(context.n),
        static_cast<long long>(context.k),
        static_cast<long long>(context.lda),
        static_cast<long long>(context.ldb),
        static_cast<long long>(context.ldc), const_cast<void*>(context.a),
        is_aligned(context.a) ? 1 : 0, const_cast<void*>(context.b),
        is_aligned(context.b) ? 1 : 0, const_cast<void*>(context.c),
        is_aligned(context.c) ? 1 : 0,
        const_cast<void*>(context.workspace),
        is_aligned(context.workspace) ? 1 : 0,
        context.workspace_doubles);
    std::fflush(stderr);

    if (mpi_usable) {
        const int error_code = status < 0 ? -status : status;
        MPI_Abort(MPI_COMM_WORLD,
                  error_code == 0 ? EXIT_FAILURE : error_code);
    }
    std::abort();
}

}  // namespace Xlsdft_backend
