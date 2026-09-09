#include "lvtx_backend.h"

#include "lvtx_blas_common.hpp"
#include "lvtx_dgemm_poj_omp.hpp"
#include "lvtx_dgemm_rot_omp_compute_packed.hpp"
#include "lvtx_dsterf_f64.hpp"
#include "lvtx_dsyrk_internal_omp.hpp"
#include "lvtx_nloc_back_f64.hpp"
#include "lvtx_nloc_forward_f64.hpp"
#include "lvtx_pulay_gelsd_f64.hpp"
#include "lvtx_pulay_gelsd_work_doubles.hpp"
#include "lvtx_pulay_gram_f64.hpp"
#include "lvtx_pulay_tmv_f64.hpp"
#include "lvtx_pulay_update_f64.hpp"

#include "lvtx_dsygvd_local/lvtx_dsygvd_internal_config.hpp"
#include "lvtx_dsygvd_local/lvtx_dsygvd_upper_f64.hpp"
#include "lvtx_dsygvd_local/lvtx_dsygvd_upper_profiled_f64.hpp"
#include "lvtx_dsygvd_local/lvtx_dsygvd_work_doubles.hpp"

#include <arm_sve.h>
#include <omp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <unistd.h>

namespace Lvtx_backend {
namespace {

struct Address_range {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool make_range(const double* pointer, const std::size_t doubles,
                Address_range* range) noexcept {
    if (pointer == nullptr || range == nullptr ||
        doubles > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    const std::size_t bytes = doubles * sizeof(double);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        return false;
    }
    range->begin = begin;
    range->end = begin + bytes;
    return true;
}

bool overlap(const Address_range& left, const Address_range& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <std::size_t Count>
bool pairwise_disjoint(const std::array<Address_range, Count>& ranges) noexcept {
    for (std::size_t left = 0; left < Count; ++left) {
        for (std::size_t right = left + 1U; right < Count; ++right) {
            if (overlap(ranges[left], ranges[right])) {
                return false;
            }
        }
    }
    return true;
}

static __attribute__((noinline, used)) int streaming_fp64_lanes_impl()
    __arm_streaming {
    return static_cast<int>(svcntsd());
}

const char* common_status_diagnostic(const int status) noexcept {
    switch (status) {
        case 0:
            return "success";
        case -1:
            return "null pointer";
        case -2:
            return "invalid dimension or thread count";
        case -3:
            return "invalid leading dimension";
        case -4:
            return "buffer is not 64-byte aligned";
        case -5:
            return "buffer ranges overlap";
        case -6:
            return "workspace is too small";
        case -7:
            return "call is not permitted inside an OpenMP region";
        case -8:
            return "input contains a non-finite value";
        default:
            return status > 0 ? "positive numerical failure detail"
                              : "unknown LVTX status";
    }
}

const char* poj_status_diagnostic(const int status) noexcept {
    switch (status) {
        case 0:
            return "success";
        case -1:
            return "invalid, unaligned, overlapping, or out-of-range buffer";
        case -2:
            return "invalid or undersized POJ workspace";
        case -3:
            return "unsupported streaming vector length";
        case -4:
            return "POJ cannot run inside an OpenMP region";
        case -5:
            return "OpenMP did not form the required 36-worker team";
        default:
            return "unknown POJ status";
    }
}

const char* dsyrk_status_diagnostic(const int status) noexcept {
    switch (status) {
        case 0:
            return "success";
        case -1:
            return "invalid, overlapping, or out-of-range buffer";
        case -2:
            return "buffer is not 64-byte aligned";
        case -3:
            return "DSYRK workspace is too small";
        case -5:
            return "unsupported streaming vector length";
        case -6:
            return "DSYRK cannot run inside an OpenMP region";
        case -7:
            return "OpenMP did not form the required 36-worker team";
        case -8:
            return "internal DSYRK scheduling error";
        default:
            return "unknown DSYRK status";
    }
}

const char* rot_status_diagnostic(const int status) noexcept {
    switch (status) {
        case 0:
            return "success";
        case -1:
            return "null pointer";
        case -2:
            return "invalid ROT thread request";
        case -3:
            return "ROT cannot run inside an OpenMP region";
        case -4:
            return "OpenMP did not form the required 36-worker team";
        case -5:
            return "unsupported streaming vector length";
        case rot_invalid_alignment:
            return "ROT buffer is not 64-byte aligned";
        case rot_overlap:
            return "ROT buffer ranges overlap";
        case rot_invalid_range:
            return "ROT buffer address range overflows";
        default:
            return "unknown ROT status";
    }
}

}  // namespace

static_assert(alignment_bytes == LVTX_BUFFER_ALIGNMENT,
              "LVTX alignment contract changed");
static_assert(sizeof(lvtx_dsygvd_phase_times_f64) == 10U * sizeof(double),
              "LVTX DSYGVD phase record fields changed");
static_assert(sizeof(Dsygvd_phase_times) == sizeof(lvtx_dsygvd_phase_times_f64),
              "DSYGVD phase record must match LVTX layout");
static_assert(pulay_default_threads == LVTX_PULAY_DEFAULT_THREADS,
              "LVTX Pulay team contract changed");
static_assert(pulay_max_threads == LVTX_PULAY_MAX_THREADS,
              "LVTX Pulay maximum team contract changed");
static_assert(dsygvd_default_threads == LVTX_DSYGVD_DEFAULT_THREADS,
              "LVTX DSYGVD team contract changed");
static_assert(dsygvd_max_threads == LVTX_DSYGVD_MAX_THREADS,
              "LVTX DSYGVD maximum team contract changed");
static_assert(fixed_threads == LVTX_DGEMM_POJ_OMP_THREADS,
              "LVTX POJ team contract changed");
static_assert(fixed_threads == LVTX_DSYRK_INTERNAL_OMP_THREADS,
              "LVTX DSYRK team contract changed");
static_assert(fixed_threads == LVTX_DGEMM_ROT_OMP_THREADS,
              "LVTX ROT team contract changed");
static_assert(poj_m == LVTX_DGEMM_POJ_M && poj_n == LVTX_DGEMM_POJ_N &&
                  poj_k == LVTX_DGEMM_POJ_K,
              "LVTX POJ shape changed");
static_assert(poj_a_doubles == LVTX_DGEMM_POJ_A_DOUBLES &&
                  poj_b_doubles == LVTX_DGEMM_POJ_B_DOUBLES &&
                  poj_c_doubles == LVTX_DGEMM_POJ_C_DOUBLES,
              "LVTX POJ buffer sizes changed");
static_assert(poj_workspace_doubles ==
                  LVTX_DGEMM_POJ_OMP_WORKSPACE_DOUBLES,
              "LVTX POJ workspace changed");
static_assert(poj_workspace_bytes == LVTX_DGEMM_POJ_OMP_WORKSPACE_BYTES,
              "LVTX POJ workspace byte count changed");
static_assert(dsyrk_n == LVTX_DSYRK_N && dsyrk_k == LVTX_DSYRK_K,
              "LVTX DSYRK shape changed");
static_assert(dsyrk_a_doubles == LVTX_DSYRK_A_DOUBLES &&
                  dsyrk_c_doubles == LVTX_DSYRK_C_DOUBLES,
              "LVTX DSYRK buffer sizes changed");
static_assert(dsyrk_workspace_doubles ==
                  LVTX_DSYRK_INTERNAL_OMP_WORKSPACE_DOUBLES,
              "LVTX DSYRK workspace changed");
static_assert(rot_m % rot_mr == 0 && rot_n % rot_nr == 0,
              "ROT tile geometry must divide the fixed problem");

bool is_aligned(const void* pointer) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) &
            (alignment_bytes - 1U)) == 0U;
}

int streaming_fp64_lanes() noexcept {
    return streaming_fp64_lanes_impl();
}

int fixed_team_size_probe() noexcept {
    if (omp_in_parallel() != 0) {
        return 0;
    }
    omp_set_dynamic(0);
    int actual_team = 0;
#pragma omp parallel num_threads(fixed_threads) shared(actual_team)
    {
#pragma omp single
        actual_team = omp_get_num_threads();
    }
    return actual_team;
}

std::size_t pulay_gelsd_workspace_doubles(const int history) noexcept {
    return lvtx_pulay_gelsd_work_doubles(history);
}

std::size_t dsygvd_workspace_doubles(const int n) noexcept {
    return lvtx_dsygvd_work_doubles(n);
}

int nloc_forward(const int nr, const int nchi, const int bands,
                 const double dv, const double* chi, const int ldchi,
                 const double* x, const int ldx, double* coefficients,
                 const int ldcoefficients) noexcept {
    return lvtx_nloc_forward_f64(nr, nchi, bands, dv, chi, ldchi, x, ldx,
                                 coefficients, ldcoefficients);
}

int nloc_back(const int nr, const int nchi, const int bands,
              const double* chi, const int ldchi,
              const double* coefficients, const int ldcoefficients,
              double* y, const int ldy) noexcept {
    return lvtx_nloc_back_f64(nr, nchi, bands, chi, ldchi, coefficients,
                              ldcoefficients, y, ldy);
}

int pulay_gram(const int ngrid, const int history, const double* f,
               const int ldf, double* gram, const int ldgram,
               const int nthreads) noexcept {
    return lvtx_pulay_gram_f64(ngrid, history, f, ldf, gram, ldgram,
                               nthreads);
}

int pulay_tmv(const int ngrid, const int history, const double* f,
              const int ldf, const double* vector, double* g,
              const int nthreads) noexcept {
    return lvtx_pulay_tmv_f64(ngrid, history, f, ldf, vector, g, nthreads);
}

int pulay_update(const int ngrid, const int history, const double* f,
                 const int ldf, const double* g, double* y,
                 const int nthreads) noexcept {
    return lvtx_pulay_update_f64(ngrid, history, f, ldf, g, y, nthreads);
}

int pulay_gelsd(const int history, double* a, const int lda, double* rhs,
                double* singular_values, const double rcond, int* rank,
                double* workspace,
                const std::size_t workspace_doubles) noexcept {
    return lvtx_pulay_gelsd_f64(history, a, lda, rhs, singular_values, rcond,
                                rank, workspace, workspace_doubles);
}

int dsterf(const int n, double* diagonal, double* off_diagonal) noexcept {
    return lvtx_dsterf_f64(n, diagonal, off_diagonal);
}

int dsygvd_upper(const int n, double* a, const int lda, double* b,
                 const int ldb, double* eigenvalues, double* workspace,
                 const std::size_t workspace_doubles,
                 const int nthreads) noexcept {
    return lvtx_dsygvd_upper_f64(n, a, lda, b, ldb, eigenvalues, workspace,
                                 workspace_doubles, nthreads);
}

int dsygvd_upper_opt(const int n, double* a, const int lda, double* b,
                     const int ldb, double* eigenvalues, double* workspace,
                     const std::size_t workspace_doubles,
                     const int nthreads) noexcept {
    return lvtx_dsygvd_upper_f64_opt(n, a, lda, b, ldb, eigenvalues, workspace,
                                     workspace_doubles, nthreads);
}

static int dsygvd_upper_profiled_impl(
    const int n, double* a, const int lda, double* b, const int ldb,
    double* eigenvalues, double* workspace,
    const std::size_t workspace_doubles, Dsygvd_phase_times* phase_times,
    const int nthreads) noexcept {
    if (n < 0 || n > dsygvd_max_n || nthreads < 0 ||
        nthreads > dsygvd_max_threads) {
        return LVTX_ERR_DIM;
    }
    if (phase_times == nullptr) {
        return LVTX_ERR_NULL;
    }
    if (!is_aligned(phase_times)) {
        return LVTX_ERR_ALIGN;
    }
    if (n > 0) {
        if (a == nullptr || b == nullptr || eigenvalues == nullptr ||
            workspace == nullptr) {
            return LVTX_ERR_NULL;
        }
        if (lda < n || ldb < n) {
            return LVTX_ERR_LD;
        }
        if (!is_aligned(a) || !is_aligned(b) ||
            !is_aligned(eigenvalues) || !is_aligned(workspace)) {
            return LVTX_ERR_ALIGN;
        }
        if (workspace_doubles < dsygvd_workspace_doubles(n)) {
            return LVTX_ERR_WORKSPACE;
        }

        Address_range a_range{};
        Address_range b_range{};
        Address_range eigenvalue_range{};
        Address_range workspace_range{};
        Address_range phase_range{};
        if (!make_range(a, static_cast<std::size_t>(lda) * n, &a_range) ||
            !make_range(b, static_cast<std::size_t>(ldb) * n, &b_range) ||
            !make_range(eigenvalues, static_cast<std::size_t>(n),
                        &eigenvalue_range) ||
            !make_range(workspace, workspace_doubles, &workspace_range) ||
            !make_range(reinterpret_cast<const double*>(phase_times),
                        sizeof(Dsygvd_phase_times) / sizeof(double),
                        &phase_range)) {
            return LVTX_ERR_DIM;
        }
        if (overlap(phase_range, a_range) || overlap(phase_range, b_range) ||
            overlap(phase_range, eigenvalue_range) ||
            overlap(phase_range, workspace_range)) {
            return LVTX_ERR_OVERLAP;
        }
    }

    alignas(alignment_bytes) lvtx_dsygvd_phase_times_f64 lvtx_times{};
    const int status = lvtx_dsygvd_upper_profiled_f64(
        n, a, lda, b, ldb, eigenvalues, workspace, workspace_doubles, nthreads,
        &lvtx_times);
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

int dsygvd_upper_profiled(const int n, double* a, const int lda, double* b,
                          const int ldb, double* eigenvalues,
                          double* workspace,
                          const std::size_t workspace_doubles,
                          Dsygvd_phase_times* phase_times,
                          const int nthreads) noexcept {
    return dsygvd_upper_profiled_impl(
        n, a, lda, b, ldb, eigenvalues, workspace, workspace_doubles,
        phase_times, nthreads);
}

int dgemm_poj(const double* a, const double* b, double* c,
              double* workspace,
              const std::size_t workspace_doubles) noexcept {
    return lvtx_dgemm_poj_omp(a, b, c, workspace, workspace_doubles);
}

int dgemm_poj(const double* a, const double* b, double* c,
              double* workspace) noexcept {
    return lvtx_dgemm_poj_omp(a, b, c, workspace);
}

int dsyrk_upper(const double* a, double* c, double* workspace,
                const std::size_t workspace_doubles) noexcept {
    return lvtx_dsyrk_internal_omp(a, c, workspace, workspace_doubles);
}

int dgemm_rot_pack_raw(const double* a, const double* b, double* packed_a,
                       double* packed_b) noexcept {
    if (a == nullptr || b == nullptr || packed_a == nullptr ||
        packed_b == nullptr) {
        return LVTX_DGEMM_ROT_OMP_NULL_ARGUMENT;
    }
    if (!is_aligned(a) || !is_aligned(b) || !is_aligned(packed_a) ||
        !is_aligned(packed_b)) {
        return rot_invalid_alignment;
    }

    std::array<Address_range, 4> ranges{};
    if (!make_range(a, rot_a_doubles, &ranges[0]) ||
        !make_range(b, rot_b_doubles, &ranges[1]) ||
        !make_range(packed_a, rot_packed_a_doubles, &ranges[2]) ||
        !make_range(packed_b, rot_packed_b_doubles, &ranges[3])) {
        return rot_invalid_range;
    }
    if (!pairwise_disjoint(ranges)) {
        return rot_overlap;
    }
    if (streaming_fp64_lanes() != fixed_streaming_fp64_lanes) {
        return LVTX_DGEMM_ROT_OMP_UNSUPPORTED_VECTOR_LENGTH;
    }
    if (omp_in_parallel() != 0) {
        return LVTX_DGEMM_ROT_OMP_NESTED;
    }

    omp_set_dynamic(0);
    int actual_team = 0;
#pragma omp parallel num_threads(fixed_threads) shared(actual_team)
    {
        const int thread = omp_get_thread_num();
        const int team = omp_get_num_threads();
#pragma omp single
        actual_team = team;

        if (team == fixed_threads) {
            const int mt_base = rot_m_tiles / team;
            const int mt_extra = rot_m_tiles % team;
            const int mt_begin =
                thread * mt_base + (thread < mt_extra ? thread : mt_extra);
            const int mt_end =
                mt_begin + mt_base + (thread < mt_extra ? 1 : 0);
            for (int mt = mt_begin; mt < mt_end; ++mt) {
                for (int k = 0; k < rot_k; ++k) {
                    for (int row = 0; row < rot_mr; ++row) {
                        packed_a[(static_cast<std::size_t>(mt) * rot_k + k) *
                                     rot_mr +
                                 row] =
                            a[static_cast<std::size_t>(mt * rot_mr + row) +
                              static_cast<std::size_t>(k) * rot_m];
                    }
                }
            }

            const int nt_begin = rot_n_tiles * thread / team;
            const int nt_end = rot_n_tiles * (thread + 1) / team;
            for (int nt = nt_begin; nt < nt_end; ++nt) {
                for (int k = 0; k < rot_k; ++k) {
                    for (int column = 0; column < rot_nr; ++column) {
                        packed_b[(static_cast<std::size_t>(nt) * rot_k + k) *
                                     rot_nr +
                                 column] =
                            b[static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(nt * rot_nr + column) *
                                  rot_k];
                    }
                }
            }
        }
    }
    return actual_team == fixed_threads ? LVTX_DGEMM_ROT_OMP_SUCCESS
                                        : LVTX_DGEMM_ROT_OMP_TEAM_MISMATCH;
}

int dgemm_rot_compute_packed(const double* packed_a, const double* packed_b,
                             double* c) noexcept {
    if (packed_a == nullptr || packed_b == nullptr || c == nullptr) {
        return LVTX_DGEMM_ROT_OMP_NULL_ARGUMENT;
    }
    if (!is_aligned(packed_a) || !is_aligned(packed_b) || !is_aligned(c)) {
        return rot_invalid_alignment;
    }

    std::array<Address_range, 3> ranges{};
    if (!make_range(packed_a, rot_packed_a_doubles, &ranges[0]) ||
        !make_range(packed_b, rot_packed_b_doubles, &ranges[1]) ||
        !make_range(c, rot_c_doubles, &ranges[2])) {
        return rot_invalid_range;
    }
    if (!pairwise_disjoint(ranges)) {
        return rot_overlap;
    }
    return lvtx_dgemm_rot_omp_compute_packed(packed_a, packed_b, c);
}

const char* operation_name(const Operation operation) noexcept {
    switch (operation) {
        case Operation::nloc_forward:
            return "nloc_forward";
        case Operation::nloc_back:
            return "nloc_back";
        case Operation::pulay_gram:
            return "pulay_gram";
        case Operation::pulay_tmv:
            return "pulay_tmv";
        case Operation::pulay_update:
            return "pulay_update";
        case Operation::pulay_gelsd:
            return "pulay_gelsd";
        case Operation::dsterf:
            return "dsterf";
        case Operation::dsygvd_upper:
            return "dsygvd_upper";
        case Operation::dsygvd_upper_profiled:
            return "dsygvd_upper_profiled";
        case Operation::dgemm_poj:
            return "dgemm_poj";
        case Operation::dsyrk_upper:
            return "dsyrk_upper";
        case Operation::dgemm_rot_pack:
            return "dgemm_rot_pack";
        case Operation::dgemm_rot_compute:
            return "dgemm_rot_compute";
    }
    return "unknown_operation";
}

const char* status_diagnostic(const Operation operation,
                              const int status) noexcept {
    if (status == fixed_configuration_error) {
        return "fixed LVTX shape, communicator, or mode contract violated";
    }
    switch (operation) {
        case Operation::dgemm_poj:
            return poj_status_diagnostic(status);
        case Operation::dsyrk_upper:
            return dsyrk_status_diagnostic(status);
        case Operation::dgemm_rot_pack:
        case Operation::dgemm_rot_compute:
            return rot_status_diagnostic(status);
        case Operation::pulay_gelsd:
            if (status > 0) {
                return "unconverged Pulay Jacobi column pairs";
            }
            return common_status_diagnostic(status);
        case Operation::dsterf:
            if (status > 0) {
                return "unresolved tridiagonal off-diagonal entries";
            }
            return common_status_diagnostic(status);
        case Operation::dsygvd_upper:
        case Operation::dsygvd_upper_profiled:
            if (status ==
                xlsdft_dsygvd_internal::replica_mismatch_status) {
                return "DSYGVD replicated QL worker state mismatch";
            }
            if (status > 0) {
                return "DSYGVD nonconvergence or non-SPD leading minor";
            }
            return common_status_diagnostic(status);
        case Operation::nloc_forward:
        case Operation::nloc_back:
        case Operation::pulay_gram:
        case Operation::pulay_tmv:
        case Operation::pulay_update:
            return common_status_diagnostic(status);
    }
    return "unknown operation or status";
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

    const int in_parallel = omp_in_parallel();
    const int omp_thread = omp_get_thread_num();
    const int omp_team = omp_get_num_threads();
    const int omp_max_threads = omp_get_max_threads();
    std::fprintf(
        stderr,
        "LVTX backend failure: operation=%s status=%d diagnostic=%s "
        "host=%s rank=%d omp_in_parallel=%d omp_thread=%d omp_team=%d "
        "omp_max_threads=%d label=%s m=%lld n=%lld k=%lld lda=%lld "
        "ldb=%lld ldc=%lld a=%p/a64=%d b=%p/a64=%d c=%p/a64=%d "
        "workspace=%p/a64=%d workspace_doubles=%zu\n",
        operation_name(operation), status,
        status_diagnostic(operation, status), hostname, rank, in_parallel,
        omp_thread, omp_team, omp_max_threads,
        context.label != nullptr ? context.label : "-",
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

}  // namespace Lvtx_backend
