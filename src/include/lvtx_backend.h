#ifndef XLSDFT_LVTX_BACKEND_H_
#define XLSDFT_LVTX_BACKEND_H_

#include <cstddef>
#include <cstdint>

#include <mpi.h>

namespace Lvtx_backend {

inline constexpr std::size_t alignment_bytes = 64U;
inline constexpr int nloc_max_bands = 8;
inline constexpr int dsterf_max_n = 101;
inline constexpr int dsygvd_max_n = 640;
inline constexpr int pulay_default_threads = 36;
inline constexpr int pulay_max_threads = 64;
inline constexpr int dsygvd_default_threads = 24;
inline constexpr int dsygvd_max_threads = 64;

// All fixed SME kernels require a 512-bit streaming vector length and an
// exact 36-worker OpenMP team.
inline constexpr int fixed_streaming_fp64_lanes = 8;
inline constexpr int fixed_threads = 36;
inline constexpr int fixed_configuration_error = -1000;

// Fixed POJ: C[640,640] = A[640,200704] * B[640,200704]^T.
// A and B are row-major; C is column-major. alpha=1 and beta=0.
inline constexpr int poj_m = 640;
inline constexpr int poj_n = 640;
inline constexpr int poj_k = 200704;
inline constexpr int poj_lda = poj_k;
inline constexpr int poj_ldb = poj_k;
inline constexpr int poj_ldc = poj_m;
inline constexpr std::size_t poj_a_doubles =
    static_cast<std::size_t>(poj_m) * poj_k;
inline constexpr std::size_t poj_b_doubles =
    static_cast<std::size_t>(poj_n) * poj_k;
inline constexpr std::size_t poj_c_doubles =
    static_cast<std::size_t>(poj_m) * poj_n;
inline constexpr std::size_t poj_workspace_doubles = 37748736U;
inline constexpr std::size_t poj_workspace_bytes =
    poj_workspace_doubles * sizeof(double);

// Fixed DSYRK: upper(C[640,640]) = A[640,200704] * A^T.
// A is row-major (the same bytes as column-major [200704,640]); C is
// column-major. alpha=1 and beta=0. The strict lower triangle is preserved.
inline constexpr int dsyrk_n = 640;
inline constexpr int dsyrk_k = 200704;
inline constexpr int dsyrk_lda = dsyrk_k;
inline constexpr int dsyrk_ldc = dsyrk_n;
inline constexpr std::size_t dsyrk_a_doubles =
    static_cast<std::size_t>(dsyrk_n) * dsyrk_k;
inline constexpr std::size_t dsyrk_c_doubles =
    static_cast<std::size_t>(dsyrk_n) * dsyrk_n;
inline constexpr std::size_t dsyrk_workspace_doubles = 589824U;
inline constexpr std::size_t dsyrk_workspace_bytes =
    dsyrk_workspace_doubles * sizeof(double);

// POJ and DSYRK run in distinct phases and may reuse the same aligned prefix.
inline constexpr std::size_t poj_dsyrk_workspace_doubles =
    poj_workspace_doubles;

// Fixed ROT: C[200704,640] = A[200704,640] * B[640,640]. Raw A, raw B,
// and C are column-major; alpha=1 and beta=0. The packed layouts are
// depth-major 32-row A panels and 16-column B panels.
inline constexpr int rot_m = 200704;
inline constexpr int rot_n = 640;
inline constexpr int rot_k = 640;
inline constexpr int rot_lda = rot_m;
inline constexpr int rot_ldb = rot_k;
inline constexpr int rot_ldc = rot_m;
inline constexpr int rot_mr = 32;
inline constexpr int rot_nr = 16;
inline constexpr int rot_m_tiles = rot_m / rot_mr;
inline constexpr int rot_n_tiles = rot_n / rot_nr;
inline constexpr std::size_t rot_a_doubles =
    static_cast<std::size_t>(rot_m) * rot_k;
inline constexpr std::size_t rot_b_doubles =
    static_cast<std::size_t>(rot_k) * rot_n;
inline constexpr std::size_t rot_c_doubles =
    static_cast<std::size_t>(rot_m) * rot_n;
inline constexpr std::size_t rot_packed_a_doubles = rot_a_doubles;
inline constexpr std::size_t rot_packed_b_doubles = rot_b_doubles;
inline constexpr std::size_t rot_packed_a_bytes =
    rot_packed_a_doubles * sizeof(double);
inline constexpr std::size_t rot_packed_b_bytes =
    rot_packed_b_doubles * sizeof(double);

// Adapter-only ROT validation results. Values -1 through -5 are reserved by
// the public LVTX ROT entry point and are reported operation-specifically.
inline constexpr int rot_invalid_alignment = -6;
inline constexpr int rot_overlap = -7;
inline constexpr int rot_invalid_range = -8;

struct Dsygvd_phase_times {
    double cholesky_seconds;
    double standard_transform_seconds;
    double tridiagonal_reduction_seconds;
    double householder_product_seconds;
    double tridiagonal_eigensolve_seconds;
    double backtransform_seconds;
    double total_seconds;
    double tridiagonal_matvec_seconds;
    double tridiagonal_panel_correction_seconds;
    double tridiagonal_trailing_rank2_seconds;
};

static_assert(sizeof(Dsygvd_phase_times) == 10U * sizeof(double),
              "DSYGVD phase record must contain exactly ten doubles");

enum class Operation {
    nloc_forward,
    nloc_back,
    pulay_gram,
    pulay_tmv,
    pulay_update,
    pulay_gelsd,
    dsterf,
    dsygvd_upper,
    dsygvd_upper_profiled,
    dgemm_poj,
    dsyrk_upper,
    dgemm_rot_pack,
    dgemm_rot_compute,
};

struct Failure_context {
    const char* label = nullptr;
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::int64_t k = 0;
    std::int64_t lda = 0;
    std::int64_t ldb = 0;
    std::int64_t ldc = 0;
    const void* a = nullptr;
    const void* b = nullptr;
    const void* c = nullptr;
    const void* workspace = nullptr;
    std::size_t workspace_doubles = 0U;
};

bool is_aligned(const void* pointer) noexcept;

// These probes use no private LVTX API. The team probe returns the actual
// size of a requested fixed team, or zero when called from an outer team.
int streaming_fp64_lanes() noexcept;
int fixed_team_size_probe() noexcept;

std::size_t pulay_gelsd_workspace_doubles(int history) noexcept;
std::size_t dsygvd_workspace_doubles(int n) noexcept;

int nloc_forward(int nr, int nchi, int bands, double dv,
                 const double* chi, int ldchi, const double* x, int ldx,
                 double* coefficients, int ldcoefficients) noexcept;

int nloc_back(int nr, int nchi, int bands, const double* chi, int ldchi,
              const double* coefficients, int ldcoefficients, double* y,
              int ldy) noexcept;

int pulay_gram(int ngrid, int history, const double* f, int ldf,
               double* gram, int ldgram,
               int nthreads = pulay_default_threads) noexcept;

int pulay_tmv(int ngrid, int history, const double* f, int ldf,
              const double* vector, double* g,
              int nthreads = pulay_default_threads) noexcept;

int pulay_update(int ngrid, int history, const double* f, int ldf,
                 const double* g, double* y,
                 int nthreads = pulay_default_threads) noexcept;

int pulay_gelsd(int history, double* a, int lda, double* rhs,
                double* singular_values, double rcond, int* rank,
                double* workspace,
                std::size_t workspace_doubles) noexcept;

int dsterf(int n, double* diagonal, double* off_diagonal) noexcept;

int dsygvd_upper(int n, double* a, int lda, double* b, int ldb,
                 double* eigenvalues, double* workspace,
                 std::size_t workspace_doubles,
                 int nthreads = dsygvd_default_threads) noexcept;

int dsygvd_upper_opt(int n, double* a, int lda, double* b, int ldb,
                     double* eigenvalues, double* workspace,
                     std::size_t workspace_doubles,
                     int nthreads = dsygvd_default_threads) noexcept;

int dsygvd_upper_profiled(int n, double* a, int lda, double* b, int ldb,
                          double* eigenvalues, double* workspace,
                          std::size_t workspace_doubles,
                          Dsygvd_phase_times* phase_times,
                          int nthreads = dsygvd_default_threads) noexcept;

int dgemm_poj(const double* a, const double* b, double* c,
              double* workspace, std::size_t workspace_doubles) noexcept;

int dgemm_poj(const double* a, const double* b, double* c,
              double* workspace) noexcept;

int dsyrk_upper(const double* a, double* c, double* workspace,
                std::size_t workspace_doubles) noexcept;

// Pack raw column-major ROT operands with exactly 36 OpenMP workers:
//   packed_a[(mt*rot_k+k)*rot_mr+r] =
//       a[mt*rot_mr+r+k*rot_m]
//   packed_b[(nt*rot_k+k)*rot_nr+j] =
//       b[k+(nt*rot_nr+j)*rot_k]
// No output is written unless an exact 36-worker team is formed.
int dgemm_rot_pack_raw(const double* a, const double* b, double* packed_a,
                       double* packed_b) noexcept;

int dgemm_rot_compute_packed(const double* packed_a, const double* packed_b,
                             double* c) noexcept;

const char* operation_name(Operation operation) noexcept;
const char* status_diagnostic(Operation operation, int status) noexcept;

// Return normally only for status==0. Otherwise the failing process prints
// hostname/rank/OpenMP state plus the optional shape, pointer-alignment, and
// workspace context, then aborts the supplied communicator.
void require_success(Operation operation, int status, MPI_Comm comm,
                     const Failure_context& context = Failure_context{});

}  // namespace Lvtx_backend

#endif  // XLSDFT_LVTX_BACKEND_H_
