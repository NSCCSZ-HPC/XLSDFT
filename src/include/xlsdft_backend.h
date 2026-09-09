#ifndef XLSDFT_BACKEND_H_
#define XLSDFT_BACKEND_H_

#include <cstddef>
#include <cstdint>

#include <mpi.h>

#if !defined(XLSDFT_BACKEND_FREE) || defined(XLSDFT_BACKEND_KBLAS)
#error "Only the kBLAS-free XLSDFT backend is supported"
#endif

// Backend-neutral dense/small-kernel interface used by the XLSDFT application.
// The production build links only the audited kBLAS-free implementation.
namespace Xlsdft_backend {

inline constexpr std::size_t alignment_bytes = 64U;
inline constexpr int nloc_max_bands = 8;
inline constexpr int dsterf_max_n = 101;
inline constexpr int dsygvd_max_n = 640;
inline constexpr int pulay_default_threads = 36;
inline constexpr int pulay_max_threads = 64;
inline constexpr int dsygvd_default_threads = 24;
inline constexpr int dsygvd_max_threads = 64;
inline constexpr int dsygvd_production_threads = 36;

inline constexpr int fixed_streaming_fp64_lanes = 8;
inline constexpr int fixed_threads = 36;
inline constexpr int fixed_configuration_error = -1000;

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
inline constexpr int dsyrk_n = 640;
inline constexpr int dsyrk_k = 200704;
inline constexpr int dsyrk_lda = dsyrk_k;
inline constexpr int dsyrk_ldc = dsyrk_n;
inline constexpr std::size_t dsyrk_a_doubles =
    static_cast<std::size_t>(dsyrk_n) * dsyrk_k;
inline constexpr std::size_t dsyrk_c_doubles =
    static_cast<std::size_t>(dsyrk_n) * dsyrk_n;
inline constexpr int rot_m = 200704;
inline constexpr int rot_n = 640;
inline constexpr int rot_k = 640;
inline constexpr int rot_lda = rot_m;
inline constexpr int rot_ldb = rot_k;
inline constexpr int rot_ldc = rot_m;
inline constexpr std::size_t rot_a_doubles =
    static_cast<std::size_t>(rot_m) * rot_k;
inline constexpr std::size_t rot_b_doubles =
    static_cast<std::size_t>(rot_k) * rot_n;
inline constexpr std::size_t rot_c_doubles =
    static_cast<std::size_t>(rot_m) * rot_n;
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
    dgemm_rotation,
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

const char* backend_name() noexcept;
void initialize() noexcept;
bool requires_streaming_fp64_lanes() noexcept;

bool is_aligned(const void* pointer) noexcept;
int streaming_fp64_lanes() noexcept;
int fixed_team_size_probe() noexcept;

std::size_t pulay_gelsd_workspace_doubles(int history) noexcept;
std::size_t dsygvd_workspace_doubles(int n) noexcept;
int dsygvd_profile_tile() noexcept;
std::size_t projection_workspace_doubles() noexcept;
std::size_t rotation_workspace_doubles() noexcept;

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
                double* workspace, std::size_t workspace_doubles) noexcept;

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

// Backend-neutral fused rotation entry point. The kBLAS-free implementation
// uses workspace for the packed A and B operands; another backend may ignore
// it. Callers query rotation_workspace_doubles() for the selected backend.
int rotate_raw(const double* a, const double* b, double* c,
               double* workspace, std::size_t workspace_doubles) noexcept;

const char* operation_name(Operation operation) noexcept;
const char* status_diagnostic(Operation operation, int status) noexcept;

// The communicator argument is retained at the call boundary for context,
// but failures abort MPI_COMM_WORLD so a rank-local backend error cannot leave
// peer ranks blocked in later collectives.
void require_success(Operation operation, int status, MPI_Comm comm,
                     const Failure_context& context = Failure_context{});

}  // namespace Xlsdft_backend

#endif  // XLSDFT_BACKEND_H_
