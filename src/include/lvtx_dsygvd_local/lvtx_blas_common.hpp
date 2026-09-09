#ifndef LVTX_BLAS_COMMON_HPP_
#define LVTX_BLAS_COMMON_HPP_

#include <cstddef>

// Common return codes. Numerical algorithms may additionally return a
// positive value to describe non-convergence or a non-SPD leading minor.
typedef enum lvtx_status {
  LVTX_SUCCESS = 0,
  LVTX_ERR_NULL = -1,
  LVTX_ERR_DIM = -2,
  LVTX_ERR_LD = -3,
  LVTX_ERR_ALIGN = -4,
  LVTX_ERR_OVERLAP = -5,
  LVTX_ERR_WORKSPACE = -6,
  LVTX_ERR_NESTED_OMP = -7,
  LVTX_ERR_NONFINITE = -8,
  LVTX_ERR_UNSUPPORTED_VL = -9
} lvtx_status;

enum {
  LVTX_BUFFER_ALIGNMENT = 64,
  LVTX_PULAY_DEFAULT_THREADS = 36,
  LVTX_PULAY_MAX_THREADS = 64,
  LVTX_DSYGVD_DEFAULT_THREADS = 24,
  LVTX_DSYGVD_MAX_THREADS = 64
};

typedef struct lvtx_dsygvd_phase_times_f64 {
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
} lvtx_dsygvd_phase_times_f64;

static_assert(sizeof(lvtx_dsygvd_phase_times_f64) == 10U * sizeof(double),
              "phase timing record must contain exactly ten doubles");

typedef struct lvtx_dsygvd_tridiag_kernel_times_f64 {
  double matvec_seconds;
  double panel_correction_seconds;
  double trailing_rank2_seconds;
} lvtx_dsygvd_tridiag_kernel_times_f64;

#endif  // LVTX_BLAS_COMMON_HPP_
