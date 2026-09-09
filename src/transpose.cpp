#include <omp.h>

#include <cassert>
#include <cstring>

#include "chefsi_layout.h"
#include "sme.hpp"

namespace transpose_opt {

#ifndef NT
#define NT 38
#endif

inline size_t k_in_pad(const size_t K) { return (16 - K % 16) % 16; }

inline size_t k_out_pad(const size_t K) {
  return (NT * 16 - K % (NT * 16)) % (NT * 16);
}

inline size_t compute_k_in(const size_t K) { return K + k_in_pad(K); }

inline size_t compute_k_out(const size_t K) { return K + k_out_pad(K); }

__arm_locally_streaming __arm_new("za") void transpose_16t_to_tile16_worker(
    const size_t M, const size_t K_IN, const size_t K_OUT,
    double* __restrict__ out, const double* __restrict__ in,
    const size_t worker, const size_t nworkers) {
  (void)K_OUT;
  const size_t KLOCAL_IN = K_IN / NT;

  const size_t GBLKS_IN = (K_IN + 15) / 16;

  const svbool_t pg = svptrue_b64();

  for (size_t gb = worker; gb < GBLKS_IN; gb += nworkers) {
    const size_t g0 = gb * 16;

    for (size_t b0 = 0; b0 < M; b0 += 32) {
      const bool partial = (g0 + 16 > K_IN);

      if (partial) {
        svzero();
      }

      const size_t b16_lo = b0 / 16;
      const size_t b16_hi = b16_lo + 1;

#pragma clang loop unroll(disable)
      for (size_t r = 0; r < 8; ++r) {
        const size_t g = g0 + r;

        if (g >= K_IN) continue;

        const size_t tid = g % NT;
        const size_t ploc = g / NT;

        const double* lo = in + tid * (M / 16) * KLOCAL_IN * 16 +
                           b16_lo * KLOCAL_IN * 16 + ploc * 16;

        const double* hi = in + tid * (M / 16) * KLOCAL_IN * 16 +
                           b16_hi * KLOCAL_IN * 16 + ploc * 16;

        svld1_hor_za64(0, r, pg, lo + 0);
        svld1_hor_za64(1, r, pg, lo + 8);

        svld1_hor_za64(2, r, pg, hi + 0);
        svld1_hor_za64(3, r, pg, hi + 8);
      }

#pragma clang loop unroll(disable)
      for (size_t r = 8; r < 16; ++r) {
        const size_t g = g0 + r;

        if (g >= K_IN) continue;

        const size_t tid = g % NT;
        const size_t ploc = g / NT;

        const double* lo = in + tid * (M / 16) * KLOCAL_IN * 16 +
                           b16_lo * KLOCAL_IN * 16 + ploc * 16;

        const double* hi = in + tid * (M / 16) * KLOCAL_IN * 16 +
                           b16_hi * KLOCAL_IN * 16 + ploc * 16;

        const size_t s = r - 8;

        svld1_hor_za64(4, s, pg, lo + 0);
        svld1_hor_za64(5, s, pg, lo + 8);

        svld1_hor_za64(6, s, pg, hi + 0);
        svld1_hor_za64(7, s, pg, hi + 8);
      }

      double* out_base = out + gb * M * 16 + b0 * 16;

#pragma clang loop unroll(disable)
      for (size_t s = 0; s < 8; ++s) {
        svst1_ver_za64(0, s, pg, out_base + (s + 0) * 16 + 0);
        svst1_ver_za64(4, s, pg, out_base + (s + 0) * 16 + 8);

        svst1_ver_za64(1, s, pg, out_base + (s + 8) * 16 + 0);
        svst1_ver_za64(5, s, pg, out_base + (s + 8) * 16 + 8);

        svst1_ver_za64(2, s, pg, out_base + (s + 16) * 16 + 0);
        svst1_ver_za64(6, s, pg, out_base + (s + 16) * 16 + 8);

        svst1_ver_za64(3, s, pg, out_base + (s + 24) * 16 + 0);
        svst1_ver_za64(7, s, pg, out_base + (s + 24) * 16 + 8);
      }
    }
  }
}

}  // namespace transpose_opt

void transpose_16t_to_16(const size_t M, const size_t K,
                         double* __restrict__ out,
                         const double* __restrict__ in) {
  const size_t k_in = chefsi_layout::k_in(K);
  const size_t k_out = chefsi_layout::k_out(K);

#pragma omp parallel
  {
    const size_t worker = omp_get_thread_num();
    const size_t nworkers = omp_get_num_threads();
    transpose_opt::transpose_16t_to_tile16_worker(M, k_in, k_out, out, in,
                                                  worker, nworkers);
  }

  const size_t first_zero_gb = (k_in + 15) / 16;
  const size_t total_gb = k_out / 16;
  if (first_zero_gb < total_gb) {
    std::memset(out + first_zero_gb * M * 16, 0,
                (total_gb - first_zero_gb) * M * 16 * sizeof(double));
  }
}

namespace transpose_square_opt {

constexpr size_t TILE = 16;

// q is viewed here simply as a raw n x n array with row stride n.
// In-place raw transpose converts:
//
//   original column-major Q
//
// into
//
//   row-major Q
//
// because these are the same physical permutation for a square matrix.
__arm_locally_streaming __arm_new("za") void transpose_worker(
    double* q, const size_t n, const size_t worker, const size_t nworkers) {
  const size_t nt = n / TILE;
  const svbool_t pg = svptrue_b64();

  // Upper triangle including diagonal:
  //
  //   (0,0) (0,1) ... (0,T-1)
  //         (1,1) ... (1,T-1)
  //                 ...
  //
  // Split this triangular task space approximately evenly between threads.
  const size_t ntasks = nt * (nt + 1) / 2;

  const size_t task_begin = ntasks * worker / nworkers;
  const size_t task_end = ntasks * (worker + 1) / nworkers;

  // Convert task_begin into (ti, tj).
  size_t ti = 0;
  size_t rem = task_begin;

  while (rem >= nt - ti) {
    rem -= nt - ti;
    ++ti;
  }

  size_t tj = ti + rem;

  for (size_t task = task_begin; task < task_end; ++task) {
    if (ti == tj) {
      // ---------------------------------------------------------------
      // Diagonal 16x16 tile.
      //
      // Load complete tile horizontally into:
      //
      //   ZA0 ZA1
      //   ZA4 ZA5
      //
      // then store vertically back to the same location.
      // ---------------------------------------------------------------

      double* tile = q + (ti * TILE) * n + ti * TILE;

#pragma clang loop unroll(full)
      for (size_t s = 0; s < 8; ++s) {
        const double* row0 = tile + s * n;
        const double* row1 = tile + (s + 8) * n;

        svld1_hor_za64(0, s, pg, row0 + 0);
        svld1_hor_za64(1, s, pg, row0 + 8);

        svld1_hor_za64(4, s, pg, row1 + 0);
        svld1_hor_za64(5, s, pg, row1 + 8);
      }

#pragma clang loop unroll(full)
      for (size_t s = 0; s < 8; ++s) {
        double* row0 = tile + s * n;
        double* row1 = tile + (s + 8) * n;

        // Columns 0..7 -> rows 0..7.
        svst1_ver_za64(0, s, pg, row0 + 0);
        svst1_ver_za64(4, s, pg, row0 + 8);

        // Columns 8..15 -> rows 8..15.
        svst1_ver_za64(1, s, pg, row1 + 0);
        svst1_ver_za64(5, s, pg, row1 + 8);
      }

    } else {
      // ---------------------------------------------------------------
      // Off-diagonal pair:
      //
      //        upper = (ti,tj)
      //        lower = (tj,ti)
      //
      // Load both BEFORE writing anything:
      //
      //   upper -> ZA0,1,4,5
      //   lower -> ZA2,3,6,7
      //
      // Then:
      //
      //   lower <- transpose(upper)
      //   upper <- transpose(lower)
      //
      // ---------------------------------------------------------------

      double* upper = q + (ti * TILE) * n + tj * TILE;
      double* lower = q + (tj * TILE) * n + ti * TILE;

#pragma clang loop unroll(full)
      for (size_t s = 0; s < 8; ++s) {
        const double* u0 = upper + s * n;
        const double* u1 = upper + (s + 8) * n;

        const double* l0 = lower + s * n;
        const double* l1 = lower + (s + 8) * n;

        // Upper tile.
        svld1_hor_za64(0, s, pg, u0 + 0);
        svld1_hor_za64(1, s, pg, u0 + 8);
        svld1_hor_za64(4, s, pg, u1 + 0);
        svld1_hor_za64(5, s, pg, u1 + 8);

        // Lower tile.
        svld1_hor_za64(2, s, pg, l0 + 0);
        svld1_hor_za64(3, s, pg, l0 + 8);
        svld1_hor_za64(6, s, pg, l1 + 0);
        svld1_hor_za64(7, s, pg, l1 + 8);
      }

      // ---------------------------------------------------------------
      // lower <- upper^T
      // ---------------------------------------------------------------

#pragma clang loop unroll(full)
      for (size_t s = 0; s < 8; ++s) {
        double* l0 = lower + s * n;
        double* l1 = lower + (s + 8) * n;

        svst1_ver_za64(0, s, pg, l0 + 0);
        svst1_ver_za64(4, s, pg, l0 + 8);

        svst1_ver_za64(1, s, pg, l1 + 0);
        svst1_ver_za64(5, s, pg, l1 + 8);
      }

      // ---------------------------------------------------------------
      // upper <- lower^T
      //
      // This is still the OLD lower tile because it lives in ZA.
      // ---------------------------------------------------------------

#pragma clang loop unroll(full)
      for (size_t s = 0; s < 8; ++s) {
        double* u0 = upper + s * n;
        double* u1 = upper + (s + 8) * n;

        svst1_ver_za64(2, s, pg, u0 + 0);
        svst1_ver_za64(6, s, pg, u0 + 8);

        svst1_ver_za64(3, s, pg, u1 + 0);
        svst1_ver_za64(7, s, pg, u1 + 8);
      }
    }

    // Advance triangular coordinates.
    ++tj;
    if (tj == nt) {
      ++ti;
      tj = ti;
    }
  }
}

}  // namespace transpose_square_opt

void transpose_square_colmajor_to_rowmajor(double* q, const size_t n) {
  assert(q != nullptr);
  assert(n % 16 == 0);

#pragma omp parallel
  {
    const size_t worker = omp_get_thread_num();
    const size_t nworkers = omp_get_num_threads();

    transpose_square_opt::transpose_worker(q, n, worker, nworkers);
  }
}