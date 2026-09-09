#pragma once

#include <cstddef>

#ifndef NT
#define NT 38
#endif

namespace chefsi_layout {

inline size_t k_sp_pad(const size_t K) { return (NT - K % NT) % NT; }

inline size_t k_sr_pad(const size_t K) {
  return (NT * 16 - K % (NT * 16)) % (NT * 16);
}

inline size_t k_in(const size_t K) { return K + k_sp_pad(K); }

inline size_t k_out(const size_t K) { return K + k_sr_pad(K); }

inline size_t wf_stride(const size_t K, const size_t M) { return k_out(K) * M; }

// Natural [M x K] in[p + b*K] -> SR tile_16 [band_tile][grid][16] (K_out stride).
void tile_16(const size_t M, const size_t K, double* __restrict__ out,
             const double* __restrict__ in);

// SR tile_16 -> natural [M x K] out[p + b*K].
void untile_16(const size_t M, const size_t K, double* __restrict__ out,
               const double* __restrict__ in);

// Reference untile_16 (scalar loops); used for verification.
void untile_16_ref(const size_t M, const size_t K, double* __restrict__ out,
                   const double* __restrict__ in);

// Optimized untile_16 (OpenMP + SME hor-load/ver-store when available).
void untile_16_opt(const size_t M, const size_t K, double* __restrict__ out,
                   const double* __restrict__ in);

// Production untile: dispatches to untile_16_opt unless CHEFSI_UNTILE_REF=1.
void untile_16_production(const size_t M, const size_t K, double* __restrict__ out,
                          const double* __restrict__ in);

// Natural -> SP tile_16t [NT][M/16][K_in/NT][16].
void tile_16t(const size_t M, const size_t K, double* __restrict__ out,
              const double* __restrict__ in);

// Natural -> SP tile_32t [NT][M/32][K_in/NT][32].
void tile_32t(const size_t M, const size_t K, double* __restrict__ out,
              const double* __restrict__ in);

// SR tile_16 [band_tile][grid][16] (K_out stride) -> SP tile_16t.
void tile16_to_tile16t(const size_t M, const size_t K, double* __restrict__ out,
                       const double* __restrict__ in);

// SP tile_16t -> SR tile_16 [band_tile][grid][16] (K_out stride).
void tile16t_to_tile16_sr(const size_t M, const size_t K, double* __restrict__ out,
                          const double* __restrict__ in);

// SP tile_16t -> natural [M x K] out[p + b*K].
void untile_16t(const size_t M, const size_t K, double* __restrict__ out,
                const double* __restrict__ in);

// SP tile_32t -> natural [M x K] out[p + b*K].
void untile_32t(const size_t M, const size_t K, double* __restrict__ out,
                const double* __restrict__ in);

void scale_tile16_sr(double* __restrict__ packed, double a, size_t M, size_t K);

// Zero SP tile slots with g >= K (k_sp_pad region). Safe no-op when K % NT == 0.
void zero_sp_padding_tile16t(size_t M, size_t K, double* __restrict__ packed);
void zero_sp_padding_tile32t(size_t M, size_t K, double* __restrict__ packed);

}  // namespace chefsi_layout
