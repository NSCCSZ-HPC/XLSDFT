#include "chefsi_layout.h"

#include <cstring>
#include <omp.h>

namespace chefsi_layout {

constexpr int B_TILE = 16;

void tile_16(const size_t M, const size_t K, double* __restrict__ out,
             const double* __restrict__ in) {
  const size_t K_OUT = k_out(K);
#pragma omp parallel for collapse(2)
  for (size_t bt = 0; bt < M / B_TILE; ++bt) {
    for (size_t g = 0; g < K; ++g) {
      for (size_t bl = 0; bl < B_TILE; ++bl) {
        const size_t b = bt * B_TILE + bl;
        out[bt * K_OUT * B_TILE + g * B_TILE + bl] = in[g + b * K];
      }
    }
  }
}

void untile_16(const size_t M, const size_t K, double* __restrict__ out,
               const double* __restrict__ in) {
  untile_16_production(M, K, out, in);
}

void untile_16_ref(const size_t M, const size_t K, double* __restrict__ out,
                   const double* __restrict__ in) {
  const size_t K_OUT = k_out(K);
#pragma omp parallel for collapse(2)
  for (size_t g = 0; g < K; ++g) {
    for (size_t b = 0; b < M; ++b) {
      const size_t bt = b / B_TILE;
      const size_t bl = b % B_TILE;
      out[g + b * K] = in[bt * K_OUT * B_TILE + g * B_TILE + bl];
    }
  }
}

void tile_16t(const size_t M, const size_t K, double* __restrict__ out,
              const double* __restrict__ in) {
  const size_t K_IN = k_in(K);
  const size_t KL = K_IN / NT;
#pragma omp parallel for collapse(2)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < M / 16; ++bt) {
      for (size_t ploc = 0; tid + NT * ploc < K; ++ploc) {
        const size_t p = tid + NT * ploc;
        double* dst = out + tid * (M / 16) * KL * 16 + bt * KL * 16 + ploc * 16;
        for (size_t x = 0; x < 16; ++x) {
          dst[x] = in[p + (bt * 16 + x) * K];
        }
      }
    }
  }
  zero_sp_padding_tile16t(M, K, out);
}

void tile_32t(const size_t M, const size_t K, double* __restrict__ out,
              const double* __restrict__ in) {
  const size_t K_IN = k_in(K);
  const size_t KL = K_IN / NT;
#pragma omp parallel for collapse(2)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < M / 32; ++bt) {
      for (size_t ploc = 0; tid + NT * ploc < K; ++ploc) {
        const size_t p = tid + NT * ploc;
        double* dst =
            out + tid * (M / 32) * KL * 32 + bt * KL * 32 + ploc * 32;
        for (size_t x = 0; x < 32; ++x) {
          dst[x] = in[p + (bt * 32 + x) * K];
        }
      }
    }
  }
  zero_sp_padding_tile32t(M, K, out);
}

void tile16_to_tile16t(const size_t M, const size_t K, double* __restrict__ out,
                       const double* __restrict__ in) {
  const size_t K_IN = k_in(K);
  const size_t K_OUT = k_out(K);
  const size_t KL = K_IN / NT;
#pragma omp parallel for collapse(2)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < M / 16; ++bt) {
      for (size_t ploc = 0; tid + NT * ploc < K; ++ploc) {
        const size_t p = tid + NT * ploc;
        double* dst = out + tid * (M / 16) * KL * 16 + bt * KL * 16 + ploc * 16;
        for (size_t x = 0; x < 16; ++x) {
          dst[x] = in[bt * K_OUT * B_TILE + p * B_TILE + x];
        }
      }
    }
  }
}

void tile16t_to_tile16_sr(const size_t M, const size_t K, double* __restrict__ out,
                          const double* __restrict__ in) {
  const size_t K_IN = k_in(K);
  const size_t K_OUT = k_out(K);
  const size_t KL = K_IN / NT;
#pragma omp parallel for collapse(2)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < M / 16; ++bt) {
      for (size_t ploc = 0; tid + NT * ploc < K; ++ploc) {
        const size_t p = tid + NT * ploc;
        const double* src =
            in + tid * (M / 16) * KL * 16 + bt * KL * 16 + ploc * 16;
        for (size_t x = 0; x < 16; ++x) {
          out[bt * K_OUT * B_TILE + p * B_TILE + x] = src[x];
        }
      }
    }
  }
}

void untile_16t(const size_t M, const size_t K, double* __restrict__ out,
                const double* __restrict__ in) {
  const size_t K_IN = k_in(K);
  const size_t KL = K_IN / NT;
#pragma omp parallel for collapse(2)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < M / 16; ++bt) {
      for (size_t ploc = 0; tid + NT * ploc < K; ++ploc) {
        const size_t p = tid + NT * ploc;
        const double* src =
            in + tid * (M / 16) * KL * 16 + bt * KL * 16 + ploc * 16;
        for (size_t x = 0; x < 16; ++x) {
          const size_t b = bt * 16 + x;
          out[p + b * K] = src[x];
        }
      }
    }
  }
}

void untile_32t(const size_t M, const size_t K, double* __restrict__ out,
                const double* __restrict__ in) {
  const size_t K_IN = k_in(K);
  const size_t KL = K_IN / NT;
#pragma omp parallel for collapse(2)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < M / 32; ++bt) {
      for (size_t ploc = 0; tid + NT * ploc < K; ++ploc) {
        const size_t p = tid + NT * ploc;
        const double* src =
            in + tid * (M / 32) * KL * 32 + bt * KL * 32 + ploc * 32;
        for (size_t x = 0; x < 32; ++x) {
          const size_t b = bt * 32 + x;
          out[p + b * K] = src[x];
        }
      }
    }
  }
}

void scale_tile16_sr(double* __restrict__ packed, double a, size_t M, size_t K) {
  const size_t K_OUT = k_out(K);
#pragma omp parallel for collapse(2)
  for (size_t bt = 0; bt < M / B_TILE; ++bt) {
    for (size_t g = 0; g < K; ++g) {
      for (size_t bl = 0; bl < B_TILE; ++bl) {
        packed[bt * K_OUT * B_TILE + g * B_TILE + bl] *= a;
      }
    }
  }
}

namespace {

void zero_sp_padding_tile(const size_t M, const size_t K, const size_t band_tile,
                          double* __restrict__ packed) {
  if (k_sp_pad(K) == 0) {
    return;
  }
  const size_t KL = k_in(K) / NT;
  const size_t n_bt = M / band_tile;
#pragma omp parallel for collapse(3)
  for (size_t tid = 0; tid < NT; ++tid) {
    for (size_t bt = 0; bt < n_bt; ++bt) {
      for (size_t ploc = 0; ploc < KL; ++ploc) {
        const size_t g = tid + NT * ploc;
        if (g < K) {
          continue;
        }
        double* dst = packed + tid * n_bt * KL * band_tile + bt * KL * band_tile +
                      ploc * band_tile;
        std::memset(dst, 0, band_tile * sizeof(double));
      }
    }
  }
}

}  // namespace

void zero_sp_padding_tile16t(const size_t M, const size_t K,
                             double* __restrict__ packed) {
  zero_sp_padding_tile(M, K, 16, packed);
}

void zero_sp_padding_tile32t(const size_t M, const size_t K,
                             double* __restrict__ packed) {
  zero_sp_padding_tile(M, K, 32, packed);
}

}  // namespace chefsi_layout
